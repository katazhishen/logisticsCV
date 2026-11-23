#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比赛读码核心模块：预处理阶梯 → 解码 → 对齐（透视变换）→ 任务码解析。

设计要点（全部来自教程 zbar二维码从零到赛场 的实测结论）：
- 解码器：pyzbar（zbar）为主；全阶梯都失败时再试 OpenCV QRCodeDetector 兜底。
- 预处理阶梯按『实测耗时从便宜到贵』排序，找到即停；带单帧时间预算 budget_ms。
- 对齐：用 zbar polygon 的四个真角（数据区四角）做透视变换，把码『摊平』成
  240×240 的正视图 + OTSU 二值图 —— 这就是窗口里『定帧 + 变换图』的来源。
- 任务码解析：NFKC 归一化 → 正则 → 语义校验（组内颜色/位置不重复）。
- 距离估算：d = f * W / w，f=1475px（第 12 课标定），W=80mm，w=码边长像素。
"""
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# ── zbar 引入：系统装了 libzbar0 就直接用；没装则借教程目录的 vendor 兜底 ──
try:
    from pyzbar.pyzbar import decode as _zbar_decode
except ImportError:                                        # 罕见：conda 环境缺 pyzbar
    _TUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "zbar二维码从零到赛场", "code")
    if os.path.isdir(_TUT):
        sys.path.insert(0, _TUT)
        import zbar_bootstrap
        zbar_bootstrap.ensure()
    from pyzbar.pyzbar import decode as _zbar_decode

WARP_SIZE = 240            # 对齐后输出图边长（正方形）
COLOR_NAME = {1: "红", 2: "黄", 3: "蓝", 4: "绿", 5: "黑", 6: "浅蓝"}
TASK_RE = re.compile(r"^[1-6]{3}(?:\+[1-6]{3}){3}$")
FOCAL_PX = 1475.0          # 第 12 课标定：d0=500mm, w0=236px → f = w0*d0/W
CODE_W_MM = 80.0           # 比赛规格：码 80×80mm


# ────────────────────────── 任务码解析 ──────────────────────────
class TaskCodeError(ValueError):
    """任务码不合法（附『错在哪』的中文说明）。"""


def parse_taskcode(raw: str) -> dict:
    """把 '156+123+516+231' 解析成结构化任务单；不合法就抛 TaskCodeError。"""
    s = unicodedata.normalize("NFKC", raw)     # 全角→半角
    s = re.sub(r"\s+", "", s)                  # 去空白
    groups = s.split("+")
    if len(groups) != 4:
        raise TaskCodeError(f"应有 4 组三位数，实际分了 {len(groups)} 组")
    for gi, g in enumerate(groups, 1):
        bad = [ch for ch in g if ch not in "123456"]
        if bad:
            raise TaskCodeError(f"第 {gi} 组『{g}』里有非法字符 {bad}（只能 1~6）")
    if not TASK_RE.match(s):
        raise TaskCodeError("格式不对：应为 4 组三位数、用 + 连接（例 156+123+516+231）")
    for gi, g in ((1, groups[0]), (3, groups[2])):   # 颜色组不许重复
        if len(set(g)) != 3:
            raise TaskCodeError(f"第 {gi} 组『{g}』颜色重复（三个物料颜色应互不相同）")
    for gi, g in ((2, groups[1]), (4, groups[3])):   # 位置组不许重复
        if len(set(g)) != 3:
            raise TaskCodeError(f"第 {gi} 组『{g}』位置重复（三个位置应互不相同）")

    def colors(g):
        return [COLOR_NAME[int(c)] for c in g]

    return {
        "raw": s,
        "batch1": {"colors": colors(groups[0]), "positions": [int(c) for c in groups[1]]},
        "batch2": {"colors": colors(groups[2]), "positions": [int(c) for c in groups[3]]},
        "summary": (
            f"第一批：颜色顺序 {'-'.join(colors(groups[0]))}，位置 {'-'.join(groups[1])}\n"
            f"第二批：颜色顺序 {'-'.join(colors(groups[2]))}，位置 {'-'.join(groups[3])}"
        ),
    }


# ────────────────────────── 读码结果 ──────────────────────────
@dataclass
class ReadResult:
    text: str                       # 原始字符串（bytes→str）
    type: str                       # QRCODE / CODE128 ...
    level: str                      # 命中的预处理级别名
    ms: float                       # 本帧解码耗时（毫秒）
    polygon: np.ndarray             # 数据区四角 (4,2)，已排成 左上/右上/右下/左下
    center: tuple                   # 码中心 (cx, cy)
    side_px: float                  # 数据区平均边长（像素）
    per_module_px: float            # 每格像素 ≈ side_px / 21（按 version1）
    warped: np.ndarray              # 对齐后的码（WARP_SIZE×WARP_SIZE 灰度）
    warped_bin: np.ndarray          # 对齐后 + OTSU 二值
    distance_mm: float              # 估算距离（f*W/w）
    task: Optional[dict] = None     # parse_taskcode 成功的结构化任务单
    task_error: Optional[str] = None  # 不是合法任务码时的原因


# ────────────────────────── 预处理阶梯 ──────────────────────────
def _gray(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr.ndim == 2:
        return frame_bgr
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def _lv_raw(g):
    return g


def _lv_median(g):
    return cv2.medianBlur(g, 3)                    # 椒盐噪声的专用药


def _lv_otsu(g):
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def _lv_clahe(g):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(g)


def _lv_invert(g):
    return 255 - g                                 # 浅码深底的专用药


def _lv_center2x(g):
    h, w = g.shape
    crop = g[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    return cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)


def _lv_up2(g):
    return cv2.resize(g, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)


def _lv_up4(g):
    return cv2.resize(g, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)


def _lv_up4_cubic(g):
    return cv2.resize(g, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)


# 顺序就是第 10 课实测的顺序：便宜的在前，放大 4x 的两级放最后
PIPELINE = [
    ("原图", _lv_raw),
    ("中值", _lv_median),
    ("OTSU", _lv_otsu),
    ("CLAHE", _lv_clahe),
    ("反色", _lv_invert),
    ("中心2x", _lv_center2x),
    ("放大2x", _lv_up2),
    ("放大4x", _lv_up4),
    ("三次4x", _lv_up4_cubic),
]


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """把任意顺序的 4 个角排成 [左上, 右上, 右下, 左下]。"""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()               # y - x
    tl = pts[np.argmin(s)]                         # x+y 最小 → 左上
    br = pts[np.argmax(s)]                         # x+y 最大 → 右下
    tr = pts[np.argmin(d)]                         # y-x 最小 → 右上
    bl = pts[np.argmax(d)]                         # y-x 最大 → 左下
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _align(gray: np.ndarray, polygon) -> tuple:
    """用数据区四角做透视变换，返回 (正视图, 二值图, side_px)。"""
    corners = _order_corners(polygon)
    dst = np.array([[0, 0], [WARP_SIZE - 1, 0],
                    [WARP_SIZE - 1, WARP_SIZE - 1], [0, WARP_SIZE - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(gray, M, (WARP_SIZE, WARP_SIZE))
    _, binimg = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
    return warped, binimg, float(np.mean(edges))


class RaceQRReader:
    """比赛读码器：一帧进去，一个 ReadResult（或 None）出来。"""

    def __init__(self, budget_ms: float = 40.0, use_opencv_fallback: bool = True):
        self.budget_ms = budget_ms
        self.use_opencv_fallback = use_opencv_fallback
        self.det = cv2.QRCodeDetector()
        self.stats = Counter()                     # 各级别命中次数
        self.frames = 0
        self.hits = 0
        self.skipped = 0                           # 因超预算被放弃的帧数

    # ---- 单级别尝试：解码成功就立刻对齐并返回 ----
    def _try(self, gray: np.ndarray, level: str, t0: float) -> Optional[ReadResult]:
        res = _zbar_decode(gray)
        if not res:
            return None
        r = res[0]                                 # 比赛场景一次只看一块码板
        poly = np.array(r.polygon, dtype=np.float32)
        if len(poly) != 4:
            return None                            # 一维码 5 点等，不在本模块范围
        warped, binimg, side = _align(gray, poly)
        text = r.data.decode("utf-8", "replace")
        task, terr = None, None
        try:
            task = parse_taskcode(text)
        except TaskCodeError as e:
            terr = str(e)
        per_module = side / 21.0
        out = ReadResult(
            text=text, type=str(r.type), level=level,
            ms=(time.perf_counter() - t0) * 1000.0,
            polygon=poly,
            center=tuple(float(v) for v in np.mean(poly, axis=0)),
            side_px=side, per_module_px=per_module,
            warped=warped, warped_bin=binimg,
            distance_mm=FOCAL_PX * CODE_W_MM / max(side, 1e-6),
            task=task, task_error=terr,
        )
        self.stats[level] += 1
        return out

    # ---- 主入口 ----
    def read(self, frame_bgr: np.ndarray) -> Optional[ReadResult]:
        self.frames += 1
        t0 = time.perf_counter()
        gray = _gray(frame_bgr)

        for name, fn in PIPELINE:                  # 找到即停
            if self.budget_ms and (time.perf_counter() - t0) * 1000.0 > self.budget_ms:
                self.skipped += 1
                return None                        # 超预算：宁可不读也不能拖死主循环
            out = self._try(fn(gray), name, t0)
            if out:
                self.hits += 1
                return out

        # 兜底：OpenCV 自己的检测器（对 zbar 读不出的图偶尔有效）
        if self.use_opencv_fallback:
            if self.budget_ms and (time.perf_counter() - t0) * 1000.0 > self.budget_ms:
                self.skipped += 1
                return None
            try:
                text, pts, _ = self.det.detectAndDecode(gray)
            except cv2.error:
                text, pts = "", None
            if text and pts is not None:
                poly = pts.reshape(-1, 2).astype(np.float32)
                warped, binimg, side = _align(gray, poly)
                task, terr = None, None
                try:
                    task = parse_taskcode(text)
                except TaskCodeError as e:
                    terr = str(e)
                self.stats["OpenCV兜底"] += 1
                self.hits += 1
                return ReadResult(
                    text=text, type="QRCODE(cv)", level="OpenCV兜底",
                    ms=(time.perf_counter() - t0) * 1000.0,
                    polygon=poly, center=tuple(float(v) for v in np.mean(poly, axis=0)),
                    side_px=side, per_module_px=side / 21.0,
                    warped=warped, warped_bin=binimg,
                    distance_mm=FOCAL_PX * CODE_W_MM / max(side, 1e-6),
                    task=task, task_error=terr,
                )
        return None

    def report(self) -> str:
        if self.frames == 0:
            return "还没读过任何帧"
        rate = 100.0 * self.hits / self.frames
        return (f"处理 {self.frames} 帧 | 命中 {self.hits}（{rate:.1f}%）| "
                f"超预算放弃 {self.skipped} 帧 | 级别分布 {dict(self.stats) or '无'}")


def save_read(out_dir: str, result: ReadResult, frame: np.ndarray,
              frozen_frame: Optional[np.ndarray] = None) -> dict:
    """把一次成功的读码存盘（快照 + 对齐图 + JSON/CSV 一行），返回记录。"""
    os.makedirs(out_dir, exist_ok=True)
    tag = re.sub(r"[^0-9A-Za-z]+", "", result.text)[:20] or "code"
    ts = f"{tag}_{time.strftime('%H%M%S')}"
    rec = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "text": result.text,
        "level": result.level,
        "ms": round(result.ms, 2),
        "side_px": round(result.side_px, 1),
        "per_module_px": round(result.per_module_px, 2),
        "distance_mm": round(result.distance_mm, 0),
        "center": [round(result.center[0], 1), round(result.center[1], 1)],
        "task": result.task,
        "task_error": result.task_error,
    }
    cv2.imwrite(os.path.join(out_dir, f"snapshot_{ts}.png"), frame)
    if frozen_frame is not None:
        cv2.imwrite(os.path.join(out_dir, f"frozen_{ts}.png"), frozen_frame)
    cv2.imwrite(os.path.join(out_dir, f"aligned_{ts}.png"), result.warped)
    cv2.imwrite(os.path.join(out_dir, f"aligned_bin_{ts}.png"), result.warped_bin)
    with open(os.path.join(out_dir, "read_log.csv"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "latest.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return rec


if __name__ == "__main__":                          # 自测：读一张码并打印解析结果
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "codes", "taskcode_156123516231.png")
    img = cv2.imread(path)
    rd = RaceQRReader()
    r = rd.read(img)
    print(rd.report())
    if r:
        print(f"内容 {r.text} | 级别 {r.level} | {r.ms:.1f}ms | 边长 {r.side_px:.0f}px | "
              f"每格 {r.per_module_px:.2f}px | 估算距离 {r.distance_mm:.0f}mm")
        print(r.task["summary"] if r.task else f"不是合法任务码：{r.task_error}")
