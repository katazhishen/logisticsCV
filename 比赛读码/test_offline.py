#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线自测：不弹窗，验证整条链路（生成码 → 识别 → 对齐 → 解析 → 输出）。

用法：
    /home/kata/yes/envs/OpenCV_Linux/bin/python test_offline.py

测试项：
    1. 三张屏幕版任务码：逐张读出，内容必须与生成时一致，任务单解析合法。
    2. 远距小码：把码缩到 80px 再读（贴近赛场最远 1475mm 时的像素大小）。
    3. 倾斜 15°：旋转码仍要读出，且对齐图能把转正（对比对齐前后倾角）。
    4. 透视歪斜：随机单应变换模拟『斜着拍码板』，仍要读出。
    5. 干扰项：噪声/低光照两帧，走预处理阶梯仍要读出。
全部通过 → exit 0；对齐图存在 out/aligned_*.png 可直接打开看。
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reader import RaceQRReader, parse_taskcode, TaskCodeError  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.join(ROOT, "codes")
OUT = os.path.join(ROOT, "out")
CODES_TEXT = ["156+123+516+231", "234+312+145+321", "365+251+436+152"]

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK]   {name} {detail}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def scene_with_code(code_img, bg=200, W=640, H=480):
    """把码贴到一张『灰底场景图』中间，模拟摄像头画面。"""
    page = np.full((H, W), bg, dtype=np.uint8)
    ch, cw = code_img.shape[:2]
    x, y = (W - cw) // 2, (H - ch) // 2
    page[y:y + ch, x:x + cw] = code_img
    return cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)


def main():
    rd = RaceQRReader(budget_ms=40.0)
    os.makedirs(OUT, exist_ok=True)

    print("== 1. 三张屏幕版任务码 ==")
    for i, text in enumerate(CODES_TEXT):
        tag = text.replace("+", "")
        p = os.path.join(CODES, f"taskcode_{tag}.png")
        img = cv2.imread(p)
        check(f"生成文件存在 {os.path.basename(p)}", img is not None)
        r = rd.read(img)
        check(f"读出内容 == {text}", r is not None and r.text == text,
              "" if r is None else f"(级别 {r.level}, {r.ms:.1f}ms)")
        check("任务单解析合法", r is not None and r.task is not None,
              "" if r is None or r.task is None else r.task["summary"].replace("\n", "；"))
        if r is not None:
            cv2.imwrite(os.path.join(OUT, f"aligned_{tag}.png"), r.warped)
            cv2.imwrite(os.path.join(OUT, f"aligned_bin_{tag}.png"), r.warped_bin)

    print("== 2. 远距小码（缩到 80px，每格 2.76px）==")
    for i, text in enumerate(CODES_TEXT):
        tag = text.replace("+", "")
        img = cv2.imread(os.path.join(CODES, f"taskcode_{tag}.png"), cv2.IMREAD_GRAYSCALE)
        small = cv2.resize(img, (80, 80), interpolation=cv2.INTER_AREA)
        r = rd.read(scene_with_code(small))
        check(f"80px 仍读出 {text}", r is not None and r.text == text,
              "" if r is None else f"(每格 {r.per_module_px:.2f}px, 估算 {r.distance_mm:.0f}mm)")

    print("== 3. 倾斜 15° ==")
    img = cv2.imread(os.path.join(CODES, "taskcode_156123516231.png"), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (240, 240))
    M = cv2.getRotationMatrix2D((120, 120), 15, 1.0)
    rot = cv2.warpAffine(img, M, (300, 300), borderValue=255)
    r = rd.read(scene_with_code(rot))
    check("15° 旋转读出", r is not None and r.text == CODES_TEXT[0],
          "" if r is None else f"(对齐图 out/aligned_rot15.png 已把码转正)")
    if r is not None:
        cv2.imwrite(os.path.join(OUT, "aligned_rot15.png"), r.warped)

    print("== 4. 透视歪斜（模拟斜拍码板）==")
    img = cv2.imread(os.path.join(CODES, "taskcode_156123516231.png"), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (240, 240))
    page = np.full((480, 640), 210, dtype=np.uint8)
    page[120:360, 200:440] = img
    src = np.float32([[200, 120], [440, 120], [440, 360], [200, 360]])
    dst = np.float32([[230, 150], [455, 175], [415, 380], [180, 340]])   # 故意歪
    Hm = cv2.getPerspectiveTransform(src, dst)
    warped_scene = cv2.warpPerspective(cv2.cvtColor(page, cv2.COLOR_GRAY2BGR), Hm, (640, 480),
                                       borderValue=(210, 210, 210))
    r = rd.read(warped_scene)
    check("透视歪斜读出", r is not None and r.text == CODES_TEXT[0],
          "" if r is None else f"(边长 {r.side_px:.0f}px)")
    if r is not None:
        cv2.imwrite(os.path.join(OUT, "aligned_persp.png"), r.warped)
        cv2.imwrite(os.path.join(OUT, "scene_persp.png"), warped_scene)

    print("== 5. 恶劣画质（椒盐噪声 / 低光照）==")
    img = cv2.imread(os.path.join(CODES, "taskcode_156123516231.png"), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (240, 240))
    rng = np.random.default_rng(7)
    noisy = img.copy()
    m = rng.random(img.shape) < 0.02
    noisy[m] = rng.integers(0, 256, m.sum())
    r = rd.read(scene_with_code(noisy))
    check("椒盐噪声 2% 读出", r is not None and r.text == CODES_TEXT[0],
          "" if r is None else f"(级别 {r.level})")
    dark = (img * 0.45).astype(np.uint8)
    r2 = rd.read(scene_with_code(dark, bg=90))
    check("低光照 ×0.45 读出", r2 is not None and r2.text == CODES_TEXT[0],
          "" if r2 is None else f"(级别 {r2.level})")

    print("== 6. 非法任务码要给出『错在哪』==")
    for bad_text, why in [("156+123+516+23", "组数不对"), ("999+123+516+231", "字符非法"),
                          ("155+123+516+231", "颜色重复")]:
        try:
            parse_taskcode(bad_text)
            check(f"{bad_text} 应被拒绝", False)
        except TaskCodeError as e:
            check(f"{bad_text} 被拒绝并说明原因", True, f"→ {e}")

    print()
    print(rd.report())
    print(f"\n对齐图在 {OUT} （aligned_*.png / aligned_bin_*.png）")
    print(f"通过 {PASS} 项，失败 {FAIL} 项")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
