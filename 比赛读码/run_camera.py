#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比赛读码主程序：实时摄像头窗口 + 画框 + 定帧 + 透视变换图 + 数据输出。

用法：
    /home/kata/yes/envs/OpenCV_Linux/bin/python run_camera.py            # 默认摄像头 0
    /home/kata/yes/envs/OpenCV_Linux/bin/python run_camera.py --cam 3    # 指定摄像头
    /home/kata/yes/envs/OpenCV_Linux/bin/python run_camera.py --frames 40  # 自动退出（自测用）
    /home/kata/yes/envs/OpenCV_Linux/bin/python run_camera.py --image codes/taskcode_156123516231.png

按键：
    q / ESC  退出
    空格     手动定帧 / 解除定帧（读到码会自动定帧 5 秒）
    s        把当前帧和识别数据存盘到 out/
    r        清空确认状态，重新开始连续确认

窗口布局：
    左边 = 摄像头实时画面（绿框 = zbar rect，青色四边形 = 数据区真角，白十字 = 中心）
    右上 = 定帧那一刻的码（原图 → 透视对齐正视图 → 二值图，三联画）
    右下 = 识别出的任务码数据（第一批/第二批的颜色顺序与位置、距离、每格像素）
"""
import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reader import RaceQRReader, save_read, WARP_SIZE, FOCAL_PX, CODE_W_MM  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "out")
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

# ── 界面尺寸 ──
CAM_W, CAM_H = 640, 480        # 左侧画面
RIGHT_W = 430                  # 右侧信息列
PANEL_H = 480                  # 右侧总高
WIN_W, WIN_H = CAM_W + RIGHT_W, max(CAM_H, PANEL_H)


def _font(size):
    return ImageFont.truetype(FONT_PATH, size)


def _wrap(text, font, max_w):
    """按像素宽度折行（中文没有空格可断，只能一个字一个字量）。"""
    out, cur = [], ""
    for ch in text:
        if font.getlength(cur + ch) <= max_w:
            cur += ch
        else:
            out.append(cur)
            cur = ch
    if cur:
        out.append(cur)
    return out


def draw_text_panel(lines, w=RIGHT_W, h=PANEL_H):
    """中文信息面板（cv2.putText 不支持中文，用 PIL 画；自动折行，不会超出边界）。"""
    img = Image.new("RGB", (w, h), (18, 18, 24))
    dr = ImageDraw.Draw(img)
    y = 12
    max_w = w - 28                                    # 左右各留 14px 边距
    for text, size, color in lines:
        f = _font(size)
        for ln in text.split("\n"):
            for piece in _wrap(ln, f, max_w):
                dr.text((14, y), piece, font=f, fill=color)
                y += size + 8
        y += 6
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# cv2.putText 只能画 ASCII，级别名转成英文
LEVEL_ASCII = {
    "原图": "raw", "中值": "median", "OTSU": "otsu", "CLAHE": "clahe",
    "反色": "invert", "中心2x": "center2x", "放大2x": "up2",
    "放大4x": "up4", "三次4x": "cubic4x", "OpenCV兜底": "cv-fallback",
}


def draw_hud(frame, fps, ms, level, streak, confirm, frozen, budget):
    """左上角 HUD：深色底 + 三行 ASCII 信息（中文走右侧面板）。"""
    lv = LEVEL_ASCII.get(level, level or "-")
    lines = [
        (f"FPS {fps:5.1f}   decode {ms:5.1f} ms   budget {budget:.0f} ms", (235, 235, 235)),
        (f"level {lv:<11s} streak {streak}/{confirm}", (235, 235, 235)),
        ("FROZEN  space=resume" if frozen else "LIVE  auto-freeze on read",
         (60, 200, 255) if frozen else (120, 220, 120)),
    ]
    h = 18 * len(lines) + 14
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (440, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)   # 半透明底，不挡画面
    y = 22
    for text, color in lines:
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    color, 1, cv2.LINE_AA)
        y += 18
    return frame


def draw_boxes(frame, result):
    """画框：zbar rect 没直接给，直接用四角画外接框 + 真角多边形。"""
    if result is None:
        return frame
    poly = result.polygon.astype(int)
    cv2.polylines(frame, [poly], True, (255, 255, 0), 2)          # 青：数据区真角
    x, y, w, h = cv2.boundingRect(poly)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 220, 80), 2)  # 绿：包围盒
    cx, cy = int(result.center[0]), int(result.center[1])
    cv2.drawMarker(frame, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
    return frame


def compose_right(right, frozen_pack):
    """右侧列：三联对齐图 + 中文数据面板。"""
    if frozen_pack is None:
        return draw_text_panel([
            ("等待读到任务码…", 22, (200, 200, 200)),
            ("把码板举到摄像头前", 18, (150, 150, 150)),
            ("读到后会自动定帧并显示", 18, (150, 150, 150)),
            ("透视对齐图和数据", 18, (150, 150, 150)),
            ("", 10, (0, 0, 0)),
            ("按键：", 18, (120, 200, 255)),
            ("空格  定帧/继续", 16, (180, 180, 180)),
            ("s     存盘快照", 16, (180, 180, 180)),
            ("r     清空确认重新来", 16, (180, 180, 180)),
            ("q     退出", 16, (180, 180, 180)),
        ])
    res = frozen_pack["result"]
    # 三联画：定帧原图裁剪 → 对齐正视图 → 二值图
    x, y, w, h = cv2.boundingRect(res.polygon.astype(int))
    pad = max(6, int(w * 0.08))
    H, W = frozen_pack["frame"].shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    crop = frozen_pack["frame"][y0:y1, x0:x1]
    crop = cv2.resize(crop, (RIGHT_W // 3 - 8, RIGHT_W // 3 - 8))
    warp = cv2.cvtColor(res.warped, cv2.COLOR_GRAY2BGR)
    warp = cv2.resize(warp, (RIGHT_W // 3 - 8, RIGHT_W // 3 - 8))
    binimg = cv2.cvtColor(res.warped_bin, cv2.COLOR_GRAY2BGR)
    binimg = cv2.resize(binimg, (RIGHT_W // 3 - 8, RIGHT_W // 3 - 8))
    row = np.hstack([crop, warp, binimg])
    cell = RIGHT_W // 3 - 8
    for i, label in enumerate(["FROZEN", "ALIGNED", "BINARY"]):
        x0 = i * cell
        cv2.rectangle(row, (x0, row.shape[0] - 18), (x0 + cell - 1, row.shape[0] - 1),
                      (20, 20, 20), -1)
        cv2.putText(row, label, (x0 + 6, row.shape[0] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    right[0:row.shape[0], 0:row.shape[1]] = row

    if res.task:
        t = res.task
        lines = [
            (f"任务码  {t['raw']}", 26, (80, 255, 120)),
            (t["summary"], 20, (230, 230, 230)),
            (f"解码器 {res.type}  预处理『{res.level}』  {res.ms:.1f} ms", 16, (150, 170, 200)),
            (f"码边长 {res.side_px:.0f}px   每格 {res.per_module_px:.2f}px", 16, (150, 170, 200)),
            (f"估算距离 {res.distance_mm:.0f} mm（f={FOCAL_PX:.0f}px, W={CODE_W_MM:.0f}mm）",
             16, (150, 170, 200)),
            ("已写入 out/latest.json 与 read_log.csv", 14, (120, 120, 120)),
        ]
    else:
        lines = [
            (f"读到码但不是合法任务码：", 20, (0, 160, 255)),
            (res.text, 20, (230, 230, 230)),
            (f"原因：{res.task_error}", 16, (200, 120, 120)),
        ]
    panel = draw_text_panel(lines, w=RIGHT_W, h=PANEL_H - row.shape[0])
    right[row.shape[0]:PANEL_H, 0:RIGHT_W] = panel[0:PANEL_H - row.shape[0]]
    return right


def make_demo_frames(code_png: str, n: int, W=CAM_W, H=CAM_H):
    """合成『摄像头画面』：把码贴在灰底上，随机位置/大小/倾角 + 轻微噪声。

    没有实车/打印码时用它验证整条链路（窗口、画框、定帧、对齐、数据输出）。
    """
    code = cv2.imread(code_png, cv2.IMREAD_GRAYSCALE)
    if code is None:
        raise SystemExit(f"读不到 {code_png}，先跑 gen_codes.py")
    rng = np.random.default_rng(2027)
    for i in range(n):
        side = int(rng.integers(150, 260))                 # 由远到近地变化
        img = cv2.resize(code, (side, side), interpolation=cv2.INTER_AREA)
        ang = float(rng.uniform(-12, 12))                  # 轻微倾斜
        M = cv2.getRotationMatrix2D((side / 2, side / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (side, side), borderValue=255)
        page = np.full((H, W), int(rng.integers(170, 215)), dtype=np.uint8)
        x = int(rng.integers(20, W - side - 20))
        y = int(rng.integers(20, H - side - 20))
        page[y:y + side, x:x + side] = img
        noise = rng.normal(0, 4, page.shape)
        page = np.clip(page.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        yield cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)


def main():
    ap = argparse.ArgumentParser(description="比赛读码：实时窗口 + 定帧 + 透视对齐 + 数据输出")
    ap.add_argument("--cam", type=int, default=0, help="摄像头编号（默认 0）")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--budget", type=float, default=40.0, help="单帧解码时间预算 ms")
    ap.add_argument("--confirm", type=int, default=2, help="连续 N 帧一致才算数")
    ap.add_argument("--freeze-sec", type=float, default=5.0, help="自动定帧时长（秒）")
    ap.add_argument("--frames", type=int, default=0, help="处理 N 帧后自动退出（0=不限，自测用）")
    ap.add_argument("--image", type=str, default="", help="读一张图片而不是摄像头（自测用）")
    ap.add_argument("--demo", type=int, default=-1,
                    help="不用摄像头，合成画面演示：0/1/2 选哪张任务码")
    ap.add_argument("--no-window", action="store_true", help="不弹窗（无显示器自测用）")
    args = ap.parse_args()

    reader = RaceQRReader(budget_ms=args.budget)
    streak = deque(maxlen=8)
    confirmed_text = None
    frozen_pack = None          # {"frame": 定帧那刻的画面, "result": ReadResult}
    frozen_until = 0.0
    last_result = None

    # ── 图片模式：读一张就走 ──
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"读不到图片：{args.image}")
            return 2
        res = reader.read(frame)
        print(reader.report())
        if res:
            print(f"内容 {res.text} | 级别 {res.level} | {res.ms:.1f}ms")
            print(res.task["summary"] if res.task else f"非法任务码：{res.task_error}")
            rec = save_read(OUT_DIR, res, frame)
            print(f"已存盘 → {os.path.join(OUT_DIR, 'latest.json')}")
            _ = rec
        return 0

    # ── 帧来源：真实摄像头 或 合成演示画面 ──
    cap = None
    demo_iter = None
    if args.demo >= 0:
        tags = ["156123516231", "234312145321", "365251436152"]
        tag = tags[args.demo % len(tags)]
        p = os.path.join(ROOT, "codes", f"taskcode_{tag}.png")
        need = args.frames or 120
        demo_iter = make_demo_frames(p, need)
        print(f"[演示模式] 合成摄像头画面，任务码 {tag}，共 {need} 帧（不用打印码也能看效果）")
    else:
        cap = cv2.VideoCapture(args.cam)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            print(f"打不开摄像头 {args.cam}（可以先 --image 测，或换 --cam 1/3，或 --demo 0）")
            return 2
        t_open = time.perf_counter()
        ok, _ = cap.read()
        print(f"摄像头 {args.cam} 打开耗时 {time.perf_counter() - t_open:.2f}s，出图 {ok}")
        if not ok:
            print("摄像头能开但读不出帧，换 --cam 试试")
            return 2

    win_name = "比赛读码 (q=exit space=freeze s=save)"
    show_window = not args.no_window
    if show_window:
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    t_prev = time.perf_counter()
    fps_hist = deque(maxlen=30)
    ms_hist = deque(maxlen=120)
    n_done = 0
    try:
        while True:
            if demo_iter is not None:
                try:
                    frame = next(demo_iter)
                except StopIteration:
                    print("演示画面放完了，退出")
                    break
            else:
                ok, frame = cap.read()
                if not ok:
                    print("摄像头断流，退出")
                    break
            frame = cv2.resize(frame, (CAM_W, CAM_H))

            t0 = time.perf_counter()
            res = reader.read(frame)
            ms = (time.perf_counter() - t0) * 1000.0
            ms_hist.append(ms)

            now = time.perf_counter()
            fps_hist.append(1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            fps = float(np.mean(fps_hist))

            # ── 连续确认：防单帧误读 ──
            if res is not None:
                streak.append(res.text)
                last_result = res
            else:
                streak.clear()
            if len(streak) >= args.confirm and len(set(streak)) == 1:
                if confirmed_text != streak[0]:
                    confirmed_text = streak[0]
                    print("=" * 56)
                    print(f"[确认] 任务码 {confirmed_text}（级别 {res.level}，{res.ms:.1f}ms）")
                    if res.task:
                        print(res.task["summary"])
                    else:
                        print(f"[警告] 不是合法任务码：{res.task_error}")
                    print("=" * 56)
                    rec = save_read(OUT_DIR, res, frame)
                    _ = rec
                if frozen_pack is None or now >= frozen_until:
                    frozen_pack = {"frame": frame.copy(), "result": res}
                    frozen_until = now + args.freeze_sec

            # ── 定帧到期自动恢复 ──
            if frozen_pack is not None and now >= frozen_until:
                frozen_pack = None
                confirmed_text = None
                streak.clear()

            # ── 画 ──
            display = frozen_pack["frame"].copy() if frozen_pack else frame
            cur = frozen_pack["result"] if frozen_pack else res
            draw_boxes(display, cur)
            draw_hud(display, fps, ms, (cur.level if cur else "-"),
                     len(streak), args.confirm,
                     frozen_pack is not None, args.budget)
            right = compose_right(np.zeros((PANEL_H, RIGHT_W, 3), dtype=np.uint8), frozen_pack)
            canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
            canvas[0:CAM_H, 0:CAM_W] = display
            canvas[0:PANEL_H, CAM_W:WIN_W] = right
            if show_window:
                cv2.imshow(win_name, canvas)

            n_done += 1
            if args.frames and n_done >= args.frames:
                break

            if show_window:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    if frozen_pack is not None:
                        frozen_pack = None
                        confirmed_text = None
                        streak.clear()
                    elif last_result is not None:
                        frozen_pack = {"frame": frame.copy(), "result": last_result}
                        frozen_until = now + args.freeze_sec
                elif key == ord("s"):
                    tgt = frozen_pack["result"] if frozen_pack else res
                    if tgt is not None:
                        rec = save_read(OUT_DIR, tgt, frame)
                        print(f"[存盘] {rec['text']} → out/")
                    else:
                        print("[存盘] 当前没有识别结果，不存")
                elif key == ord("r"):
                    confirmed_text = None
                    streak.clear()
                    frozen_pack = None
    finally:
        if cap is not None:
            cap.release()
        if show_window:
            cv2.destroyAllWindows()
        print(reader.report())
        if ms_hist:
            arr = np.array(ms_hist)
            print(f"单帧解码耗时：平均 {arr.mean():.1f}ms  中位 {np.median(arr):.1f}ms  "
                  f"最坏 {arr.max():.1f}ms（预算 {args.budget:.0f}ms）")


if __name__ == "__main__":
    sys.exit(main())
