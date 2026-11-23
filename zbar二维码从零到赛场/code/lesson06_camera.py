# -*- coding: utf-8 -*-
"""
lesson06_camera.py —— 第 06 课：让摄像头一直看，一直读

这一课把"读一张图"变成"读连续视频流"，这是机器人真实的工作方式。

运行方式（在教程根目录下）：
    python code/lesson06_camera.py --replay        # 用教程自带的假摄像头（回放 60 帧）
    python code/lesson06_camera.py                 # 用真摄像头 /dev/video0
    python code/lesson06_camera.py --show          # 真摄像头 + 弹窗显示（需要桌面环境）
    python code/lesson06_camera.py --bench         # 只测速，不找码

参数：
    --replay      不用摄像头，逐帧播放 assets/replay/ 里的图片
    --camera N    指定第几个摄像头，默认 0
    --frames N    最多处理多少帧就停（不给就一直跑，按 Ctrl+C 结束）
    --show        边跑边弹窗显示画面（没有图形界面时别加）
    --bench       不弹窗预览（无人值守跑一遍）
"""
import os
import sys
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "assets")

sys.path.insert(0, HERE)
import zbar_bootstrap
zbar_bootstrap.ensure()

import cv2
import numpy as np
from PIL import Image
from pyzbar.pyzbar import decode


# ---------------------------------------------------------------------
# 工具函数 1：把 OpenCV 的一帧（BGR 的 numpy 数组）交给 pyzbar
# ---------------------------------------------------------------------
def decode_frame(frame_bgr):
    """输入一帧 BGR 图，返回 pyzbar 的结果列表。

    为什么要绕一圈？pyzbar 只认 PIL 图片或灰度 numpy 数组。
      · frame_bgr 是三通道 BGR（OpenCV 的默认顺序）
      · zbar 只需要"黑白信息"，所以先转灰度最省事也最快
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return decode(gray)


# ---------------------------------------------------------------------
# 工具函数 2：在画面上把读到的码框出来（调试时看这个最直观）
# ---------------------------------------------------------------------
def draw_results(frame_bgr, results):
    for r in results:
        pts = np.array([[p.x, p.y] for p in r.polygon], dtype=np.int32)
        cv2.polylines(frame_bgr, [pts], True, (0, 255, 0), 2)
        text = r.data.decode("utf-8", "replace")
        x, y = r.rect.left, r.rect.top
        cv2.putText(frame_bgr, text, (x, max(y - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame_bgr


# ---------------------------------------------------------------------
# 取帧来源 1：真摄像头
# ---------------------------------------------------------------------
class CameraSource:
    def __init__(self, index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"打不开摄像头 /dev/video{index}")
        # 设置分辨率只是"请求"，摄像头可能不给，所以要读回来看看
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total = None

    def read(self):
        ok, frame = self.cap.read()
        return (frame if ok else None)

    def release(self):
        self.cap.release()


# ---------------------------------------------------------------------
# 取帧来源 2：回放图片目录（假装自己是摄像头）
# ---------------------------------------------------------------------
class ReplaySource:
    def __init__(self, folder):
        self.files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                            if f.endswith(".png"))
        if not self.files:
            raise RuntimeError(f"{folder} 里没有 png，先跑 code/make_assets.py")
        self.i = 0
        self.total = len(self.files)
        first = cv2.imread(self.files[0])
        self.h, self.w = first.shape[:2]

    def read(self):
        if self.i >= len(self.files):
            return None
        frame = cv2.imread(self.files[self.i])
        self.i += 1
        return frame

    def release(self):
        pass


# ---------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------
def open_source(args):
    """按命令行参数打开取帧来源。

    摄像头打不开时**不崩**，自动退回回放模式 —— 
    这不是"偷懒"，这是工程习惯：
      · 你的机器人主程序里，摄像头开机失败必须能降级/重试，不能整机卡死
      · 学习时没摄像头（或者被别的程序占着）也能把这一课学完
    """
    if getattr(args, "_replay_forced", False):
        return ReplaySource(os.path.join(ASSETS, "replay")), "回放模式"
    if args.replay:
        return ReplaySource(os.path.join(ASSETS, "replay")), "回放模式"
    try:
        return CameraSource(args.camera), f"摄像头 /dev/video{args.camera}"
    except RuntimeError as e:
        print(f"[提示] {e} —— 自动改用回放模式（assets/replay 里的图）。")
        print("       想用真摄像头：确认没有别的程序占用它，再试 --camera 0。")
        print()
        args._replay_forced = True
        return ReplaySource(os.path.join(ASSETS, "replay")), "回放模式（摄像头不可用）"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--bench", action="store_true",
                    help="不弹窗预览（无人值守跑一遍）")
    args = ap.parse_args()

    src, src_name = open_source(args)
    if src.total:
        src_name += f"（{src.total} 帧）"

    print("=" * 66)
    print(f"第 06 课 · 视频流读码    来源：{src_name}   画面：{src.w}x{src.h}")
    print("=" * 66)

    # ---------------- 第 1 步：先看看一帧要花多久 ----------------
    probe = src.read()
    if probe is None:
        print("取帧失败，直接退出。")
        src.release()
        return
    t0 = time.perf_counter()
    r = decode_frame(probe)
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000
    print(f"\n【第 1 步】单帧解码耗时 {ms:.2f} ms  "
          f"→ 理论上每秒能处理 {1000 / ms:.1f} 帧")
    print(f"           这一帧读出 {len(r)} 个码。")
    print("           注意：这是『算法耗时』，不含摄像头本身的等待。")

    # ---------------- 第 2 步：主要循环 ----------------
    print("\n【第 2 步】进入循环：一帧一帧地读，读到就记住")
    src.release()
    src, _ = open_source(args)          # 重新开一次，从第 1 帧开始

    first_hit = None          # 第一次读到的帧号
    final_text = None         # 读到的内容（记住第一次成功的，别被空帧覆盖）
    n = 0
    hits = 0
    t_start = time.perf_counter()
    decode_time_total = 0.0

    while True:
        frame = src.read()
        if frame is None:
            break
        n += 1

        t0 = time.perf_counter()
        # 注意：--bench 只是"不弹窗预览"，解码照做 ——
        # 早期版本这里写成"bench 就跳过解码"，结果测出来 400fps，
        # 其实测的是"空循环有多快"，毫无意义。（真实踩过的坑）
        results = decode_frame(frame)
        decode_time_total += time.perf_counter() - t0

        if results:
            hits += 1
            if final_text is None:
                final_text = results[0].data.decode("utf-8", "replace")
                first_hit = n
                print(f"           第 {n} 帧：第一次读到 → {final_text}")

        if args.show:
            vis = draw_results(frame.copy(), results)
            cv2.imshow("lesson06", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if args.frames and n >= args.frames:
            break

    t_end = time.perf_counter()
    src.release()
    if args.show:
        cv2.destroyAllWindows()

    elapsed = t_end - t_start
    # ---------------- 第 3 步：结果汇总 ----------------
    print("\n" + "=" * 66)
    print("【第 3 步】这一段跑完的统计")
    print("=" * 66)
    print(f"  处理帧数        : {n}")
    print(f"  读到码的帧数    : {hits}"
          + (f"（命中率 {hits / n * 100:.1f}%）" if n else ""))
    print(f"  第一次读到      : 第 {first_hit} 帧" if first_hit else "  一次都没读到")
    print(f"  最终内容        : {final_text}")
    print(f"  总耗时          : {elapsed * 1000:.1f} ms")
    if n:
        print(f"  端到端帧率      : {n / elapsed:.1f} 帧/秒（fps）")
        print(f"  平均每帧解码    : {decode_time_total / n * 1000:.2f} ms")
    print()
    print("看这几个数的意思：")
    print("  · 『命中率』低 → 要么码太小/太糊，要么车没停稳（第 07、09 课解决）")
    print("  · 『第一次读到』晚 → 车开得太快或对焦慢，靠降低车速/提前扫描解决")
    print("  · 端到端帧率远低于第 1 步的理论值 → 摄像头出图本身慢（USB 带宽/曝光）")


if __name__ == "__main__":
    main()
