#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 3 张比赛用的任务码二维码（print 版 + 屏幕版）。

用法：
    /home/kata/yes/envs/OpenCV_Linux/bin/python gen_codes.py

输出（都在本目录的 codes/ 下）：
    codes/taskcode_156.png   屏幕版（方便直接对着摄像头展示）
    codes/taskcode_234.png
    codes/taskcode_365.png
    codes/a4_156.png         打印版：A4 300DPI，码正好 80×80mm（比赛规格）
    codes/a4_234.png
    codes/a4_365.png

任务码格式（来自命题文件）：四组三位数用 + 连接
    第 1 组 = 第一批三个物料的颜色与搬运顺序（颜色编号：红1 黄2 蓝3 绿4 黑5 浅蓝6）
    第 2 组 = 第一批物料在粗加工区和暂存区的放置位置
    第 3 组 = 第二批颜色与搬运顺序
    第 4 组 = 第二批在粗加工区的放置位置
"""
import os
import sys

import qrcode
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.join(ROOT, "codes")

# 三张码：1 张是命题原文示例，另外 2 张随机但合法（组内颜色/位置不重复）
TASK_CODES = ["156+123+516+231", "234+312+145+321", "365+251+436+152"]

# 打印版参数：与第 12 课 C 节同一套算法（300DPI，码 80mm，红线外扩 5mm）
DPI = 300
MM_PER_INCH = 25.4
PX_PER_MM = DPI / MM_PER_INCH          # 11.811 px/mm
CODE_MM = 80.0                          # 比赛规定：二维码 80×80mm（含边框）
A4_W, A4_H = int(round(210 * PX_PER_MM)), int(round(297 * PX_PER_MM))  # 2480×3507
RED_PAD_MM = 5.0                        # 红线画在码外 5mm（千万别压住安静区！）


def make_screen_code(text: str, box_size: int = 10, border: int = 4) -> Image.Image:
    """屏幕版：box_size 大一点，摄像头近拍方便。"""
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M,
                       box_size=box_size, border=border)
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("L")
    return img


def make_a4_code(text: str) -> Image.Image:
    """打印版：整张 A4 @300DPI，码逐格画到正好 80mm，外扩红线标出裁剪边界。"""
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M,
                       box_size=1, border=4)
    qr.add_data(text)
    qr.make(fit=True)
    n = qr.modules_count                       # v1 是 29（21 数据 + 8 白边）
    code_px = int(round(CODE_MM * PX_PER_MM))  # 945px
    step = code_px / n                         # 每格像素（浮点，逐格 round 才不歪）

    page = Image.new("L", (A4_W, A4_H), 255)
    dr = ImageDraw.Draw(page)
    ox = (A4_W - code_px) // 2                 # 居中
    oy = (A4_H - code_px) // 2
    for r in range(n):                         # 逐格按目标像素画矩形
        for c in range(n):                     # （不要先画小图再 resize，会抽行抽列）
            if qr.get_matrix()[r][c]:
                x0 = ox + round(c * step)
                x1 = ox + round((c + 1) * step)
                y0 = oy + round(r * step)
                y1 = oy + round((r + 1) * step)
                dr.rectangle([x0, y0, x1 - 1, y1 - 1], fill=0)

    page_rgb = page.convert("RGB")
    dr2 = ImageDraw.Draw(page_rgb)
    pad = round(RED_PAD_MM * PX_PER_MM)        # 红线必须画在码『外面』
    dr2.rectangle([ox - pad, oy - pad, ox + code_px + pad, oy + code_px + pad],
                  outline=(255, 0, 0), width=6)
    return page_rgb


def main():
    os.makedirs(CODES, exist_ok=True)
    for text in TASK_CODES:
        tag = text.replace("+", "")
        scr = make_screen_code(text)
        a4 = make_a4_code(text)
        p1 = os.path.join(CODES, f"taskcode_{tag}.png")
        p2 = os.path.join(CODES, f"a4_{tag}.png")
        scr.save(p1)
        a4.save(p2)
        n = 21 + 8
        print(f"任务码 {text}")
        print(f"  屏幕版 {p1}  尺寸 {scr.size[0]}×{scr.size[1]}px（box_size=10）")
        print(f"  打印版 {p2}  A4@300DPI {a4.size[0]}×{a4.size[1]}px，码 {n} 格 = {code_mm():.0f}mm")
        print(f"  打印时：100% 缩放（不要『适合页面』），码板裁到红线，码距摄像头越近每格像素越多")
        print()
    print(f"共生成 {len(TASK_CODES)} 张码（每张 2 个文件）→ {CODES}")


def code_mm() -> float:
    return CODE_MM


if __name__ == "__main__":
    sys.exit(main())
