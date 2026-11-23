# -*- coding: utf-8 -*-
"""
lesson05_inspect_result.py —— 第 05 课：读懂 decode() 的返回值

第 03 课你只用了结果里的 .data。其实 zbar 还告诉了你很多别的东西：
  这个码在画面的哪个位置、有多大、朝哪个方向、是二维码还是条形码……
这些信息才是让机器人"边看边算"的关键。

运行（在教程根目录下）：
    python code/lesson05_inspect_result.py
"""
import os
import sys
import math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "assets")

sys.path.insert(0, HERE)
import zbar_bootstrap
zbar_bootstrap.ensure()

from PIL import Image
from pyzbar.pyzbar import decode


def show(path, title=None):
    """把一张图的所有解码结果，逐字段摊开打印。"""
    img = Image.open(path).convert("RGB")
    results = decode(img)
    print("-" * 70)
    print(f"文件：{os.path.basename(path)}   图片尺寸：{img.size[0]}x{img.size[1]}")
    if not results:
        print("  （没有读到任何码）")
        return
    for i, r in enumerate(results):
        print(f"  ── 第 {i + 1} 个码 ──")
        print(f"    .data        = {r.data!r}")
        print(f"    .data.decode = {r.data.decode('utf-8', 'replace')}")
        print(f"    .type        = {r.type}")
        print(f"    .rect        = {r.rect}")
        print(f"                     （左上角 x={r.rect.left}, y={r.rect.top}, "
              f"宽 {r.rect.width}, 高 {r.rect.height}）")
        print(f"    .polygon     = {r.polygon}")
        print(f"    .quality     = {r.quality}")
        print(f"    .orientation = {r.orientation}")
        # 顺手算两个对机器人有用的量
        cx = r.rect.left + r.rect.width // 2
        cy = r.rect.top + r.rect.height // 2
        print(f"    → 码的中心点 = ({cx}, {cy})  "
              f"（机器人要转向这里对准）")
        if r.polygon and len(r.polygon) >= 4:
            # polygon 的相邻两点连起来就是码的一条边；四条边都算一遍
            edges = []
            for i in range(4):
                a, b = r.polygon[i], r.polygon[(i + 1) % 4]
                dx, dy = b.x - a.x, b.y - a.y
                edges.append((math.degrees(math.atan2(dy, dx)),
                              math.hypot(dx, dy)))
            txt = "  ".join(f"{ang:6.1f}°/{ln:5.1f}px" for ang, ln in edges)
            print(f"    → 四条边(倾角/长度) = {txt}")
            # 四条边里"离水平线最近"的那个夹角，就是码被转了多少（范围 0~45）
            tilt = min(min(abs(ang), abs(abs(ang) - 180)) for ang, _ in edges)
            print(f"    → 码相对画面水平线的夹角 ≈ {tilt:.1f}°  "
                  f"（0~45；0 = 某条边与画面水平线平行，也就是'摆正'）")


print("=" * 70)
print("第 1 步：把一个 QR 的每个字段摊开看")
print("=" * 70)
show(os.path.join(ASSETS, "01_clean.png"))

print()
print("=" * 70)
print("第 2 步：码在画面里挪了位置 / 变大了，哪些字段跟着变？")
print("=" * 70)
for name in ("03_size_120.png", "03_size_200.png", "03_size_400.png"):
    show(os.path.join(ASSETS, name))
print("观察要点：")
print("  · .data / .type 不变 —— 它们是『码里装的内容』，与拍摄无关。")
print("  · .rect / .polygon 变了 —— 它们是『码在画面上的几何位置』。")

print()
print("=" * 70)
print("第 3 步：二维码被转了个角度，哪个字段能告诉你？")
print("=" * 70)
for name in ("10_rot30.png", "10_rot45.png", "10_rot90.png"):
    show(os.path.join(ASSETS, name))
print("注意：.orientation 在 0°/30° 时是 UP，但转到 45°/90° 时会变成 LEFT ——")
print("     它不是'真实倾角'，而是 zbar 内部枚举出的四种朝向之一（UP/LEFT/RIGHT/DOWN）。")
print("     想知道码到底歪了多少度，只能用 .polygon 的四个顶点坐标自己算。")

print()
print("=" * 70)
print("第 4 步：zbar 不止认二维码，条形码也认")
print("=" * 70)
show(os.path.join(ASSETS, "14_barcode_code128.png"))
show(os.path.join(ASSETS, "15_barcode_ean13.png"))

print()
print("=" * 70)
print("第 5 步：一张图里多个码，怎么逐个取出来用")
print("=" * 70)
multi = os.path.join(ASSETS, "generated", "multi_three.png")
if os.path.exists(multi):
    show(multi)
    results = decode(Image.open(multi))
    print("常见用法：用字典收集，key 用内容，value 用位置")
    table = {r.data.decode(): (r.rect.left, r.rect.top) for r in results}
    for k, v in table.items():
        print(f"   {k}  ->  左上角 {v}")
else:
    print("先跑 code/lesson04_make_qr.py 生成 multi_three.png")

print()
print("=" * 70)
print("第 6 步：由 polygon 反推『每格多少像素』——判断码够不够清楚")
print("=" * 70)
p = os.path.join(ASSETS, "13_scene.png")
r = decode(Image.open(p))[0]
pts = r.polygon
side = math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y)
modules = 21          # 任务码是 version 1 = 21x21 有效格
print(f"检测到的码『数据区』边长 = {side:.1f}px")
print(f"任务码是 version 1 = {modules}x{modules} 有效格"
      f"（注意：polygon 只框数据区，不含四周白边）")
print(f"→ 每格约 {side / modules:.2f}px")
print("经验线：清洁合成图上每格 1.4px 就能读；真实摄像头画面建议 >= 3px。")
print("用法：每格像素与距离成反比 —— 每格像素翻倍，就等于靠近一半的距离。")
