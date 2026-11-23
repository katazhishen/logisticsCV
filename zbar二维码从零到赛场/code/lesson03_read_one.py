# -*- coding: utf-8 -*-
"""
lesson03_read_one.py —— 第 03 课：第一次读出二维码

这一课只做一件事：让一张图片里的二维码，变成屏幕上的一行字。

运行（在教程根目录下）：
    python code/lesson03_read_one.py

如果你还没生成素材图，先跑：
    python code/make_assets.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "assets")

# ---------------------------------------------------------------------
# 第 0 步：把 zbar 本体找出来（这一段是"插件"，不是重点，第 02 课解释过）
# ---------------------------------------------------------------------
sys.path.insert(0, HERE)
import zbar_bootstrap
zbar_bootstrap.ensure()

# ---------------------------------------------------------------------
# 第 1 步：三行核心代码
# ---------------------------------------------------------------------
from PIL import Image                 # ① 负责"把图片文件打开成图片对象"
from pyzbar.pyzbar import decode      # ② 负责"在图片里找码并翻译成文字"

img = Image.open(os.path.join(ASSETS, "01_clean.png"))   # ③ 打开一张图
results = decode(img)                                    # ④ 让 zbar 读它

print("=" * 62)
print("第 1 步：最小可运行版本（4 行）")
print("=" * 62)
print("decode() 返回的是：", type(results))
print("返回内容：", results)
print("里面第一个元素的类型是：", type(results[0]))
print("它带着的原始字节是：", results[0].data)
print("→ 用 .decode() 转成人类看的字符串：", results[0].data.decode())

# ---------------------------------------------------------------------
# 第 2 步：把"读出来"包装成一个函数，顺便处理读不出的情况
# ---------------------------------------------------------------------
def read_qr(path):
    """读一张图里的第一个二维码。读到返回字符串，读不到返回 None。

    为什么要写成函数？
      · 后面第 10 课要做成一个能反复调用的模块，现在就先养成习惯。
      · 读不到时返回 None（而不是直接崩），调用的人可以决定"再试一次"。
    """
    img = Image.open(path)
    found = decode(img)
    if not found:                       # 空列表 = 一个码都没找到
        return None
    return found[0].data.decode()


print()
print("=" * 62)
print("第 2 步：包成函数，再批量读全部素材图")
print("=" * 62)
print(f"{'文件名':<26}{'读到的内容'}")
print("-" * 62)
names = sorted(n for n in os.listdir(ASSETS) if n.endswith(".png"))
success = 0
for name in names:
    text = read_qr(os.path.join(ASSETS, name))
    if text:
        success += 1
        print(f"{name:<26}{text}")
    else:
        print(f"{name:<26}（读不出来）")
print("-" * 62)
print(f"共 {len(names)} 张，成功 {success} 张，失败 {len(names) - success} 张。")

# ---------------------------------------------------------------------
# 第 3 步：故意喂一张"没有二维码"的图，看它怎么反应
# ---------------------------------------------------------------------
print()
print("=" * 62)
print("第 3 步：喂一张没有码的图（体会『失败』长什么样）")
print("=" * 62)
from PIL import ImageDraw
blank = Image.new("RGB", (300, 300), (255, 255, 255))
ImageDraw.Draw(blank).rectangle([100, 100, 200, 200], fill=(0, 0, 0))
blank_path = os.path.join(ASSETS, "_blank_no_qr.png")
blank.save(blank_path)

print("decode(白底黑方块图) 返回：", decode(blank))
print("len(返回值) =", len(decode(blank)), "  ← 0 表示列表是空的，不是报错")
print("read_qr(同一个文件) 返回：", read_qr(blank_path))

# ---------------------------------------------------------------------
# 第 4 步：赛场那台码板，用同样的代码照样能读
# ---------------------------------------------------------------------
print()
print("=" * 62)
print("第 4 步：读模拟赛场的 A4 二维码板")
print("=" * 62)
board = os.path.join(ASSETS, "12_board_a4.png")
scene = os.path.join(ASSETS, "13_scene.png")
print("A4 板（1753x1240，二维码只占中间 472x472）：", read_qr(board))
print("合成场景（640x480，整块板子只占画面一部分）：", read_qr(scene))
print()
print("结论：zbar 不要求二维码铺满整张图，它自己会去找。")
print("     这一点很重要——机器人摄像头拍到的画面里，码永远只占一小块。")
