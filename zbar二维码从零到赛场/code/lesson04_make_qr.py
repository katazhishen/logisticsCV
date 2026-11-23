# -*- coding: utf-8 -*-
"""
lesson04_make_qr.py —— 第 04 课：自己造二维码

为什么算法组要会"造码"？
  因为比赛现场发给你的码是别人印的，你没法改；
  但你自己调试时，需要成百上千张不同难度的码来"喂"你的程序。
  会造码 = 你拥有无限的测试数据。

运行（在教程根目录下）：
    python code/lesson04_make_qr.py
"""
import os
import sys
import qrcode
from qrcode.constants import (ERROR_CORRECT_L, ERROR_CORRECT_M,
                              ERROR_CORRECT_Q, ERROR_CORRECT_H)
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "assets")
GEN = os.path.join(ASSETS, "generated")
os.makedirs(GEN, exist_ok=True)

sys.path.insert(0, HERE)
import zbar_bootstrap
zbar_bootstrap.ensure()
from pyzbar.pyzbar import decode

TASK_CODE = "156+123+516+231"
EC_MAP = {"L": ERROR_CORRECT_L, "M": ERROR_CORRECT_M,
          "Q": ERROR_CORRECT_Q, "H": ERROR_CORRECT_H}


def build(text, ec="M", box=10, border=4, version=None):
    """造一张二维码，返回 (图片, QRCode对象)。

    QRCode 的五个参数，就是"造码"的全部旋钮：
      version          —— 二维码版本 1~40。版本越高格子越多、能装越多字。
                          None = 让库自己算（fit=True 时必须 None 或整数）
      error_correction —— 纠错等级 L/M/Q/H。重复的东西越少 → 能装字越少、越抗坏
      box_size         —— 每个小方块画成几像素。只影响图片分辨率，不影响信息
      border           —— 四周留几格白边。标准要求 ≥4，小于 4 很多库直接放弃
      fill_color / back_color —— 颜色。注意：必须是"深色码 + 浅色底"
    """
    qr = qrcode.QRCode(version=version,
                       error_correction=EC_MAP[ec],
                       box_size=box,
                       border=border)
    qr.add_data(text)
    qr.make(fit=(version is None))
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    return img, qr


print("=" * 66)
print("第 1 步：造出第一张码，并读懂它的『体检报告』")
print("=" * 66)
img, qr = build(TASK_CODE)
path = os.path.join(GEN, "task_clean.png")
img.save(path)
modules = qr.modules_count          # 不含白边的有效格子数（边长上有几格）
print(f"内容        : {TASK_CODE}（{len(TASK_CODE)} 个字符）")
print(f"版本 version: {qr.version}   ← 版本 1 = 21x21 格；每升一版每边 +4 格")
print(f"有效格数    : {modules} x {modules} = {modules * modules} 个方块")
print(f"纠错等级    : {qr.error_correction}  (1=L 0=M 3=Q 2=H)")
print(f"白边 border : 4 格")
print(f"图片尺寸    : {img.size[0]}x{img.size[1]} px")
print(f"验算        : (有效格 {modules} + 白边 {4 * 2}) x 每格 {10}px = "
      f"{(modules + 8) * 10}px  ← 与图片尺寸一致")
print(f"读回来验证  : {decode(img)[0].data.decode()}")

print()
print("=" * 66)
print("第 2 步：同样的内容，四档纠错等级，代价是什么？")
print("=" * 66)
print(f"{'等级':<6}{'版本':<6}{'有效格':<10}{'图片边长':<10}{'能救回多少':<12}")
print("-" * 66)
for ec in "LMQH":
    im, q = build(TASK_CODE, ec=ec)
    im.save(os.path.join(GEN, f"ec_{ec}.png"))
    label = {"L": "约 7%", "M": "约 15%", "Q": "约 25%", "H": "约 30%"}[ec]
    print(f"{ec:<6}{q.version:<6}{q.modules_count:<10}{im.size[0]:<10}{label:<12}")
print("-" * 66)
print("看出来的规律：")
print("  · L→Q 阶段版本没变（都是 version 1，21x21 格），因为内容够短，装得下。")
print("  · 到 H，同样的字塞不进 21x21 了，库自动升到 version 2（25x25 格）。")
print("  · 格数一多，同样印成 80mm 的话每格就更小 → 摄像头更难看清。")
print("  → 这就是工程上的权衡：抗污损 ↑ 但可读距离 ↓。赛场建议 M 或 Q。")

print()
print("=" * 66)
print("第 3 步：只改『每格几像素』和『白边』，信息量会不会变？")
print("=" * 66)
for box in (2, 4, 10, 20):
    im, q = build(TASK_CODE, box=box)
    print(f"box_size={box:<3} 图片 {im.size[0]:>4}x{im.size[1]:<4} 有效格还是 "
          f"{q.modules_count}x{q.modules_count}  读回：{decode(im)[0].data.decode()}")
print()
for border in (0, 1, 2, 4):
    im, q = build(TASK_CODE, border=border)
    im.save(os.path.join(GEN, f"border_{border}.png"))
    r = decode(im)
    print(f"border={border:<2} 图片 {im.size[0]:>3}px  单独白底读回："
          f"{r[0].data.decode() if r else '读不出来'}")

print()
print("但是——把码贴到板子上，旁边紧挨着深色块时，白边就变成生死线了：")
for border in (0, 2, 4):
    im, _ = build(TASK_CODE, border=border)
    canvas = Image.new("RGB", (im.size[0] + 60, im.size[1] + 60), (255, 255, 255))
    canvas.paste(im, (30, 30))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, 30, im.size[1] + 60], fill=(0, 0, 0))   # 左边紧贴一条黑
    r = decode(canvas)
    print(f"border={border:<2} 左侧紧贴黑条  ："
          f"{r[0].data.decode() if r else '读不出来 ← 安静区被黑条吃掉了'}")
print()
print("结论：单独放在白底上时，border=0 也能读出来（zbar 很宽容）；")
print("     但只要旁边有深色东西（板子边框、支架、阴影），白边不够就立刻崩。")
print("     二维码标准规定 ≥4 格，就是为这种情况留的余量。")
print("     你在排版打印时不要为了『省地方』把白边裁掉。")

print()
print("=" * 66)
print("第 4 步：一张图里放三个码（机器人一次拍到多张码怎么办）")
print("=" * 66)
multi = Image.new("RGB", (900, 320), (255, 255, 255))
texts = ["156+123+516+231", "231+312+456+123", "666+111+222+333"]
for i, t in enumerate(texts):
    im, _ = build(t, box=6)
    multi.paste(im, (20 + i * 290, 20))
multi.save(os.path.join(GEN, "multi_three.png"))
found = decode(multi)
print(f"decode() 返回 {len(found)} 个结果：")
for d in found:
    print(f"   内容={d.data.decode():<20} 中心点=({d.rect.left + d.rect.width // 2}, "
          f"{d.rect.top + d.rect.height // 2})")
print("→ 返回值永远是列表。只想要一个时写 decode(img)[0]，"
      "但要先确认列表不空。")

print()
print("=" * 66)
print("第 5 步：坑 —— 二维码里存中文，读回来为什么报错？")
print("=" * 66)
im, _ = build("搬运任务：红黑浅蓝", box=8)
im.save(os.path.join(GEN, "chinese.png"))
raw = decode(im)[0].data
print("原始字节     :", raw)
print("字节长度     :", len(raw), "（如果不按字节存，7 个汉字不可能塞进这么点）")
try:
    print("按 UTF-8 解  :", raw.decode("utf-8"))
except UnicodeDecodeError as e:
    print("按 UTF-8 解  : 失败 ->", e)
print()
print("解释：二维码本身只存『一串字节』，不规定编码。qrcode 库默认按 UTF-8 写，")
print("     所以读的时候也要用 UTF-8 解。赛场任务码是纯数字和加号，不会踩这个坑，")
print("     但如果你以后要读中文标签，记得统一用 .decode('utf-8')。")

print()
print("=" * 66)
print("第 6 步：造一张『赛场同款』80x80mm 的码（按 150 DPI 算）")
print("=" * 66)
DPI = 150
px_per_mm = DPI / 25.4
side = int(80 * px_per_mm)
im, q = build(TASK_CODE, ec="M", box=1, border=4)
im = im.resize((side, side), Image.NEAREST)
p = os.path.join(GEN, "task_80mm_150dpi.png")
im.save(p)
print(f"80mm @150DPI = {side}px 见方，实际存成 {im.size[0]}x{im.size[1]}")
print(f"每格宽度     = {side / (q.modules_count + 8):.2f}px  "
      f"← 这是『能不能被看清』的物理下限")
print("读回验证     :", decode(im)[0].data.decode())
print()
print("提示：每格小于 2px 时，很多解码器就开始不稳。")
print("     所以摄像头分辨率不是越高越好，而是要让『码在画面里够大』。")

print()
print(f"本课生成的文件都在：{GEN}")
