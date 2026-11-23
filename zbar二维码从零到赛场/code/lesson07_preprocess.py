# -*- coding: utf-8 -*-
"""
lesson07_preprocess.py —— 第 07 课：图像预处理，把读不出的码救回来

核心思想（非常重要，记住这一句就够本课了）：
    zbar 不是万能的，但它是"免费的"。你多花 1 毫秒做点预处理，
    常常能换回 30% 的命中率。而命中率就是比赛分数。

本课用一张"体检表"的方式，把 8 种预处理手段 × 11 张难图 全部跑一遍，
让你亲眼看到哪种手段救哪种病。

运行（在教程根目录下）：
    python code/lesson07_preprocess.py
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "assets")

sys.path.insert(0, HERE)
import zbar_bootstrap
zbar_bootstrap.ensure()

import cv2
import numpy as np
import unicodedata
from pyzbar.pyzbar import decode


def dwidth(text):
    """算字符串在终端里占几格。中文占 2 格、英文占 1 格，
    这样打印出来的表格才是真正对齐的（不然中文表头会歪）。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def pad(text, width):
    """把 text 右补空格到指定显示宽度。"""
    return text + " " * max(0, width - dwidth(text))


# =====================================================================
# 一、预处理"药方"：每个函数都吃一张 BGR 图，吐一张新图
#     （zbar 只关心明暗，所以大部分药方最后都转成灰度单通道）
# =====================================================================
def p_raw(img):
    """不吃药：直接把原图转灰度喂给 zbar。
    这是基准线，所有比较都以它为参照。"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def p_median3(img):
    """中值滤波 3x3：专治"椒盐噪声"（随机黑白麻点）。
    原理：把每个像素换成它周围 9 个像素的"中位数"。
         孤立的黑点/白点会被周围的中位数挤掉，而码的黑白块是大片连续的，
         中位数不会被改变 —— 这就是它比高斯模糊聪明的地方。"""
    return cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 3)


def p_gauss3(img):
    """高斯模糊 3x3：专治轻微模糊/锯齿，但会把细节一起抹平。
    对比着记：中值滤波"删掉孤立的坏点"，高斯模糊"把大家都搅匀"。"""
    return cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)


def p_otsu(img):
    """OTSU 自动阈值二值化：把灰度图变成纯黑纯白。
    原理：它自己统计整张图的灰度分布，找一个"最能分开两堆"的界线。
    好处：不用你猜阈值。
    坏处：如果整张图亮度不均（有阴影/反光），一个全局阈值会一刀切错。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def p_adaptive(img):
    """自适应阈值：每个像素跟"它附近一小片的平均值"比，而不是跟全图比。
    好处：天生抗光照不均（半边亮半边暗也能分开）。
    坏处：码的格子太小时，"附近一小片"里既有黑又有白，容易糊成噪点。
    参数 31 = 考察窗口大小（必须是奇数），5 = 减去的常数（调大更"狠"）。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 5)


def p_clahe(img):
    """CLAHE 局部对比度增强：把"暗部"拉亮、"亮部"压暗，让浅灰的字变清楚。
    专治：整体发灰、低对比度（比如摄像头曝光不足、或者浅色码印在白纸上）。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)


def p_upscale2(img):
    """最近邻放大 2 倍：专治"码太小，一个格子不到 2 像素"。
    为什么必须用最近邻（INTER_NEAREST）而不是双线性/三次样条？
      最近邻 = 每个像素原样复制成 2x2 方块，黑白边界保持锐利。
      双线性和三次样条 = 会插值出灰色过渡带，反而把边界糊掉。
    这是本课最容易踩的坑，第 09 课还会再看到实测数据。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)


def p_invert(img):
    """反色：专治"浅色码 + 深色底"。
    zbar 默认只认"深色码 + 浅色底"。遇到反色的码，把整张图 255-x 翻过来就行。
    公式很朴素：白(255) 变黑(0)，黑(0) 变白(255)。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return 255 - gray


def p_upscale4(img):
    """最近邻放大 4 倍：给"实在太小"的码用。第 09 课会看到它的临界点。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)


def p_median_otsu(img):
    """组合药方：先去噪，再二值化。
    为什么组合？因为二值化会把噪声"固化"成硬黑硬白，之后就再也救不回来了。
    顺序必须是"先柔后刚"：先中值去噪，再 OTSU 切黑白。反过来就晚了。"""
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 3)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


MEDICINES = [
    ("原图", p_raw),
    ("中值3", p_median3),
    ("高斯3", p_gauss3),
    ("OTSU", p_otsu),
    ("自适应", p_adaptive),
    ("CLAHE", p_clahe),
    ("放大2x", p_upscale2),
    ("放大4x", p_upscale4),
    ("反色", p_invert),
    ("中值+OTSU", p_median_otsu),
]

# 参与体检的"病人"
PATIENTS = [
    ("01_clean.png", "干净基准"),
    ("03_size_30.png", "码只有 30px"),
    ("04_blur.png", "高斯模糊"),
    ("05_lowcontrast.png", "低对比度"),
    ("06_noise.png", "椒盐噪声"),
    ("07_dark.png", "光线很暗"),
    ("08_glare.png", "反光斑"),
    ("09_occluded.png", "局部遮挡"),
    ("11_perspective.png", "透视歪斜"),
]


def try_decode(gray_or_bgr):
    """统一的解码入口。zbar 能吃单通道灰度，也能吃三通道 BGR。"""
    r = decode(gray_or_bgr)
    return r[0].data.decode("utf-8", "replace") if r else None


def main():
    print("=" * 78)
    print("第 07 课 · 预处理体检表：横着是药方，竖着是病症")
    print("=" * 78)

    # ---------------- 第 1 步：打印体检表 ----------------
    COL = 12
    header = pad("病人", 16) + "".join(pad(n, COL) for n, _ in MEDICINES)
    print("\n【第 1 步】体检表（OK=读出来了，X=读不出）\n")
    print(header)
    print("-" * dwidth(header))

    rescue_map = {}      # 病症 -> 能读出来的药方列表
    for fname, desc in PATIENTS:
        path = os.path.join(ASSETS, fname)
        if not os.path.exists(path):
            print(f"{desc}（缺 {fname}，先跑 make_assets.py）")
            continue
        img = cv2.imread(path)
        row = pad(desc, 16)
        good = []
        for name, fn in MEDICINES:
            try:
                got = try_decode(fn(img))
            except Exception:
                got = None
            row += pad("OK" if got else "X", COL)
            if got:
                good.append(name)
        print(row)
        rescue_map[desc] = good

    # ---------------- 第 2 步：得出结论 ----------------
    print("\n【第 2 步】每张图分别被谁救了？\n")
    for desc, good in rescue_map.items():
        head = pad(desc, 14)
        if not good:
            print(f"  {head}全部药方都失败 → 这种病只能靠『换硬件/换角度』治，见下方说明")
        elif good[0] == "原图":
            extra = [g for g in good if g != "原图"]
            print(f"  {head}原图就读得出" + (f"，另外 {extra} 也行" if extra else ""))
        else:
            print(f"  {head}原图读不出 → 被 {good} 救活 ★")

    print()
    print("  为什么『反色』那一列几乎全是大 X？")
    print("    反色是『专用药』：它只治『浅码深底』这一种病。")
    print("    正常的『深码浅底』图被一反色，就变成 zbar 不认的样子了。")
    print("    → 通用药方（中值/OTSU/放大）可以无脑试；专用药方必须有条件地用。")

    print("\n【第 3 步】几种药方的耗时（1 毫秒的取舍，很关键）\n")
    test_img = cv2.imread(os.path.join(ASSETS, "01_clean.png"))
    print(f"{'药方':<12}{'单次耗时(ms)':>14}{'放大倍数':>10}")
    print("-" * 36)
    base = None
    for name, fn in MEDICINES:
        t0 = time.perf_counter()
        for _ in range(100):
            fn(test_img)
        dt = (time.perf_counter() - t0) / 100 * 1000
        if base is None:
            base = dt
        print(f"{name:<12}{dt:>14.3f}{dt / base:>9.1f}x")
    print()
    print("读法：自适应阈值最贵（十几倍），放大4x / CLAHE 次之，中值/反色最便宜。")
    print("     正确策略是：先吃便宜药（原图），不行再吃贵的。第 10 课会写这个策略。")
    print("     （这些数字每次跑都会有 ±20% 的抖动，是 CPU 调度造成的，看数量级就行。）")

    # ---------------- 第 4 步：把对比图存下来，自己看一眼 ----------------
    print("\n【第 4 步】把『治病前后』拼成图存下来，你可以直接打开看\n")
    pairs = [("06_noise.png", "中值3", p_median3),
             ("05_lowcontrast.png", "CLAHE", p_clahe),
             ("08_glare.png", "自适应", p_adaptive),
             ("03_size_30.png", "放大4x", p_upscale4)]
    tiles = []
    for fname, label, fn in pairs:
        p = os.path.join(ASSETS, fname)
        if not os.path.exists(p):
            continue
        img = cv2.imread(p)
        before = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        after = fn(img)
        # 统一高度，方便拼在一起
        h = 200
        b = cv2.resize(before, (int(before.shape[1] * h / before.shape[0]), h))
        a = cv2.resize(after, (b.shape[1], h))
        if a.ndim == 2:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        if b.ndim == 2:
            b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
        tile = np.hstack([b, np.full((h, 8, 3), 128, np.uint8), a])
        tiles.append(tile)
        print(f"  {pad(fname, 20)}用『{label}』处理  结果："
              f"{'救活了' if try_decode(after) else '还是读不出'}")

    if tiles:
        w = max(t.shape[1] for t in tiles)
        padded = [np.hstack([t, np.full((t.shape[0], w - t.shape[1], 3), 255, np.uint8)])
                  for t in tiles]
        out = np.vstack(padded)
        outp = os.path.join(ASSETS, "20_before_after.png")
        cv2.imwrite(outp, out)
        print(f"\n  对比图已保存：{outp}")
        print("  （每一行：左边=原图，右边=处理后的图）")

    # ---------------- 第 5 步：反色码这个特殊病 ----------------
    print("\n【第 5 步】特殊病：浅色码印在深色底上\n")
    import qrcode
    qr = qrcode.QRCode(box_size=8, border=4)
    qr.add_data("156+123+516+231")
    qr.make(fit=True)
    inv_img = qr.make_image(fill_color="white", back_color="black").convert("RGB")
    arr = cv2.cvtColor(np.asarray(inv_img), cv2.COLOR_RGB2BGR)
    print(f"  反色码 原始    ：{'读出来了' if try_decode(p_raw(arr)) else '读不出'}")
    print(f"  反色码 + 反色  ："
          f"{'读出来了 ← 救活' if try_decode(p_invert(arr)) else '还是读不出'}")
    print("  记这一条：zbar 只认『深码浅底』。遇到反色的板子（比如深色亚克力 + 白码），")
    print("           把图反色再读，几乎零成本。")

    print("\n" + "=" * 78)
    print("本课要背下来的三句话：")
    print("  1. 便宜药先吃：原图 → 中值 → OTSU → 放大 → 反色，按成本从低到高排队。")
    print("  2. 顺序不能反：先去噪再二值化，先柔后刚。")
    print("  3. 放大要用最近邻，别用三次样条（会把黑白边界插成灰色）。")
    print("=" * 78)


if __name__ == "__main__":
    main()
