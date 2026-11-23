# -*- coding: utf-8 -*-
"""
lesson10_reader.py —— 第 10 课：把它做成一个能上车的"读码器"

前 9 课你手里有一堆零散技巧。这一课把它们组装成一个模块：
    · 分级流水线：便宜的先试，成功了立刻收工（省时间）
    · 二次确认：连续读到同一个内容才认账（防误读/防读到别人的码）
    · 自带统计：每一级救了多少次，让"该怎么调"有数据支撑
    · 一个命令行入口：读单图 / 读回放 / 读摄像头，一种代码全都能跑

用法（在教程根目录下）：
    python code/lesson10_reader.py --image assets/08_glare.png
    python code/lesson10_reader.py --replay
    python code/lesson10_reader.py --camera 0 --frames 100
    python code/lesson10_reader.py --self-test

在你自己的机器人工程里，直接 `from lesson10_reader import RobustReader`，
把 read(frame) 挂到主循环里就行。
"""
import os
import sys
import time
import argparse
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "assets")

sys.path.insert(0, HERE)
import zbar_bootstrap
zbar_bootstrap.ensure()

import cv2
import numpy as np
from pyzbar.pyzbar import decode

TASK_CODE = "156+123+516+231"


# =====================================================================
# 一、分级流水线
# =====================================================================
# 每一级是一个 (名字, 函数)。函数吃 BGR 或灰度图，吐一张"喂给 zbar 的图"。
# 顺序就是"从便宜到贵"，这是因为越贵的预处理越慢，
# 而 90% 的正常帧在 L0 就成功了 —— 那 90% 的帧不该为小概率的坏帧付钱。
def _gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def lv_raw(img):
    """L0 原图灰度。成本 ≈ 0.01ms。"""
    return _gray(img)


def lv_median(img):
    """L1 中值滤波 3x3。治椒盐噪声。成本 ≈ 0.02ms。"""
    return cv2.medianBlur(_gray(img), 3)


def lv_otsu(img):
    """L2 OTSU 二值化。治低对比度/光照不均。成本 ≈ 0.05ms。"""
    g = _gray(img)
    return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def lv_clahe(img):
    """L3 CLAHE 局部对比度增强。治整体发灰/偏暗。成本 ≈ 0.07ms。"""
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(_gray(img))


def lv_invert(img):
    """L4 反色。治"浅码深底"。成本 ≈ 0.02ms。
    它很便宜，但故意排在便宜的通用药之后：因为正常码反色后就读不出了，
    它是"专用药"（只治罕见病），让它排在昂贵的放大之前就够了。"""
    return 255 - _gray(img)


def lv_center_crop(img):
    """L5 中心裁剪 + 放大。治"码在画面正中但太小"。
    裁剪掉四周能显著减少 zbar 的搜索面积，等于给它减负。"""
    g = _gray(img)
    h, w = g.shape[:2]
    crop = g[h // 4:h * 3 // 4, w // 4:w * 3 // 4]
    return cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)


def lv_up2(img):
    """L6 最近邻放大 2x。治码偏小。成本 ≈ 0.05ms（预处理）+ 解码变贵。"""
    return cv2.resize(_gray(img), None, fx=2, fy=2,
                      interpolation=cv2.INTER_NEAREST)


def lv_up4(img):
    """L7 最近邻放大 4x。治码很小。成本 ≈ 0.07ms（预处理）+ 解码变贵。"""
    return cv2.resize(_gray(img), None, fx=4, fy=4,
                      interpolation=cv2.INTER_NEAREST)


def lv_up4_cubic(img):
    """L8 三次样条放大 4x。第 09 课实测：有些采样方式下它比最近邻好得多。"""
    return cv2.resize(_gray(img), None, fx=4, fy=4,
                      interpolation=cv2.INTER_CUBIC)


def cv_direct(img):
    """L9 OpenCV 直接解码。zbar 和 OpenCV 是两套独立实现，
    一方的盲区常是另一方的强项（第 09 课：30° 斜拍时反而 OpenCV 更弱，
    但 JPEG 压缩/某些透视下它更稳）。失败时返回 None。"""
    g = _gray(img)
    data, _, _ = cv2.QRCodeDetector().detectAndDecode(g)
    return data or None


def cv_straight(img):
    """L10 OpenCV 摆正后交给 zbar。
    straight 是"纯码点阵"（21×21，每格 1 像素，无白边），1px/格 时 zbar 读不出
    → 必须放大到每格 ≥2 像素（用最近邻），补 4 格白边是稳妥做法（第 12 课 D 实测）。"""
    g = _gray(img)
    _, _, st = cv2.QRCodeDetector().detectAndDecode(g)
    if st is None or getattr(st, "size", 0) == 0:
        return None
    m = st[:, :, 0] if st.ndim == 3 else st
    m = (m > 127).astype(np.uint8) * 255
    m = np.pad(m, 4, constant_values=255)
    m = cv2.resize(m, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
    r = decode(m)
    return r[0].data.decode("utf-8", "replace") if r else None


PIPELINE = [
    ("L0 原图", lv_raw),
    ("L1 中值", lv_median),
    ("L2 OTSU", lv_otsu),
    ("L3 CLAHE", lv_clahe),
    ("L4 反色", lv_invert),
    ("L5 中心裁剪", lv_center_crop),
    ("L6 放大2x", lv_up2),
    ("L7 放大4x", lv_up4),
    ("L8 三次4x", lv_up4_cubic),
]
# 排序理由来自实测耗时（第 10 课 md 里有剖析表），不是猜的：
#   L0~L5 每一级"预处理 + zbar"约 2ms（640x480 干净画面）
#   L6 放大2x ≈ 9ms，L7 放大4x ≈ 26ms，L8 三次4x ≈ 26ms
#   放大之所以贵，是因为喂给 zbar 的图变成 4 倍边长（16 倍像素）。
#   而放大只对"码很小"这一种病有用 —— 所以它必须排到最后。
#
# 历史教训：第一版把三级放大排在 L3~L5，注释却写着"按成本从低到高"。
#   注释是谎话，代码才会骗人。这种"号称按成本排序其实没排"的错误，
#   只有真的去 profile 一次才会发现。

# 最后两级不是"预处理"，而是"换一个解码器/换一条解码路线"，
# 单列出来是为了让统计表里一眼看出"兜底到底有没有派上用场"。
FALLBACKS = [
    ("L9 OpenCV识别", cv_direct),
    ("L10 CV摆正", cv_straight),
]


# =====================================================================
# 二、读码器本体
# =====================================================================
class RobustReader:
    """能上车的读码器。

    参数
      confirm   要连续读到几次相同内容才认账。1 = 读到就认（最快，最容易误判）；
                2~3 = 推荐值（多花 1~2 帧，换来稳）。
      use_opencv_all  True 时，zbar 全失败后让 OpenCV 兜底。
      budget_ms 单帧时间上限（毫秒）。None = 不限。设了以后，跑到预算就收工，
                宁可这一帧读不到，也不让主循环被拖住（上车必设，推荐 30~50）。
      verbose    打印每一级的过程（调试用）。
    """

    def __init__(self, confirm=2, use_opencv_fallback=True, verbose=False,
                 budget_ms=None):
        self.confirm = max(1, confirm)
        self.use_opencv_fallback = use_opencv_fallback
        self.verbose = verbose
        self.budget_ms = budget_ms
        self.stats = Counter()        # 每一级成功了多少次
        self.frames = 0
        self.confirmed = None         # 最终确认的任务码
        self._streak = Counter()      # 连续命中计数
        self.skipped = 0              # 因为超预算而被跳过的帧数

    def read(self, frame):
        """读一帧。返回 (内容 或 None, 用了哪一级)。
        '内容'是字符串；还没达到 confirm 次数时返回 None。"""
        self.frames += 1
        text, level = self._read_raw(frame)
        if text is None:
            self._streak.clear()      # 一旦读不到，连续计数清零（要求"连续"）
            return None, None

        self._streak[text] += 1
        if self.verbose:
            print(f"    第 {self.frames} 帧 命中『{text}』"
                  f"（{self.confirm} 次里第 {self._streak[text]} 次，来自 {level}）")
        if self._streak[text] >= self.confirm:
            self.confirmed = text
            return text, level
        return None, level

    def _read_raw(self, frame):
        """跑一遍流水线。返回 (内容 或 None, 成功的级别名)。"""
        t0 = time.perf_counter()
        for name, fn in PIPELINE:
            if (self.budget_ms is not None
                    and (time.perf_counter() - t0) * 1000 > self.budget_ms):
                self.skipped += 1          # 超预算：这一帧放弃，别拖住主循环
                if self.verbose:
                    print(f"    超出 {self.budget_ms}ms 预算，在 {name} 前收工")
                return None, None
            try:
                r = decode(fn(frame))
            except Exception as e:      # 预处理本身不该炸掉主循环
                if self.verbose:
                    print(f"    {name} 预处理异常：{e}")
                continue
            if r:
                self.stats[name] += 1
                return r[0].data.decode("utf-8", "replace"), name

        # 全失败 → 换解码器兜底
        if self.use_opencv_fallback:
            for name, fn in FALLBACKS:
                try:
                    text = fn(frame)
                except Exception:
                    continue
                if text:
                    self.stats[name] += 1
                    return text, name
        return None, None

    # ---------------- 统计报告 ----------------
    def report(self):
        print("\n" + "-" * 60)
        print("读码器统计报告")
        print("-" * 60)
        print(f"  处理帧数        : {self.frames}")
        total_hit = sum(self.stats.values())
        print(f"  命中帧数        : {total_hit}"
              + (f"（{total_hit / self.frames * 100:.1f}%）" if self.frames else ""))
        print(f"  最终确认的码    : {self.confirmed}")
        if self.stats:
            print("\n  每一级贡献了多少次命中（这张表告诉你该保留哪几级）：")
            for name, _ in PIPELINE + FALLBACKS:
                c = self.stats.get(name, 0)
                if c:
                    bar = "█" * min(40, c)
                    print(f"    {name:<12}{c:>5} 次  {bar}")
            unused = [n for n, _ in PIPELINE + FALLBACKS if not self.stats.get(n)]
            if unused:
                print(f"\n  一次都没用上的级别：{unused}")
                print("  （别急着删！它们防的是『偶尔才出现』的坏帧，")
                print("    比赛现场只要出现一次就够你失分了。）")


# =====================================================================
# 三、三种输入源，一种调用方式
# =====================================================================
def source_from_args(args):
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"读不到图片：{args.image}")
        return [img], f"单张图片 {os.path.basename(args.image)}"
    if args.replay:
        folder = os.path.join(ASSETS, "replay")
        files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                       if f.endswith(".png"))
        return [cv2.imread(f) for f in files], f"回放 {len(files)} 帧"
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"打不开摄像头 /dev/video{args.camera}")
    frames = []
    n = args.frames or 60
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, f"摄像头 /dev/video{args.camera} 抓 {len(frames)} 帧"


def self_test():
    """自检：把资产里 4 张"难题"过一遍流水线，看是否都被救回来。
    这类自检脚本在工程上叫 smoke test（冒烟测试）—— 
    改完代码先跑它，30 秒内知道有没有把东西改坏。"""
    print("=" * 66)
    print("自检：4 张难图 + 1 张正常图")
    print("=" * 66)
    cases = [("01_clean.png", True), ("06_noise.png", True),
             ("03_size_30.png", True), ("09_occluded.png", False),
             ("08_glare.png", False)]
    allok = True
    for fname, should_pass in cases:
        path = os.path.join(ASSETS, fname)
        if not os.path.exists(path):
            print(f"  缺文件 {fname}")
            continue
        reader = RobustReader(confirm=1)
        text, level = reader.read(cv2.imread(path))
        ok = text is not None
        flag = "✓" if ok == should_pass else "✗ 预期不符"
        if ok != should_pass:
            allok = False
        print(f"  {fname:<20} 读到={str(text):<20} 级别={str(level):<12} {flag}")
    print()
    print("  说明：09_occluded 和 08_glare 本来就救不回来（定位角被完全挡住），")
    print("        它们在这里是『负样本』——如果哪天它们突然能读了，")
    print("        反而要怀疑是不是读错了（假阳性）。")
    return allok


def survey():
    """体检模式：把 assets 里每张图单独过一遍读码器，
    打印"哪张图是哪一级救回来的"。写 md 报告、给别人解释时特别有用。"""
    files = sorted(f for f in os.listdir(ASSETS)
                   if f.endswith(".png") and f[0].isdigit())
    files += [f for f in sorted(os.listdir(ASSETS))
              if f == "_blank_no_qr.png"]
    print("=" * 70)
    print("体检：每张图分别是靠哪一级读出来的")
    print("=" * 70)
    print(f"  {'文件':<24}{'读到的内容':<22}{'级别'}")
    print("  " + "-" * 66)
    level_of = {}
    ok = 0
    for f in files:
        reader = RobustReader(confirm=1)
        text, level = reader.read(cv2.imread(os.path.join(ASSETS, f)))
        if text:
            ok += 1
        level_of[f] = level or "（读不出）"
        print(f"  {f:<24}{str(text or '（读不出）'):<22}{level_of[f]}")
    print("  " + "-" * 66)
    print(f"  共 {len(files)} 张，成功 {ok} 张，失败 {len(files) - ok} 张")
    print()
    cnt = Counter(level_of.values())
    print("  各级贡献（按级别统计张数）：")
    for name, _ in PIPELINE + FALLBACKS:
        c = cnt.get(name, 0)
        if c:
            print(f"    {name:<14}{c:>3} 张  " + "█" * c)
    fail = cnt.get("（读不出）", 0)
    if fail:
        print(f"    {'（读不出）':<14}{fail:>3} 张")
    print()
    print("  读这张表要知道的事：")
    print("    · L7 放大4x 救回了 03_size_30 —— 小码就靠它")
    print("    · L8 三次4x 在回放帧里立功 —— 插值方式要备两种（第 09 课）")
    print("    · L1 中值 救回了 06_noise —— 椒盐噪声的唯一解药")
    print("    · 08_glare / 09_occluded 谁都救不了 —— 这是物理世界的信息丢失")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--confirm", type=int, default=2)
    ap.add_argument("--budget", type=float, default=None,
                    help="单帧时间上限（毫秒），例如 40；不填=不限")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--survey", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if args.survey:
        survey()
        return

    frames, name = source_from_args(args)
    reader = RobustReader(confirm=args.confirm, verbose=args.verbose,
                          budget_ms=args.budget)

    print("=" * 66)
    print(f"第 10 课 · 工程化读码器    输入：{name}   需连续确认：{args.confirm} 次")
    print("=" * 66)

    t0 = time.perf_counter()
    for f in frames:
        text, level = reader.read(f)
        if text:
            print(f"\n  ★ 第 {reader.frames} 帧确认任务码：{text}   （来自 {level}）")
            break
    elapsed = (time.perf_counter() - t0) * 1000

    if reader.confirmed is None:
        print("\n  这一段没读到能确认的码。")

    reader.report()
    n = max(1, reader.frames)
    print(f"\n  耗时：{elapsed:.1f} ms / 实际处理 {reader.frames} 帧"
          f"（平均 {elapsed / n:.1f} ms 每帧）")
    if reader.skipped:
        print(f"  ⚠ 有 {reader.skipped} 帧因为超出 {args.budget}ms 预算被主动放弃。")
    if reader.confirmed is None:
        print("  ⚠ 这一轮是『最坏情况』：没有码，所以流水线 11 级全跑完才放弃。")
        print("     实测（640×480）：纯灰空画面 ≈ 82ms/帧；带噪声的空画面 ≈ 540ms/帧。")
        print("     噪声越大，zbar 越会去追那些『像码的假目标』，所以慢得离谱。")
        print("     上车必做两件事：① 开 budget_ms（默认建议 30~50ms）")
        print("                     ② 没码的帧降低扫描频率（比如每 3 帧才全流程扫一次）")
    else:
        print("  注意：这是『一命中就停』的耗时，不代表最坏情况 —— 最坏情况见 --image 空图测试。")
    print()
    print("  为什么需要『连续 confirm 次』？")
    print("    · 单次命中可能是误读（虽然二维码有 CRC，但屏摄/反光也可能骗过它）")
    print("    · 也可能是你读到了隔壁赛场的板子、或者码板上贴了别的码")
    print("    · 要求『连续读到同一个内容』，能把这两类错误都挡掉，代价只是 1~2 帧")


if __name__ == "__main__":
    main()
