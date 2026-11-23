# -*- coding: utf-8 -*-
"""
lesson08_parse_task.py —— 第 08 课：把字符串变成"机器人能执行的任务单"

前 7 课你解决的是"怎么把码读出来"。但从 zbar 手里接过来的只是一串字符：
    156+123+516+231
如果直接拿它去控制机械臂，你根本不知道该先抓哪个颜色、放到几号圆环。

这一课做的是"翻译"：字符串 → 结构化任务单 + 合法性检查。
这一步在算法组里叫「解析（parse）」和「校验（validate）」。

运行（在教程根目录下）：
    python code/lesson08_parse_task.py
"""
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

# =====================================================================
# 一、先把赛题规则写成代码里的"常量表"
#    命题原文：颜色的编号如下：红色1、黄色2、蓝色3、绿色4、黑色5、浅蓝6
# =====================================================================
COLOR_NAME = {
    "1": "红色",
    "2": "黄色",
    "3": "蓝色",
    "4": "绿色",
    "5": "黑色",
    "6": "浅蓝",
}

# 命题原文：任务码由四组三位数组成，四组之间以 "+" 连接
#   第一组 = 第一批三个物料的颜色和搬运顺序
#   第二组 = 第一批物料在粗加工区和暂存区的放置位置
#   第三组 = 第二批三个物料的颜色和搬运顺序
#   第四组 = 第二批物料在粗加工区的放置位置
GROUP_MEANING = ["第一批·颜色顺序", "第一批·放置位置",
                 "第二批·颜色顺序", "第二批·放置位置"]

VALID_DIGITS = set(COLOR_NAME)      # {'1','2','3','4','5','6'}


# =====================================================================
# 二、解析结果的数据结构
#    用 dataclass 而不是普通 dict 的理由：
#      · 拼错字段名时 Python 会立刻报错，而不是悄悄拿到 None
#      · 打印出来自带可读格式，调试时一眼看懂
# =====================================================================
@dataclass
class TaskCode:
    raw: str                                  # 原始读到的字符串
    colors_first: List[str] = field(default_factory=list)   # 第一批颜色 id
    pos_first: List[int] = field(default_factory=list)      # 第一批位置
    colors_second: List[str] = field(default_factory=list)  # 第二批颜色 id
    pos_second: List[int] = field(default_factory=list)     # 第二批位置

    # ---- 给人看的摘要 ----
    def colors_first_cn(self):
        return [COLOR_NAME[c] for c in self.colors_first]

    def colors_second_cn(self):
        return [COLOR_NAME[c] for c in self.colors_second]

    def summary(self):
        return (f"第1批：依次搬 {self.colors_first_cn()} → 放到圆环 "
                f"{self.pos_first}\n"
                f"第2批：依次搬 {self.colors_second_cn()} → 放到圆环 "
                f"{self.pos_second}")


# =====================================================================
# 三、归一化：把"人可能输错/摄像头可能带来的脏东西"清掉
# =====================================================================
def normalize(text):
    """在解析之前先做一次"洗澡"。

    真实会遇到的情况：
      · 手机上输入法打出来的全角字符：１５６＋１２３ （人眼一模一样，机器完全不同）
      · 复制粘贴带进来的空格、换行、制表符
      · 有人用小写字母 o 代替数字 0（打错，但很常见）

    这一步叫 normalize（归一化）：把各种等价写法统一成一种标准形式。
    """
    if text is None:
        return ""
    # NFKC 会把全角数字/加号转成半角，这是标准库里现成的工具
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()            # 去掉首尾空白
    text = re.sub(r"\s+", "", text)  # 去掉中间所有空白（含换行、制表符）
    return text


# =====================================================================
# 四、解析 + 校验
# =====================================================================
class TaskCodeError(ValueError):
    """专门表示"任务码不合法"。自己定义异常类型的好处：
    调用方可以精确地只捕获这一种错误，不会把别的 bug 一起吞掉。"""


def parse_task_code(text):
    """把字符串解析成 TaskCode。不合法就抛 TaskCodeError，并说清哪里不对。"""
    raw = text
    text = normalize(text)

    if not text:
        raise TaskCodeError("内容为空")

    # ---- 1. 形状检查：必须正好是 4 组、每组 3 个字符、用 + 连接 ----
    # 正则逐段解释：
    #   ^ 开头   $ 结尾
    #   [1-6]{3}  连续 3 个 1~6 的数字
    #   \+        一个字面意义的加号（正则里 + 有特殊含义，所以要转义）
    #   (...){3}  前面这块重复 3 次，也就是"加号+三位数"再来三遍
    pattern = r"^[1-6]{3}(?:\+[1-6]{3}){3}$"
    if not re.match(pattern, text):
        # 细分报错原因，别只丢一句"格式错"
        groups = text.split("+")
        if len(groups) != 4:
            raise TaskCodeError(
                f"应该用 + 分成 4 组，实际分了 {len(groups)} 组：{groups}")
        for i, g in enumerate(groups):
            if len(g) != 3:
                raise TaskCodeError(
                    f"第 {i + 1} 组『{g}』长度是 {len(g)}，必须是 3 位")
            bad = [ch for ch in g if ch not in VALID_DIGITS]
            if bad:
                raise TaskCodeError(
                    f"第 {i + 1} 组『{g}』里有非法字符 {bad}，只能是 1~6")
        raise TaskCodeError(f"格式不匹配：{text}")

    g = text.split("+")

    # ---- 2. 语义检查：赛题给的额外约束 ----
    # 命题原文：初赛"确定需要搬运的三种不同颜色的物料"，
    #          物料"放置在粗加工区和暂存区的 1、2、3 号圆环上"。
    # 所以：同一批的三个颜色必须互不相同；三个位置也必须互不相同。
    for idx in (0, 2):
        if len(set(g[idx])) != 3:
            raise TaskCodeError(
                f"第 {idx + 1} 组『{g[idx]}』里有重复颜色 —— "
                f"赛题要求三种物料颜色不同")
    for idx in (1, 3):
        if len(set(g[idx])) != 3:
            raise TaskCodeError(
                f"第 {idx + 1} 组『{g[idx]}』里有重复位置 —— "
                f"赛题要求放在 1、2、3 号不同圆环上")

    return TaskCode(
        raw=raw,
        colors_first=list(g[0]),
        pos_first=[int(c) for c in g[1]],
        colors_second=list(g[2]),
        pos_second=[int(c) for c in g[3]],
    )


# =====================================================================
# 五、测试：把"应该成功"和"应该失败"的例子都写出来
#    这叫「测试用例」。工程上先写用例再写代码，能少踩一半坑。
# =====================================================================
def run_tests():
    print("=" * 70)
    print("第 1 步：正常任务码 —— 能不能翻译成人话？")
    print("=" * 70)

    ok_cases = [
        "156+123+516+231",        # 命题原文给的例子
        "123+123+456+321",
        "１５６＋１２３＋５１６＋２３１",   # 全角字符，人眼一样，机器不同
        "  156 + 123 + 516 + 231 ",  # 带空格
        "156+123+516+231\n",      # 带换行（从文件/串口读进来很常见）
    ]
    for c in ok_cases:
        try:
            t = parse_task_code(c)
            print(f"\n  输入：{c!r}")
            print(f"  结果：合法 ✓")
            print("        " + t.summary().replace("\n", "\n        "))
        except TaskCodeError as e:
            print(f"\n  输入：{c!r}\n  结果：不合法 ✗ —— {e}")

    print()
    print("=" * 70)
    print("第 2 步：错误输入 —— 能不能说清『错在哪』？")
    print("=" * 70)
    bad_cases = [
        ("156+123+516",          "只有 3 组"),
        ("156+123+516+231+222",  "5 组"),
        ("156+12+516+231",       "第 2 组只有 2 位"),
        ("156+123+516+23X",      "含字母 X"),
        ("156+123+516+230",      "有 0，颜色编号只有 1~6"),
        ("156+789+516+231",      "有 7、8、9，超出颜色编号"),
        ("155+123+516+231",      "颜色 5 重复了两次"),
        ("156+113+516+231",      "位置 1 重复了两次"),
        ("",                     "空字符串"),
        (None,                   "None（比如读码失败直接传进来）"),
        ("hello",                "完全不是任务码"),
    ]
    for c, why in bad_cases:
        try:
            parse_task_code(c)
            print(f"  输入 {c!r:<26} 竟然通过了 ← 说明校验写漏了！")
        except TaskCodeError as e:
            print(f"  输入 {c!r:<26} 被拦下：{e}")

    print()
    print("=" * 70)
    print("第 3 步：为什么必须有『语义校验』？")
    print("=" * 70)
    print("  zbar 出来的二维码自带 CRC 校验，『读错一个数字』的概率极低。")
    print("  但下面这两种情况是真的会发生：")
    print("    ① 你读到了别人的/上一轮的码 —— 内容完全合法，但任务不是你的")
    print("    ② 板子上印的码本身有问题 —— 比如印错/排版错")
    print("  语义校验（颜色不重复、位置不重复）能挡掉其中一部分。")
    print("  严格来说它挡不住 ①，所以正式比赛还要加『确认机制』：")
    print("  读到码 → 显示在屏幕上 → 等 0.5 秒再读一次，两次一致才认。")
    print("  这套机制第 10 课会实装。")

    print()
    print("=" * 70)
    print("第 4 步：把任务单渲染成『显示装置』上要显示的样子")
    print("=" * 70)
    # 命题原文：必须配备任务码显示装置…亮光显示…字体高度不小于 12mm
    # 这里只演示"画图"，真机上你要把这套逻辑搬到 OLED / LCD / 点阵屏
    from PIL import Image, ImageDraw, ImageFont
    t = parse_task_code("156+123+516+231")

    W, H = 800, 240
    img = Image.new("RGB", (W, H), (0, 0, 0))     # 黑底
    d = ImageDraw.Draw(img)
    fp = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    try:
        f_big = ImageFont.truetype(fp, 64)
        f_small = ImageFont.truetype(fp, 28)
    except OSError:
        f_big = f_small = ImageFont.load_default()

    d.text((20, 20), t.raw, font=f_big, fill=(0, 255, 120))     # 亮绿=亮光显示
    d.text((20, 110), "第1批 " + " ".join(t.colors_first_cn())
           + "  位" + str(t.pos_first), font=f_small, fill=(255, 255, 255))
    d.text((20, 155), "第2批 " + " ".join(t.colors_second_cn())
           + "  位" + str(t.pos_second), font=f_small, fill=(255, 255, 255))
    d.text((20, 200), "抓取 0/6   放置 0/6   状态：待出发", font=f_small,
           fill=(0, 180, 255))

    here = os.path.dirname(os.path.abspath(__file__))
    outp = os.path.join(os.path.dirname(here), "assets", "21_display_panel.png")
    img.save(outp)
    print(f"  已生成显示装置示意图：{outp}")
    print(f"  尺寸 {W}x{H}；按 150DPI 换算，64px 高 ≈ 10.8mm，")
    print("  正式作品请把字号调到 ≥12mm 对应的像素数（150DPI 下约 71px 起）。")
    print()
    print("  注意命题这句：『该装置必须放置在搬运机器人上部醒目位置，")
    print("  亮光显示，且不被任何物体遮挡，字体高度不小于 12mm』")
    print("  —— 这是硬件组的活，但显示什么内容、什么时候刷新，是算法组定。")

    print()
    print("=" * 70)
    print("第 5 步：和读码串起来 —— 从一张图到一份任务单")
    print("=" * 70)
    sys.path.insert(0, here)
    import zbar_bootstrap
    zbar_bootstrap.ensure()
    from PIL import Image as PILImage
    from pyzbar.pyzbar import decode
    asset = os.path.join(os.path.dirname(here), "assets", "01_clean.png")
    text = decode(PILImage.open(asset))[0].data.decode("utf-8")
    print(f"  第 1 步 读码 → {text!r}")
    task = parse_task_code(text)
    print(f"  第 2 步 解析 → 合法 ✓")
    print(f"  第 3 步 生成动作序列：")
    for i, (c, p) in enumerate(zip(task.colors_first, task.pos_first), 1):
        print(f"         [第一批 {i}/3] 去原料区抓 {COLOR_NAME[c]}(id={c}) "
              f"→ 放到 {p} 号圆环")
    for i, (c, p) in enumerate(zip(task.colors_second, task.pos_second), 1):
        print(f"         [第二批 {i}/3] 去原料区抓 {COLOR_NAME[c]}(id={c}) "
              f"→ 码垛放到 {p} 号圆环")
    print("  —— 到这里，『读码』这件事才真正对机器人有用。")


if __name__ == "__main__":
    run_tests()
