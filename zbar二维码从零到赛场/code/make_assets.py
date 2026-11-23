# -*- coding: utf-8 -*-
"""
make_assets.py —— 一键生成本套教程要用到的全部实验图片

运行方式（在教程根目录下）：
    python code/make_assets.py

生成的东西全部放在 assets/ 目录里。
每一张图都对应正文里某一个"动手做"环节，文件名就是索引。
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import qrcode

# ---------- 路径：不管从哪里运行，都把 assets 定位到教程根目录 ----------
HERE = os.path.dirname(os.path.abspath(__file__))      # .../code
ROOT = os.path.dirname(HERE)                            # 教程根目录
ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)

# 赛场用的任务码（取自命题文件里的例子）
TASK_CODE = "156+123+516+231"

# 中文字体（用来在图片上写说明文字，不带中文字体的话会画成方块）
CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def font(size):
    for p in (CJK_FONT,
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
              "/usr/share/fonts/truetype/arphic/uming.ttc"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def out(name):
    return os.path.join(ASSETS, name)


def make_qr(text, ec="M", box=10, border=4, version=None):
    """生成一张干净的二维码 PIL 图。

    参数说明见第 04 课：
      ec     —— 纠错等级 L<M<Q<H，级别越高越抗污损，但格子越密
      box    —— 每个小方块放大成多少像素
      border —— 四周留几格白边（标准要求 >=4 格，否则很多库读不出）
      version—— 固定版本号；None 表示让库按内容自动选
    """
    ec_map = {
        "L": qrcode.constants.ERROR_CORRECT_L,
        "M": qrcode.constants.ERROR_CORRECT_M,
        "Q": qrcode.constants.ERROR_CORRECT_Q,
        "H": qrcode.constants.ERROR_CORRECT_H,
    }
    qr = qrcode.QRCode(
        version=version,
        error_correction=ec_map[ec],
        box_size=box,
        border=border,
    )
    qr.add_data(text)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white").convert("RGB")


# =====================================================================
# 一、干净基准图
# =====================================================================
def build_clean():
    img = make_qr(TASK_CODE)
    img.save(out("01_clean.png"))
    return ("01_clean.png", img.size)


# =====================================================================
# 二、不同纠错等级（第 04 课用）
# =====================================================================
def build_ec_levels():
    rows = []
    for ec in "LMQH":
        img = make_qr(TASK_CODE, ec=ec)
        img.save(out(f"02_ec_{ec}.png"))
        rows.append((ec, img.size))
    return rows


# =====================================================================
# 三、不同尺寸（第 09 课用）：把干净图缩到很小
# =====================================================================
def build_sizes():
    base = make_qr(TASK_CODE, box=10, border=4)          # 290x290（version 1，21+8=29 格）
    rows = []
    for side in (400, 200, 120, 80, 60, 40, 30):
        img = base.resize((side, side), Image.NEAREST)
        img.save(out(f"03_size_{side}.png"))
        rows.append((side, img.size))
    return rows


# =====================================================================
# 四、各种"劣化"图（第 07、09 课用）
# =====================================================================
def build_degraded():
    base = make_qr(TASK_CODE, box=10, border=4)
    rows = []

    # 1) 高斯模糊：模拟手抖 / 对焦没对准
    img = base.filter(__import__("PIL.ImageFilter", fromlist=["ImageFilter"]).GaussianBlur(3))
    img.save(out("04_blur.png")); rows.append(("04_blur.png", img.size))

    # 2) 低对比度：浅灰码写在白纸上，模拟曝光过度
    arr = np.asarray(base).astype(np.int16)
    arr = (arr * 0.25 + 255 * 0.68).clip(0, 255).astype(np.uint8)   # 黑->浅灰，白->白
    img = Image.fromarray(arr); img.save(out("05_lowcontrast.png"))
    rows.append(("05_lowcontrast.png", img.size))

    # 3) 椒盐噪声：模拟高 ISO 噪点 / 便宜的摄像头
    arr = np.asarray(base).astype(np.uint8).copy()
    rng = np.random.default_rng(2027)
    mask = rng.random(arr.shape[:2])
    arr[mask < 0.05] = 0
    arr[mask > 0.95] = 255
    img = Image.fromarray(arr); img.save(out("06_noise.png"))
    rows.append(("06_noise.png", img.size))

    # 4) 整体变暗：模拟车间光线不足
    arr = (np.asarray(base).astype(np.float32) * 0.45).astype(np.uint8)
    img = Image.fromarray(arr); img.save(out("07_dark.png"))
    rows.append(("07_dark.png", img.size))

    # 5) 反光斑：模拟亚克力板/灯光反射，一块白色高光盖住局部
    img = base.copy()
    d = ImageDraw.Draw(img)
    d.ellipse([150, 60, 330, 200], fill=(255, 255, 255))
    img.save(out("08_glare.png")); rows.append(("08_glare.png", img.size))

    # 6) 局部遮挡：模拟污渍 / 贴纸挡住一角
    img = base.copy()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 110, 110], fill=(255, 255, 255))
    img.save(out("09_occluded.png")); rows.append(("09_occluded.png", img.size))

    return rows


# =====================================================================
# 五、旋转 / 透视（第 07、09 课用）
# =====================================================================
def build_geometry():
    import cv2
    base = make_qr(TASK_CODE, box=10, border=4)
    rows = []

    for deg in (10, 15, 30, 45, 90, 180):
        img = base.rotate(deg, expand=True, fillcolor=(255, 255, 255),
                          resample=Image.BICUBIC)
        img.save(out(f"10_rot{deg}.png")); rows.append((f"10_rot{deg}.png", img.size))

    # 透视畸变：模拟摄像头斜着看二维码板（比赛现场最常见的情况）
    arr = np.asarray(base)
    h, w = arr.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * 0.18, h * 0.05],
                      [w * 0.96, h * 0.22],
                      [w * 0.80, h * 0.98],
                      [w * 0.02, h * 0.72]])
    M = cv2.getPerspectiveTransform(src, dst)
    big = cv2.warpPerspective(arr, M, (w, h), borderValue=(255, 255, 255))
    cv2.imwrite(out("11_perspective.png"), big); rows.append(("11_perspective.png", (w, h)))

    return rows


# =====================================================================
# 六、模拟赛场：A4 横放二维码板（第 03、10 课用）
#   命题原文：内侧垂直安装 1 个 A4 大小的二维码板（横放），
#            二维码位于板的中间，尺寸为 80x80mm（包含边框）
# =====================================================================
def build_competition_board():
    DPI = 150
    mm = DPI / 25.4                       # 1 毫米 = 多少像素
    W, H = int(297 * mm), int(210 * mm)   # A4 横放
    board = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(board)

    # 板子上写点说明（真实赛场板子上也会有方位提示，这里做个示意）
    d.text((int(20 * mm), int(12 * mm)), "TASK CODE", font=font(int(10 * mm)),
           fill=(30, 30, 30))
    d.text((int(20 * mm), int(180 * mm)), "智能物流搬运 · 任务码板（模拟）",
           font=font(int(8 * mm)), fill=(90, 90, 90))

    # 80mm 的二维码，贴在正中间
    side = int(80 * mm)
    qr_img = make_qr(TASK_CODE, box=10, border=4).resize((side, side), Image.NEAREST)
    x, y = (W - side) // 2, (H - side) // 2
    board.paste(qr_img, (x, y))

    board.save(out("12_board_a4.png"))

    # 再合成一张"机器人摄像头看到的样子"：远一点 + 轻微透视 + 灰色地面背景
    import cv2
    arr = cv2.cvtColor(np.asarray(board), cv2.COLOR_RGB2BGR)
    scene = np.full((480, 640, 3), 150, np.uint8)          # 灰色车道
    small = cv2.resize(arr, (420, 297))
    scene[90:90 + 297, 110:110 + 420] = small
    # 加一点透视，让板子看起来是斜的
    h, w = scene.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[30, 20], [w - 10, 0], [w, h], [0, h]])
    scene = cv2.warpPerspective(scene, cv2.getPerspectiveTransform(src, dst), (w, h))
    cv2.imwrite(out("13_scene.png"), scene)

    return [("12_board_a4.png", board.size), ("13_scene.png", (640, 480))]


# =====================================================================
# 七、一条"扫码演示带"：给第 06 课的回放模式当假摄像头用
#    把同一张码在不同位置/角度画 60 帧，模拟机器人边开边扫
# =====================================================================
def build_replay_frames():
    import cv2
    folder = os.path.join(ASSETS, "replay")
    os.makedirs(folder, exist_ok=True)

    qr_img = np.asarray(make_qr(TASK_CODE, box=6, border=4))      # RGB
    qr_bgr = cv2.cvtColor(qr_img, cv2.COLOR_RGB2BGR)

    n = 60
    for i in range(n):
        t = i / (n - 1)
        frame = np.full((480, 640, 3), 160, np.uint8)
        # 位置：从左上飘到中间
        cx = int(120 + 260 * t)
        cy = int(110 + 120 * t)
        # 大小：从很小长到正常（模拟机器人开近了）
        s = int(70 + 200 * t)
        # 旋转：一路转 20 度
        ang = 20 * t
        small = cv2.resize(qr_bgr, (s, s), interpolation=cv2.INTER_NEAREST)
        M = cv2.getRotationMatrix2D((s / 2, s / 2), ang, 1.0)
        rot = cv2.warpAffine(small, M, (s, s), borderValue=(255, 255, 255))
        x0, y0 = cx - s // 2, cy - s // 2
        if y0 >= 0 and x0 >= 0 and y0 + s <= 480 and x0 + s <= 640:
            frame[y0:y0 + s, x0:x0 + s] = rot
        # 前 10 帧故意加模糊，模拟刚上电还没对焦
        if i < 10:
            frame = cv2.GaussianBlur(frame, (5, 5), 1.5)
        cv2.imwrite(os.path.join(folder, f"frame_{i:03d}.png"), frame)

    return folder, n


# =====================================================================
# 主流程
# =====================================================================
if __name__ == "__main__":
    print("[1/7] 干净基准图 ...", build_clean())
    print("[2/7] 纠错等级图 ...")
    for ec, size in build_ec_levels():
        print(f"      02_ec_{ec}.png  {size[0]}x{size[1]}")
    print("[3/7] 尺寸梯度图 ...")
    for side, size in build_sizes():
        print(f"      03_size_{side}.png  实际{size[0]}x{size[1]}")
    print("[4/7] 劣化图 ...")
    for n, size in build_degraded():
        print(f"      {n}  {size[0]}x{size[1]}")
    print("[5/7] 几何变换图 ...")
    for n, size in build_geometry():
        print(f"      {n}  {size[0]}x{size[1]}")
    print("[6/7] 模拟赛场 A4 板 ...")
    for n, size in build_competition_board():
        print(f"      {n}  {size[0]}x{size[1]}")
    print("[7/7] 回放帧序列 ...")
    folder, n = build_replay_frames()
    print(f"      {folder}  共 {n} 帧")
    print("\n全部素材已生成到:", ASSETS)
