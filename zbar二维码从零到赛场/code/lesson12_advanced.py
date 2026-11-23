# -*- coding: utf-8 -*-
"""
lesson12_advanced.py —— 第 12 课：五个"进阶小尝试"

基础课学完，你会发现"能读出来"只是及格线。真正拉开差距的是这些：
    A. 从码的大小反推距离 —— 让车知道自己该不该再往前开
    B. 多帧投票 —— 用时间换准确率
    C. 按毫米精确打印 A4 码板 —— 直接能上赛场的实物
    D. 用 OpenCV 定位码的四个角 —— 给机械臂/底盘对准用
    E. 顺手把条形码也读了 —— 分拣赛项要用

每个都可以单独跑，改参数、看结果、自己加实验。

用法：
    .venv/bin/python code/lesson12_advanced.py            # 五个全跑
    .venv/bin/python code/lesson12_advanced.py --only C   # 只跑 C
"""
import os
import sys
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
# A. 从码的像素尺寸反推距离
# =====================================================================
def try_A():
    print("=" * 68)
    print("A. 从码的大小反推距离 —— 让车知道『该不该再往前开』")
    print("=" * 68)
    print("""
  原理（小孔成像，初中物理就够）：
      物体实际宽度 W、成像宽度 w（像素）、焦距 f（像素）、距离 d
      满足比例关系：    w / f = W / d
      移项得到：        d = f × W / w

  赛题给的数：二维码板上的码 80mm×80mm（含边框）→ W = 80 mm
  未知的是 f（你摄像头的焦距，单位是像素），它必须标定一次。
""")
    # 标定：把码放在已知距离 d0，量出它占多少像素 w0，则 f = w0 * d0 / W
    W_mm = 80.0
    d0_mm, w0_px = 500.0, 236.0        # 假设：500mm 处量到 236 像素宽
    f = w0_px * d0_mm / W_mm
    print(f"  标定示例：距离 {d0_mm:.0f}mm 时码宽 {w0_px:.0f}px")
    print(f"            → 焦距 f = w0 × d0 / W = {w0_px:.0f} × {d0_mm:.0f} / {W_mm:.0f}"
          f" = {f:.1f} px\n")
    print(f"  {'码在画面里的宽度':>16}{'推算距离':>14}{'每格像素(29格码)':>18}")
    print("  " + "-" * 50)
    for w_px in (40, 60, 80, 100, 140, 200, 300, 400):
        d = f * W_mm / w_px
        per_module = w_px / 29          # 21 格 + 8 格白边 = 29
        print(f"  {w_px:>13} px{d:>12.0f} mm{per_module:>17.2f} px")
    print("""
  怎么用这张表：
    · 第 09 课实测：640×480 下码宽 80px（2.76px/格）时 zbar 命中 100%，
      60px（2.07px/格）只剩 75%，40px（1.38px/格）直接 0%。
      查表就是：想稳读，距离要 ≤1475mm；想更保险（≥120px），距离要 ≤983mm。
    · 于是你可以写：if 码宽 < 80: 车往前挪 100mm，重读。
    · 这叫『闭环』：不是猜，是量了再动。
  怎么标定你自己的摄像头：
    1) 把 80mm 的码板摆在你量得准的距离（比如 500mm）
    2) 拍一张，用 pyzbar 读 .rect.width（或 polygon 边长）
    3) 代进 f = w0 × d0 / W。拍 3 次取平均，误差能到 5% 以内。
""")


# =====================================================================
# B. 多帧投票
# =====================================================================
def try_B():
    print("=" * 68)
    print("B. 多帧投票 —— 用时间换准确率")
    print("=" * 68)
    print("""
  第 10 课的『连续 2 次相同』是投票的简化版。这里写通用的『滑动窗口投票』：
  攒最近 N 帧的读取结果，票数过半才认。
  好处：偶发的误读会被多数票压下去。
  代价：慢 N 帧。N=5、每帧 20ms 时，要多等 100ms。
""")
    # 模拟一段"读出来的内容"流：中间混进 2 个误读
    stream = [TASK_CODE, TASK_CODE, "156+123+516+231", TASK_CODE,
              "156+123+5l6+231", TASK_CODE, TASK_CODE, "999+999+999+999",
              TASK_CODE, TASK_CODE, None, TASK_CODE]
    window = 5
    print(f"  模拟 {len(stream)} 帧的读取结果（None = 这一帧没读到）：")
    for i, s in enumerate(stream, 1):
        print(f"    第 {i:>2} 帧：{s}")
    print(f"\n  滑动窗口 N={window} 的投票过程：")
    buf = []
    decided = None
    for i, s in enumerate(stream, 1):
        if s:
            buf.append(s)
            buf = buf[-window:]
        if len(buf) >= (window + 1) // 2:
            cnt = Counter(buf)
            top, n = cnt.most_common(1)[0]
            if n * 2 > len(buf):        # 严格过半
                decided = top
        print(f"    第 {i:>2} 帧后：窗口={buf}  当前判定={decided}")
    print(f"\n  最终判定：{decided}")
    print("  对比一下：")
    print("    · 只认『最后一帧』→ 如果是第 12 帧之前那帧 None，就白等了")
    print("    · 只认『单帧』→ 会读到 156+123+5l6+231 这种误读")
    print(f"    · 5 帧投票 → 稳稳拿到 {decided}，误读被压掉")
    print("""
  实战建议：窗口 3~5 帧，并且**要求票数过半**（不是"最多票"）。
  因为"最多票"在 3 个不同结果各 1 票时也会选出一个，那就等于赌。
""")


# =====================================================================
# C. 按毫米精确打印 A4 码板
# =====================================================================
def try_C():
    print("=" * 68)
    print("C. 按毫米精确打印 A4 二维码板（可以直接上赛场）")
    print("=" * 68)
    import qrcode
    from PIL import Image, ImageDraw, ImageFont

    A4_MM = (210.0, 297.0)
    CODE_MM = 80.0                     # 赛题：二维码 80×80mm（含边框）
    DPI = 300
    mm = DPI / 25.4                    # 1mm 等于多少像素
    print(f"  A4 = {A4_MM[0]:.0f}×{A4_MM[1]:.0f} mm，打印 {DPI} DPI → "
          f"1mm = {mm:.3f} px，整页 {int(A4_MM[0] * mm)}×{int(A4_MM[1] * mm)} px")

    # 码 = 21 格数据 + 四周各 4 格白边 = 29 格
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M,
                       box_size=1, border=4)
    qr.add_data(TASK_CODE)
    qr.make(fit=True)
    matrix = qr.get_matrix()            # 二维 bool 表：True = 黑格
    n = len(matrix)                     # 已含四周各 4 格白边
    modules = qr.modules_count          # 不含白边的数据区边长
    print(f"  这个任务码是 version {qr.version}：数据区 {modules}×{modules} 格，"
          f"加四周各 4 格白边 = {modules} + 8 = {n} 格")
    print(f"  （为什么是 version {qr.version}？『{TASK_CODE}』里的字符都属于"
          f"字母数字表 0-9A-Z 加 + - . / 等，库会用 alphanumeric 模式，很省空间）")
    print(f"  → 每格 = {CODE_MM:.0f}mm / {n} 格 = {CODE_MM / n:.3f} mm "
          f"= {CODE_MM / n * mm:.2f} px")

    # 关键：不要"先生成再缩放"！缩放会让某些行/列被抽掉，格子宽度不均，
    # 边缘会出现"半格黑半格白"，zbar 的采样点一旦落在这种地方就误读。
    # 正确做法：按目标像素直接算每个格子的坐标，逐格画矩形。
    target_px = int(round(CODE_MM * mm))
    step = target_px / n
    code_img = Image.new("L", (target_px, target_px), 255)
    dc = ImageDraw.Draw(code_img)
    for r in range(n):
        for c in range(n):
            if matrix[r][c]:
                x0 = int(round(c * step))
                x1 = int(round((c + 1) * step))
                y0 = int(round(r * step))
                y1 = int(round((r + 1) * step))
                dc.rectangle([x0, y0, x1 - 1, y1 - 1], fill=0)
    print(f"  → 逐格绘制，成品正好 {target_px}px = "
          f"{target_px / mm:.1f}mm 见方（不是\"约 80mm\"，是 80.0mm）")

    W, H = int(round(A4_MM[0] * mm)), int(round(A4_MM[1] * mm))
    page = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(page)
    x = (W - target_px) // 2
    y = int(round(60 * mm))
    page.paste(code_img.convert("RGB"), (x, y))

    # 量尺寸用的标注框：画在码外面 5mm 处，绝不能压到码的白边！
    pad = int(round(5 * mm))
    d.rectangle([x - pad, y - pad, x + target_px + pad, y + target_px + pad],
                outline=(255, 0, 0), width=2)

    fp = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    try:
        f12 = ImageFont.truetype(fp, int(12 * mm))
        f8 = ImageFont.truetype(fp, int(7 * mm))
        f6 = ImageFont.truetype(fp, int(5 * mm))
    except OSError:
        f12 = f8 = f6 = ImageFont.load_default()
    d.text((x - pad, y - pad - int(9 * mm)),
           f"二维码 {CODE_MM:.0f}mm 含白边 · 每格 {CODE_MM / n:.2f}mm · {DPI}DPI",
           font=f6, fill=(0, 0, 0))
    d.text((x - pad, y - pad - int(4 * mm)),
           "红线在码外 5mm 处，用来量尺寸",
           font=f6, fill=(120, 120, 120))
    d.text((x - pad, y + target_px + pad + int(6 * mm)),
           "任务码（字体高度应 ≥12mm）", font=f6, fill=(80, 80, 80))
    d.text((x - pad, y + target_px + pad + int(14 * mm)),
           TASK_CODE, font=f12, fill=(0, 0, 0))

    outp = os.path.join(ASSETS, "30_print_board_a4.png")
    page.save(outp, dpi=(DPI, DPI))
    print(f"\n  已生成：{outp}")
    print("  打印要点：")
    print("    1) 打印机缩放必须选『100% / 实际大小』，不能选『适合页面』")
    print("       （选了会缩到 97%，80mm 变 77.6mm，第 A 节的距离估算就全错了）")
    print("    2) 白纸黑墨，别用彩色；亚光纸 > 光面纸（反光是读码头号杀手）")
    print("    3) 打出来用尺子量红色框：应该是 90mm（80 + 左右各 5mm）。")
    print("       不是就调 DPI 重打，别凑合")
    print("    4) 贴板子时码四周至少留 10mm 白边 —— 第 04 课实测过：")
    print("       白边被深色边框吃掉，border=0 时直接读不出")

    got = decode(page)
    print(f"\n  验证：读一下这张 A4 图 → "
          f"{got[0].data.decode() if got else '读不出'}")

# =====================================================================
# D. 用 OpenCV 定位码的四个角
# =====================================================================
def try_D():
    print("=" * 68)
    print("D. 用 OpenCV 定位码的四个角 —— 给底盘对准用")
    print("=" * 68)
    path = os.path.join(ASSETS, "11_perspective.png")
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    det = cv2.QRCodeDetector()
    data, pts, straight = det.detectAndDecode(gray)
    print(f"  OpenCV 读到：{data!r}")
    if pts is not None:
        print(f"  四个角（顺序：左上/右上/右下/左下）：")
        for i, (px, py) in enumerate(pts[0]):
            print(f"    角{i + 1}: ({px:.1f}, {py:.1f})")
        # 由四个角算中心 —— 机器人转向就靠这个点
        cx, cy = pts[0][:, 0].mean(), pts[0][:, 1].mean()
        print(f"  中心点：({cx:.1f}, {cy:.1f})")
        # 由"上边"算偏航角
        dx = pts[0][1][0] - pts[0][0][0]
        dy = pts[0][1][1] - pts[0][0][1]
        ang = np.degrees(np.arctan2(dy, dx))
        print(f"  上边偏角：{ang:.1f}°（0 = 码摆正；负数 = 画面里逆时针歪了）")

    print(f"\n  zbar 也给了同样的信息，对比一下：")
    r = decode(gray)
    if r:
        poly = [(p.x, p.y) for p in r[0].polygon]
        print(f"    zbar polygon：{poly}")
    # ---- 一个真能用上的技巧：把 OpenCV 摆正后的码，补白边放大再交给 zbar ----
    print("\n  技巧验证：OpenCV 摆正 + 补白边 + 放大 → 再交给 zbar 读")
    print("  " + "-" * 56)

    def straight_to_zbar(g, scale=8, border=4):
        """OpenCV 只负责"检测+摆正"，解码仍交给 zbar。
        straight 是 21×21 的纯码点阵：没有白边，每格才 1 像素 ——
        1px/格 时 zbar 读不出，必须先放大到每格 ≥2 像素。
        这里补 4 格白边再放大 8 倍，是"稳妥配置"（对真实帧更保险）。"""
        _, _, st = cv2.QRCodeDetector().detectAndDecode(g)
        if st is None or getattr(st, "size", 0) == 0:
            return None, "OpenCV 没检测到码"
        m = st[:, :, 0] if st.ndim == 3 else st
        m = (m > 127).astype(np.uint8) * 255
        m = np.pad(m, border, constant_values=255)
        m = cv2.resize(m, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_NEAREST)
        r = decode(m)
        return (r[0].data.decode() if r else None), f"{m.shape[0]}×{m.shape[1]}"

    for fname in ("11_perspective.png", "13_scene.png", "10_rot45.png",
                  "03_size_40.png", "08_glare.png", "09_occluded.png"):
        gp = os.path.join(ASSETS, fname)
        if not os.path.exists(gp):
            continue
        g2 = cv2.cvtColor(cv2.imread(gp), cv2.COLOR_BGR2GRAY)
        txt, info = straight_to_zbar(g2)
        print(f"    {fname:<22} → {str(txt):<20} {info}")

    print("""
  straight 到底需要什么？（同一张 11_perspective.png 实测，逐条都跑过）
    · 裸码 1x（21×21，1.00 px/格）          → 读不出
    · 裸码 2x（42×42，2.00 px/格，最近邻）  → 读得出
    · 补白边 1x（29×29，1.00 px/格）        → 读不出
    · 补白边 2x（58×58，最近邻）            → 读得出
    · 补白边 2x（58×58，双线性）            → 读不出 ← 插值把边界糊成灰了
    · 补白边 8x（232×232，最近邻）          → 读得出

  → 真正的硬要求是『每格 ≥2 像素』且『插值用最近邻』；
    白边是保险（对真实照片更稳），不是唯一关键。

  实测结论（别照抄，要在你自己的图上跑一遍）：
    · 只要 OpenCV 能『检测到』码，摆正+补白边+放大 这条路过 zbar 是通的
    · 但 OpenCV 检测不到时（30px 小码、强眩光、定位角被挡），这条路也走不通 ——
      它救的是"能看见但读不出"，救不了"根本找不到"
    · 所以它是兜底里的一层，不是银弹

  两个工具的差别：
    · zbar 给你 polygon（沿边界的点，可能是 4 个或 5 个），够用来算中心和角度
    · OpenCV 的 detectAndDecode 还额外给一张 straight（把码"摆正"后的图），
      可以直接当"码点阵"用 —— 甚至不必解码，做模板匹配都行
  机器人怎么用：
    · 中心点 → 底盘转向（把中心推到画面正中就对准了板子）
    · 角点/偏航角 → 判断车是不是斜着对着板子，斜太多就影响抓取精度
""")


# =====================================================================
# E. 条形码
# =====================================================================
def try_E():
    print("=" * 68)
    print("E. 顺手也把条形码读了 —— 分拣赛项要用")
    print("=" * 68)
    for name in ("14_barcode_code128.png", "15_barcode_ean13.png"):
        p = os.path.join(ASSETS, name)
        r = decode(cv2.imread(p))
        if r:
            print(f"  {name:<26} → {r[0].type:<8} {r[0].data.decode()}")
        else:
            print(f"  {name:<26} → 读不出")
    print("""
  注意 .type 字段：QRCODE / CODE128 / EAN13 / DATABAR / PDF417 ……
  分拣赛项的货物上可能同时有二维码和条形码，所以判型很重要：

      results = decode(img)
      for r in results:
          if r.type == "QRCODE":    处理二维码
          elif r.type in ("CODE128", "EAN13", "EAN8"):  处理条形码

  zbar 支持的类型里，二维码和常见一维码都有；QR 之外的二维码（DataMatrix、
  PDF417）支持有限，需要时换 zxing-cpp 库（接口几乎一样：decode()）。
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list("ABCDE"), default=None)
    args = ap.parse_args()
    todo = {"A": try_A, "B": try_B, "C": try_C, "D": try_D, "E": try_E}
    for k, fn in todo.items():
        if args.only and k != args.only:
            continue
        fn()
        print()


if __name__ == "__main__":
    main()
