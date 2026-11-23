# -*- coding: utf-8 -*-
"""
env_check.py —— 第 02 课：环境自检

作用：一条命令告诉你"这台电脑到底能不能跑本教程"。
每一项都会打印 [OK] 或 [X]，最后给一句结论。

运行（在教程根目录下）：
    .venv/bin/python code/env_check.py
"""
import os
import sys
import platform

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ok_count = 0
bad = []


def check(name, fn, hint=""):
    """跑一项检查。fn 正常返回就给 [OK]，抛异常就给 [X]。"""
    global ok_count
    try:
        detail = fn()
        print(f"  [OK] {name}" + (f" —— {detail}" if detail else ""))
        ok_count += 1
        return True
    except Exception as e:
        print(f"  [ X] {name} —— {type(e).__name__}: {e}")
        if hint:
            print(f"       修法：{hint}")
        bad.append(name)
        return False


print("=" * 62)
print(" zbar 二维码教程 · 环境自检")
print("=" * 62)

print("\n【一】皮（Python 解释器）")
check("Python 版本", lambda: f"{sys.version.split()[0]}（{platform.system()}）"
      if sys.version_info >= (3, 8) else (_ for _ in ()).throw(RuntimeError("需要 3.8+")))


def _numpy():
    import numpy
    return f"numpy {numpy.__version__}"


def _pillow():
    import PIL
    return f"Pillow {PIL.__version__}"


def _cv2():
    import cv2
    return f"OpenCV {cv2.__version__}"


def _qrcode():
    import qrcode
    return f"qrcode {getattr(qrcode, '__version__', '已装')}"


print("\n【二】料（图像处理三件套：算得快、存得住、看得见）")
check("numpy（数组运算）", _numpy)
check("Pillow（图像读写）", _pillow)
check("OpenCV（图像处理/摄像头）", _cv2, "pip install opencv-python")
check("qrcode（造二维码）", _qrcode, "pip install qrcode")

print("\n【三】骨（zbar 本体 + 它的 Python 翻译官）")


def _bootstrap():
    sys.path.insert(0, os.path.join(ROOT, "code"))
    import zbar_bootstrap
    mode = zbar_bootstrap.ensure()
    if mode == "system":
        return "使用系统安装的 libzbar（apt 装成功的那种）"
    return "使用教程自带的 vendor/lib/libzbar.so.0（没 sudo 权限时的兜底）"


check("libzbar 能找到", _bootstrap,
      "sudo apt install -y libzbar0，或者确认 vendor/lib/libzbar.so.0 存在")


def _pyzbar():
    from pyzbar.pyzbar import decode
    from PIL import Image
    p = os.path.join(ROOT, "assets", "01_clean.png")
    if not os.path.exists(p):
        return "pyzbar 可用（素材图还没生成，先跑 code/make_assets.py）"
    r = decode(Image.open(p))
    assert r, "解码返回空"
    return f"pyzbar 可用，试读 01_clean.png 得到：{r[0].data.decode()}"


check("pyzbar 能读出一张真实图片", _pyzbar, "pip install pyzbar")

print("\n【四】眼睛（摄像头，没有也能学，只是第 06 课要改用回放模式）")


def _camera():
    import cv2
    for i in range(4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            good, frame = cap.read()
            cap.release()
            if good and frame is not None:
                h, w = frame.shape[:2]
                return f"/dev/video{i} 可用，画面 {w}x{h}"
    raise RuntimeError("没找到可用摄像头")


check("摄像头", _camera, "接上 USB 摄像头；或在第 06 课加 --replay 参数用回放模式")

print("\n" + "=" * 62)
if not bad:
    print(f" 结论：{ok_count} 项检查全部通过。可以开始第 03 课了。")
else:
    print(f" 结论：{ok_count} 项通过，{len(bad)} 项需要修：{'、'.join(bad)}")
    print(" 修完再重新跑一次本脚本。")
print("=" * 62)
