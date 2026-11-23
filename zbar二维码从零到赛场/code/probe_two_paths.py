# -*- coding: utf-8 -*-
"""
probe_two_paths.py —— 02 课 §5.5 的练习：在一台机器上把两条路都验一遍

为什么需要它：zbar_bootstrap 的"兜底路线"平时永远不执行（因为系统库装好了）。
从来没被执行过的代码 = 从没被验证过的代码。这个脚本故意把"正路"堵死，
逼 ensure() 走兜底分支，证明它真的能用。

用法（在教程根目录下）：
    python code/probe_two_paths.py
"""
import os
import sys
import ctypes.util

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import zbar_bootstrap

# ① 正路：什么都不改，让 bootstrap 自己判断
print("默认情况 ensure() ->", zbar_bootstrap.ensure())

# ② 兜底路：把 find_library 篡改成"永远找不到"，模拟一台没有 sudo 的机器
ctypes.util.find_library = lambda name, *args: None
print("强制兜底 ensure() ->", zbar_bootstrap.ensure())

# ③ 两条路都要能真读出一张码，才算验完
from pyzbar.pyzbar import decode
from PIL import Image

path = os.path.join(ROOT, "assets", "01_clean.png")
if not os.path.exists(path):
    print("素材图不存在，先跑 python code/make_assets.py")
    sys.exit(1)

results = decode(Image.open(path))
if results:
    print("兜底路线读码 ->", results[0].data.decode())
else:
    print("兜底路线读码 -> 失败（这说明兜底代码有 bug，必须修）")
    sys.exit(1)
