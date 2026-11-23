# ══════ 环境准备（每个示例都一样，照抄，不用改任何东西）══════
import os, sys
ROOT = "/home/kata/Desktop/logisticsCV/zbar二维码从零到赛场"   # 教程根目录
os.chdir(ROOT)                                    # 让示例里的 "assets/xxx" 找得到
sys.path.insert(0, os.path.join(ROOT, "code"))    # 让 import zbar_bootstrap 找得到
import zbar_bootstrap
zbar_bootstrap.ensure()                           # 确认 zbar 已就绪（第 02 课验证过）
from PIL import Image
from pyzbar.pyzbar import decode
# ══════════════════════════════════════════════════════════

import os, sys                                              # 重复导入无副作用，片段可单独跑
import cv2                                                  # cv2 负责读图和转灰度
sys.path.insert(0, "code")                                  # 让 import 在 code 目录里找
import zbar_bootstrap; zbar_bootstrap.ensure()              # 再确认一次 zbar 可用
from pyzbar.pyzbar import decode                            # decode() 是找码的入口

folder = "assets/replay"                                    # 60 张合成帧都放这里
# 排序让顺序固定，两次跑结果才能对比
files = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png"))
line = ""                                                   # 一个字符代表一帧
for f in files:                                             # 60 帧逐张处理
    gray = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)  # 读图并转灰度
    line += "#" if decode(gray) else "."                    # 命中加 #，没命中加 .
print(line)                                                 # 一行 60 个字符，前 10 个是点
print(f"共 {len(files)} 帧，命中 {line.count('#')} 帧")           # 60 帧命中 50 帧，与正文一致