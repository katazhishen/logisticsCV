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

from PIL import Image
from pyzbar.pyzbar import decode

one = Image.open("assets/03_size_120.png")          # 这是 120x120 的码
page = Image.new("RGB", (400, 140), (255, 255, 255))
for x in (10, 140, 270):                            # 横着排三张
    page.paste(one, (x, 10))
page.save("assets/_page3.png")

r = decode(page)
print("找到几个码：", len(r))
for d in r:
    print(" ", d.data.decode(), d.rect)