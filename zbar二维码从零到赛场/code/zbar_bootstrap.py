import ctypes, ctypes.util, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(os.path.dirname(HERE), "vendor", "lib")
def ensure():
    if ctypes.util.find_library("zbar"):
        return "system"          # 系统里已经装好了，什么都不用做
    cand = os.path.join(LIB, "libzbar.so.0")
    if not os.path.exists(cand):
        raise RuntimeError("找不到 " + cand)
    ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
    ctypes.util.find_library = lambda name, *a: cand if name == "zbar" else None
    return "bundled"
