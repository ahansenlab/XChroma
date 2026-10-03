"""Compare EPCOT and XChrom on matched 500-kb, 1-kb submatrices.
The three EPCOT heads share GT and XChrom inputs. EPCOT native log2(O/E)
outputs are converted to linear O/E and then self-normalized and clipped.
GT receives whole-window ICE before subdivision."""
from ..paths import require_directory, epcot_directory


import os, json
import numpy as np
from ..space import ice, snl, clip

RES_KB = 1.0
NB = 1000
BS = 500          # 子块边长(bin); 1Mb/1kb = 1000bin -> 2 个 500 子块
_CELLS = ["GM12878", "K562", "HepG2", "HCT116"]

_HEADS = {"microc": "epcot_wg_pred", "intacthic": "epcot_wg_pred_intacthic", "rcmc": "epcot_wg_pred_rcmc"}
from ..config import selected_epcot_head
_HEAD = selected_epcot_head()
assert _HEAD in _HEADS, f"EPCOT_HEAD 必须是 {list(_HEADS)}, 收到 {_HEAD}"
LABEL = f"EPCOT {_HEAD}"
KEY = f"epcot_{_HEAD}"     # perwin 路径用 KEY(带 head), 避免三 head 互相覆盖
OFFDIAG = False     # 打分口径: True=只在 |i-j|>2 上算(该线旧脚本口径), False=全图
_CONV = os.environ.get("EPCOT_CONV", "log2")

_GTD = require_directory('GT_TILES')
_EPD = epcot_directory(_HEAD)
from ..config import require_prediction_store
_OURS_BASE = require_prediction_store()


def _ours_dir(cell):
    return f"{_OURS_BASE}/{cell}"


def _conv_to_oe(eb):
    """EPCOT 输出 -> 线性 O/E(供 snl/selfclip 用)。"""
    x = np.nan_to_num(np.asarray(eb, np.float64))
    if _CONV == "log2":  return np.exp(x * np.log(2.0))
    if _CONV == "ln":    return np.exp(x)
    if _CONV == "linear":return np.maximum(x, 1e-6)
    raise ValueError(_CONV)


def _selfclip_obs(obs):
    """observed(线性) -> log(O/E) -> clip[-2,2]。snl() 里已含 per-diagonal E + log。"""
    return clip(snl(np.asarray(obs, np.float64)))


def windows(cell):
    out = []
    for c in [f"chr{i}" for i in list(range(1, 23))]:
        m = f"{_GTD}/{cell}/{c}.w1000k.s1000k.manifest.json"
        if not os.path.exists(m):
            continue
        starts = json.load(open(m))["starts"]
        for s in starts:
            for blk in (0, 1):
                out.append((c, int(s) + blk * BS * 1000))
    return out


from functools import lru_cache as _lru


@_lru(maxsize=8)
def _ice_gt_win(cell, chrom, win, n, k):
    gt = np.memmap(f"{_GTD}/{cell}/{chrom}.w1000k.s1000k.f16.memmap", np.float16, "r", shape=(n, NB, NB))
    return ice(np.asarray(gt[k], np.float64), NB)     # None = 该窗有效 bin 太少


def _fetch(cell, chrom, start, which):
    """which ∈ {gt, ours, epcot}。返回该子块进公共空间后的图, 取不到 -> None。"""
    win = (start // 1_000_000) * 1_000_000
    blk = 0 if start - win < BS * 1000 else 1
    sl = slice(blk * BS, (blk + 1) * BS)
    m = f"{_GTD}/{cell}/{chrom}.w1000k.s1000k.manifest.json"
    if not os.path.exists(m):
        return None
    starts = [int(s) for s in json.load(open(m))["starts"]]
    if win not in starts:
        return None
    k = starts.index(win); n = len(starts)

    if which == "gt":
        Bal = _ice_gt_win(cell, chrom, win, n, k)     # 整窗 ICE(缓存); 子块只是切片
        if Bal is None:
            return None
        return _selfclip_obs(Bal[sl, sl])
    if which == "ours":
        P = _ours_dir(cell)
        with open(f"{P}/{chrom}.pred.manifest.json") as stream:
            pred_starts = [int(s) for s in json.load(stream)["starts"]]
        if pred_starts != starts:
            raise ValueError("EPCOT comparison requires XChrom and GT manifests with identical window starts/order.")
        pr = np.memmap(f"{P}/{chrom}.pred.f16.memmap", np.float16, "r", shape=(n, NB, NB))
        return _selfclip_obs(np.exp(np.asarray(pr[k], np.float64)[sl, sl]))   # ours 存 log(O/E)
    if which == "epcot":
        ep = np.memmap(f"{_EPD}/{cell}/{chrom}.epcot.f16.memmap", np.float16, "r", shape=(n, NB, NB))
        return _selfclip_obs(_conv_to_oe(ep[k])[sl, sl])
    raise ValueError(which)


def load(cell, chrom, start, out_res_kb=1.0):
    return _fetch(cell, chrom, start, "epcot")


def load_gt(cell, chrom, start):
    return _fetch(cell, chrom, start, "gt")


def load_ours(cell, chrom, start):
    """XChrom 侧(同口径, _fetch 的 ours 分支)。缺这个 -> run.py 的 xchrom 列全 nan -> 被判全输。"""
    return _fetch(cell, chrom, start, "ours")
