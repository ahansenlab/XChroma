"""Compare 10-kb Epiphany maps with matched GT and XChrom.
GT receives ICE and aggregation; Epiphany is stored as linear O/E.
GM12878 evaluation uses chr3, chr11 and chr17; other cells use cached chromosomes."""
from ..paths import require_directory, epcot_directory


import os, json
import numpy as np
from ..space import ice, snl, clip

RES_KB = 10.0
LABEL = "Epiphany"
OFFDIAG = False     # 打分口径: True=只在 |i-j|>2 上算(该线旧脚本口径), False=全图
NB = 100
_CELLS = ["GM12878", "K562", "HCT116", "HepG2"]
_HELDOUT = ["chr3", "chr11", "chr17"]     # Epiphany 训在 GM12878, 论文只这三条 completely held-out
def _chroms(cell):
    if cell == "GM12878":
        return _HELDOUT
    import glob
    return sorted({os.path.basename(f).split(".")[0]
                   for f in glob.glob(f"{_EPD}/{cell}/chr*.epiphany.manifest.json")},
                  key=lambda c: (len(c), c))

_GTD = require_directory('GT_TILES')
_EPD = require_directory('EPIPHANY_DIR')
from ..config import require_prediction_store
_OURS_BASE = require_prediction_store()


def _ours_dir(cell):
    return f"{_OURS_BASE}/{cell}"


def _to10k(M):
    """1000x1000 @1kb -> 100x100 @10kb, 求和(保接触计数语义)。"""
    return M.reshape(NB, 10, NB, 10).sum(axis=(1, 3))


def _selfclip_obs(obs):
    return clip(snl(np.asarray(obs, np.float64)))


def windows(cell):
    out = []
    for c in _chroms(cell):
        m = f"{_GTD}/{cell}/{c}.w1000k.s1000k.manifest.json"
        if os.path.exists(m):
            out += [(c, int(s)) for s in json.load(open(m))["starts"]]
    return out


def _fetch(cell, chrom, start, which):
    m = f"{_GTD}/{cell}/{chrom}.w1000k.s1000k.manifest.json"
    if not os.path.exists(m):
        return None
    starts = [int(s) for s in json.load(open(m))["starts"]]
    if start not in starts:
        return None
    k = starts.index(start); n = len(starts)

    if which == "gt":
        gt = np.memmap(f"{_GTD}/{cell}/{chrom}.w1000k.s1000k.f16.memmap", np.float16, "r", shape=(n, 1000, 1000))
        Bal = ice(np.asarray(gt[k], np.float64), 1000)
        if Bal is None:
            return None
        return _selfclip_obs(_to10k(Bal))
    if which == "ours":
        P = _ours_dir(cell)
        ps = [int(s) for s in json.load(open(f"{P}/{chrom}.pred.manifest.json"))["starts"]]
        if start not in ps:
            return None
        pr = np.memmap(f"{P}/{chrom}.pred.f16.memmap", np.float16, "r", shape=(len(ps), 1000, 1000))
        return _selfclip_obs(_to10k(np.exp(np.asarray(pr[ps.index(start)], np.float64))))
    if which == "epiphany":
        E = f"{_EPD}/{cell}"
        eman = json.load(open(f"{E}/{chrom}.epiphany.manifest.json")) if os.path.exists(f"{E}/{chrom}.epiphany.manifest.json") else None
        ne = len(eman["starts"]) if eman else n
        ep = np.memmap(f"{E}/{chrom}.epiphany.f16.memmap", np.float16, "r", shape=(ne, NB, NB))
        ek = eman["starts"].index(start) if eman else k
        return _selfclip_obs(np.nan_to_num(np.asarray(ep[ek], np.float64), nan=1.0))   # 线性 O/E
    raise ValueError(which)


def load(cell, chrom, start, out_res_kb=10.0):
    return _fetch(cell, chrom, start, "epiphany")


def load_gt(cell, chrom, start):
    return _fetch(cell, chrom, start, "gt")


def load_ours(cell, chrom, start):
    return _fetch(cell, chrom, start, "ours")
