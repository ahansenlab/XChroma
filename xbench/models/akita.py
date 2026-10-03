"""Compare cached Akita maps with GT and XChrom at 2-kb resolution.
The cache contains Akita outputs resampled to the 500-by-500 comparison grid.
GT receives ICE and 2-by-2 aggregation before self-normalizing. GM12878 and
HCT116 are the overlapping cell types."""
from ..paths import require_directory, epcot_directory



import os, json
import numpy as np
from ..space import ice, snl, clip

RES_KB = 2.0
LABEL = "Akita"
OFFDIAG = True     # 打分口径: True=只在 |i-j|>2 上算(该线旧脚本口径), False=全图
_CELLS = ["GM12878", "HCT116"]     # Akita 的 5 个 head 里只有这两个和我们重合
_CACHE = require_directory('AKITA_MAP_CACHE')
_GTD = require_directory('GT_TILES')
from ..config import require_prediction_store
_OURS_BASE = require_prediction_store()


def _ours_dir(cell):
    return f"{_OURS_BASE}/{cell}"


def _agg2(M):
    """1000x1000 @1kb -> 500x500 @2kb(2x2 均值)。旧脚本 agg2 逐字。"""
    return M[:1000, :1000].reshape(500, 2, 500, 2).mean((1, 3))


def windows(cell):
    """只返回 AG cache 里真有 .npy 的窗口(否则要调云 API, 本模块不做)。"""
    out = []
    for c in [f"chr{i}" for i in list(range(1, 23))]:
        gm = f"{_GTD}/{cell}/{c}.w1000k.s1000k.manifest.json"
        if not os.path.exists(gm):
            continue
        for s in json.load(open(gm))["starts"]:
            if os.path.exists(f"{_CACHE}/{cell}/{c}_{int(s)}.npy"):
                out.append((c, int(s)))
    return out


def _gt_ours(cell, chrom, start, which):
    gm = f"{_GTD}/{cell}/{chrom}.w1000k.s1000k.manifest.json"
    if not os.path.exists(gm):
        return None
    starts = [int(s) for s in json.load(open(gm))["starts"]]
    if start not in starts:
        return None
    k = starts.index(start); n = len(starts)
    if which == "gt":
        gt = np.memmap(f"{_GTD}/{cell}/{chrom}.w1000k.s1000k.f16.memmap", np.float16, "r", shape=(n, 1000, 1000))
        gb = ice(np.nan_to_num(np.asarray(gt[k], np.float64)), 1000)
        if gb is None:
            return None
        return clip(snl(_agg2(gb)))
    if which == "ours":
        P = _ours_dir(cell)
        ps = [int(s) for s in json.load(open(f"{P}/{chrom}.pred.manifest.json"))["starts"]]
        if start not in ps:
            return None
        pr = np.memmap(f"{P}/{chrom}.pred.f16.memmap", np.float16, "r", shape=(len(ps), 1000, 1000))
        return clip(snl(_agg2(np.exp(np.asarray(pr[ps.index(start)], np.float64)))))
    raise ValueError(which)


def load(cell, chrom, start, out_res_kb=2.0):
    """AG 预测图(cache .npy, 已公共空间)。"""
    f = f"{_CACHE}/{cell}/{chrom}_{int(start)}.npy"
    if not os.path.exists(f):
        return None
    return clip(np.load(f))     # cache 已 clip 过; 再 clip 一次纯保险, 无副作用


def load_gt(cell, chrom, start):
    return _gt_ours(cell, chrom, start, "gt")


def load_ours(cell, chrom, start):
    """XChrom 侧(本线自带, 和 GT 同口径 agg2->snl->clip)。"""
    return _gt_ours(cell, chrom, start, "ours")
