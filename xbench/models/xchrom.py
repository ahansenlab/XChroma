"""Read native 1-Mb, 1-kb XChrom predictions stored as natural-log O/E.
Coarsening proceeds in linear O/E space before diagonal renormalization."""
import json, os
import numpy as np
from ..space import snl, agg, clip

RES_KB = 1.0
LABEL = "XChrom"
OFFDIAG = False     # 打分口径: True=只在 |i-j|>2 上算(该线旧脚本口径), False=全图
NB_RAW = 1000
CKPT = "explicit-cache-identity"  # Generating model is declared by the cache manifest.

from ..config import require_prediction_store
_OURS_BASE = require_prediction_store()


def _dir(cell):
    return f"{_OURS_BASE}/{cell}"


def windows(cell):
    out = []
    for c in [f"chr{i}" for i in list(range(1, 23))]:
        m = f"{_dir(cell)}/{c}.pred.manifest.json"
        if os.path.exists(m):
            out += [(c, int(s)) for s in json.load(open(m))["starts"]]
    return out


def load(cell, chrom, start, out_res_kb=1.0):
    man = f"{_dir(cell)}/{chrom}.pred.manifest.json"
    if not os.path.exists(man):
        return None
    starts = [int(s) for s in json.load(open(man))["starts"]]
    if start not in starts:
        return None
    M = np.memmap(f"{_dir(cell)}/{chrom}.pred.f16.memmap", np.float16, "r",
                  shape=(len(starts), NB_RAW, NB_RAW))
    p = np.asarray(M[starts.index(start)], np.float64)
    f = int(round(out_res_kb / RES_KB))
    if f > 1:
        p = snl(agg(np.exp(p), f))      # log(O/E) -> O/E -> 聚合 -> 重算 per-diag O/E
    return clip(p)
