"""Read C.Origami cached maps and compare aligned 1-Mb submatrices at 10 kb.
Explicit stores retain separate released and retrained model identities.
The adapter applies matched self-normalization and excludes |i-j| <= 2."""
from ..paths import require_directory, epcot_directory


import glob, json, os
from functools import lru_cache
import numpy as np
from ..space import ice as _ice, snl as _snl, clip, agg as _agg

from ..comparators.corigami.stores import resolve_stores
SOURCE_IDENTITY = resolve_stores()
STORE = SOURCE_IDENTITY['stores']['K562']  # compatibility name; reads use per-cell roots

def _store(cell):
    return SOURCE_IDENTITY['stores'][cell]
GTDIR = require_directory('GT_TILES')
from ..config import require_prediction_store
_OURS_BASE = require_prediction_store()
_OURS_PRED = {c: f"{_OURS_BASE}/{c}" for c in ("GM12878", "K562", "HepG2", "HCT116")}
LABEL = SOURCE_IDENTITY["label"]
KEY = SOURCE_IDENTITY["key"]
RES_KB = 10.0
OFFDIAG = True
_CELLS = SOURCE_IDENTITY["selected_cells"]
NB1 = 1000; AGG = 10; NB = 100; RES = 10_000


@lru_cache(maxsize=8)
def _gt(cell, chrom):
    man = f"{GTDIR}/{cell}/{chrom}.w1000k.s1000k.manifest.json"
    if not os.path.exists(man):
        return None
    gs = json.load(open(man))["starts"]; ng = len(gs)
    mm = np.memmap(f"{GTDIR}/{cell}/{chrom}.w1000k.s1000k.f16.memmap",
                   np.float16, "r", shape=(ng, NB1, NB1))
    return {int(s): i for i, s in enumerate(gs)}, mm


@lru_cache(maxsize=8)
def _co(cell, chrom):
    f = f"{_store(cell)}/{cell}_{chrom}.maps.npz"
    if not os.path.exists(f):
        return None
    d = np.load(f)
    return {int(s): i for i, s in enumerate(d["starts"].astype(int))}, d["corigami"]


def windows(cell):
    """(chrom, tile_start) —— encode 1Mb tile, 且落在某 CO 2Mb 窗的对齐子块里。"""
    out = []
    for f in sorted(glob.glob(f"{_store(cell)}/{cell}_chr*.maps.npz")):
        ch = f.split("_")[-1].split(".")[0]
        g = _gt(cell, ch); co = _co(cell, ch)
        if g is None or co is None:
            continue
        gidx = g[0]; coidx = co[0]
        op = _ourspred(cell, ch); oidx = op[0] if op else {}
        for s in sorted(gidx):
            S_co = (s // 2_000_000) * 2_000_000; ofb = (s - S_co) // RES
            if S_co in coidx and ofb in (0, 100) and int(s) in oidx:
                out.append((ch, int(s)))
    return out


def load(cell, chrom, s, out_res_kb=None):
    """C.Origami 预测: 从 2Mb 图取对齐 1Mb 子块 -> exp -> snl -> clip(与 GT 同空间)。"""
    co = _co(cell, chrom)
    if co is None:
        return None
    coidx, cor = co
    S_co = (s // 2_000_000) * 2_000_000; ofb = (s - S_co) // RES
    if S_co not in coidx or ofb not in (0, 100):
        return None
    sub = np.asarray(cor[coidx[S_co]][ofb:ofb + NB, ofb:ofb + NB], np.float64)
    return clip(_snl(np.exp(sub)))


@lru_cache(maxsize=8)
def _ourspred(cell, chrom):
    pdir = _OURS_PRED.get(cell)
    if pdir is None:
        return None
    man = f"{pdir}/{chrom}.pred.manifest.json"
    if not os.path.exists(man):
        return None
    ps = json.load(open(man))["starts"]
    mm = np.memmap(f"{pdir}/{chrom}.pred.f16.memmap", np.float16, "r", shape=(len(ps), NB1, NB1))
    return {int(s): i for i, s in enumerate(ps)}, mm


def load_ours(cell, chrom, s):
    """XChrom 原生 1Mb 预测(encattn_wg_pred): exp -> agg10 -> snl -> clip。"""
    op = _ourspred(cell, chrom)
    if op is None:
        return None
    pidx, mm = op
    if s not in pidx:
        return None
    return clip(_snl(_agg(np.exp(np.asarray(mm[pidx[s]], np.float64)), AGG)))


def load_gt(cell, chrom, s):
    """干净 deep intact GT: ice -> agg10 -> snl -> clip。"""
    g = _gt(cell, chrom)
    if g is None:
        return None
    gidx, mm = g
    if s not in gidx:
        return None
    raw = np.nan_to_num(np.asarray(mm[gidx[s]], np.float64))
    if (raw > 0).mean() < 0.10:
        return None
    gb = _ice(raw, NB1)
    if gb is None:
        return None
    return clip(_snl(_agg(gb, AGG)))
