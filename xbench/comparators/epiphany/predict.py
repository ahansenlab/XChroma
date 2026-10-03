"""Generate 1-Mb, 10-kb Epiphany caches from explicit five-track inputs.
See README.md for required paths and input-channel order.
"""

import os, sys, json
import numpy as np
import torch
import pyBigWig

from xbench.paths import require_directory
UP = require_directory("EPIPHANY_UPSTREAM")
sys.path.insert(0, f"{UP}/epiphany")
import types                                    # model_10kb 顶部 import torchvision 但一个都没用到
for _m in ("torchvision", "torchvision.datasets", "torchvision.transforms"):
    sys.modules.setdefault(_m, types.ModuleType(_m))
sys.modules["torchvision"].datasets = sys.modules["torchvision.datasets"]
sys.modules["torchvision"].transforms = sys.modules["torchvision.transforms"]
from model_10kb import Net


CELL = os.environ["CELL"]
CHROM = os.environ["CHROM"]
BWDIR = require_directory("BWDIR")          # 该细胞的 5 条 bigwig
OUT = os.environ["OUTDIR"]
if any(os.path.exists(os.path.join(OUT, f"{CHROM}.epiphany.{suffix}")) for suffix in ("f16.memmap", "manifest.json")):
    raise ValueError("Output files already exist for this chromosome.")
ORDER = ["DNase", "H3K27ac", "H3K4me3", "H3K27me3", "CTCF"]   # ← 模型定出来的顺序,别改
SEQ, WIN, RES, NB = 200, 14000, 10_000, 100
BP = 100                                        # 输入分辨率
NIN = SEQ * (RES // BP) + WIN                   # 34,000 个 100bp bin
FLANK = (WIN // 2) * BP                         # 700 kb 侧翼
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = Net()
sd = torch.load(os.environ["EPIPHANY_CHECKPOINT"], map_location="cpu")
sd = sd.get("model_state_dict", sd.get("model", sd))
sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
miss, _ = net.load_state_dict(sd, strict=False)
assert not miss, f"权重没装全: {miss[:3]}"
net.eval().to(dev)

_bw0 = pyBigWig.open(f"{BWDIR}/{ORDER[0]}.bw")
L = _bw0.chroms()[CHROM]
_bw0.close()
NBIN = L // 100
print(f"[cache] 读入 {CHROM} 的 5 条轨道 ({NBIN:,} 个 100bp bin, ~{5*NBIN*4/1e6:.0f} MB)…", flush=True)
TRACK = np.zeros((5, NBIN), np.float32)
for _i, t in enumerate(ORDER):
    bw = pyBigWig.open(f"{BWDIR}/{t}.bw")
    TRACK[_i] = np.nan_to_num(np.array(bw.stats(CHROM, 0, NBIN * 100, type="mean", nBins=NBIN), float))
    bw.close()
print("[cache] 完成", flush=True)
os.makedirs(OUT, exist_ok=True)

G = os.path.join(require_directory("GT_TILES"), CELL)
man = json.load(open(f"{G}/{CHROM}.w1000k.s1000k.manifest.json"))
starts = man["starts"]

mp = f"{OUT}/{CHROM}.epiphany.f16.memmap"
mm = np.memmap(mp, np.float16, "w+", shape=(len(starts), NB, NB))
nfail = 0
for k, s in enumerate(starts):
    M = np.full((NB, NB), np.nan, np.float32)
    lo = s - FLANK
    hi = lo + NIN * BP
    if lo < 0 or hi > L:
        nfail += 1
        mm[k] = M
        continue
    b0 = lo // BP
    if b0 < 0 or b0 + NIN > NBIN:
        nfail += 1
        mm[k] = M
        continue
    X = np.ascontiguousarray(TRACK[:, b0:b0 + NIN])
    assert X.shape == (5, NIN), X.shape
    with torch.no_grad():
        y, _ = net(torch.from_numpy(np.ascontiguousarray(X)).to(dev), seq_length=SEQ)
    y = y[0].cpu().numpy()                       # (200, 100)
    #    upstream/epiphany/utils.py::contact_extraction:
    #        for k in range(100): a1=k//2; a2=k%2; value = diag[k][m-a1-a2]
    for a in range(NB):                    # 窗口内的 100 个 anchor(绝对 bin = 窗口起点 + a)
        for kk in range(NB):
            i = a - (kk // 2) - (kk % 2)
            j = i + kk
            if 0 <= i < NB and 0 <= j < NB:
                M[i, j] = M[j, i] = y[a, kk]
    mm[k] = M
mm.flush()
json.dump({"chrom": CHROM, "starts": starts, "n_bins": NB, "res": RES, "cell": CELL,
           "model": "Epiphany-10kb", "channel_order": ORDER, "n_failed": nfail,
           "evaluation_chromosomes": (["chr3", "chr11", "chr17"] if CELL == "GM12878" else "all_cached")},
          open(f"{OUT}/{CHROM}.epiphany.manifest.json", "w"))
print(f"[done] {CELL} {CHROM}: {len(starts)} 窗, {nfail} 个边界窗跳过 -> {mp}", flush=True)
