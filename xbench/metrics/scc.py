"""SCC —— Pollard 官方 scoring.scc = hicrep 的 sccByDiag(分层调整相关)。

内部: fill_missing_points_map(nan->0) -> coo_matrix -> sccByDiag,算全部对角线、上三角。
⛔ 别自己写 HiCRep 近似。C.Origami 线曾手搓一版(自做 3x3 平滑、从 d=3 起、权重 n·std·std),
   换成官方后四细胞 SCC 全部下修一档(GM12878 0.726->0.592),胜负零翻转 —— 手搓的平滑把
   双方都抬高了。这就是"看起来差不多"的手搓有多危险的实例。
"""
from . import pollard

POLLARD = True
LOWER_BETTER = False
DOC = "HiCRep 分层调整相关(官方 sccByDiag)"


def score(pred, gt):
    return float(pollard.scc(pred, gt))
