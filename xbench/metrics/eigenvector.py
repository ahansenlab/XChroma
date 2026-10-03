"""eigenvector(区室 A/B)track 的 Spearman 相关。

⛔ 官方 `pollard.eigenvector_track` 有两处浪费:`fit()` 之后又 `fit_transform()`
   (拟合两遍),且默认 full SVD。合计 **8.97 s/窗**,全量一格(5730 窗 × raw/dog)要 30 小时
   —— 这就是原来 `pollard_full.py` 里 `SLOW_EVERY=5`(每 5 窗才算一次)的原因。

⛔ 这里用 **sklearn 自己的 `svd_solver="arpack"`**:与默认 full solver
   **逐点 max|Δ| = 2e-13**(数值等价),但快 **20.7×**,且**符号约定完全一致**。

⛔ 2026-09-06 教训:曾改用 `scipy.sparse.linalg.svds` 手算(641× 更快),但
   **PCA 主成分方向有正负两解,svds 与 sklearn 挑的符号不同**。最终指标是两条 track 的
   Spearman,符号翻转会让相关系数直接变号 —— 12 个窗里 8 个反号,EPCOT 的
   eigenvector 比值从 1.11 变成 4.41。当初只用 `|Spearman|` 验证等价性,恰好掩盖了它。
   自己复刻 `svd_flip` 的符号约定也没对齐(20 个窗里仍有 9 个反号)。
   **结论:不要手搓符号约定,直接用 sklearn 的求解器。**
"""
import os
import numpy as np
from ..config import keep_missing
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

POLLARD = False        # 官方算法的等价加速实现(去掉重复 fit + 换 solver), 不是逐字复刻
LOWER_BETTER = False
DOC = "eigenvector(区室 A/B) track 的 Spearman 相关(sklearn arpack, 与 full solver 数值等价)"


def _pc1(M):
    X = np.nan_to_num(np.asarray(M, np.float64), nan=0.0)
    if X.shape[0] < 3 or not np.isfinite(X).all() or X.std() < 1e-12:
        return None
    try:
        return PCA(1, svd_solver="arpack", random_state=0).fit_transform(X).ravel()
    except Exception:
        try:
            return PCA(1).fit_transform(X).ravel()      # arpack 偶发不收敛 -> 回退 full
        except Exception:
            return None


def score(pred, gt):
    if keep_missing():
        A = np.asarray(pred, np.float64); B = np.asarray(gt, np.float64)
        m = np.isnan(A) | np.isnan(B)
        A = np.where(m, np.nan, A); B = np.where(m, np.nan, B)
        pred, gt = A, B
    a = _pc1(pred); b = _pc1(gt)
    if a is None or b is None:
        return float("nan")
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or a[m].std() < 1e-12 or b[m].std() < 1e-12:
        return float("nan")
    return float(spearmanr(a[m], b[m])[0])
