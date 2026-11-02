"""Spearman agreement of first-principal-component tracks."""
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

POLLARD = False  # Pollard-style PCA with the existing sklearn ARPACK solver.
LOWER_BETTER = False
DOC = "Spearman correlation of first-principal-component tracks"


def _pc1(M):
    X = np.nan_to_num(np.asarray(M, np.float64), nan=0.0)
    if X.shape[0] < 3 or not np.isfinite(X).all() or X.std() < 1e-12:
        return None
    try:
        return PCA(1, svd_solver="arpack", random_state=0).fit_transform(X).ravel()
    except Exception:
        try:
            return PCA(1).fit_transform(X).ravel()      # Fall back to the full solver if ARPACK fails.
        except Exception:
            return None


def score(pred, gt):
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
