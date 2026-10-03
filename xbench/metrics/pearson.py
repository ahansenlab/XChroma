"""Pearson correlation of finite paired contact pixels."""
import numpy as np
from scipy.stats import pearsonr

POLLARD = False
LOWER_BETTER = False
DOC = "Pearson correlation of paired finite pixels"


def score(pred, gt):
    x, y = np.asarray(pred).reshape(-1), np.asarray(gt).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 20 or x[m].std() < 1e-9 or y[m].std() < 1e-9:
        return np.nan
    return float(pearsonr(x[m], y[m])[0])
