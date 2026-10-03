"""Cooltools diamond insulation followed by Spearman track correlation.
Input ln(O/E) maps are exponentiated before diamond scoring; genomic scale is
window_bins multiplied by the comparison grid's resolution."""
import numpy as np
from scipy.stats import spearmanr
from cooltools.api.insulation import _insul_diamond_dense as _idd

POLLARD = False  # The operator is cooltools, not Pollard insulation.
LOWER_BETTER = False
DOC = "Spearman correlation of cooltools insulation tracks"


def make(window_bins, res_kb):
    """返回 (指标名, score 函数)。指标名里的数字是**换算好的 kb**。"""
    name = f"Insul{int(window_bins * res_kb)}kb"
    w = int(window_bins)

    def score(pred, gt):
        try:
            vo = _idd(np.exp(np.asarray(pred, np.float64)), window=w, ignore_diags=2, norm_by_median=True)
            vg = _idd(np.exp(np.asarray(gt, np.float64)), window=w, ignore_diags=2, norm_by_median=True)
            m = np.isfinite(vo) & np.isfinite(vg)
            if m.sum() < 10 or vo[m].std() < 1e-9 or vg[m].std() < 1e-9:
                return np.nan
            return float(spearmanr(vo[m], vg[m])[0])
        except Exception:
            return np.nan
    return name, score
