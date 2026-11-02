"""Focal loop strength and Pearson agreement at shared experimentally defined sites."""
import numpy as np
from scipy.stats import pearsonr

from .fine_structure import dog


def focal_strength(filtered, ia, ib):
    """Central 3x3 mean minus the mean of four 7-bin edges; corners count twice."""
    n = filtered.shape[0]
    if ia - 3 < 0 or ia + 3 >= n or ib - 3 < 0 or ib + 3 >= n:
        return np.nan
    patch = filtered[ia - 3:ia + 4, ib - 3:ib + 4]
    if np.isfinite(patch).mean() < 0.8:
        return np.nan
    border = np.concatenate([patch[0].ravel(), patch[-1].ravel(),
                             patch[:, 0].ravel(), patch[:, -1].ravel()])
    return float(np.nanmean(patch[2:5, 2:5]) - np.nanmean(border))


def score_maps(gt, ours, opp, ia, ib, res_kb):
    """Input is already comparison-grid ln(O/E), clipped to [-2,2]."""
    maps = [np.asarray(x, np.float64) for x in (gt, ours, opp)]
    if any(x.ndim != 2 or x.shape != maps[0].shape or x.shape[0] != x.shape[1] for x in maps):
        raise ValueError('Expected three square maps on one scoring grid.')
    if float(res_kb) not in (1.0, 2.0):
        raise ValueError('Loop-scoring grids must be 1 or 2 kb.')
    if any(np.isinf(x).any() or (np.abs(x[np.isfinite(x)]) > 2).any() for x in maps):
        raise ValueError('Inputs must already be clipped ln(O/E); infinities are invalid.')
    ia, ib = np.asarray(ia), np.asarray(ib)
    if ia.ndim != 1 or ib.shape != ia.shape or not np.issubdtype(ia.dtype, np.integer) or not np.issubdtype(ib.dtype, np.integer):
        raise ValueError('ia and ib must be matching integer vectors of experimentally defined loop sites.')
    filtered = [dog(x, res_kb) for x in maps]
    retained, values = [], []
    for k, (a, b) in enumerate(zip(ia, ib)):
        if a >= b:
            raise ValueError('Loop positions must be in the upper triangle (ia < ib).')
        v = [focal_strength(x, int(a), int(b)) for x in filtered]
        if np.isfinite(v).all():
            retained.append(k)
            values.append(v)
    # Use float32 strength vectors for the subsequent Pearson comparison.
    a = np.asarray(values, np.float32).reshape(-1, 3)
    return {'gt': a[:, 0], 'ours': a[:, 1], 'opp': a[:, 2],
            'ia': ia[retained], 'ib': ib[retained], 'retained_index': np.asarray(retained, np.int64)}


def correlations(gt, ours, opp):
    a = [np.asarray(x, np.float64) for x in (gt, ours, opp)]
    if any(x.ndim != 1 or x.shape != a[0].shape for x in a):
        raise ValueError('Scalar vectors must be one-dimensional and position-paired.')
    if not all(np.isfinite(x).all() for x in a):
        raise ValueError('Strength vectors must contain only jointly retained finite sites.')
    def corr(x):
        return float(pearsonr(a[0], x).statistic) if len(x) >= 2 and np.ptp(a[0]) > 0 and np.ptp(x) > 0 else None
    return {'n': len(a[0]), 'XChroma_Pearson_r': corr(a[1]), 'comparison_model_Pearson_r': corr(a[2])}


def score(pred, gt, ia, ib, res_kb):
    """Pearson agreement of predicted and measured focal strengths at shared experimental sites.

    Uses the same extraction, joint retention and float32 strength export as
    score_maps. Return NaN if fewer than two variable strengths remain.
    """
    values = score_maps(gt, pred, pred, ia, ib, res_kb)
    value = correlations(values['gt'], values['ours'], values['opp'])['XChroma_Pearson_r']
    return np.nan if value is None else value
