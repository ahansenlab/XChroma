"""Spearman agreement of contact-directionality tracks."""
import numpy as np

from . import pollard

POLLARD = True
LOWER_BETTER = False
DOC = "Spearman correlation of contact-directionality tracks"


def _prep(X):
    X = np.asarray(X, np.float64)
    return X


def score(pred, gt):
    try:
        v = pollard.vectorMethodToScalar(pollard.contact_directionality_track,
                                         _prep(pred), _prep(gt),
                                         finalCompMetric="corr")
        return float(v)
    except Exception:
        return float("nan")
