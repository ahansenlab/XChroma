"""Spearman correlation using the preserved Pollard scoring implementation."""
from . import pollard

POLLARD = True
LOWER_BETTER = False
DOC = "Spearman correlation of paired pixels"


def score(pred, gt):
    return float(pollard.correlation(pred, gt))
