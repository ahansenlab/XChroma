"""Stratum-adjusted correlation using the preserved Pollard HiCRep operator."""
from . import pollard

POLLARD = True
LOWER_BETTER = False
DOC = 'HiCRep stratum-adjusted correlation'


def score(pred, gt):
    return float(pollard.scc(pred, gt))
