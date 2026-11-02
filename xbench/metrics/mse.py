"""Mean squared error of paired contact pixels (whole maps only)."""
from . import pollard

POLLARD = True
LOWER_BETTER = True
DOC = 'Mean squared pixel error; lower is better'


def score(pred, gt):
    return float(pollard.mse(pred, gt))
