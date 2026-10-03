"""SSIM uses the preserved Pollard operator with GT as the first argument.

The data range is defined by GT after the pair-specific missing mask is filled
with zero. Different prediction masks can therefore change that range. Pairing
of finite window scores is enforced by the scoring driver.
"""
from . import pollard

POLLARD = True
LOWER_BETTER = False
DOC = "SSIM (GT-derived data_range within each scoring call)"


def score(pred, gt):
    return float(pollard.ssim_map(gt, pred))
