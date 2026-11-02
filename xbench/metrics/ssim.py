"""SSIM uses the preserved Pollard operator with the experimental map first.

The data range is defined by the experimental map after the pair-specific missing mask is filled
with zero. Different prediction masks can therefore change that range. When comparing model summaries, retain the same windows with finite scores
for all models in that comparison.
"""
from . import pollard

POLLARD = True
LOWER_BETTER = False
DOC = "SSIM (data_range from the experimental map within each scoring call)"


def score(pred, gt):
    return float(pollard.ssim_map(gt, pred))
