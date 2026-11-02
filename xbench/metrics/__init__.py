"""Array-based whole-map, fine-structure and focal-loop evaluation.

Scalar modules expose score(pred, gt), where gt is the measured contact map.
The insulation factory additionally
requires the window size in bins and map resolution in kb.
"""
import importlib

from ._pollard import pollard

_NAMES = ["spearman", "pearson", "scc", "mse", "ssim", "insulation",
          "distance_enrichment", "eigenvector", "contact_directionality"]


def load(name):
    """Return one of the nine supported metric modules."""
    if name not in _NAMES:
        raise ValueError(f"Unknown metric {name!r}; choose from {', '.join(_NAMES)}.")
    module = importlib.import_module(f"{__name__}.{name}")
    if getattr(module, "POLLARD", False):
        pollard._load()
    return module


def all_names():
    """Return the nine available metric names."""
    return list(_NAMES)
