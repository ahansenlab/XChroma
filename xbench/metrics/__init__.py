"""Metric registry. Each scalar module exports score(pred, gt), POLLARD,
LOWER_BETTER and DOC. The insulation factory additionally takes a bin width
and resolution. Metric implementations operate on adapter-prepared maps."""
from ..config import configure_protocol, legacy_mode
configure_protocol()
import importlib
import sys, os

from ._pollard import pollard  # Explicit preserved source; lazy dependency loading.

_NAMES = ["spearman", "pearson", "scc", "mse", "ssim", "insulation",
          "distance_enrichment", "eigenvector", "contact_directionality"]


def load(name):
    """按名字取指标模块。insulation 是参数化的,见 insulation.make(window_bins)。"""
    if name == "pollard_insulation" and not legacy_mode():
        raise RuntimeError("Pollard insulation is historical; current paper scoring uses cooltools. Set XCHROM_BENCH_LEGACY=1 only for an explicit historical analysis.")
    module = importlib.import_module(f"{__name__}.{name}")
    if getattr(module, "POLLARD", False):
        pollard._load()  # Fail before window iteration if the declared dependency is unavailable.
    return module


def all_names():
    return list(_NAMES)
