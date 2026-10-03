"""Cached prediction adapters. Each comparator defines RES_KB, OFFDIAG, _CELLS,
windows(cell), load(cell, chrom, start), load_gt(...), and load_ours(...).
All returned maps share that adapter's comparison grid and clipped ln(O/E) space."""
import importlib

_NAMES = ["xchrom", "epcot", "epiphany", "corigami", "alphagenome", "akita"]


def load_model(name):
    return importlib.import_module(f"{__name__}.{name}")


def all_names():
    return list(_NAMES)
