"""Data pipeline for XChroma training: fixed-window Hi-C sampler
with an ATAC 5th input channel, plus the ATAC load/normalize helpers.

The default normalization is ATAC_NORM=global with ATAC_NORM_SCALE=1.0.

Public API:
  _load_atac_memmaps(atac_dir)         load per-chromosome ATAC stores
  normalize_atac(atac)                normalize ATAC using env ATAC_NORM
  fetch_atac(chrom, s, e, atac_mm)     float32 ATAC slice, 0-padded
  stack_seq_atac(seq_oh, ...)          concat (L,4) DNA one-hot + (L,1) ATAC -> (L,5)
  make_sampler(...)                  build a configured sampler
"""
import os
import json as _json
import numpy as np

try:
    from .x_utils import FixedWindowSampler
    from .inputs import ZAtac as _ZAtac
except ImportError:  # Direct train.py execution.
    from x_utils import FixedWindowSampler
    from inputs import ZAtac as _ZAtac

# === ATAC configuration from environment variables =======================
ATAC_DIR = os.environ.get("ATAC_DIR", "")
ATAC_DTYPE = np.float16
ATAC_NORM = os.environ.get("ATAC_NORM", "global").lower()
ATAC_NORM_SCALE = float(os.environ.get("ATAC_NORM_SCALE", "1.0"))
ATAC_TOPFRAC = float(os.environ.get("ATAC_TOPFRAC", "0") or 0)
ATAC_ZERO = os.environ.get("ATAC_ZERO", "0") == "1"   # seq-only: feed a zero ATAC channel

ATAC_REF = None          # Unused reference-table placeholder

# ---- Genome-wide p99 for ATAC_NORM=global --------------------------------
# One normalization constant per cell, cached in <ATAC directory>/.global_p99.
# A shared constant preserves relative accessibility levels across windows.
# It is estimated from regularly spaced 1,000-bp slices, sampling approximately
# one million bases per chromosome, and cached for subsequent runs.
_GP99 = None             # Current cell's normalization constant, set by bind_gp99
_GP99_BY_DIR = {}


def gp99_for_dir(atac_dir, memmaps=None):
    """Read an ATAC directory's cached p99, or estimate and cache it."""
    if atac_dir in _GP99_BY_DIR:
        return _GP99_BY_DIR[atac_dir]
    v = None
    cache = os.path.join(atac_dir, ".global_p99") if atac_dir else ""
    if cache and os.path.exists(cache):
        try:
            v = float(open(cache).read().strip())
        except Exception:
            v = None
    if v is None:
        if not memmaps:            # Load stores for either None or an empty dict
            memmaps = _load_atac_memmaps(atac_dir)
        # Sample deterministic, regularly spaced slices from each chromosome.
        samp = []
        for mm in memmaps.values():
            # Slice indexing works with both dense memmaps and .az stores.
            n_tot = mm.shape[0]
            k = min(1_000_000, n_tot)
            stride = max(1, n_tot // max(1, k // 1000))
            for off in range(0, n_tot - 1000, stride):
                samp.append(np.asarray(mm[off:off + 1000]).astype(np.float32))
        v = float(np.percentile(np.concatenate(samp), 99))
        if cache:
            try:
                open(cache, "w").write(f"{v:.6f}\n")
            except OSError:
                pass
    _GP99_BY_DIR[atac_dir] = v
    print(f"[atac] global p99[{atac_dir}] = {v:.4f}", flush=True)
    return v


def bind_gp99(atac_dir, memmaps=None):
    """Set the current cell's p99 constant at training initialization."""
    global _GP99
    _GP99 = gp99_for_dir(atac_dir, memmaps)
    return _GP99


def _chrom_entries(man):
    """Yield (chrom, length) from an ATAC manifest, tolerating both schemas and metadata keys.

    Two schemas exist in practice:
      A) flat   {"chr1": {"length": N, ...}, ..., "_source": {...}}
      B) nested {"chroms": {"chr1": N, ...}, ...}
    Metadata keys such as `_source` are skipped.
    """
    if isinstance(man.get("chroms"), dict):                 # schema B
        for _c, _v in man["chroms"].items():
            if isinstance(_v, dict):
                if "length" in _v:
                    yield _c, int(_v["length"])
            else:
                yield _c, int(_v)
        return
    for _c, _info in man.items():                            # schema A
        if _c.startswith("_"):                               # metadata, e.g. _source
            continue
        if isinstance(_info, dict) and "length" in _info:
            yield _c, int(_info["length"])


def _load_atac_memmaps(atac_dir):
    """Load per-chromosome log1p ATAC signal from float16 memmaps or AZ1 stores."""
    with open(os.path.join(atac_dir, "manifest.json")) as f:
        man = _json.load(f)
    mm = {}
    for _c, _len in _chrom_entries(man):
        _paz = os.path.join(atac_dir, f"{_c}.atac.log1p.az")
        if os.path.exists(_paz):
            mm[_c] = _ZAtac(_paz)
        else:
            _p = os.path.join(atac_dir, f"{_c}.atac.log1p.f16.memmap")
            mm[_c] = np.memmap(_p, dtype=ATAC_DTYPE, mode="r", shape=(_len,))
    print(f"[atac] loaded {len(mm)} chromosome stores from {atac_dir}", flush=True)
    return mm


def normalize_atac(atac):
    """Normalize log1p ATAC signal for the network.

    ``global`` uses one genome-wide p99 estimate per cell, preserving relative
    accessibility across windows. This is the default training configuration.
    ``log1p_p99`` uses a separate p99 per window for checkpoints trained with
    that normalization.
    """
    if ATAC_NORM == "global":
        # Require a valid constant before applying global normalization.
        if _GP99 is None or not np.isfinite(_GP99) or _GP99 <= 1e-6:
            raise RuntimeError(
                "ATAC_NORM=global requires a valid global p99 constant. Call "
                "data.bind_gp99(ATAC_DIR) before training and ensure the cached "
                "<ATAC directory>/.global_p99 is finite and positive.")
        return (atac / _GP99).astype(np.float32) * ATAC_NORM_SCALE
    if ATAC_NORM == "log1p_p99":
        p = float(np.percentile(atac, 99))
        return atac if p <= 1e-6 else (atac / p).astype(np.float32) * ATAC_NORM_SCALE
    raise ValueError(f"Unknown ATAC_NORM={ATAC_NORM!r}; expected 'global' or 'log1p_p99'.")


def fetch_atac(chrom, start, end, atac_mm=None):
    """Return float32 ATAC slice of length (end-start), padded with 0 if OOB.

    Missing chromosome tracks and out-of-bounds positions are filled with zero.
    The bigWig converter also replaces NaN with zero before log1p. ATAC therefore
    encodes missing data and measured zero signal identically, without a separate
    missingness channel. DNA padding uses 0.25 for each of the four bases.
    """
    L = end - start
    arr = (atac_mm or {}).get(chrom)
    if arr is None:
        return np.zeros(L, dtype=np.float32)
    n = arr.shape[0]
    if start >= n or end <= 0:
        return np.zeros(L, dtype=np.float32)
    s = max(0, start); e = min(n, end)
    sl = np.asarray(arr[s:e]).astype(np.float32)
    if sl.shape[0] == L:
        return sl
    out = np.zeros(L, dtype=np.float32)
    out[max(0, -start):max(0, -start) + sl.shape[0]] = sl
    return out


def stack_seq_atac(seq_oh, chrom, start, end, strand="+", atac_mm=None):
    """Concatenate (L, 4) DNA one-hot with (L, 1) ATAC -> (L, 5)."""
    L = seq_oh.shape[0]
    if ATAC_ZERO:
        return np.concatenate([seq_oh, np.zeros((L, 1), np.float32)], axis=1)
    atac = fetch_atac(chrom, start, end, atac_mm)
    if atac.shape[0] != L:
        if atac.shape[0] > L:
            atac = atac[:L]
        else:
            pad = np.zeros(L - atac.shape[0], dtype=np.float32)
            atac = np.concatenate([atac, pad])
    if strand == "-":
        atac = atac[::-1].copy()
    # train.py binds the current cell's p99 before constructing training batches.
    atac = normalize_atac(atac)
    return np.concatenate([seq_oh, atac[:, None].astype(np.float32)], axis=1)


class _ATACTransform:
    """Picklable adapter that binds the sampler's current accessibility store."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __call__(self, chrom, start, end, dna, strand):
        return stack_seq_atac(dna, chrom, start, end, strand, self.sampler.atac_mm)


def make_sampler(genome, target, feat_name, core_bp, resolution,
                 holdout_test, holdout_val, atac_mm):
    """Build fixed-window batches with the established DNA-only 100-bp jitter."""
    sampler = FixedWindowSampler(
        reference_sequence=genome, target=target, sequence_length=core_bp,
        position_resolution=resolution, test_holdout=holdout_test,
        validation_holdout=holdout_val, random_shift=100,
    )
    sampler.atac_mm = atac_mm
    sampler.sequence_transform = _ATACTransform(sampler)
    return sampler
