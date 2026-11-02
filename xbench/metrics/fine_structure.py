"""Difference-of-Gaussians (DoG) fine-structure transform.

Inputs use the prepared map grid, in natural-log observed/expected units.
The current scoring convention excludes NaN and exact-zero input locations.
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def dog(matrix, res_kb):
    """Apply mask-normalized DoG; preserve the nonzero-finite mask and use nearest boundaries."""
    x = np.asarray(matrix, np.float64)
    m = np.isfinite(x) & (x != 0)
    mask = m.astype(np.float64)
    xf = np.where(mask > 0, x, 0.0)
    s1 = 3.2 / float(res_kb)
    def smooth(s):
        den = gaussian_filter(mask, s, mode='nearest')
        return np.where(den > 1e-6,
                        gaussian_filter(xf, s, mode='nearest') / np.maximum(den, 1e-6), np.nan)
    return np.where(m, smooth(s1) - smooth(s1 * 2 ** (1.0 / 10)), np.nan)
