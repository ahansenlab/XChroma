"""Common ICE, per-diagonal observed/expected and grid-aggregation operators.
Zero observed counts become missing during ln(O/E) transformation. Display
colour scaling is not part of these numerical transformations."""
import os
import numpy as np
from .config import keep_missing

CLIP = 2.0          # log(O/E) 的裁剪范围 [-CLIP, +CLIP],Pollard 口径


def ice(M, nb, it=20, mad=5.0):
    """逐窗 ICE 均衡。铁律:GT 一律 ICE,never SCALE/KR/VC。

    mad 过滤:按覆盖度的 log 值剔除低覆盖 bin(中位数下方 mad 个 MAD 之外),置零后迭代均衡。
    返回 None = 该窗有效 bin 太少(<50),调用方应跳过此窗。
    """
    A = M.astype(np.float64).copy()
    cov = A.sum(1); good = cov > 0
    if good.sum() < 50:
        return None
    lc = np.log(cov[good]); md = np.median(np.abs(lc - np.median(lc)))
    keep = np.zeros(nb, bool); keep[good] = lc > np.median(lc) - mad * 1.4826 * md
    A[~keep, :] = 0; A[:, ~keep] = 0
    for _ in range(it):
        s = A.sum(1); s[s == 0] = 1
        d = s / np.mean(s[keep]); d[d == 0] = 1
        A /= d[:, None]; A /= d[None, :]
    return A


def snl(M, nb=None):
    """observed -> log(O / per-diagonal E)。E = 该窗自己每条对角线的均值(只统计 >0 的像素)。

    ⚠ O = 0 的像素 -> **nan**(视为缺失,被 Pollard 的 remove_missing_points_flat 排除),
      不是变成一个极小的负数。这与"mask 掉 GT 的洞"是同一个决定,别改成 eps。
    """
    nb = M.shape[0] if nb is None else nb
    o = np.full_like(M, np.nan, np.float64)
    for d in range(nb):
        i = np.arange(nb - d); v = M[i, i + d]
        m = np.nanmean(v[np.isfinite(v) & (v > 0)]) if np.isfinite(v).any() else np.nan
        if np.isfinite(m) and m > 0:
            o[i, i + d] = v / m; o[i + d, i] = o[i, i + d]
    return np.log(np.where(o > 0, o, np.nan))


def clip(M, c=CLIP):
    """Clip log(O/E) to [-c,c], preserving native missing values by default.

    KEEP_MISSING=0 reproduces historical zero fill only in explicit legacy mode.
    Historical pre-guard code is archived; existing published data are unchanged.
    """
    X = np.asarray(M, np.float64)
    if keep_missing():
        return np.clip(X, -c, c)          # nan 原样保留
    return np.clip(np.nan_to_num(X), -c, c)


def agg(M, factor):
    """把 (n, n) 图按 factor 聚合(均值)。1kb -> 2kb 用 factor=2;1kb -> 10kb 用 10。"""
    n = (M.shape[0] // factor) * factor
    k = n // factor
    return M[:n, :n].reshape(k, factor, k, factor).mean((1, 3))


def offdiag_mask(nb, k=2):
    """|i-j| > k 的掩码。近对角线是 log(O/E)≈0 的平凡区,算进去等于给所有模型白送分。"""
    ii = np.arange(nb)
    return np.abs(ii[:, None] - ii[None, :]) > k
