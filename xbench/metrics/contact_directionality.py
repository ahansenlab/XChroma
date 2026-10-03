"""contact directionality track 的 Spearman 相关

⛔ Pollard 官方实现, 经 vectorMethodToScalar(finalCompMetric='corr') -> track 的 Spearman。
⛔ 2026-09-05 新增: 把原来 pollard_full.py 那条独立链路并进 xbench, 从此只有一套 benchmark。
   注册进 metrics/__init__.py 的 _NAMES 之前, 确保没有 array 正在排队 ——
   见 [[feedback_never_edit_script_while_array_drains]]。
⛔ 2026-09-07: 原来传进去的是 np.nan_to_num(...)。而 vectorMethodToScalar 第一件事就是
   fill_missing_points_map(a, b, fill=np.nan) —— 把两图缺失取**并集**再置 nan, 这是
   Pollard 保证两边对称的地方。预先填 0 会把这一步废掉, 也会让 run.py 的 offdiag 掩码
   (它把 |i-j|<=2 设成 nan)失效。KEEP_MISSING=1 时不预填, 交给 Pollard 自己处理。
   ⛔ 换口径要整表重算, 别和旧数混。
"""
import os
import numpy as np
from ..config import keep_missing

from . import pollard

POLLARD = True
LOWER_BETTER = False
DOC = "contact directionality track 的 Spearman 相关"


def _prep(X):
    X = np.asarray(X, np.float64)
    return X if keep_missing() else np.nan_to_num(X)


def score(pred, gt):
    try:
        v = pollard.vectorMethodToScalar(pollard.contact_directionality_track,
                                         _prep(pred), _prep(gt),
                                         finalCompMetric="corr")
        return float(v)
    except Exception:
        return float("nan")
