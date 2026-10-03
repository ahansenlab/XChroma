"""MSE —— Pollard 官方 scoring.mse(逐像素)。

⚠ LOWER_BETTER=True: 越小越好, 和所有相关类指标相反。汇总表的 win 判据必须为它反向。
⚠ MSE 对"预测得平"很宽容 —— 一张过度平滑的图 MSE 可以很低。别拿它当卖点。
   (历史: 它曾因 POLLARD 集合漏写而被自动标注成 "hand-rolled (NOT Pollard)",标注是错的。)
"""
from . import pollard

POLLARD = True
LOWER_BETTER = True
DOC = "逐像素均方误差(越小越好)"


def score(pred, gt):
    return float(pollard.mse(pred, gt))
