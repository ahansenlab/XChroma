"""训练曲线 —— **只用同细胞的留出染色体**, 不碰任何用于 test 的细胞系。

⛔ 这里刻意**没有** zero-shot 人类细胞曲线。用 4 个人类细胞系画训练曲线, 等于拿它们选
   checkpoint, 那它们就成了 validation 而不是 held-out test —— 发布出去这个说法站不住。
   人类细胞系的评测在训练之外单独跑, 用最终冻结的 ckpt, 只跑一次。

曲线的两条线都来自 mESC 自己:
  median_r  —— `HOLDOUT_TEST` 那几条染色体上, 逐 1 Mb 区域的 log(O/E) Pearson 取中位数
  loss      —— 训练损失

数据来源是 append-only 的 TSV, 所以 preempt/resume 不会把曲线截断:
  <rundir>/metrics.tsv    step, median_r, ...
  <rundir>/val_loss.tsv   step, val_pear, loss
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path):
    """读 append-only TSV。同一个 step 出现多次时以最后一次为准 (resume 会重复写)。"""
    if not os.path.exists(path):
        return None, None
    rows = {}
    with open(path) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if not p or not p[0]:
                continue
            try:
                step = int(float(p[0]))
            except ValueError:
                continue
            vals = []
            for x in p:
                try:
                    vals.append(float(x))
                except ValueError:
                    vals.append(np.nan)
            rows[step] = vals
    if not rows:
        return hdr, None
    steps = sorted(rows)
    arr = np.full((len(steps), len(hdr)), np.nan)
    for i, s in enumerate(steps):
        v = rows[s]
        arr[i, :min(len(v), len(hdr))] = v[:len(hdr)]
    return hdr, arr


def _col(hdr, arr, name):
    if hdr is None or arr is None or name not in hdr:
        return None
    return arr[:, hdr.index(name)]


def _smooth(y, w=5):
    y = np.asarray(y, float)
    o = np.full_like(y, np.nan)
    for i in range(len(y)):
        seg = y[max(0, i - w // 2):min(len(y), i + w // 2 + 1)]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            o[i] = seg.mean()
    return o


def plot_curve(rundir):
    """画 <rundir>/training_curve.png。返回 PNG 路径; 没有可画的数据时返回 None。"""
    name = os.path.basename(rundir.rstrip("/"))
    mh, ma = _load(os.path.join(rundir, "metrics.tsv"))
    lh, la = _load(os.path.join(rundir, "val_loss.tsv"))
    mr = _col(mh, ma, "median_r")
    if mr is None or not np.isfinite(mr).any():
        return None
    step = _col(mh, ma, "step")
    # 点少的时候固定 w=5 会把整条线抹成一根平线
    w = 1 if len(step) < 8 else 5
    sm = (lambda y: np.asarray(y, float)) if w <= 1 else (lambda y: _smooth(y, w))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(step, mr, "-", lw=0.9, color="tab:blue", alpha=0.35)
    ax.plot(step, sm(mr), "-o", ms=3, lw=2.0, color="tab:blue", label="mESC median_r (1 kb, held-out chroms)")
    ax.set_xlabel("step (true cumulative, survives preempt)")
    ax.set_ylabel("median_r")
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)
    best = float(np.nanmax(mr))
    ax.set_title(f"{name}   best median_r = {best:.4f}", fontsize=11)

    loss = _col(lh, la, "loss")
    if loss is not None and np.isfinite(loss).any():
        axL = ax.twinx()
        axL.plot(_col(lh, la, "step"), loss, "--", lw=1.0, color="tab:red", alpha=0.55, label="train loss")
        axL.set_ylabel("train loss", color="tab:red")
        axL.tick_params(axis="y", labelcolor="tab:red")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axL.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
    else:
        ax.legend(fontsize=8, loc="lower right")

    out = os.path.join(rundir, "training_curve.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
