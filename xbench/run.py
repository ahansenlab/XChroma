"""Score cached predictions by model, cell and metric.

Current mode requires an explicit prediction store, a producer cache manifest
matching the locked ten-member model, and a fresh OUT directory. It preserves
missing values and uses the nine current metrics, including cooltools insulation.
No prediction or GPU forward pass is performed here.

Model-specific OFFDIAG and RES_KB definitions determine geometry. MODELS, METRICS,
CELLS, CHROMS and LIMIT select evaluation subsets; INSUL_BINS controls physical
insulation scales through each grid's resolution. EPCOT_HEAD must be explicit.
See README.md for the complete current configuration contract.

Per-window files are keyed by model/cell/metric, not complete source identity.
The historical incremental behavior is therefore available only through explicit
XCHROM_BENCH_LEGACY=1 with a deliberately configured store and output directory.
"""
import os, sys, csv
import numpy as np

from .config import configure_protocol, require_prediction_store, prepare_output_directory
configure_protocol()

from . import metrics as MET
from .models import load_model, all_names as all_models

OUT = os.environ.get("OUT", os.path.join(os.path.dirname(__file__), "out"))
DEFAULT_INSUL_BINS = [5, 10, 15, 25, 50, 100]   # insulation 缺省窗宽(bin); bins × 各线 RES_KB(1/2/10) 并成实际 kb 尺度
                                          # RES_KB=1: 5/10/15/25/50/100 · RES_KB=2: 10/20/30/50/100/200 · RES_KB=10: 50/100/150/250/500/1000
DEFAULT_INSUL_MAX_KB = 250   # 缺省绝缘尺度上限(kb); >250kb(500/1000, 来自 RES_KB=10 的 bin50/100)默认不报, env INSUL_MAX_KB 可调


def _env_list(key):
    v = os.environ.get(key, "").strip()
    return [x.strip() for x in v.split(",") if x.strip()] if v else None


def _offdiag_mask(nb, k=2):
    ii = np.arange(nb)
    return np.abs(ii[:, None] - ii[None, :]) > k


def _metric_fns(names):
    """把指标名展开成 [(metric_name, score_fn, lower_better), ...]。
    insulation 是参数化的 -> 每个 window_bin 展开成一个具名指标(名字带真实 kb)。"""
    bins = [int(x) for x in (_env_list("INSUL_BINS") or DEFAULT_INSUL_BINS)]
    out = []
    for nm in names:
        mod = MET.load(nm)
        if nm == "insulation":
            out.append(("__insul__", bins, mod))     # 占位, RES_KB 在打分时才知道 -> 延后展开
        else:
            out.append((nm.capitalize() if nm != "scc" and nm != "ssim" and nm != "mse"
                        else nm.upper(), mod.score, getattr(mod, "LOWER_BETTER", False)))
    return out


def _expand_insul(bins, mod, res_kb):
    """把 insulation 的 bins 展开成具名 score_fn(名字 = 真实 kb = window_bins*res_kb)。
    ⛔ insulation.make() 返回 **(name, score_fn) 元组**, 不是带 .name/.score 的对象。"""
    max_kb = float(os.environ.get("INSUL_MAX_KB", DEFAULT_INSUL_MAX_KB))
    fns = []
    for wb in bins:
        if wb * res_kb > max_kb:                 # 超上限的尺度(默认 >250kb)不报
            continue
        name, score_fn = mod.make(wb, res_kb)
        fns.append((name, score_fn, False))      # name 已是 Insul{kb}kb
    return fns


def _apply_mask(P, G, offdiag):
    if not offdiag:
        return P, G
    m = _offdiag_mask(P.shape[0])
    return np.where(m, P, np.nan), np.where(m, G, np.nan)


def _offdiag_override():
    """OFFDIAG 环境变量: "1"=强制只在 |i-j|>2 上打分, "0"=强制全图, 留空=各模型自己的默认。

    ⛔ 不设时行为和以前**逐字节相同** —— 各模型仍用 models/*.py 里写死的 OFFDIAG。
       只有显式设了 0/1 才覆盖, 而且此时 perwin 会写到带后缀的独立文件, 不碰已有结果。
    """
    v = os.environ.get("OFFDIAG", "").strip()
    return v if v in ("0", "1") else None


def _perwin_path(model, cell, metric):
    ov = _offdiag_override()
    sfx = f"__offdiag{ov}" if ov is not None else ""     # 覆盖时另存, 绝不覆盖已有 perwin
    return os.path.join(OUT, "perwin", model, f"{cell}__{metric}{sfx}.tsv")


def run():
    require_prediction_store()
    prepare_output_directory()
    models = _env_list("MODELS") or [m for m in all_models() if m != "xchrom"]
    metric_names = _env_list("METRICS") or MET.all_names()
    cells_ovr = _env_list("CELLS")
    chroms_ovr = _env_list("CHROMS")
    limit = int(os.environ.get("LIMIT", "0"))
    force = os.environ.get("FORCE", "0") == "1"

    print(f"[run] models={models}", flush=True)
    print(f"[run] metrics={metric_names}  cells={cells_ovr or '(each model default)'}  "
          f"chroms={chroms_ovr or 'all'}  limit={limit or '全跑'}  force={force}", flush=True)

    for mname in models:
        M = load_model(mname)
        from .provenance import record_adapter
        record_adapter(M, OUT, mode="whole-map")
        pkey = getattr(M, 'KEY', None) or mname     # perwin 路径键: 带 variant(epcot_rcmc/cleopatra_2kb), 避免覆盖
        cells = cells_ovr or getattr(M, "_CELLS", None) or ["GM12878", "K562", "HepG2", "HCT116"]
        res_kb = getattr(M, "RES_KB", 1.0)
        _ov = _offdiag_override()
        offdiag = (_ov == "1") if _ov is not None else getattr(M, "OFFDIAG", False)
        if _ov is not None:
            print(f"[offdiag] {mname}: 环境变量覆盖 -> {'|i-j|>2' if offdiag else 'whole map'}"
                  f" (默认是 {getattr(M, 'OFFDIAG', False)})", flush=True)

        flat = []
        for item in _metric_fns(metric_names):
            if item[0] == "__insul__":
                flat += _expand_insul(item[1], item[2], res_kb)
            else:
                flat.append(item)

        has_ours = hasattr(M, "load_ours")     # 对手线才有: 同口径下我们的图
        for cell in cells:
            todo = [f for f in flat if force or not os.path.exists(_perwin_path(pkey, cell, f[0]))]
            if not todo:
                print(f"  [skip] {mname}/{cell}: 全部 perwin 已存在", flush=True)
                continue
            wins = M.windows(cell)
            if chroms_ovr:
                wins = [w for w in wins if w[0] in chroms_ovr]
            if limit:
                wins = wins[:limit]
            if not wins:
                print(f"  [skip] {mname}/{cell}: 无窗口", flush=True); continue
            print(f"  [{mname}/{cell}] {len(wins)} 窗 × {len(todo)} 指标待算", flush=True)

            acc = {f[0]: [] for f in todo}     # metric -> [(chrom, key, opp_val, ours_val)]
            import time as _time; _t0 = _time.time()
            for _wi, (chrom, key) in enumerate(wins):
                if (_wi + 1) % 200 == 0:
                    _el = _time.time() - _t0; _rate = (_wi + 1) / max(_el, 1e-9)
                    _eta = (len(wins) - (_wi + 1)) / max(_rate, 1e-9)
                    print(f"    ..{_wi+1}/{len(wins)} 窗  {_el:.0f}s  {_rate:.1f} win/s  ETA {_eta:.0f}s", flush=True)
                O = M.load(cell, chrom, key)                              # 对手
                G = M.load_gt(cell, chrom, key) if hasattr(M, "load_gt") else None
                U = M.load_ours(cell, chrom, key) if has_ours else None   # 我们(同口径)
                if O is None or G is None:
                    continue
                if not np.isfinite(G).any() or np.nanstd(G) < 1e-9:
                    continue
                Om, Gm = _apply_mask(O, G, offdiag)
                Um = _apply_mask(U, G, offdiag)[0] if U is not None else None
                for (mn, fn, lower) in todo:
                    try: ov = fn(Om, Gm)
                    except Exception: ov = float("nan")
                    try: uv = fn(Um, Gm) if Um is not None else float("nan")
                    except Exception: uv = float("nan")
                    if np.isfinite(ov) or np.isfinite(uv):
                        acc[mn].append((chrom, key, ov, uv))

            for (mn, fn, lower) in todo:
                rows = acc[mn]
                p = _perwin_path(pkey, cell, mn)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w", newline="") as fh:
                    w = csv.writer(fh, delimiter="\t", lineterminator="\n")
                    w.writerow(["chrom", "key", "xchrom", "opp"])
                    for (c, k, ov, uv) in rows:
                        w.writerow([c, k, f"{uv:.6f}", f"{ov:.6f}"])
                om = np.nanmean([r[2] for r in rows]) if rows else float("nan")
                um = np.nanmean([r[3] for r in rows]) if rows else float("nan")
                print(f"    -> {mn:12s} n={len(rows):>4}  XChrom={um:+.4f} vs {M.LABEL}={om:+.4f}", flush=True)

    if os.environ.get("NO_AGG") != "1":
        _aggregate()


def _aggregate():
    """从 perwin 文件聚合成 summary.tsv(model / cell / metric / n / mean)。"""
    root = os.path.join(OUT, "perwin")
    if not os.path.isdir(root):
        print("[agg] 无 perwin, 跳过"); return
    rows = []
    for model in sorted(os.listdir(root)):
        md = os.path.join(root, model)
        if not os.path.isdir(md):
            continue
        for fn in sorted(os.listdir(md)):
            if not fn.endswith(".tsv"):
                continue
            cell, metric = fn[:-4].split("__", 1)
            xc, op = [], []
            with open(os.path.join(md, fn)) as fh:
                r = csv.reader(fh, delimiter="\t"); next(r, None)
                for line in r:
                    if len(line) < 4:
                        continue
                    try:
                        a, b = float(line[2]), float(line[3])
                    except ValueError:
                        continue
                    xc.append(a); op.append(b)
            pair = [(a, b) for a, b in zip(xc, op) if np.isfinite(a) and np.isfinite(b)]
            xc = [a for a, _ in pair]; op = [b for _, b in pair]
            xm = float(np.mean(xc)) if xc else float("nan")
            om = float(np.mean(op)) if op else float("nan")
            from xbench.aggregate_full10 import base_metric
            _base = base_metric(metric.split("|")[0]).lower()
            try:
                lower = bool(getattr(MET.load(_base), "LOWER_BETTER", False))
            except Exception:
                lower = _base == "mse"
            win = "Y" if (xm < om if lower else xm > om) else "N"
            rows.append((model, cell, metric, len(op), xm, om, win))
    sp = os.path.join(OUT, "summary.tsv")
    with open(sp, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["model", "cell", "metric", "n", "XChrom", "opp", "win"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.6f}", f"{r[5]:.6f}", r[6]])
    nwin = sum(1 for r in rows if r[6] == "Y")
    print(f"\n[agg] {len(rows)} 行 ({nwin} 胜) -> {sp}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, epilog="Configure input paths and selectors with environment variables; see xbench/README.md.")
    parser.parse_args()
    run()
