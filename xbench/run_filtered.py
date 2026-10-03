"""Score whole maps and fine structure (difference of Gaussians).
Input paths, model/cell selection and output directory are explicit environment
settings documented in xbench/README.md. No training or inference is performed."""
import os, sys, csv, json, numpy as np
from scipy.ndimage import gaussian_filter
from xbench.config import configure_protocol, require_prediction_store, prepare_output_directory
configure_protocol()
NEW_BASE = None
from xbench.models import load_model
from xbench import metrics as MET
from xbench.run import _apply_mask, _expand_insul, _offdiag_mask   # 复用主线的实现

OUT = os.environ.get("OUT")
LINES = os.environ.get("LINES", "epcot:intacthic,epcot:microc,epcot:rcmc,alphagenome,akita,corigami,epiphany").split(",")
ONLY_CELL = os.environ.get("ONLY_CELL", "")
ONLY_CHROM = {c.strip() for c in os.environ.get("ONLY_CHROM","").split(",") if c.strip()}
DOG_LINES = set(os.environ.get("DOG_LINES", "epcot,alphagenome,akita").split(","))
LIMIT = int(os.environ.get("LIMIT", "0"))
PERWIN = os.environ.get("PERWIN", "0") == "1"
PW = []       # >0 时每 cell 只取前 N 窗(调试)

#     1kb/bin -> lambda*=7.36 kb ; 2kb/bin -> 14.72 kb ; 10kb/bin -> 73.6 kb
SIGMA1_KB = float(os.environ.get("SIGMA1_KB", "3.20"))   # = mustache 默认 1.6 bin @ 2kb 网格
S_OCT = 10                  # mustache 每倍频程档数 (mustache.py:713)

def sigmas_for(res_kb):
    """该线的 (sigma1, sigma2), 单位 = bin。lambda* = 4.601 * sigma1 恒为 14.72 kb。"""
    s1 = SIGMA1_KB / float(res_kb)
    return s1, s1 * 2 ** (1.0 / S_OCT)

S1, S2 = sigmas_for(1.0)    # 默认值; 每条线在 _mk 里按自己的 RES_KB 覆盖

# SSIM is evaluated for whole maps and DoG features. The SSIM wrapper passes
# GT first; data_range is GT-derived after each pair-specific missing mask.
# MSE remains a whole-map-only metric under the current benchmark policy.
NO_DOG_METRIC = {"MSE"}

def _valid(X):
    return np.isfinite(X) & (X != 0)

def _gs(X, s, M=None):
    if M is None: M = _valid(X)
    M = M.astype(np.float64)
    Xf = np.where(M > 0, X, 0.0)
    den = gaussian_filter(M, s, mode="nearest")
    return np.where(den > 1e-6, gaussian_filter(Xf, s, mode="nearest") / np.maximum(den, 1e-6), np.nan)

def dog(X):
    """Mustache 第一层带通, 掩膜归一化; 缺失(NaN 或恰好 0)处保持 NaN。"""
    if X is None: return None
    X = np.asarray(X, np.float64)
    m = _valid(X)
    return np.where(m, _gs(X, S1, m) - _gs(X, S2, m), np.nan)

def main():
    global NEW_BASE
    NEW_BASE = require_prediction_store()
    if os.environ.get('EPCOT_DIR') and sum(spec.partition(':')[0] == 'epcot' for spec in LINES) > 1:
        raise ValueError('Multiple EPCOT heads cannot share the global EPCOT_DIR override. Unset it or run one explicit head per scoring command.')
    prepare_output_directory()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for spec in LINES:
        mod_name, _, head = spec.partition(":")
        if head: os.environ["EPCOT_HEAD"] = head          # ⛔ 必须在 import 前设
        import importlib
        M = load_model(mod_name)
        importlib.reload(M)                                # head 变了要重载, 否则 KEY/路径还是上一个
        lname = getattr(M, "KEY", mod_name)
        if hasattr(M, "_OURS_BASE"):
            M._OURS_BASE = NEW_BASE
        if hasattr(M, "_OURS_PRED"):
            M._OURS_PRED = {c: f"{NEW_BASE}/{c}" for c in M._OURS_PRED}
        from xbench.provenance import record_adapter
        record_adapter(M, OUT, mode="whole-map-and-fine-structure")
        res_kb = float(M.RES_KB); offdiag = bool(getattr(M, "OFFDIAG", False))
        do_dog = mod_name in DOG_LINES
        global S1, S2
        S1, S2 = sigmas_for(res_kb)
        if do_dog and S1 < 1.0:
            raise SystemExit(f"⛔ {mod_name}: sigma1={S1:.3f} bin < 1, 该网格({res_kb} kb)上 DoG 无意义")
        if do_dog:
            print(f"[dog] {mod_name}: res={res_kb} kb/bin  sigma1={S1:.3f} bin ({SIGMA1_KB} kb)  "
                  f"sigma2={S2:.3f} bin  lambda*={4.6014*SIGMA1_KB:.2f} kb", flush=True)
        flat = []
        for item in MET.make_all() if hasattr(MET, "make_all") else _mk():
            if item[0] == "__insul__": flat += _expand_insul(item[1], item[2], res_kb)
            else: flat.append(item)
        for cell in M._CELLS:
            if ONLY_CELL and cell != ONLY_CELL: continue
            wins = M.windows(cell)
            if LIMIT: wins = wins[:LIMIT]
            acc = {}
            if ONLY_CHROM: wins = [w for w in wins if w[0] in ONLY_CHROM]
            for wi, (chrom, key) in enumerate(wins):
                try:
                    O = M.load(cell, chrom, key); G = M.load_gt(cell, chrom, key)
                    U = M.load_ours(cell, chrom, key)
                except Exception: continue
                if O is None or G is None or U is None: continue
                if not np.isfinite(G).any() or np.nanstd(G) < 1e-9: continue
                Om, Gm = _apply_mask(O, G, offdiag); Um = _apply_mask(U, G, offdiag)[0]
                variants = [("raw", Om, Gm, Um)]
                if do_dog: variants.append(("dog", dog(Om), dog(Gm), dog(Um)))
                for tag, Ov, Gv, Uv in variants:
                    for (mn, fn, lower) in flat:
                        if tag == "dog" and mn in NO_DOG_METRIC:   # 见 NO_DOG_METRIC 注释
                            continue
                        k = (cell, f"{mn}|{tag}", bool(lower))
                        try: ov = fn(Ov, Gv)
                        except Exception: ov = float("nan")
                        try: uv = fn(Uv, Gv)
                        except Exception: uv = float("nan")
                        if np.isfinite(ov) or np.isfinite(uv):
                            acc.setdefault(k, []).append((ov, uv))
                            if PERWIN:
                                _span = int(round(Gm.shape[0] * res_kb * 1000))
                                PW.append((lname, cell, mn, tag, chrom, int(key),
                                           int(key) + _span, uv, ov))
                if (wi + 1) % 500 == 0:
                    print(f"  [{lname}/{cell}] {wi+1}/{len(wins)}", flush=True)
            for (cell_, mt, lower), vals in sorted(acc.items()):
                _p = [(a, b) for a, b in vals if np.isfinite(a) and np.isfinite(b)]
                if not _p:
                    continue
                ov = float(np.mean([a for a, _ in _p])); uv = float(np.mean([b for _, b in _p]))
                mn, tag = mt.split("|")
                win = "Y" if ((uv < ov) if lower else (uv > ov)) else "N"
                rows.append(dict(model=lname, cell=cell_, metric=mn, filt=tag,
                                 n=len(_p), XChrom=f"{uv:.6f}", opp=f"{ov:.6f}", win=win))
                print(f"    -> {lname:<16}{cell_:<9}{mn:<12}{tag:<4} n={len(_p):>5} "
                      f"ours={uv:+.4f} opp={ov:+.4f} {win}", flush=True)
    tag_ = os.environ.get("PART", "")
    p = f"{OUT}/part_{tag_}.tsv" if tag_ else f"{OUT}/summary.tsv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model","cell","metric","filt","n","XChrom","opp","win"],
                           delimiter="\t", lineterminator="\n")
        w.writeheader(); [w.writerow(r) for r in rows]
    print(f"\n-> {p}  ({len(rows)} 行)")
    if PERWIN:
        pwf = f"{OUT}/perwin_{tag_}.tsv" if tag_ else f"{OUT}/perwin.tsv"
        with open(pwf, "w", newline="") as fh2:
            w2 = csv.writer(fh2, delimiter="\t", lineterminator="\n")
            w2.writerow(["model", "cell", "metric", "filt", "chrom", "start", "end", "xchrom", "opp"])
            for r_ in PW:
                w2.writerow(list(r_[:7]) + [f"{r_[7]:.6f}", f"{r_[8]:.6f}"])
        print(f"-> {pwf}  ({len(PW)} 行, 逐窗)")

def _mk():
    """⛔ insulation 的 bins 与上限**逐字取主线 run.py 的常量**, 不自己编 ——
    我最初写死 [2,5,...] 多出一个 Insul2kb(1kb 网格上只有 2 bin, 无意义), 主线没有这一档。"""
    from xbench.run import DEFAULT_INSUL_BINS
    out = []
    only = {x.strip().lower() for x in os.environ.get("ONLY_METRIC","").split(",") if x.strip()}
    for nm in MET.all_names():
        if only and nm.lower() not in only: continue
        mod = MET.load(nm)
        if nm == "insulation": out.append(("__insul__", [int(x) for x in os.environ.get("INSUL_BINS", "").split(",") if x.strip()] or list(DEFAULT_INSUL_BINS), mod))
        else:
            nice = nm.upper() if nm in ("scc","ssim","mse") else nm.capitalize()
            out.append((nice, mod.score, getattr(mod, "LOWER_BETTER", False)))
    return out

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, epilog="Configure input paths and selectors with environment variables; see xbench/README.md.")
    parser.parse_args()
    main()
