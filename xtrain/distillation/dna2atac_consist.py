"""Target-specific DNA-to-accessibility distillation.

Dna2Atac and Dna2AtacCBP implement accessibility predictors; run(net) trains
a predictor through a separately supplied frozen contact network. Model and
normalization bindings are specific to each student, not the main ensemble.
"""

from pathlib import Path as _XChromPath
import sys as _xchrom_sys
_xchrom_code_root = _XChromPath(__file__).resolve().parents[2]
if str(_xchrom_code_root) not in _xchrom_sys.path:
    _xchrom_sys.path.insert(0, str(_xchrom_code_root))
import os, json, time, numpy as np, torch, pysam, cooler
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from torch import nn
from torch.utils.checkpoint import checkpoint
from xtrain.inputs import ZAtac

NT = {"A": 0, "C": 1, "G": 2, "T": 3}


def encode(s):
    s = s.upper(); oh = np.zeros((len(s), 4), np.float32); a = np.frombuffer(s.encode("ascii"), np.uint8)
    for b, i in NT.items(): oh[a == ord(b), i] = 1.0
    return oh


def log1p_p99(a):   # == normalize_atac log1p_p99 with scale=1.0 (what encattn was trained with)
    p = float(np.percentile(a, 99)); return a if p <= 1e-6 else (a / p).astype(np.float32)


# --- AlphaGenome-style count loss (verbatim port of train_atac.py _ag_*), for the ATAC track ---
# count/coverage signal -> models WHERE signal piles up (multinomial over MRES-bp segments) + total depth
# (poisson). Designed for peaky tracks like ATAC; does NOT regress to the conditional mean like MSE -> no flattening.
def _ag_poisson_loss(y_true, y_pred):
    y_true = y_true.abs().float(); y_pred = y_pred.float()
    logits = torch.log(y_pred + 1e-7); min_value = y_true - y_true * torch.log(y_true + 1e-7)
    return ((y_pred - y_true * logits) - min_value).mean()
def _ag_multinomial_loss(y_true, y_pred, mres, pos_w=1.0):
    """(B,S,C): poisson(segment totals)/mres + pos_w * KL(positional distribution within each segment)."""
    B, S, C = y_true.shape
    if S % mres != 0: return _ag_poisson_loss(y_true, y_pred)
    ns = S // mres
    yt = y_true.clamp(min=0).reshape(B, ns, mres, C); yp = y_pred.reshape(B, ns, mres, C)
    tot_t = yt.sum(dim=2, keepdim=True); tot_p = yp.sum(dim=2, keepdim=True)
    loss_total = _ag_poisson_loss(tot_t, tot_p) / mres
    prob = yp / (tot_p + 1e-7); loss_pos = (-yt * torch.log(prob + 1e-7))
    p_true = yt / (tot_t + 1e-7); loss_pos = loss_pos - (-yt * torch.log(p_true + 1e-7))   # CE -> KL (0 at optimum)
    return loss_total + pos_w * loss_pos.mean()


def _gn(c):
    g = 8 if c % 8 == 0 else (4 if c % 4 == 0 else (2 if c % 2 == 0 else 1))
    return nn.GroupNorm(g, c)


class DilBlock(nn.Module):
    def __init__(s, c, d):
        super().__init__()
        s.gn1 = _gn(c); s.c1 = nn.Conv1d(c, c, 3, padding=d, dilation=d)
        s.gn2 = _gn(c); s.c2 = nn.Conv1d(c, c, 3, padding=d, dilation=d)

    def forward(s, x):
        h = s.c1(torch.nn.functional.gelu(s.gn1(x)))
        h = s.c2(torch.nn.functional.gelu(s.gn2(h)))
        return x + h


class Dna2Atac(nn.Module):
    """DNA one-hot (B,4,L) -> ATAC track (B,1,L), non-negative, in the log1p_p99 space."""
    def __init__(s, c=64, nb=9):
        super().__init__()
        s.stem = nn.Conv1d(4, c, 7, padding=3)
        s.blocks = nn.ModuleList([DilBlock(c, 2 ** (i % 9)) for i in range(nb)])
        s.head = nn.Conv1d(c, 1, 1)

    def forward(s, x):
        h = s.stem(x)
        for b in s.blocks:
            h = checkpoint(b, h, use_reentrant=False)
        return torch.nn.functional.softplus(s.head(h))   # (B,1,L) >= 0


class _CBPBlock(nn.Module):
    """ChromBPNet-style dilated residual block: relu -> dilated conv(k3), SAME padding, residual add."""
    def __init__(s, c, d):
        super().__init__()
        s.conv = nn.Conv1d(c, c, 3, padding=d, dilation=d)   # same padding keeps length -> equal-length trunk
    def forward(s, x):
        return x + torch.nn.functional.relu(s.conv(x))


class Dna2AtacCBP(nn.Module):
    """ChromBPNet's deep dilated trunk (wide k21 stem, C=512, 8 residual blocks dil 2^1..2^8) but
    EQUAL-LENGTH (SAME padding) so one forward covers the whole 1Mb window and MSE-through-frozen-general
    distillation can backprop in a single pass. RF = 21 + 2*sum(2^1..2^8) ~= 1041bp (matches ChromBPNet).
    DNA one-hot (B,4,L) -> ATAC (B,1,L) >= 0 in log1p_p99 space. Same I/O contract as Dna2Atac."""
    def __init__(s, c=512, nb=8):
        super().__init__()
        s.stem = nn.Conv1d(4, c, 21, padding=10)
        s.blocks = nn.ModuleList([_CBPBlock(c, 2 ** (i + 1)) for i in range(nb)])
        s.head = nn.Conv1d(c, 1, 1)
    def forward(s, x):
        h = torch.nn.functional.relu(s.stem(x))
        for b in s.blocks:
            h = checkpoint(b, h, use_reentrant=False)
        return torch.nn.functional.softplus(s.head(h))   # (B,1,L) >= 0


def run(net):
    dev = "cuda"
    WIN = 1_000_000; RES = 1000; N = WIN // RES; DIAG = 2
    CELL = os.environ.get("D2A_CELL", "GM12878")
    GENOME_FA = os.environ["D2A_GENOME_FA"]
    ATAC_DIR = os.environ["D2A_ATAC_DIR"]
    COOL = os.environ.get("D2A_COOL", "")   # capture RCMC GT (optional; only covers a few regions incl. MYC)
    OUTDIR = os.environ.get("OUTDIR", f"./runs/distillation_{CELL}")
    os.makedirs(OUTDIR + "/models", exist_ok=True); os.makedirs(OUTDIR + "/figs", exist_ok=True)
    MAX_STEPS = int(os.environ.get("D2A_MAX_STEPS", "30000"))
    LR = float(os.environ.get("D2A_LR", "1e-3"))
    BATCH = int(os.environ.get("D2A_BATCH", "2"))
    WARMUP = int(os.environ.get("D2A_WARMUP", "500"))
    VAL_EVERY = int(os.environ.get("D2A_VAL_EVERY", "500"))
    SAVE_EVERY = int(os.environ.get("D2A_SAVE_EVERY", "250"))
    CKPT_EVERY = int(os.environ.get("D2A_CKPT_EVERY", "2000"))
    SEED = int(os.environ.get("D2A_SEED", "0"))
    SKIP_INITIAL_VAL = os.environ.get("SKIP_INITIAL_VAL", "1") == "1"
    HOLDOUT = set(os.environ.get("D2A_HOLDOUT", "chr8").split(",")) - {""}
    PRED_CH = int(os.environ.get("D2A_PRED_CH", "64"))
    PRED_BLOCKS = int(os.environ.get("D2A_PRED_BLOCKS", "9"))
    ATAC_AUX_W = float(os.environ.get("D2A_ATAC_AUX_W", "1.0"))
    ATAC_WARMUP = int(os.environ.get("D2A_ATAC_WARMUP", "800"))
    FRAGBP_MIN = float(os.environ.get("D2A_FRAGBP_MIN", "0"))   # HARD-gate: only train on windows with native frag/bp>=this (0=off)
    HAP1_HIC = os.environ.get("D2A_HAP1_HIC_VAL", "") == "1"   # val = HAP1 CTCF-edit preds (predictor->ATAC->frozen contact) vs Sanborn capture Hi-C
    rng = np.random.default_rng(SEED); torch.manual_seed(SEED)

    # ---- target-cell I/O (pysam seq + per-chrom ATAC store), independent of train_atac's mESC setup ----
    fa = pysam.FastaFile(GENOME_FA)
    _ZATAC = {}

    def _zatac(chrom):
        if chrom not in _ZATAC:
            zp = os.path.join(ATAC_DIR, f"{chrom}.atac.log1p.az")
            _ZATAC[chrom] = ZAtac(zp) if os.path.exists(zp) else None
        return _ZATAC[chrom]

    def load_atac(chrom, s, e):
        z = _zatac(chrom)
        if z is not None: return z.read(s, e).astype(np.float32)
        m = json.load(open(os.path.join(ATAC_DIR, "manifest.json")))
        L = m[chrom]["length"] if isinstance(m.get(chrom), dict) else m["chroms"][chrom]
        mm = np.memmap(os.path.join(ATAC_DIR, f"{chrom}.atac.log1p.f16.memmap"), dtype=np.float16, mode="r", shape=(L,))
        return np.asarray(mm[s:e]).astype(np.float32)

    def window_fragbp(chrom, start):
        # native frag/bp = mean(expm1(RAW log1p ATAC)). CHEAP (ATAC only, no seq fetch) -> gate BEFORE get_window.
        a = load_atac(chrom, start, start + WIN)
        if len(a) < WIN: return 0.0
        return float(np.expm1(a[:WIN]).mean())

    # ---- ATAC normalization: log1p_p99 (default) OR global (÷ ONE genome-wide log1p-p99 constant,
    #      matches the ATAC_NORM=global teacher; preserves cross-window absolute accessibility) ----
    D2A_NORM = os.environ.get("D2A_ATAC_NORM", "log1p_p99")
    ATAC_LINEAR = os.environ.get("D2A_ATAC_LINEAR", "") == "1"   # predictor 在 RAW COUNT 空间训(loss/target=count); 喂 general 前在 gen_contact 里转回 log1p_p99. 只在 ATAC_ONLY 下有意义
    if ATAC_LINEAR: print("[d2a] ATAC_LINEAR: target/pred/loss in raw-count space; gen_contact converts count->log1p_p99 before the frozen general", flush=True)
    _GP99 = [None]
    if D2A_NORM == "global":
        _gp = os.environ.get("D2A_GP99", "")
        if _gp:
            _GP99[0] = float(_gp)
        else:
            _man = json.load(open(os.path.join(ATAC_DIR, "manifest.json")))
            _chroms = list((_man.get("chroms", _man)).keys())
            _rng2 = np.random.RandomState(0); _samp = []
            for _c in [c for c in _chroms if c not in HOLDOUT][:6]:
                try:
                    _L = _man[_c]["length"] if isinstance(_man.get(_c), dict) else _man["chroms"][_c]
                    _a = load_atac(_c, 0, _L)
                except Exception:
                    continue
                _idx = _rng2.choice(len(_a), min(1_000_000, len(_a)), replace=False)
                _samp.append(_a[_idx])
            _GP99[0] = float(np.percentile(np.concatenate(_samp), 99)) if _samp else 1.0
        print(f"[d2a] GLOBAL atac norm: genome-wide log1p-p99={_GP99[0]:.4f}", flush=True)

    def _norm_atac(a):
        if ATAC_LINEAR:
            return np.expm1(a).astype(np.float32)      # a=memmap log1p(count) -> raw count; general 空间转换延后到 gen_contact
        if D2A_NORM == "global":
            g = _GP99[0]
            return a if (g is None or g <= 1e-6) else (a / g).astype(np.float32)
        return log1p_p99(a)

    def get_window(chrom, start):
        seq = fa.fetch(chrom, start, start + WIN).upper()
        if len(seq) != WIN or seq.count("N") > WIN * 0.5: return None
        atac = load_atac(chrom, start, start + WIN)
        if len(atac) < WIN: return None
        return encode(seq), _norm_atac(atac[:WIN])

    # ---- general model: freeze + eval (teacher and student use the SAME frozen net) ----
    gen = net.module if hasattr(net, "module") else net
    gen.eval()
    for p in gen.parameters():
        p.requires_grad_(False)

    # D2A_GENERAL_CKPTS="a.pt,b.pt,...": 每步从这些 general 里**随机抽一个**, 而不是固定一个。
    # 目的: 让 predictor 不去迎合某一个 general 的特质 —— 单个 general 的 idiosyncrasy 会被
    #       predictor 当成信号学进去, 换个 general 就失效。
    # ⛔ 同一步里 teacher 和 student 必须用**同一个** general: loss 是两条分支穿过同一个 general
    #    的一致性, 用不同的 general 会把 ckpt 之间的差异也算进 loss, 那不是我们要优化的东西。
    #    -> 训练循环里每步抽一次 gi, 显式传给两次 gen_contact。
    # ⛔ 所有 val / 出图路径**不传 gi**, 默认 gi=0 固定用第一个 general: 否则每次评估用的
    #    general 不同, 曲线和图会因为抽到谁而抖动, 没法跨 step 比。
    _GENS = [gen]
    _CKS = [c.strip() for c in os.environ.get("D2A_GENERAL_CKPTS", "").split(",") if c.strip()]
    if _CKS:
        import copy as _copy
        _GENS = []
        for _p in _CKS:
            if not os.path.exists(_p):
                print(f"[d2a][general] 缺 ckpt, 跳过: {_p}", flush=True); continue
            g2 = _copy.deepcopy(gen)
            _ck = torch.load(_p, map_location="cpu", weights_only=False)
            _sd = _ck.get("model", _ck); _sd = {k.replace("module.", "", 1): v for k, v in _sd.items()}
            _msd = g2.state_dict()
            _keep = {k: v for k, v in _sd.items() if k in _msd and v.shape == _msd[k].shape}
            # ⛔ 加载率不足 = 这个 ckpt 和当前架构对不上, 剩下的层还是随机初始化的。
            #    半随机的 general 会把 teacher 变成噪声, 而且不会报错 —— 必须硬失败。
            _frac = len(_keep) / max(len(_msd), 1)
            if _frac < 0.95:
                raise SystemExit(f"[d2a] {os.path.basename(_p)} 只匹配上 {len(_keep)}/{len(_msd)} "
                                 f"({_frac:.1%}) 个张量, 架构不一致, 拒绝当 general")
            g2.load_state_dict(_keep, strict=False)
            g2.eval()
            for p in g2.parameters(): p.requires_grad_(False)
            _GENS.append(g2.to(dev))
            print(f"[d2a][general] loaded {len(_keep)}/{len(_msd)} ({_frac:.1%}) <- {os.path.basename(_p)}", flush=True)
        if not _GENS:
            raise SystemExit("[d2a] D2A_GENERAL_CKPTS 里一个都没加载成功")
    print(f"[d2a][general] pool size = {len(_GENS)}"
          + ("  (每步随机抽一个; val 固定用 #0)" if len(_GENS) > 1 else "  (固定一个)"), flush=True)

    # ---- 层级 KL 作为 contact loss (D2A_HIER_KL=1) ----
    # 病灶: 现在的 contact 监督是 MSE(student, teacher) 穿过一个**MSE 训出来的**冻结 general,
    #       即"对 MSE 学出来的东西继续 MSE", 每叠一层压平一次。
    # 为什么这里适合 KL(而从零训不适合): 序列->接触图由冻结的 general 完成, KL 不必学序列表示;
    #       它只塑造 1 通道 ATAC 输入, 而 predictor 仅 0.23M 参数, 弱梯度要推动的空间小得多。
    #
    # 输出适配(关键): general 输出的是**无界 log(O/E)**, 要变成分布才能做 KL。
    #   D2A_KL_EXPECTED=0 (默认): p ∝ exp(logOE), 在 off-diag 上归一。各像素按自身 O/E 加权,
    #       结构本身就是信号。蒸馏两边都是同一个 general 的输出、同空间, expected 在这里不承担
    #       任何统计意义, 只起"按距离加权"的作用 —— 而那个加权正是把可约信号压到 0.26% 的元凶。
    #   D2A_KL_EXPECTED=1: p ∝ exp(logOE)*E, 忠于 Manta 的计数空间(需 D2A_KL_EXP_NPY 指向 expected)。
    # 防塌平机制在两种下都保留: 梯度 -q/p, teacher 说有峰而 student 给背景时代价发散。
    D2A_HIER_KL   = os.environ.get("D2A_HIER_KL", "") == "1"
    D2A_HIER_LV   = int(os.environ.get("D2A_HIER_LEVELS", "4"))
    D2A_HIER_SC   = float(os.environ.get("D2A_HIER_SCALE", "0.5"))
    D2A_HIER_SUMW = float(os.environ.get("D2A_HIER_SUMW", "0.2"))
    D2A_KL_EXPECTED = os.environ.get("D2A_KL_EXPECTED", "0") == "1"
    _DK = {}
    def _d2a_hier_kl(s2d, t2d, vmask):
        """s2d/t2d: (B,N,N) log(O/E), 同一冻结 general 的两次输出。vmask: (B,N,N) 或 (N,N) bool。"""
        e = 1e-6
        m = vmask.float() if vmask.dim() == 3 else vmask.float().unsqueeze(0)
        if D2A_KL_EXPECTED:
            if "E" not in _DK:
                _bd = np.exp(np.load(os.environ["D2A_KL_EXP_NPY"]).astype(np.float32))
                _n = s2d.shape[-1]
                _d = np.abs(np.arange(_n)[:, None] - np.arange(_n)[None, :])
                _DK["E"] = torch.from_numpy(_bd[np.clip(_d, 0, len(_bd) - 1)].astype(np.float32)).to(s2d.device)
            Emat = _DK["E"].unsqueeze(0)
        else:
            Emat = 1.0
        q = torch.exp(t2d.float().clamp(-30, 30)) * Emat * m
        p = torch.exp(s2d.float().clamp(-30, 30)) * Emat * m
        qs = q.sum(dim=(1, 2)); ps = p.sum(dim=(1, 2))
        q = q / (qs[:, None, None] + e); p = p / (ps[:, None, None] + e)
        B = q.shape[0]; v = q > 0
        ce = -(q[v] * torch.log(p[v] + e)).sum() / B
        sc = 1.0
        for _l in range(D2A_HIER_LV):
            n = q.shape[1]
            if n % 2:
                n -= 1
                if n < 2: break
                q = q[:, :n, :n]; p = p[:, :n, :n]
            q = q.reshape(B, n // 2, 2, n // 2, 2).sum(dim=(2, 4))
            p = p.reshape(B, n // 2, 2, n // 2, 2).sum(dim=(2, 4))
            w = D2A_HIER_SC ** (_l + 1); v = q > 0
            ce = ce + (-(q[v] * torch.log(p[v] + e)).sum() / B) * w; sc += w
        L = ce / sc
        L = L + D2A_HIER_SUMW * ((torch.log(ps + 1) - torch.log(qs + 1)) ** 2).mean()
        return L
    if D2A_HIER_KL:
        print(f"[d2a] HIER_KL contact loss: levels={D2A_HIER_LV} scale={D2A_HIER_SC} "
              f"sumw={D2A_HIER_SUMW} use_expected={D2A_KL_EXPECTED}", flush=True)

    def gen_contact(soh_t, atac_t, gi=0):   # soh_t (B,4,L), atac_t (B,1,L) -> (B,N,N) symmetrized
        if ATAC_LINEAR:                     # 入口是 raw count(真实/预测都一样) -> 转回 general 训练时的 log1p_p99 空间
            at = torch.log1p(atac_t.clamp(min=0).float())
            p99 = torch.quantile(at.reshape(at.shape[0], -1), 0.99, dim=1).clamp(min=1e-6).view(-1, 1, 1)
            atac_t = (at / p99).to(soh_t.dtype)
        x = torch.cat([soh_t, atac_t], 1)
        o = _GENS[gi](x); m = (o[0] if isinstance(o, tuple) else o)[:, 0]
        return 0.5 * (m + m.transpose(-1, -2))

    _arch = os.environ.get("D2A_PRED_ARCH", "default")
    if _arch == "cbp":
        predictor = Dna2AtacCBP(PRED_CH, PRED_BLOCKS).to(dev)          # ChromBPNet-style deep dilated trunk, equal-length
    else:
        predictor = Dna2Atac(PRED_CH, PRED_BLOCKS).to(dev)
    predictor.train()
    npar = sum(p.numel() for p in predictor.parameters()) / 1e6
    print(f"[d2a] CELL={CELL} predictor arch={_arch} ch={PRED_CH} blocks={PRED_BLOCKS} params={npar:.2f}M batch={BATCH} lr={LR}", flush=True)
    print(f"[d2a] general=FROZEN encattn(mESC) zero-shot on {CELL}; genome={GENOME_FA}; atac={ATAC_DIR}", flush=True)
    opt = torch.optim.Adam(predictor.parameters(), lr=LR)

    start_step = 0; LATEST = f"{OUTDIR}/models/predictor.latest.ckpt.pt"
    _PUBCK = os.environ.get("D2A_RESUME_CKPT", "")        # load a SPECIFIC predictor ckpt (e.g. a frozen step) instead of latest
    if _PUBCK: LATEST = _PUBCK
    if os.environ.get("RESUME", "1") == "1" and os.path.exists(LATEST):
        rc = torch.load(LATEST, map_location="cpu", weights_only=False)
        predictor.load_state_dict(rc["model"]); start_step = int(rc.get("step", 0))
        if "opt" in rc:
            try: opt.load_state_dict(rc["opt"])
            except Exception: pass
        print(f"[d2a resume] predictor from {LATEST} step={start_step}", flush=True)

    autos = [f"chr{i}" for i in range(1, 23)] + ["chrX"]   # works for hg38 (chr1-22,X) and mm39 (chr1-19,X auto-filtered)
    chroms = [c for c in fa.references if c in autos and c not in HOLDOUT]
    clen = {c: fa.get_reference_length(c) for c in chroms}
    clen_h = {c: fa.get_reference_length(c) for c in HOLDOUT if c in fa.references}
    w = np.array([clen[c] for c in chroms], float); w /= w.sum()
    ii = np.arange(N); off = torch.tensor(np.abs(ii[:, None] - ii[None, :]) > DIAG).to(dev)
    CROP_EDGE = int(os.environ.get("D2A_CROP_EDGE", "0"))   # edge-effect test: contact MSE on CENTRAL (N-2*CROP) bins only
    if CROP_EDGE > 0:                                        # frozen-general contact near the 1Mb window edge lacks context -> unreliable label
        _cen = (ii >= CROP_EDGE) & (ii < N - CROP_EDGE)
        off = off & torch.tensor(_cen[:, None] & _cen[None, :]).to(dev)
        print(f"[d2a][CROP_EDGE] contact MSE on central {N - 2 * CROP_EDGE} bins (drop {CROP_EDGE}kb each edge)", flush=True)
    print(f"[d2a] train chroms={len(chroms)} holdout={sorted(HOLDOUT)}", flush=True)

    # ---- figure / GT helpers ----
    def _md(a, k=2):
        z = a.astype(float).copy(); jj = np.arange(z.shape[0]); z[np.abs(jj[:, None] - jj[None, :]) <= k] = np.nan; return z

    def _pctn(z):
        f = z[np.isfinite(z)]
        if f.size == 0: return z
        lo, hi = np.nanpercentile(f, 1), np.nanpercentile(f, 99)
        return np.clip((z - lo) / (hi - lo + 1e-9), 0, 1)

    FIG_SMOOTH = int(os.environ.get("D2A_FIG_SMOOTH", "0"))   # k x k box blur on the GT panel only

    def _blur(a, k=FIG_SMOOTH):
        """Deep model output is already smooth; sparse capture/Hi-C GT is not, and at 1 kb the
        real structure sits under shot noise. Blurring the GT for DISPLAY is the same operation
        that lifted the RCMC GT Pearson 0.349 -> 0.502 -- it reveals signal rather than adding it.
        Never applied to a reported number, only to the pixels that get drawn."""
        if k <= 1: return a
        from scipy.signal import convolve2d
        fin = np.isfinite(a); ker = np.ones((k, k))
        num = convolve2d(np.where(fin, a, 0.0), ker, mode="same", boundary="symm")
        den = convolve2d(fin.astype(float), ker, mode="same", boundary="symm")
        return np.where(fin, np.where(den > 0, num / np.maximum(den, 1e-9), np.nan), np.nan)

    _cool = cooler.Cooler(COOL) if COOL else None

    def fetch_raw(chrom, s):
        raw = _cool.matrix(balance=True).fetch(f"{chrom}:{s}-{s + WIN}").astype(np.float32)
        if raw.shape[0] != N: raw = raw[:N, :N]
        return raw

    def gt_logoe_from_raw(raw):
        jj = np.arange(N); dist = np.abs(jj[:, None] - jj[None, :])
        fin = np.isfinite(raw)                                   # 向量化 per-diagonal 均值(bincount), 去掉 1000 次 Python 循环
        cnt = np.bincount(dist[fin], minlength=N).astype(float)
        ssum = np.bincount(dist[fin], weights=raw[fin].astype(float), minlength=N)
        ex = np.where(cnt > 0, ssum / np.maximum(cnt, 1), np.nan)
        last = ex[0] if np.isfinite(ex[0]) else 1e-6
        for d in range(N):
            if not np.isfinite(ex[d]): ex[d] = last
            else: last = ex[d]
        nm = ex[dist]; eps = float(np.nanmin(nm[nm > 0])) if (nm > 0).any() else 1e-6
        return np.log((np.maximum(raw, 0) + eps) / (nm + eps))

    FIGW = [("MYC", "chr8", 124988000), ("R1", "chr6", 26111000)]
    if os.environ.get("FIGW", ""):
        FIGW = [(p.split(":")[0], p.split(":")[1], int(p.split(":")[2])) for p in os.environ["FIGW"].split(";") if p]

    def dump_fig(step):
        try:
            cm = plt.get_cmap("YlOrRd").copy(); cm.set_bad("white")
            fig, ax = plt.subplots(len(FIGW), 3, figsize=(15, 5 * len(FIGW)), squeeze=False, constrained_layout=True)
            for ri, (nm0, chrom, s0) in enumerate(FIGW):
                g = get_window(chrom, s0)
                if g is None: continue
                soh, atac = g
                soh_t = torch.Tensor(soh.T).unsqueeze(0).to(dev)
                ra = torch.Tensor(atac).view(1, 1, -1).to(dev)
                with torch.no_grad():
                    tp = gen_contact(soh_t, ra).cpu().numpy()[0]
                    ap = predictor(soh_t)
                    sp = gen_contact(soh_t, ap).cpu().numpy()[0]
                if _cool is not None:
                    try:
                        raw = fetch_raw(chrom, s0); cov = np.isfinite(raw) & (raw > 0)
                        gt = np.where(cov, gt_logoe_from_raw(raw), np.nan) if cov.any() else np.full((N, N), np.nan)
                    except Exception: gt = np.full((N, N), np.nan)
                else: gt = np.full((N, N), np.nan)
                for a, (m, t) in zip(ax[ri], [(_blur(gt), f"{CELL} GT" + (f" ({FIG_SMOOTH}x{FIG_SMOOTH} blur)" if FIG_SMOOTH > 1 else "")),
                                              (tp, "general(DNA, REAL ATAC)"),
                                              (sp, f"general(DNA, PRED ATAC) step{step}")]):
                    a.imshow(_pctn(_md(m)), cmap=cm, vmin=0, vmax=1, origin="lower", interpolation="nearest")
                    a.set_xticks([]); a.set_yticks([]); a.set_title(f"{nm0} {chrom}:{s0 // 10**6}M  {t}", fontsize=10)
            fig.suptitle(f"{CELL} DNA->ATAC consistency step {step}: GT / real-ATAC contact / pred-ATAC contact", fontsize=13)
            for p in (f"{OUTDIR}/figs/d2a_step{step}.png", f"{OUTDIR}/figs/d2a_latest.png"):
                fig.savefig(p, dpi=110, bbox_inches="tight")
            plt.close()
            figt, axt = plt.subplots(len(FIGW), 1, figsize=(13, 2.6 * len(FIGW)), squeeze=False)
            for ri, (nm0, chrom, s0) in enumerate(FIGW):
                g = get_window(chrom, s0)
                if g is None: continue
                soh, atac = g; soh_t = torch.Tensor(soh.T).unsqueeze(0).to(dev)
                with torch.no_grad(): ap = predictor(soh_t).cpu().numpy()[0, 0]
                rb = atac.reshape(N, WIN // N).mean(1); pb = ap.reshape(N, WIN // N).mean(1)
                r = float(np.corrcoef(rb, pb)[0, 1]) if rb.std() > 1e-6 and pb.std() > 1e-6 else float("nan")
                a = axt[ri, 0]; a.plot(rb, color="k", lw=0.7, label="real ATAC")
                a.plot(pb, color="C3", lw=0.7, alpha=0.8, label="pred ATAC")
                a.set_title(f"{nm0} {chrom}:{s0 // 10**6}M  ATAC (1kb-bin)  r={r:.3f}", fontsize=10)
                a.legend(fontsize=8); a.set_xticks([])
            figt.tight_layout()
            for p in (f"{OUTDIR}/figs/d2a_atac_step{step}.png", f"{OUTDIR}/figs/d2a_atac_latest.png"):
                figt.savefig(p, dpi=110, bbox_inches="tight")
            plt.close()
            print(f"[d2a fig step {step}] {OUTDIR}/figs/d2a_step{step}.png", flush=True)
        except Exception as ex:
            import traceback; print(f"[d2a fig step {step}] FAILED: {ex}\n{traceback.format_exc()}", flush=True)

    # ---- HAP1 CTCF-edit vs capture Hi-C val: ALL 13 Sanborn edits across 3 regions ----
    #      seq-only: edit DNA -> predictor ATAC -> frozen general contact ; GT = Sanborn capture Hi-C (O/E 5kb).
    _HIC = os.environ.get("D2A_HAP1_HIC_DIR", "")
    if HAP1_HIC and not _HIC:
        raise ValueError("D2A_HAP1_HIC_DIR is required when D2A_HAP1_HIC_VAL=1")
    _REGIONS = {
        "R1_chr8": dict(CH="chr8", S=132_600_000, MOT={"A": 132_875_646, "B": 133_209_431, "C": 133_449_646},
            EDITS=[("WT", []),
                   ("dA",        [("del", 132_875_643, 132_875_659)]),
                   ("dB/Rev",    [("del", 133_203_018, 133_203_159)]),
                   ("dB/Fwd",    [("del", 133_209_368, 133_209_526)]),
                   ("invB/Fwd",  [("inv", 133_209_368, 133_209_526)]),
                   # Clone005 / Clone006 -- the two R1 double mutants. Table S7 gives them in hg19;
                   # the -1,012,243 offset used here is the one implied EXACTLY by the two R1-B
                   # edits above (134,215,261->133,203,018 and 134,221,611->133,209,368). It is not
                   # genome-wide: R1-A implies -1,012,245, so it is applied only inside R1-B.
                   #   B-reverse motif  hg19 134,215,312-134,215,332 -> 133,203,069-133,203,089
                   #   Clone005 insert  hg19 134,215,324           -> 133,203,081
                   # Table S7 does not say WHICH base was inserted; "A" follows insG's precedent
                   # and has to be stated in the caption.
                   ("invB/Fwd+insB/Rev", [("inv", 133_209_368, 133_209_526),
                                          ("ins", 133_203_081, "A")]),
                   ("invB/Fwd+invB/Rev", [("inv", 133_209_368, 133_209_526),
                                          ("inv", 133_203_069, 133_203_089)])]),
        "R2_chr1": dict(CH="chr1", S=180_400_000, MOT={"E/Fwd": 180_882_097, "E/Rev": 180_863_366},
            EDITS=[("WT", []),
                   ("dE/Fwd",        [("del", 180_882_097, 180_882_117)]),
                   ("dE/Rev",        [("del", 180_863_366, 180_863_386)]),
                   ("invE/Fwd",      [("inv", 180_882_097, 180_882_116)]),
                   ("dE/Fwd+dE/Rev", [("del", 180_882_097, 180_882_117), ("del", 180_863_366, 180_863_386)]),
                   ("E/Fwd->Rev",    [("rep", 180_882_097, 180_882_117, 180_863_366, 180_863_386)])]),
        "R3_chr5": dict(CH="chr5", S=31_350_000, MOT={"G": 31_581_669, "H": 31_825_672},
            EDITS=[("WT", []),
                   ("insG", [("ins", 31_581_681, "A")]),
                   ("dH",   [("del", 31_825_670, 31_825_706)])]),
    }
    # each edit name -> capture Hi-C GT mcool glob(s). A value may be a list; _hap1_gt sums all
    # matches, so a pattern that reaches across genotypes silently pools them.
    #
    # ⛔ invB/Fwd USED TO BE `*inversion_of_region_1_b_forward_ctcf*`. The 4DN filenames truncate
    # to an identical prefix, so that matched FOUR experiment sets, and two of them were the
    # Clone005/Clone006 double mutants -- 966 MB of the 1145 MB behind that column, 84%. Resolved
    # against data.4dnucleome.org, the four are:
    #   4DNESAYH7WXY  B-fwd inversion (biotin-UTP)              <- the real thing
    #   4DNESWFRUPLR  B-fwd inversion (biotin-ATP)              <- the real thing
    #   4DNESWQ4K74H  B-fwd inversion + B-rev INVERSION  = Clone006
    #   4DNESWZZWT68  B-fwd inversion + B-rev INSERTION  = Clone005
    # Named explicitly now. Every other entry was checked the same way and does pool only
    # replicate libraries / size-variant clones of one genotype.
    _GTGLOB = {
        "WT": "capture_wt/*wild_type.mcool",
        "dA": "capture_edits/*deletion_of_region_1_a_forward_ctcf*.mcool",
        "dB/Rev": "capture_edits/*deletion_of_region_1_b_reverse_ctcf*.mcool",
        "dB/Fwd": "capture_edits/*deletion_of_region_1_b_forward_ctcf*.mcool",
        "invB/Fwd": ["capture_edits/4DNESAYH7WXY_*.mcool", "capture_edits/4DNESWFRUPLR_*.mcool"],
        "invB/Fwd+insB/Rev": "capture_edits/4DNESWZZWT68_*.mcool",
        "invB/Fwd+invB/Rev": "capture_edits/4DNESWQ4K74H_*.mcool",
        "dE/Fwd": "capture_edits/*deletion_of_region_2_e_forward_ctcf_s*.mcool",
        "dE/Rev": "capture_edits/*deletion_of_region_2_e_reverse_ctcf_s*.mcool",
        "dE/Fwd+dE/Rev": "capture_edits/*deletion_of_region_2_e_forward_and_re*.mcool",
        "invE/Fwd": "capture_edits/*inversion_of_region_2_e_forward*.mcool",
        "E/Fwd->Rev": "capture_edits/*region_2_e_forward_ctcf_site_replaced*.mcool",
        "insG": "capture_edits/*region_3_ctcf_site_g_insertion*.mcool",
        "dH": "capture_edits/*region_3_h_forward_ctcf_site_deletion*.mcool",
    }
    _HAP1_GTC = {}
    def _hap1_gt(rk, CH, S, E, nm, PF, NB):
        key = (rk, nm)
        if key in _HAP1_GTC: return _HAP1_GTC[key]
        import glob as _glob
        _pat = _GTGLOB.get(nm, [])
        _pat = [_pat] if isinstance(_pat, str) else list(_pat)
        mcs = sorted({p for g in _pat for p in _glob.glob(os.path.join(_HIC, g))})
        if not mcs:
            _HAP1_GTC[key] = np.full((NB, NB), np.nan); return _HAP1_GTC[key]
        def psum(M): return M[:NB*PF, :NB*PF].reshape(NB, PF, NB, PF).sum((1, 3))
        def ice(M, n=50):
            M = M.astype(np.float64).copy(); M[~np.isfinite(M)] = 0; mk = M.sum(1) > 0
            for _ in range(n):
                s = M.sum(1); mm = s[mk].mean() if mk.any() else 1.0
                sm = np.where(mk & (s > 0), s/(mm+1e-12), 1.0); M = M/sm[:, None]/sm[None, :]
            M[~mk, :] = np.nan; M[:, ~mk] = np.nan; return M
        raw = np.zeros((N, N), np.float64)
        for mc in mcs:
            cs = cooler.fileops.list_coolers(mc); r = [x for x in cs if x.endswith("/1000")]
            raw += cooler.Cooler(mc + "::" + (r[0] if r else cs[0])).matrix(balance=False).fetch(f"{CH}:{S}-{E}").astype(np.float64)[:N, :N]
        raw = ice(psum(raw)); n = raw.shape[0]; jj = np.arange(n); ex = np.full(n, np.nan)
        for k in range(n):
            f = np.diagonal(raw, k); f = f[np.isfinite(f) & (f > 0)]; ex[k] = np.nanmean(f) if len(f) else np.nan
        last = ex[0] if np.isfinite(ex[0]) else 1e-6
        for k in range(n): ex[k] = last if not np.isfinite(ex[k]) else ex[k]; last = ex[k]
        Em = ex[np.abs(jj[:, None] - jj[None, :])]
        _HAP1_GTC[key] = np.clip(np.where((Em > 0) & np.isfinite(raw) & (raw > 0), np.log(raw/Em), np.nan), -2, 2)
        return _HAP1_GTC[key]
    _RC2 = str.maketrans("ACGT", "TGCA"); _rc2 = lambda s: s.translate(_RC2)[::-1]
    def dump_hap1_hic(step):
        try:
            PF = 5; NB = N // PF; cm = plt.get_cmap("YlOrRd").copy(); cm.set_bad("white")
            jj = np.arange(NB); offm = np.abs(jj[:, None] - jj[None, :]) > 1
            def pool_mean(M):
                A = M[:NB*PF, :NB*PF].reshape(NB, PF, NB, PF).astype(float)
                import warnings; warnings.filterwarnings("ignore"); return np.nanmean(A, axis=(1, 3))
            def disp(a):
                z = a.astype(float).copy(); z[np.abs(jj[:, None]-jj[None, :]) < 1] = np.nan; f = z[np.isfinite(z)]
                if f.size == 0: return z
                a1, a2 = np.nanpercentile(f, 1), np.nanpercentile(f, 99); return np.clip((z-a1)/(a2-a1+1e-9), 0, 1)
            def pear(a, b):
                m = offm & np.isfinite(a) & np.isfinite(b); return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() >= 10 else np.nan
            def pred_edit(seq):   # edited DNA -> predictor ATAC -> frozen general contact
                soh_t = torch.Tensor(encode(seq[:WIN]).T).unsqueeze(0).to(dev)
                with torch.no_grad():
                    ap = predictor(soh_t); m = gen_contact(soh_t, ap).cpu().numpy()[0]
                return 0.5 * (m + m.T)
            all_r = {}
            dump5k = {}          # 5 kb pred + GT arrays, so the loops can be measured off the numbers
                                 # instead of eyeballed off the PNG (see hap1_loop_npz.py)
            for rk, cfg in _REGIONS.items():
                CH, S = cfg["CH"], cfg["S"]; E = S + WIN; MOT = cfg["MOT"]
                base = fa.fetch(CH, S, E + 512).upper()
                def apply_edits(ops, base=base, S=S):
                    s = list(base)
                    for op in sorted(ops, key=lambda o: -o[1]):     # right-most edit first so coords don't shift
                        k = op[0]
                        if k == "del": del s[op[1]-S:op[2]-S]
                        elif k == "inv": i, j = op[1]-S, op[2]-S; s[i:j] = list(_rc2("".join(s[i:j])))
                        elif k == "ins": s[op[1]-S:op[1]-S] = list(op[2])
                        elif k == "rep": s[op[1]-S:op[2]-S] = list(base[op[3]-S:op[4]-S])
                    return "".join(s)
                edits = cfg["EDITS"]; nc = len(edits)
                tick = [int((p - S)//(1000*PF)) for p in MOT.values()]; tlab = list(MOT.keys())
                fig, ax = plt.subplots(2, nc, figsize=(2.7*nc, 5.8), squeeze=False)
                rs = {}
                _nogt = os.environ.get("D2A_HIC_NOGT") == "1"   # skip pool GT reads (FS-hang workaround): render pred row only
                for j, (nm, ops) in enumerate(edits):
                    praw = pool_mean(pred_edit(apply_edits(ops)))
                    graw = np.full((NB, NB), np.nan) if _nogt else _hap1_gt(rk, CH, S, E, nm, PF, NB)
                    rs[nm] = pear(praw, graw); all_r[f"{rk}:{nm}"] = rs[nm]
                    dump5k[f"{rk}__{nm}__pred"] = praw.astype(np.float32)
                    dump5k[f"{rk}__{nm}__gt"] = graw.astype(np.float32)
                    for r, M in enumerate([disp(praw), disp(graw)]):
                        a = ax[r, j]; a.imshow(M, cmap=cm, vmin=0, vmax=1, origin="lower", interpolation="nearest")
                        if r == 0: a.set_title(f"{nm}\nr={rs[nm]:.3f}", fontsize=10, pad=3)
                        a.set_xticks(tick); a.set_xticklabels(tlab, fontsize=6, rotation=45, ha="right"); a.set_yticks(tick); a.set_yticklabels(tlab, fontsize=6)
                        if j == 0: a.set_ylabel(["pred (DNA->ATAC->contact)", "capture Hi-C"][r], fontsize=9)
                rmean_r = float(np.nanmean([v for v in rs.values() if np.isfinite(v)]))
                fig.suptitle(f"HAP1 {rk} CTCF-edit (seq-only DNA->ATAC->frozen contact) vs capture Hi-C (O/E 5kb) step{step}  mean r={rmean_r:.3f}", fontsize=10, y=0.99)
                fig.tight_layout()
                for p in (f"{OUTDIR}/figs/hap1_hic_{rk}_step{step}.png", f"{OUTDIR}/figs/hap1_hic_{rk}_latest.png"):
                    fig.savefig(p, dpi=120, bbox_inches="tight")
                plt.close()
            dump5k["__meta"] = np.array([str(step)] + [f"{rk}:{cfg['CH']}:{cfg['S']}:{PF*1000}"
                                                       for rk, cfg in _REGIONS.items()])
            np.savez_compressed(f"{OUTDIR}/figs/hap1_hic_maps_latest.npz", **dump5k)

            # ---- whole-window edit-minus-WT, every val. Free: reuses the maps just computed.
            #      The maps are log(O/E), so edit - WT is a log ratio (-0.69 = contact halved).
            #      Scale is diverging, symmetric, and SHARED across all columns of a row -- a
            #      per-panel scale stretches a non-responding edit until it looks like a response.
            def _blur2(a, sig=float(os.environ.get("D2A_DIFF_SMOOTH", "2"))):
                """nan-aware Gaussian blur for the DISPLAYED capture Hi-C; sig<=0 disables"""
                if sig <= 0:
                    return a
                from scipy.ndimage import gaussian_filter
                m = np.isfinite(a).astype(float); v = np.nan_to_num(a)
                num, den = gaussian_filter(v * m, sig), gaussian_filter(m, sig)
                return np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)

            try:
                _EXTRA = {"R2_chr1": {"F": 181_150_000}}   # E/Fwd's loop partner, found in the WT GT
                cols, dPs, dGs = [], [], []
                for rk, cfg in _REGIONS.items():
                    pw, gw = dump5k[f"{rk}__WT__pred"], dump5k[f"{rk}__WT__gt"]
                    for nm, _ in cfg["EDITS"][1:]:
                        dPs.append(dump5k[f"{rk}__{nm}__pred"].astype(float) - pw.astype(float))
                        # Smooth the capture Hi-C BEFORE differencing. At 5 kb the per-pixel
                        # difference of two sparse matrices is mostly shot noise, so the raw
                        # bottom row reads as salt-and-pepper even where the response is real.
                        # D2A_FIG_SMOOTH already does this for dump_fig's GT panel; this block
                        # was missed, which is why the diff figure's lower row stayed grainy.
                        # Display only -- dump5k still holds the unsmoothed arrays and every
                        # reported r is computed on those.
                        dGs.append(_blur2(dump5k[f"{rk}__{nm}__gt"].astype(float))
                                   - _blur2(gw.astype(float)))
                        mot = dict(cfg["MOT"]); mot.update(_EXTRA.get(rk, {}))
                        cols.append((rk, nm, cfg["S"], mot))
                _fin = lambda L: np.concatenate([x[np.isfinite(x)].ravel() for x in L]) if L else np.array([0.0])
                vp = float(np.nanpercentile(np.abs(_fin(dPs)), 99)) or 1e-3
                vg = float(np.nanpercentile(np.abs(_fin(dGs)), 99)) or 1e-3
                dcm = plt.get_cmap("RdBu_r").copy(); dcm.set_bad("#dddddd")
                nn = len(cols)
                fg, ax2 = plt.subplots(2, nn, figsize=(1.95*nn, 5.4), squeeze=False)
                for k, (rk, nm, S0, mot) in enumerate(cols):
                    tk = [(v - S0)//(1000*PF) for v in mot.values()]; tl = list(mot.keys())
                    for r2, (Md, v, rl) in enumerate([(dPs[k], vp, "prediction"), (dGs[k], vg, "capture Hi-C")]):
                        A = ax2[r2, k]
                        imh = A.imshow(Md, cmap=dcm, vmin=-v, vmax=v, origin="lower", interpolation="nearest")
                        A.set_xticks(tk); A.set_xticklabels(tl, fontsize=6, rotation=45, ha="right")
                        A.set_yticks(tk); A.set_yticklabels(tl, fontsize=6); A.tick_params(length=2)
                        if k == 0: A.set_ylabel(f"{rl}\n(edit - WT)", fontsize=9)
                        if r2 == 0: A.set_title(f"{rk.split('_')[0]}\n{nm}", fontsize=8, pad=4)
                        else: fg.colorbar(imh, ax=A, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
                fg.suptitle(f"HAP1 CTCF edits, whole 1 Mb window at {PF} kb: edit - WT in log(O/E)  step{step}\n"
                            f"blue = contact lost, red = gained;  shared scale +/-{vp:.2f} (pred) / +/-{vg:.2f} (Hi-C)",
                            fontsize=11, y=1.07)
                fg.tight_layout()
                for p in (f"{OUTDIR}/figs/hap1_diff_step{step}.png", f"{OUTDIR}/figs/hap1_diff_latest.png"):
                    fg.savefig(p, dpi=150, bbox_inches="tight")
                plt.close(fg)
                amp = {f"{rk}:{nm}": float(np.nanmean(np.abs(dPs[k]))) for k, (rk, nm, _, _) in enumerate(cols)}
                print(f"[hap1_diff step {step}] pred |edit-WT| mean: "
                      + " ".join(f"{k}={v:.3f}" for k, v in amp.items()), flush=True)
            except Exception as ex:
                print(f"[hap1_diff step {step}] FAILED: {ex}", flush=True)
            rmean = float(np.nanmean([v for v in all_r.values() if np.isfinite(v)]))
            print(f"[hap1_hic step {step}] mean r={rmean:.3f} ({len(all_r)} edits)  "
                  + " ".join(f"{k}={v:.2f}" for k, v in all_r.items()) + f" -> {OUTDIR}/figs/hap1_hic_R*_latest.png", flush=True)
            return rmean
        except Exception as ex:
            import traceback; print(f"[hap1_hic step {step}] FAILED: {ex}\n{traceback.format_exc()}", flush=True); return float("nan")

    def dump_hap1_pub(step):
        """PUBLICATION figure: only WT + delta-CTCF-A. Clean 2x2 (prediction / capture Hi-C),
        NaN-aware smoothed capture map (fills capture dropouts, removes 5 kb speckle), anchor
        ticks OUTSIDE the maps (no internal lines), shared colorbar, square panels."""
        from scipy.ndimage import gaussian_filter
        from matplotlib import gridspec
        MOT = {"A/Fwd": 132875646, "B/Rev": 133203068, "B/Fwd": 133209431, "C/Rev": 133449646}; CH = "chr8"; MW = 19
        allp = list(MOT.values()); S = (min(allp)+max(allp))//2 - WIN//2; E = S + WIN; PF = 5; NB = N//PF
        base = fa.fetch(CH, S, E+256).upper(); _RC = str.maketrans("ACGT", "TGCA"); rcf = lambda s: s.translate(_RC)[::-1]
        def apply_edits(ops):
            s = list(base)
            for kind, name in sorted(ops, key=lambda o: -MOT[o[1]]):
                a = MOT[name] - S - MW//2
                if kind == "del": del s[a:a+MW]
                elif kind == "inv": s[a:a+MW] = list(rcf("".join(s[a:a+MW])))
            return "".join(s[:WIN])
        def pred_edit(seq):
            soh_t = torch.Tensor(encode(seq).T).unsqueeze(0).to(dev)
            with torch.no_grad(): return gen_contact(soh_t, predictor(soh_t)).cpu().numpy()[0]
        GT = _hap1_gt(S, E, CH, PF, NB)
        jj = np.arange(NB); offm = np.abs(jj[:, None] - jj[None, :]) > 1
        diagmask = np.abs(jj[:, None] - jj[None, :]) < 1
        def pool_mean(M):
            A = M[:NB*PF, :NB*PF].reshape(NB, PF, NB, PF).astype(float)
            import warnings; warnings.filterwarnings("ignore"); return np.nanmean(A, axis=(1, 3))
        def nansmooth(a, sig):
            m = np.isfinite(a).astype(float); a0 = np.where(m > 0, a, 0.0)
            num = gaussian_filter(a0, sig); den = gaussian_filter(m, sig)
            out = num / np.maximum(den, 1e-6); out[den < 0.05] = np.nan; return out
        def disp(a):
            z = a.astype(float).copy(); z[diagmask] = np.nan; f = z[np.isfinite(z)]
            if f.size == 0: return z
            a1, a2 = np.nanpercentile(f, 2), np.nanpercentile(f, 98); return np.clip((z-a1)/(a2-a1+1e-9), 0, 1)
        def pear(a, b):
            m = offm & np.isfinite(a) & np.isfinite(b); return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() >= 10 else np.nan
        tick = {k: int((MOT[{"A": "A/Fwd", "B": "B/Fwd", "C": "C/Rev"}[k]]-S)//(1000*PF)) for k in "ABC"}
        cm = plt.get_cmap("YlOrRd").copy(); cm.set_bad("0.92")
        EDITS = [("WT", [], "wild type"), (r"$\Delta$CTCF-A", [("del", "A/Fwd")], "CTCF-A deleted")]
        fig = plt.figure(figsize=(7.4, 7.7))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.12, hspace=0.12,
                               left=0.10, right=0.92, top=0.90, bottom=0.07)
        rows = ["Prediction (sequence-only)", "Capture Hi-C"]
        im0 = None; rs = {}
        for col, (nm, ops, sub) in enumerate(EDITS):
            praw = pool_mean(pred_edit(apply_edits(ops))); graw = GT.get(nm if nm == "WT" else "dA/Fwd", np.full((NB, NB), np.nan))
            rs[nm] = pear(praw, graw)
            for row, raw in enumerate([praw, graw]):
                ax = fig.add_subplot(gs[row, col])
                M = disp(nansmooth(raw, 0.9 if row == 1 else 0.6))
                im0 = ax.imshow(M, cmap=cm, vmin=0, vmax=1, origin="lower", interpolation="bilinear", aspect="equal")
                ax.set_xticks([tick[k] for k in "ABC"]); ax.set_yticks([tick[k] for k in "ABC"])
                ax.tick_params(length=3, labelsize=8)
                if row == 1: ax.set_xticklabels(list("ABC")); ax.set_xlabel("CTCF anchor", fontsize=9)
                else: ax.set_xticklabels([])
                if col == 0: ax.set_yticklabels(list("ABC")); ax.set_ylabel(rows[row], fontsize=10)
                else: ax.set_yticklabels([])
                if row == 0: ax.set_title(f"{nm}\n({sub})", fontsize=11, pad=5)
        cax = fig.add_subplot(gs[:, 2]); cb = fig.colorbar(im0, cax=cax)
        cb.set_label("contact O/E (per-panel scaled)", fontsize=9); cb.ax.tick_params(labelsize=8)
        fig.suptitle("HAP1 engineered CTCF edit: sequence-only prediction vs capture Hi-C",
                     fontsize=12.5, y=0.965)
        out = os.path.abspath(f"{OUTDIR}/figs/hap1_pub_step{step}.png")
        fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
        print(f"[hap1_pub step {step}] WT r={rs['WT']:.3f}  dA r={rs[list(rs)[1]]:.3f} -> {out}", flush=True)
        return out

    if os.environ.get("D2A_PUB_RENDER"):
        predictor.eval(); dump_hap1_pub(int(os.environ.get("D2A_PUB_STEP", str(start_step)))); return
    if os.environ.get("D2A_HIC_RENDER"):   # render-only: all 13 CTCF edits (3 regions) from the resumed ckpt, no training
        predictor.eval(); dump_hap1_hic(int(os.environ.get("D2A_HIC_STEP", str(start_step)))); return

    # holdout 扫描的窗数。旧行为 = 整条 chr9+chr10 共 271 窗, 每窗 3 次前向 + 一次 cooler fetch,
    # 实测单窗冷读 12.5s / GPU 0% -> 每次 val 20+ 分钟, 占墙钟 ~27%。
    #   -1 = 全部(旧行为, 默认不变)   0 = 完全跳过   N = 沿 holdout 均匀取 N 窗
    VAL_NWIN = int(os.environ.get("D2A_VAL_NWIN", "-1"))
    # mESC only: render the Koska Sox2 promoter-insertion panel each val (see koska_val.py).
    KOSKA_FIG = os.environ.get("D2A_KOSKA_FIG", "") == "1"
    HARVEY_VAL = os.environ.get("HARVEY_VAL", "") == "1"
    HARVEY_EVERY = int(os.environ.get("HARVEY_VAL_EVERY", "2000"))
    if HARVEY_VAL: print(f"[d2a] HARVEY_VAL: Harvey chr2 clones every {HARVEY_EVERY} steps", flush=True)
    if KOSKA_FIG: print("[d2a] KOSKA_FIG: Sox2 insertion panel every val", flush=True)

    def validate(step):
        predictor.eval(); rs_st = []; rs_sg = []; rs_tg = []; rs_a = []
        omask = np.abs(ii[:, None] - ii[None, :]) > DIAG
        wins = [(c, s) for c in HOLDOUT if c in clen_h
                for s in range(0, clen_h[c] - WIN, WIN)]
        if VAL_NWIN == 0:
            wins = []
        elif 0 < VAL_NWIN < len(wins):
            _idx = sorted(set(np.linspace(0, len(wins) - 1, VAL_NWIN).round().astype(int).tolist()))
            wins = [wins[i] for i in _idx]
        with torch.no_grad():
            for chrom, start in wins:
                    g = get_window(chrom, start)
                    if g is None: continue
                    soh, atac = g; soh_t = torch.Tensor(soh.T).unsqueeze(0).to(dev); ra = torch.Tensor(atac).view(1, 1, -1).to(dev)
                    ap = predictor(soh_t)
                    t = gen_contact(soh_t, ra).cpu().numpy()[0]
                    s = gen_contact(soh_t, ap).cpu().numpy()[0]
                    m = omask & np.isfinite(s) & np.isfinite(t); rs_st.append(np.corrcoef(s[m], t[m])[0, 1])
                    rb = atac.reshape(N, WIN // N).mean(1); pb = ap.cpu().numpy()[0, 0].reshape(N, WIN // N).mean(1)
                    if rb.std() > 1e-6 and pb.std() > 1e-6: rs_a.append(np.corrcoef(rb, pb)[0, 1])
                    if _cool is not None:
                        try:
                            raw = fetch_raw(chrom, start)
                            # ⛔ 口径: 模型输出是 log(O/E), 所以 GT 也必须先除掉每条对角线的期望,
                            #    再取对数。这里以前直接用 log(raw), 距离衰减主导 log(raw), 相关系数被
                            #    它拖低, 而且对"把衰减吸进输出"的模型偏高 —— 两个方向的偏都有。
                            #    gt_logoe_from_raw 就是训练侧 t2d 用的同一个函数, 现在验证与训练同空间。
                            lgoe = gt_logoe_from_raw(raw)
                            mg = omask & (raw > 0) & np.isfinite(s) & np.isfinite(t) & np.isfinite(lgoe)
                            if mg.sum() > 100:
                                rs_sg.append(np.corrcoef(s[mg], lgoe[mg])[0, 1])
                                rs_tg.append(np.corrcoef(t[mg], lgoe[mg])[0, 1])
                        except Exception: pass
        f = lambda L: float(np.nanmean(L)) if L else float("nan")
        r_st, r_sg, r_tg, r_a = f(rs_st), f(rs_sg), f(rs_tg), f(rs_a)
        # ---- Harvey chr2 edited clones ----
        #   每个克隆有自己编辑过的 chr2 和在同一编辑基因组上测的 mcool, 所以这是纯序列 -> 接触的
        #   检验, 和 Koska 插入是同一类, 只是十个克隆、整条染色体。d2a 是"预测器 + 冻结 trunk"
        #   两段, 而 run_clone 要单个模块, 所以在这里现包一个; 它不需要 CLONE_ATAC_ZERO —— 那是
        #   给真正的 seq-only 模型补第五通道用的, 我们的第五通道由预测器产生。
        #   ⛔ 用 try 包住: 渲染失败不能把训练拖死, 这是 train_atac.py 里同一个钩子的既有做法。
        if HARVEY_VAL and step % HARVEY_EVERY == 0:
            try:
                import infer_per_cell as _ipc

                class _Seq2Contact(torch.nn.Module):
                    def __init__(self, pr, gn):
                        super().__init__(); self.pr = pr; self.gn = gn

                    def forward(self, x):
                        a = self.pr(x)
                        return self.gn(torch.cat([x, a.to(x.dtype)], 1))   # 不取 [0,0]: run_clone 自己索引

                os.environ.pop("CLONE_ATAC_ZERO", None)
                _hp = f"{OUTDIR}/figs/harvey_step{step}.png"
                os.environ["INFER_OUT"] = _hp
                _ipc.run_clone(_Seq2Contact(predictor, _GENS[0]).to(dev).eval(), dev, _hp)
                print(f"[harvey] step {step} -> {_hp}", flush=True)
            except Exception as _ex:
                import traceback
                print(f"[harvey] FAILED at step {step}: {_ex}\n{traceback.format_exc()}", flush=True)
        r_hic = dump_hap1_hic(step) if HAP1_HIC else None   # edit-vs-capture-HiC mean Pearson (also saves the val figure)
        if r_hic is not None: r_sg = r_hic                  # curve r_pred_gt = HAP1 capture-HiC mean Pearson
        print(f"[d2a val step {step}] r(predATAC_contact, realATAC_contact)={r_st:.4f}  "
              f"r(predATAC,{'HiC' if HAP1_HIC else 'GT'})={r_sg:.4f}  r(realATAC,GT)={r_tg:.4f}  r_atac1d={r_a:.4f}", flush=True)
        try:
            cf = f"{OUTDIR}/curve.tsv"; new = not os.path.exists(cf)
            with open(cf, "a") as h:
                if new: h.write("step\tr_pred_vs_real\tr_pred_gt\tr_real_gt\tr_atac1d\n")
                h.write(f"{step}\t{r_st:.4f}\t{r_sg:.4f}\t{r_tg:.4f}\t{r_a:.4f}\n")
            d = np.atleast_1d(np.genfromtxt(cf, delimiter="\t", names=True, dtype=None))
            ff, ax = plt.subplots(figsize=(7.5, 4.5)); st = d["step"].astype(float)
            ax.plot(st, d["r_pred_vs_real"].astype(float), "o-", label="pred-ATAC vs real-ATAC contact")
            ax.plot(st, d["r_pred_gt"].astype(float), "s-", label="pred-ATAC vs GT (capture)")
            ax.plot(st, d["r_real_gt"].astype(float), "^--", color="gray", label="real-ATAC vs GT (ceiling)")
            ax.plot(st, d["r_atac1d"].astype(float), "d-", color="C2", label="ATAC 1D track")
            ax.set_xlabel("step"); ax.set_ylabel("Pearson r"); ax.set_title(os.path.basename(OUTDIR.rstrip("/")))
            ax.grid(alpha=0.3); ax.legend(fontsize=8); ff.tight_layout(); ff.savefig(f"{OUTDIR}/curve.png", dpi=120); plt.close(ff)
        except Exception as e:
            print(f"[d2a curve] skip: {e}", flush=True)
        if not HAP1_HIC: dump_fig(step)   # HAP1: edit-vs-HiC figure already saved by dump_hap1_hic
        if KOSKA_FIG:
            # Sox2 promoter-insertion panel, 7 conditions each against its own published RCMC.
            # Costs 8 extra 1Mb forwards -- nothing next to the holdout scan above. Never let a
            # figure kill a training run, so this is the one place a broad except is right.
            try:
                import koska_val
                koska_val.render(predictor, gen_contact, dev, f"{OUTDIR}/figs/koska_step{step}.png", step)
            except Exception as e:
                print(f"[koska_val] skipped: {type(e).__name__}: {e}", flush=True)
        predictor.train()

    if not SKIP_INITIAL_VAL and start_step == 0:
        validate(0)
    TOPK = float(os.environ.get("D2A_TOPK", "0.05"))    # predictor only learns the top-TOPK ATAC peaks; rest = window mean
    ATAC_W = float(os.environ.get("D2A_ATAC_W", "1.0")) # weight on the direct ATAC loss
    ATAC_ONLY = os.environ.get("D2A_ATAC_ONLY", "") == "1"  # 只用 ATAC loss 训 predictor(不碰 contact); general 仅在 validate 评 CTCF-edit
    if ATAC_ONLY: print("[d2a] ATAC_ONLY: loss = ATAC only (no contact MSE); general used only in validate", flush=True)
    ATAC_LOSS = os.environ.get("D2A_ATAC_LOSS", "ag")   # "ag" = AlphaGenome multinomial+poisson (peak-preserving) | "mse"
    MRES = int(os.environ.get("D2A_MRES", "125"))       # multinomial segment size (bp); must divide WIN (1e6/125=8000)
    POSW = float(os.environ.get("D2A_POSW", "1.0"))     # weight of the positional (multinomial) term vs the total (poisson) term
    print(f"[d2a] TARGET = top{TOPK:.0%} real ATAC peaks + rest=window-mean; ATAC_LOSS={ATAC_LOSS} (mres={MRES} posw={POSW}) atac_w={ATAC_W}", flush=True)
    GRADNORM = os.environ.get("D2A_GRADNORM", "") == "1"   # adaptively set ATAC_W so |w*g_atac| ~= TARGET*|g_contact| on the predictor (EMA-smoothed; tracks the per-batch noise instead of a hand-picked weight)
    GN_EVERY = int(os.environ.get("D2A_GRADNORM_EVERY", "25")); GN_EMA = float(os.environ.get("D2A_GRADNORM_EMA", "0.95")); GN_TARGET = float(os.environ.get("D2A_GRADNORM_TARGET", "1.0"))
    def _pgnorm():
        sq = [(p.grad.detach().float() ** 2).sum() for p in predictor.parameters() if p.grad is not None]
        return float(torch.sqrt(torch.stack(sq).sum())) if sq else 0.0
    if GRADNORM: print(f"[d2a] GRADNORM on: every {GN_EVERY} steps, ema={GN_EMA}, target |w*g_atac|/|g_contact|={GN_TARGET}; ATAC_W starts {ATAC_W}", flush=True)
    t0 = time.time()
    if os.environ.get("D2A_GRADPROBE"):   # measure gradient-norm balance of the two loss terms on the predictor (not loss magnitude)
        NB = int(os.environ.get("D2A_GRADPROBE_N", "8"))
        def _pgnorm():
            sq = [(p.grad.detach().float() ** 2).sum() for p in predictor.parameters() if p.grad is not None]
            return float(torch.sqrt(torch.stack(sq).sum())) if sq else 0.0
        gc_l, ga_l, lc_l, la_l = [], [], [], []
        for bi in range(NB):
            sohs = []; atacs = []
            while len(sohs) < BATCH:
                ci = rng.choice(len(chroms), p=w); chrom = chroms[ci]; start = int(rng.integers(0, clen[chrom] - WIN))
                g = get_window(chrom, start)
                if g is None: continue
                sohs.append(g[0]); atacs.append(g[1])
            soh_t = torch.Tensor(np.stack([s.T for s in sohs])).to(dev)
            real_atac = torch.Tensor(np.stack(atacs)).unsqueeze(1).to(dev)
            Bn = real_atac.shape[0]; _flat = real_atac.view(Bn, -1); _mean = _flat.mean(1, keepdim=True)
            _thr = torch.quantile(_flat, 1.0 - TOPK, dim=1, keepdim=True)
            atac_tgt = torch.where(_flat >= _thr, _flat, _mean).view_as(real_atac)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                t2d = gen_contact(soh_t, atac_tgt).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                atac_pred = predictor(soh_t); s2d = gen_contact(soh_t, atac_pred)
            s2d = s2d.float()
            l_contact = ((s2d - t2d) ** 2)[:, off].mean()
            if ATAC_LOSS == "ag": l_atac = _ag_multinomial_loss(atac_tgt.transpose(1, 2), atac_pred.float().transpose(1, 2), MRES, POSW)
            else: l_atac = ((atac_pred.float() - atac_tgt) ** 2).mean()
            opt.zero_grad(set_to_none=True); l_contact.backward(retain_graph=True); gc = _pgnorm()
            opt.zero_grad(set_to_none=True); (ATAC_W * l_atac).backward(); ga = _pgnorm()
            gc_l.append(gc); ga_l.append(ga); lc_l.append(float(l_contact)); la_l.append(float(ATAC_W * l_atac))
            print(f"[gradprobe {bi+1}/{NB}] |g_contact|={gc:.4g}  |w*g_atac|={ga:.4g}  ratio(atac/contact)={ga/(gc+1e-12):.3f}   (l_contact={float(l_contact):.4f} w*l_atac={float(ATAC_W*l_atac):.4f})", flush=True)
        import numpy as _np
        print(f"[gradprobe MEAN over {NB}] |g_contact|={_np.mean(gc_l):.4g}  |w*g_atac|={_np.mean(ga_l):.4g}  "
              f"ratio(atac/contact)={_np.mean(ga_l)/(_np.mean(gc_l)+1e-12):.3f}  || loss-ratio(atac/contact)={_np.mean(la_l)/(_np.mean(lc_l)+1e-12):.3f}", flush=True)
        print("[gradprobe] DONE", flush=True); return
    GT_TARGET = os.environ.get("D2A_GT_TARGET", "") == "1"   # POC: target = REAL GT Micro-C log O/E (不是 teacher). 测架构 vs 蒸馏输出
    if GT_TARGET:
        assert _cool is not None, "D2A_GT_TARGET 需要 COOL (real GT Micro-C)"
        print("[d2a] GT_TARGET: contact loss = MSE(student, REAL GT Micro-C logO/E) —— teacher 换成真 GT", flush=True)
    # D2A_VAL_ONLY=1 -> run validate() ONCE at the resumed step and stop, without touching the
    # optimiser or writing a checkpoint. Added to re-render a figure from a FROZEN predictor ckpt
    # (D2A_RESUME_CKPT=.../predictor.stepNNNN.ckpt.pt): the per-step koska/hap1 npz caches are
    # `latest` snapshots that every val overwrites, so an old step's arrays are otherwise gone.
    # Strictly additive and inert unless the env var is set -- this file is imported by running
    # training jobs, and a requeue must not behave differently because of this block.
    if os.environ.get("D2A_VAL_ONLY") == "1":
        print(f"[d2a] VAL_ONLY: validate({start_step}) from {LATEST}, then exit — no training",
              flush=True)
        validate(start_step)
        print("[d2a] VAL_ONLY done", flush=True)
        return

    for step in range(start_step + 1, MAX_STEPS + 1):
        opt.zero_grad()
        sohs = []; atacs = []; coords = []
        while len(sohs) < BATCH:
            ci = rng.choice(len(chroms), p=w); chrom = chroms[ci]; start = int(rng.integers(0, clen[chrom] - WIN))
            if FRAGBP_MIN > 0 and window_fragbp(chrom, start) < FRAGBP_MIN: continue   # HARD frag/bp gate (cheap, before get_window)
            g = get_window(chrom, start)
            if g is None: continue
            if GT_TARGET:                                    # 需要真 GT: 该窗必须在 cool 里取得到
                try: _rawgt = fetch_raw(chrom, start)
                except Exception: continue
                coords.append(gt_logoe_from_raw(_rawgt))
            sohs.append(g[0]); atacs.append(g[1])
        soh_t = torch.Tensor(np.stack([s.T for s in sohs])).to(dev)        # (B,4,L)
        real_atac = torch.Tensor(np.stack(atacs)).unsqueeze(1).to(dev)     # (B,1,L)
        # sparsified ATAC target: keep only the top-TOPK fraction of peaks per window, set the rest to the window mean
        Bn = real_atac.shape[0]; _flat = real_atac.view(Bn, -1)
        _mean = _flat.mean(1, keepdim=True)                                # (B,1) window mean
        _thr = torch.quantile(_flat, 1.0 - TOPK, dim=1, keepdim=True)      # (B,1) top-TOPK threshold
        atac_tgt = torch.where(_flat >= _thr, _flat, _mean).view_as(real_atac)   # (B,1,L)
        gi = int(rng.integers(len(_GENS)))          # 本步用哪个 general —— teacher/student 共用这一个
        if ATAC_ONLY:                                # 纯 DNA->ATAC 预测器: 训练不碰 general, loss 只有 ATAC 项
            with torch.autocast("cuda", dtype=torch.bfloat16):
                atac_pred = predictor(soh_t)                             # (B,1,L)
            l_contact = 0.0                                             # 无 contact 项; general 只在 validate 用
        else:
            if GT_TARGET:                                               # target = REAL GT Micro-C logO/E (POC)
                t2d = torch.tensor(np.stack(coords), dtype=torch.float32, device=dev)   # (B,N,N)
                gtm = torch.isfinite(t2d); t2d = torch.nan_to_num(t2d)
            else:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    t2d = gen_contact(soh_t, real_atac, gi)             # teacher = contact from FULL real ATAC
                t2d = t2d.float(); gtm = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                atac_pred = predictor(soh_t)                             # (B,1,L)
                s2d = gen_contact(soh_t, atac_pred, gi)                  # ⛔ student 用 predictor 的 ATAC
            s2d = s2d.float()
            if GT_TARGET:
                m = gtm & off.unsqueeze(0)                              # off-diag ∩ GT 有限值
                l_contact = _d2a_hier_kl(s2d, t2d, m) if D2A_HIER_KL else ((s2d - t2d) ** 2)[m].mean()
            else:
                l_contact = (_d2a_hier_kl(s2d, t2d, off.unsqueeze(0)) if D2A_HIER_KL
                             else ((s2d - t2d) ** 2)[:, off].mean())
        if ATAC_LOSS == "ag":                                            # ATAC head still only LEARNS the top-TOPK peaks (atac_tgt); teacher is full-ATAC above
            l_atac = _ag_multinomial_loss(atac_tgt.transpose(1, 2), atac_pred.float().transpose(1, 2), MRES, POSW)
        else:
            l_atac = ((atac_pred.float() - atac_tgt) ** 2).mean()        # direct MSE to the top-TOPK sparsified target
        if GRADNORM and step % GN_EVERY == 0:                            # EMA-update ATAC_W toward gradient balance on the predictor
            opt.zero_grad(set_to_none=True); l_contact.backward(retain_graph=True); _gc = _pgnorm()
            opt.zero_grad(set_to_none=True); l_atac.backward(retain_graph=True); _ga = _pgnorm()
            if _ga > 1e-9 and _gc > 1e-9:
                _wt = GN_TARGET * _gc / _ga
                ATAC_W = GN_EMA * ATAC_W + (1 - GN_EMA) * _wt
            opt.zero_grad(set_to_none=True)
            print(f"[gradnorm step {step}] |g_contact|={_gc:.4g} |g_atac|={_ga:.4g} -> ATAC_W={ATAC_W:.3f}", flush=True)
        loss = l_contact + ATAC_W * l_atac
        loss.backward()
        if WARMUP > 0 and step <= WARMUP:
            for g in opt.param_groups: g["lr"] = LR * step / WARMUP
        opt.step()
        if step % 100 == 0 or (GT_TARGET and step % 20 == 0):
            print(f"[d2a step {step}] l_contact={float(l_contact):.4f} l_atac={float(l_atac):.4f} (top{TOPK:.0%} w={ATAC_W}) t={time.time()-t0:.0f}s", flush=True)
        if step % SAVE_EVERY == 0 or step == MAX_STEPS:
            torch.save({"model": predictor.state_dict(), "opt": opt.state_dict(), "step": step}, LATEST)
        if CKPT_EVERY and (step % CKPT_EVERY == 0 or step == MAX_STEPS):
            torch.save({"model": predictor.state_dict(), "step": step}, f"{OUTDIR}/models/predictor.step{step}.ckpt.pt")
        if step % VAL_EVERY == 0 or step == MAX_STEPS:
            validate(step)
    print("DONE", flush=True)


def verify_topk(net):
    """VERIFY hypothesis: only the top-f fraction of ATAC peaks drive the frozen general's contact;
    the rest can be set to the window AVERAGE with ~no change in output.

      ref   = general(DNA, full real ATAC)
      mod_f = general(DNA, [keep top-f ATAC positions at real value, set the rest to the window mean])

    Report corr(mod_f, ref) over off-diag, two ways: raw, and per-distance-detrended 'detail'
    (subtract each map's own mean-per-|i-j| -> isolates loop/domain structure beyond the trivial decay,
    so it actually tests whether the FINE structure is reproduced). Sweep f from 0 (all average / flat)
    to 1 (full). Saturation by f~0.2 confirms 'only top ~20% peaks participate'. No training; general FROZEN.
    """
    dev = "cuda"; WIN = 1_000_000; N = WIN // 1000; DIAG = 2
    CELL = os.environ.get("D2A_CELL", "GM12878")
    GENOME_FA = os.environ["D2A_GENOME_FA"]
    ATAC_DIR = os.environ["D2A_ATAC_DIR"]
    COOL = os.environ.get("D2A_COOL", "")
    OUTDIR = os.environ.get("OUTDIR", f"./runs/distillation_check_{CELL}")
    os.makedirs(OUTDIR, exist_ok=True)
    CHROM = os.environ.get("D2A_VERIFY_CHROM", "chr8")
    NWIN = int(os.environ.get("D2A_VERIFY_NWIN", "24"))
    FIG_S0 = int(os.environ.get("D2A_VERIFY_FIG_S0", "124988000"))   # MYC window for the visual panel (GT covered)
    fracs = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 1.0]
    rng = np.random.default_rng(0)

    fa = pysam.FastaFile(GENOME_FA)
    _Z = {}
    def _zatac(c):
        if c not in _Z:
            zp = os.path.join(ATAC_DIR, f"{c}.atac.log1p.az")
            _Z[c] = ZAtac(zp) if os.path.exists(zp) else None
        return _Z[c]
    def load_atac(c, s, e):
        z = _zatac(c)
        if z is not None: return z.read(s, e).astype(np.float32)
        m = json.load(open(os.path.join(ATAC_DIR, "manifest.json")))
        L = m[c]["length"] if isinstance(m.get(c), dict) else m["chroms"][c]
        mm = np.memmap(os.path.join(ATAC_DIR, f"{c}.atac.log1p.f16.memmap"), dtype=np.float16, mode="r", shape=(L,))
        return np.asarray(mm[s:e]).astype(np.float32)

    gen = net.module if hasattr(net, "module") else net
    gen.eval()
    for p in gen.parameters():
        p.requires_grad_(False)

    def gen_contact(soh, atac):
        x = torch.cat([soh, atac], 1)
        o = gen(x); m = (o[0] if isinstance(o, tuple) else o)[:, 0]
        return 0.5 * (m + m.transpose(-1, -2))

    ii = np.arange(N); dist = np.abs(ii[:, None] - ii[None, :]); off = dist > DIAG
    _cnt = np.bincount(dist.ravel(), minlength=N).astype(np.float64)
    def detrend(mp):
        s = np.bincount(dist.ravel(), weights=mp.ravel().astype(np.float64), minlength=N)
        mu = s / np.maximum(_cnt, 1.0)
        return mp - mu[dist]

    def amod_topk(atac, f, mean):
        if f <= 0: return np.full_like(atac, mean)
        if f >= 1: return atac
        thr = np.quantile(atac, 1.0 - f)
        return np.where(atac >= thr, atac, mean).astype(np.float32)

    # sample valid windows on CHROM
    L = fa.get_reference_length(CHROM); starts = []; tries = 0
    while len(starts) < NWIN and tries < NWIN * 30:
        tries += 1; st = int(rng.integers(0, L - WIN))
        if fa.fetch(CHROM, st, st + WIN).upper().count("N") > WIN * 0.5: continue
        starts.append(st)
    print(f"[verify_topk] CELL={CELL} general=FROZEN encattn; {len(starts)} {CHROM} windows; fracs={fracs}", flush=True)

    res_raw = {f: [] for f in fracs}; res_det = {f: [] for f in fracs}
    for wi, st in enumerate(starts):
        seq = fa.fetch(CHROM, st, st + WIN).upper(); soh = encode(seq)
        atac = log1p_p99(load_atac(CHROM, st, st + WIN)[:WIN]); mean = float(atac.mean())
        soh_t = torch.Tensor(soh.T).unsqueeze(0).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ref = gen_contact(soh_t, torch.Tensor(atac).view(1, 1, -1).to(dev)).float().cpu().numpy()[0]
        refd = detrend(ref)
        for f in fracs:
            am = amod_topk(atac, f, mean)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                mod = gen_contact(soh_t, torch.Tensor(am).view(1, 1, -1).to(dev)).float().cpu().numpy()[0]
            m = off & np.isfinite(ref) & np.isfinite(mod)
            res_raw[f].append(float(np.corrcoef(ref[m], mod[m])[0, 1]))
            md = detrend(mod)
            res_det[f].append(float(np.corrcoef(refd[m], md[m])[0, 1]))
        if (wi + 1) % 5 == 0: print(f"[verify_topk] {wi+1}/{len(starts)} windows done", flush=True)

    # report table + curve
    print("[verify_topk] === corr(mod_f, ref) vs fraction of top ATAC kept (rest=window mean) ===", flush=True)
    print("  frac   r_raw(off-diag)   r_detail(distance-detrended)", flush=True)
    tsv = f"{OUTDIR}/verify_topk.tsv"
    with open(tsv, "w") as h:
        h.write("frac\tr_raw\tr_detail\n")
        for f in fracs:
            rr = float(np.nanmean(res_raw[f])); rd = float(np.nanmean(res_det[f]))
            print(f"  {f:>5.2f}      {rr:.4f}            {rd:.4f}", flush=True)
            h.write(f"{f}\t{rr:.4f}\t{rd:.4f}\n")
    try:
        xs = fracs
        yr = [float(np.nanmean(res_raw[f])) for f in fracs]; yd = [float(np.nanmean(res_det[f])) for f in fracs]
        ff, ax = plt.subplots(figsize=(7, 4.6))
        ax.plot(xs, yr, "o-", label="raw off-diag")
        ax.plot(xs, yd, "s-", label="detail (distance-detrended)")
        ax.axvline(0.20, color="gray", ls="--", lw=1, label="f=0.20")
        ax.set_xlabel("fraction of top ATAC positions kept (rest = window mean)")
        ax.set_ylabel(f"corr(mod, full-ATAC ref)  [{CHROM}, n={len(starts)}]")
        ax.set_title(f"{CELL}: do only top ATAC peaks drive the frozen general's contact?")
        ax.grid(alpha=0.3); ax.legend(fontsize=9); ff.tight_layout()
        ff.savefig(f"{OUTDIR}/verify_topk_curve.png", dpi=130); plt.close(ff)
    except Exception as e:
        print(f"[verify_topk] curve skip: {e}", flush=True)

    # visual panel at the MYC window: contact for f in {0, 0.05, 0.20, 1.0} (+ GT if cool covers it)
    try:
        cm = plt.get_cmap("YlOrRd").copy(); cm.set_bad("white")
        def _md(a, k=2):
            z = a.astype(float).copy(); jj = np.arange(z.shape[0]); z[np.abs(jj[:, None]-jj[None, :]) <= k] = np.nan; return z
        def _pctn(z):
            fv = z[np.isfinite(z)]
            if fv.size == 0: return z
            lo, hi = np.nanpercentile(fv, 1), np.nanpercentile(fv, 99); return np.clip((z-lo)/(hi-lo+1e-9), 0, 1)
        seq = fa.fetch(CHROM, FIG_S0, FIG_S0 + WIN).upper()
        if seq.count("N") <= WIN * 0.5:
            soh = encode(seq); atac = log1p_p99(load_atac(CHROM, FIG_S0, FIG_S0 + WIN)[:WIN]); mean = float(atac.mean())
            soh_t = torch.Tensor(soh.T).unsqueeze(0).to(dev)
            panel = [("f=0.00 (all average)", 0.0), ("f=0.05", 0.05), ("f=0.20", 0.20), ("f=1.00 (full ATAC)", 1.0)]
            maps = []
            for _, f in panel:
                am = amod_topk(atac, f, mean)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    maps.append(gen_contact(soh_t, torch.Tensor(am).view(1, 1, -1).to(dev)).float().cpu().numpy()[0])
            gt = None
            if COOL:
                try:
                    raw = cooler.Cooler(COOL).matrix(balance=True).fetch(f"{CHROM}:{FIG_S0}-{FIG_S0+WIN}").astype(np.float32)
                    if raw.shape[0] != N: raw = raw[:N, :N]
                    cov = np.isfinite(raw) & (raw > 0)
                    if cov.any():
                        d2 = np.abs(ii[:, None]-ii[None, :]); ex = np.zeros(N)
                        for d in range(N):
                            v = np.diagonal(raw, d); v = v[np.isfinite(v)]; ex[d] = np.nanmean(v) if len(v) else np.nan
                        lastv = ex[0] if np.isfinite(ex[0]) else 1e-6
                        for d in range(N):
                            ex[d] = lastv if not np.isfinite(ex[d]) else ex[d]; lastv = ex[d]
                        nm = ex[d2]; eps = float(np.nanmin(nm[nm > 0])) if (nm > 0).any() else 1e-6
                        gt = np.where(cov, np.log((np.maximum(raw, 0)+eps)/(nm+eps)), np.nan)
                except Exception: gt = None
            ncol = len(panel) + (1 if gt is not None else 0)
            ff, ax = plt.subplots(1, ncol, figsize=(4.4*ncol, 4.6), squeeze=False)
            col = 0
            if gt is not None:
                ax[0, col].imshow(_pctn(_md(gt)), cmap=cm, vmin=0, vmax=1, origin="lower", interpolation="nearest")
                ax[0, col].set_title(f"{CELL} RCMC GT", fontsize=10); ax[0, col].set_xticks([]); ax[0, col].set_yticks([]); col += 1
            for (lab, _), mp in zip(panel, maps):
                ax[0, col].imshow(_pctn(_md(mp)), cmap=cm, vmin=0, vmax=1, origin="lower", interpolation="nearest")
                ax[0, col].set_title(f"general(DNA, {lab})", fontsize=10); ax[0, col].set_xticks([]); ax[0, col].set_yticks([]); col += 1
            ff.suptitle(f"{CELL} {CHROM}:{FIG_S0//10**6}M  top-ATAC ablation (rest=window mean)", fontsize=12)
            ff.tight_layout()
            ff.savefig(f"{OUTDIR}/verify_topk_maps.png", dpi=120, bbox_inches="tight"); plt.close(ff)
            print(f"[verify_topk] maps -> {OUTDIR}/verify_topk_maps.png", flush=True)
    except Exception as e:
        import traceback; print(f"[verify_topk] maps skip: {e}\n{traceback.format_exc()}", flush=True)
    print(f"[verify_topk] DONE -> {tsv}", flush=True)
