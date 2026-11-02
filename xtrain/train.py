"""Training entry point for the XChroma contact architecture.
Configuration is supplied through environment variables; see README.md.
The released inference ensemble is defined separately in checkpoints/current_model.json."""
import os
import cooler
import numpy as np
import torch

from selene_sdk.samplers.dataloader import SamplerDataLoader
from x_utils import Genomic2DFeatures, MemmapGenome

import model as _model
import data as _data

RESOLUTION = int(os.environ.get("RESOLUTION", "1000"))
CORE_BP    = int(os.environ.get("WINDOW_BP", "1000000"))
N_BINS     = CORE_BP // RESOLUTION

COOL        = os.environ["COOL"]
EXPECTED    = os.environ["EXPECTED"]
GENOME_FA   = os.environ["GENOME_FA"]
GENOME_MMAP = os.environ.get("GENOME_MMAP", "")
BLACKLIST   = os.environ.get("BLACKLIST", "")
OUTDIR      = os.environ.get("OUTDIR", "./run")
INIT_FROM   = os.environ.get("INIT_FROM", "")

BATCH_SIZE  = int(os.environ.get("BATCH_SIZE", "4"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
MAX_STEPS   = int(os.environ.get("MAX_STEPS", "150000"))
ACCUM_STEPS = int(os.environ.get("ACCUM_STEPS", "1"))
LR          = float(os.environ.get("LR", "5e-4"))
VAL_EVERY   = int(os.environ.get("VAL_EVERY", "500"))
SAVE_EVERY  = int(os.environ.get("SAVE_EVERY", "250"))
VAL_COORD   = os.environ.get("VAL_COORD", "chr8:85000000-86000000")
DIAG_BAND   = int(os.environ.get("DIAG_BAND", "2"))
SKIP_INITIAL_VAL = os.environ.get("SKIP_INITIAL_VAL", "1") == "1"
BENCH_EVERY  = int(os.environ.get("BENCH_EVERY", "0")) or None
BENCH_MAX_REGIONS = int(os.environ.get("BENCH_MAX_REGIONS", "0"))

HOLDOUT_TEST = [x for x in os.environ.get("HOLDOUT_TEST", "chr9,chr10").split(",") if x]
HOLDOUT_VAL  = [x for x in os.environ.get("HOLDOUT_VAL", "chr8").split(",") if x]

assert HOLDOUT_VAL or not HOLDOUT_TEST, (
    "HOLDOUT_TEST requires a nonempty HOLDOUT_VAL for the sampler modes. "
    "Leave both empty to train without chromosome holdouts.")
if not HOLDOUT_TEST:
    print("[config] No HOLDOUT_TEST chromosomes; held-out scoring is disabled.", flush=True)

COVERAGE_MASK   = os.environ.get("COVERAGE_MASK", "0") == "1"
MIN_VALID_FRAC  = float(os.environ.get("MIN_VALID_FRAC", "0.6"))
USE_AMP         = os.environ.get("USE_AMP", "0") == "1"

RC_AUG_PROB = float(os.environ.get("RC_AUG_PROB", "0"))
SEED = int(os.environ.get("SEED", "1"))

feat_name = f"r{RESOLUTION}"
prefix    = "xchroma_1mb_1kb"
models_dir = os.path.join(OUTDIR, "models")
os.makedirs(models_dir, exist_ok=True)

torch.manual_seed(SEED); np.random.seed(SEED)

print(f"[config] {RESOLUTION}bp x {N_BINS} bins (window {CORE_BP})", flush=True)
print(f"[config] cool={COOL}\n[config] expected={EXPECTED}\n[config] outdir={OUTDIR}", flush=True)

normmat_bydist = np.exp(np.load(EXPECTED).astype(np.float32))[:N_BINS]
normmat = normmat_bydist[np.abs(np.arange(N_BINS)[:, None] - np.arange(N_BINS)[None, :])]
eps = float(np.min(normmat))
print(f"[config] eps={eps:.3e}", flush=True)

target = Genomic2DFeatures([COOL], [feat_name], (N_BINS, N_BINS), cg=True)
genome = MemmapGenome(input_path=GENOME_FA,
                      memmapfile=GENOME_MMAP or None,
                      blacklist_regions=BLACKLIST or None)
atac_mm = _data._load_atac_memmaps(_data.ATAC_DIR)
_data.bind_gp99(_data.ATAC_DIR, atac_mm)

_cool_chroms = set(cooler.Cooler(COOL).chromsizes.index)
_orig_get_chr_lens = genome.get_chr_lens
genome.get_chr_lens = lambda *a, **k: [t for t in _orig_get_chr_lens(*a, **k) if t[0] in _cool_chroms]
sampler = _data.make_sampler(genome, target, feat_name, CORE_BP, RESOLUTION,
                             HOLDOUT_TEST, HOLDOUT_VAL, atac_mm)
sampler.mode = "train"
dataloader = SamplerDataLoader(sampler, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, seed=SEED)

net = _model.build_net().cuda()
n_par = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"[model] trainable params: {n_par/1e6:.1f} M", flush=True)

optimizer = torch.optim.Adam(_model.dcls_param_groups(net, LR), lr=LR, betas=(0.9, 0.999))
scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

start_step = 0
if INIT_FROM and os.path.isfile(INIT_FROM):
    ckpt = torch.load(INIT_FROM, map_location="cpu")
    missing, unexpected = net.load_state_dict(ckpt["model"], strict=False)
    start_step = int(ckpt.get("step", 0))
    print(f"[init] {INIT_FROM} step={start_step} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"]); print("[init] optimizer state restored", flush=True)
        except Exception as e:
            print(f"[init] optimizer state skipped: {e}", flush=True)
else:
    print("[init] from scratch (no warm-start)", flush=True)

_vc, _vr = VAL_COORD.split(":"); _vs, _ve = (int(x) for x in _vr.split("-"))
assert _ve - _vs == CORE_BP, f"VAL_COORD must span {CORE_BP} bp"
_val_gt = np.asarray(target.get_feature_data(_vc, _vs, _ve), np.float32).reshape(N_BINS, N_BINS)
gt_logoe = np.log((_val_gt + eps) / (normmat + eps))
_val_x = _data.stack_seq_atac(genome.get_encoding_from_coords(_vc, _vs, _ve), _vc, _vs, _ve, atac_mm=atac_mm)
_ii = np.arange(N_BINS); _offdiag = np.abs(_ii[:, None] - _ii[None, :]) > DIAG_BAND
loss_history = []

_bench_c = cooler.Cooler(COOL)
_bench_regions = []
for _ch in (HOLDOUT_TEST or []):
    if _ch not in _bench_c.chromsizes:
        print(f"[bench] Skipping {_ch}: absent from the Cooler target", flush=True); continue
    _cs = int(_bench_c.chromsizes[_ch])
    _bench_regions += [(_ch, _s, _s + CORE_BP) for _s in range(0, _cs - CORE_BP, CORE_BP)]
if BENCH_MAX_REGIONS > 0:
    _bench_regions = _bench_regions[:BENCH_MAX_REGIONS]
print(f"[bench] {len(_bench_regions)} held-out regions; chromosomes {HOLDOUT_TEST or '(none)'}", flush=True)

def run_benchmark(step):
    """Return median per-region Pearson r and count, excluding the diagonal band and nonfinite values."""
    net.eval()
    rs = []
    with torch.no_grad():
        for chrom, s_, e_ in _bench_regions:
            seq = genome.get_encoding_from_coords(chrom, s_, e_)
            if seq.shape[0] != CORE_BP:
                continue
            x = _data.stack_seq_atac(seq, chrom, int(s_), int(e_), atac_mm=atac_mm)
            o = net(torch.tensor(x.T[None], dtype=torch.float32).cuda())
            p = (o[0] if isinstance(o, (list, tuple)) else o)[0, 0].cpu().numpy().astype(np.float32)
            p = 0.5 * (p + p.T)
            try:
                g = _bench_c.matrix(balance=True).fetch((chrom, s_, e_)).astype(np.float32)
            except Exception:
                continue
            if g.shape[0] != N_BINS:
                continue
            g = np.log((np.maximum(g, 0.0) + eps) / (normmat + eps))
            m = _offdiag & np.isfinite(g) & np.isfinite(p)
            if m.sum() < 100:
                continue
            rs.append(float(np.corrcoef(p[m], g[m])[0, 1]))
    net.train()
    return (float(np.median(rs)) if rs else float("nan")), len(rs)

def validate(step):
    net.eval()
    with torch.no_grad():
        o = net(torch.tensor(_val_x.T[None], dtype=torch.float32).cuda())
        p = (o[0] if isinstance(o, tuple) else o)[0, 0].cpu().numpy().astype(np.float32)
    p = 0.5 * (p + p.T)
    m = _offdiag & np.isfinite(gt_logoe)
    pear = float(np.corrcoef(p[m], gt_logoe[m])[0, 1]) if m.sum() > 100 else float("nan")
    avg_loss = float(np.mean(loss_history[-VAL_EVERY:])) if loss_history else float("nan")
    _f = os.path.join(OUTDIR, "val_loss.tsv"); _new = not os.path.exists(_f)
    with open(_f, "a") as o_:
        if _new: o_.write("step\tval_pear\tloss\n")
        o_.write(f"{step}\t{pear:.5f}\t{avg_loss:.5f}\n")
    print(f"[val] step={step} val_pear={pear:.4f} loss={avg_loss:.4f}", flush=True)

    if BENCH_EVERY is None or step % BENCH_EVERY == 0:
        try:
            med = run_benchmark(step)
            _mf = os.path.join(OUTDIR, "metrics.tsv"); _new = not os.path.exists(_mf)
            with open(_mf, "a") as o_:
                if _new: o_.write("step\tmedian_r\tn_regions\n")
                o_.write(f"{step}\t{med[0]:.4f}\t{med[1]}\n")
            print(f"[bench] step={step} median_r={med[0]:.4f} over {med[1]} regions", flush=True)
        except Exception as e:
            print(f"[bench] FAILED: {e}", flush=True)
    net.train()

def save_ckpt(step):
    sd = net.state_dict()
    for p in (f"{prefix}.step{step}.ckpt.pt", f"{prefix}.latest.ckpt.pt"):
        torch.save({"step": step, "model": sd, "optimizer": optimizer.state_dict()},
                   os.path.join(models_dir, p))
    print(f"[save] step={step}", flush=True)

if not SKIP_INITIAL_VAL:
    validate(start_step)
net.train()
step = start_step
accum_count = 0
optimizer.zero_grad()
_it = iter(dataloader)
print(f"Training up to {MAX_STEPS} steps (accum={ACCUM_STEPS})...", flush=True)

while step < MAX_STEPS:
    try:
        sequence, tgt = next(_it)
    except StopIteration:
        _it = iter(dataloader); continue

    if RC_AUG_PROB > 0 and torch.rand(1).item() < RC_AUG_PROB:

        seq_rc = sequence.flip([1])
        sequence = torch.cat([seq_rc[..., :4].flip([-1]),
                              seq_rc[..., 4:]], dim=-1)
        tgt = tgt.flip([1, 2])

    tgt_np = tgt.numpy()
    target_cuda = torch.Tensor(np.log((tgt_np + eps) / (normmat + eps))).cuda()
    base_valid = ~torch.isnan(target_cuda)
    if COVERAGE_MASK:
        base_valid = base_valid & torch.Tensor((tgt_np > 0.0).astype(np.float32)).cuda().bool()
    if base_valid.float().mean().item() < MIN_VALID_FRAC:
        continue

    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=USE_AMP):
        out = net(torch.Tensor(sequence.float()).transpose(1, 2).cuda())
        pred = out[0] if isinstance(out, tuple) else out
        target_clean = torch.where(base_valid, target_cuda, torch.zeros_like(target_cuda))
        loss = ((pred[:, 0, :, :] - target_clean) ** 2)[base_valid].mean()

    scaler.scale(loss / ACCUM_STEPS).backward()
    loss_history.append(float(loss.detach().cpu()))
    accum_count += 1
    if accum_count % ACCUM_STEPS != 0:
        continue
    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

    (net.module if hasattr(net, "module") else net).clamp_positions()
    step += 1

    if step == 1 and torch.cuda.is_available():
        print(f"[mem] peak GPU memory after step 1: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
    if step % 100 == 0:
        print(f"[train] step={step} loss={np.mean(loss_history[-100:]):.4f}", flush=True)
    if step % VAL_EVERY == 0:
        validate(step)
    if step % SAVE_EVERY == 0:
        save_ckpt(step)

save_ckpt(step)
print(f"[done] step={step}", flush=True)
