"""Data pipeline for XChrom training: Selene-based random-position Hi-C sampler
with an ATAC 5th input channel, plus the ATAC load/normalize helpers.

Extracted from train_atac.py (锁定配置是 ATAC_NORM=global, ATAC_NORM_SCALE=1.0). Nothing here imports back into the day-to-day tree.

Public API:
  _load_atac_memmaps(atac_dir)         build such a dict for any ATAC dir
  normalize_atac(atac)                 per-window ATAC normalization (env ATAC_NORM)
  fetch_atac(chrom, s, e, atac_mm)     float32 ATAC slice, 0-padded
  stack_seq_atac(seq_oh, ...)          concat (L,4) DNA one-hot + (L,1) ATAC -> (L,5)
  RandomPositionsSamplerHiCWithATAC    sampler that appends ATAC as the 5th channel
  make_sampler(...)                    build a configured sampler
"""
import os
import json as _json
import numpy as np

from x_utils import (
    Genomic2DFeatures, MemmapGenome,
    RandomPositionsSamplerHiC, _adaptive_coarsegrain,
)
from x_utils import ZAtac as _ZAtac   # zstd 分块的 ATAC 存储, 切片和 memmap 兼容

# === ATAC config (env, same var names as train_atac.py) ====================
ATAC_DIR = os.environ.get("ATAC_DIR", "")
ATAC_DTYPE = np.float16
ATAC_NORM = os.environ.get("ATAC_NORM", "global").lower()
ATAC_NORM_SCALE = float(os.environ.get("ATAC_NORM_SCALE", "1.0"))
ATAC_TOPFRAC = float(os.environ.get("ATAC_TOPFRAC", "0") or 0)
ATAC_ZERO = os.environ.get("ATAC_ZERO", "0") == "1"   # seq-only: feed a zero ATAC channel

ATAC_REF = None          # qmesc 参考表, 锁定配置用不到

# ---- ATAC_NORM=global 用的 genome-wide p99 ----------------------------------
# ⛔ 这个常数**每个细胞一个**, 缓存在 <ATAC 目录>/.global_p99。
#    不能退回逐窗 p99: 逐窗归一会抹掉跨窗的绝对可及性水平, 而 global 归一保留它 ——
#    正是这条路径存在的理由。也不能静默返回未归一的值 (那会让输入整体偏大约 5.5 倍)。
#    首次计算是**等距**采样(不是随机): 每条染色体按固定 stride 取若干段 1000 bp,
#    每条约 100 万 bases, 合并后取 p99。所以它是**抽样估计**, 不是所有 base 等权的
#    精确全基因组 p99 (audit INPUT-004)。慢盘上首算可能 >1h, 一定要走缓存;
#    见 README 的 "ATAC 归一常数预热"。
_GP99 = None             # 当前正在归一化的那个细胞的常数; 由 stack_seq_atac 绑定
_GP99_BY_DIR = {}


def gp99_for_dir(atac_dir, memmaps=None):
    """某个 ATAC 目录的 genome-wide p99。优先读 <dir>/.global_p99, 缺了就采样并回写缓存。"""
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
        if not memmaps:            # None 和空 dict 都要重新加载
            memmaps = _load_atac_memmaps(atac_dir)
        # ⛔ 这里**没有随机数**: 下面是固定 stride 的等距切片。曾经有一行
        #    `rng = np.random.RandomState(0)` 但从未被使用, 却让文档写成"随机抽样"。
        samp = []
        for mm in memmaps.values():
            # ⛔ 不能用花式索引: .az store (x_utils.ZAtac) 只支持 slice, 会 TypeError。
            #    改成等距切片抽样, dense memmap 和 .az 都能走。
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
    """把某个细胞的常数设为当前值。单细胞训练时启动调一次即可。"""
    global _GP99
    _GP99 = gp99_for_dir(atac_dir, memmaps)
    return _GP99


def _chrom_entries(man):
    """Yield (chrom, length) from an ATAC manifest, tolerating both schemas and metadata keys.

    Two schemas exist in practice:
      A) flat   {"chr1": {"length": N, ...}, ..., "_source": {...}}
      B) nested {"chroms": {"chr1": N, ...}, ...}
    The converter writes a `_source` metadata block; iterating every key and reading
    ``entry["length"]`` therefore raised KeyError on that block (audit INPUT-006).
    Skip anything that is not a chromosome entry.
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
    """Per-chrom mm39 per-bp log1p(fragment count) memmaps (or zstd .az store)."""
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
    print(f"[atac] loaded {len(mm)} chrom memmaps from {atac_dir}", flush=True)
    return mm


# Primary cell ATAC store, built at import (matches train_atac.py behavior).

def normalize_atac(atac):
    """把 log1p(fragment count) 的 ATAC 通道归一到网络吃的尺度。

    只保留两条口径, 其余历史开关 (none / linear_p99 / bin1k_p99 / affine / nz_p99 /
    nz_rank / rank / qmesc / local_baseline) 在瘦身时删掉了 —— 当前配置一条都走不到。

      global      genome-wide 的一个 p99 常数, **每个细胞一个**。保留跨窗的绝对可及性水平。
                  这是锁定配置的口径。
      log1p_p99   逐窗 p99。⛔ legacy: 只为复现用这个口径训出来的旧 ckpt 保留 ——
                  它会抹掉跨窗的绝对水平, 不要用在新 run 上。
    """
    if ATAC_NORM == "global":
        # ⛔ 常数缺失时**报错**, 绝不静默返回未归一的值 —— 那样输入会整体偏大数倍而毫无提示。
        if _GP99 is None or not np.isfinite(_GP99) or _GP99 <= 1e-6:
            raise RuntimeError(
                "ATAC_NORM=global 但没有可用的 global p99 常数。训练前调用 "
                "data.bind_gp99(ATAC_DIR), 或确保 <ATAC 目录>/.global_p99 存在且是有限正数。")
        return (atac / _GP99).astype(np.float32) * ATAC_NORM_SCALE
    if ATAC_NORM == "log1p_p99":
        p = float(np.percentile(atac, 99))
        return atac if p <= 1e-6 else (atac / p).astype(np.float32) * ATAC_NORM_SCALE
    raise ValueError(f"ATAC_NORM={ATAC_NORM!r} 不认识; 只支持 'global' 和 'log1p_p99'。")


def fetch_atac(chrom, start, end, atac_mm=None):
    """Return float32 ATAC slice of length (end-start), padded with 0 if OOB.

    ⛔ 缺失一律变成 **0**, 而且不带 missingness 通道(audit INPUT-005):
      - 整条染色体没有 track  -> 全 0
      - 区间越界              -> 越界部分补 0
      - converter 在 log1p 之前把 bigWig 的 NaN 置 0
    所以模型看到的 "0" 同时表示"测到但没信号"和"根本没测"。这与 DNA 的 padding
    不同 —— DNA 越界补的是 0.25(四碱基等概率), 是可区分的标记值。
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
    # ⛔ 单细胞: 归一常数由 train.py 启动时的 bind_gp99() 设好。
    #    多细胞要按细胞切换常数的话, 别用模块全局 —— 把常数挂到 sampler 实例上, 随 atac_mm
    #    一起传进来。源头是靠给 dict 套一个能挂属性的子类做到的, 那套没有搬过来。
    atac = normalize_atac(atac)
    return np.concatenate([seq_oh, atac[:, None].astype(np.float32)], axis=1)


# ===========================================================================
#  the locked mESC config uses a .cool -> Genomic2DFeatures. Kept for release.)
# ===========================================================================
class RandomPositionsSamplerHiCWithATAC(RandomPositionsSamplerHiC):
    """Parent retrieves (B, L, 4) sequences; this appends ATAC as the 5th channel.
    Parent's sample() hardcodes a (B, L, 4) buffer, so sample() is overridden too
    (the only change is the 4 -> 5 in the np.zeros call)."""

    def _retrieve_multi(self, chroms, starts, ends, strands=None):
        result = super()._retrieve_multi(chroms, starts, ends, strands)
        retrieved_seqs = list(result[0])
        new_seqs = []
        # ⚠ 基类取 DNA 时加了 random_shift 的随机偏移 (`start + r`, r ∈ [-100, 100)),
        #   这里取 ATAC 用的是**没加 r 的** start/end。所以每个样本的 ATAC 通道相对 DNA
        #   随机错位最多 ±100 bp。这和参考实现 (train_atac.py:2878-2886) 逐字一致 —— 是
        #   忠实复刻, 不是本包引入的。改它会改变训练语义、和参考 run 不再可比, 所以原样保留。
        for i, (chrom, start, end) in enumerate(zip(chroms, starts, ends)):
            seq_oh = retrieved_seqs[i]
            strand_i = strands[i] if strands is not None else "+"
            new_seqs.append(stack_seq_atac(seq_oh, chrom, int(start), int(end),
                                           strand_i, getattr(self, "atac_mm", None)))
        return (new_seqs,) + tuple(result[1:])

    def sample(self, batch_size=1, mode=None, coordinate_only=False):
        # Verbatim copy of RandomPositionsSamplerHiC.sample with the ONLY change
        # being the sequences buffer dim: 4 -> 5 channels. Parent's @init lazy
        # first-call init is inlined here (decorator lost on override).
        if not self.initialized:
            self._partition_genome_by_chromosome()
            for _m in self.modes:
                self._update_randcache(mode=_m)
            self.initialized = True
        mode = mode if mode else self.mode
        if not coordinate_only:
            sequences = np.zeros((batch_size, self.sequence_length, 5))   # 4 DNA + 1 ATAC
            targets = np.zeros((batch_size, *self.target.shape))
            if self.bg:
                normmats = np.zeros((batch_size, *self.target.shape))
            if self.target_1d:
                target_1ds = np.zeros((batch_size, *self.target_1d.shape))

        n_samples_drawn = 0
        allcoords = []
        while n_samples_drawn < batch_size:
            current_length = 0
            chroms = []; starts = []; ends = []; strands = []
            while current_length < self.sequence_length:
                if len(chroms) == 0 or self.cross_chromosome:
                    sample_index = self._randcache[mode]["sample_next"]
                    if sample_index == len(self._randcache[mode]["cache_indices"]):
                        self._update_randcache()
                        sample_index = 0
                    rand_interval_index = self._randcache[mode]["cache_indices"][sample_index]
                    self._randcache[mode]["sample_next"] += 1
                    chrom, cstart, cend = self.sample_from_intervals[rand_interval_index]
                next_length = self.sequence_length - current_length
                if self.length_schedule is not None and self.cross_chromosome:
                    if np.random.rand() < self.length_schedule[0]:
                        next_length = np.fmin(next_length,
                            np.random.randint(self.length_schedule[1][0], self.length_schedule[1][1]))
                if self.max_seg_length is not None and self.cross_chromosome:
                    next_length = np.fmin(next_length, self.max_seg_length)
                start_position = np.random.randint(cstart, np.fmax(cstart + 1, cend - next_length))
                start_position -= start_position % self.position_resolution
                if start_position + next_length > cend:
                    if (self.cross_chromosome or (self.length_schedule is not None)
                            or (self.max_seg_length is not None)):
                        end_position = cend
                    else:
                        continue
                else:
                    end_position = start_position + next_length
                end_position -= end_position % self.position_resolution
                if end_position == start_position:
                    continue
                if not self.reference_sequence.coords_in_bounds(chrom, start_position, end_position):
                    continue
                current_length += end_position - start_position
                chroms.append(chrom); starts.append(start_position); ends.append(end_position)
                strand = self.STRAND_SIDES[np.random.randint(0, 2)] if self.random_strand else "+"
                strands.append(strand)
            if self.permute_segments:
                perm = np.random.permutation(np.arange(len(chroms)))
                chroms = [chroms[i] for i in perm]
                starts = [starts[i] for i in perm]
                ends = [ends[i] for i in perm]
                strands = [strands[i] for i in perm]
            allcoords.append([chroms, starts, ends, strands])
            n_samples_drawn += 1

        for i, (chroms, starts, ends, strands) in enumerate(allcoords):
            if not coordinate_only:
                retrieve_output = self._retrieve_multi(chroms, starts, ends, strands)
                if self.bg:
                    if self.target_1d:
                        seq, seq_targets, seq_background, seq_target_1ds = retrieve_output
                    else:
                        seq, seq_targets, seq_background = retrieve_output
                else:
                    if self.target_1d:
                        seq, seq_targets, seq_target_1ds = retrieve_output
                    else:
                        seq, seq_targets = retrieve_output

                if not isinstance(seq, list):
                    sequences[i, :, :] = seq
                    targets[i, :] = seq_targets
                    if self.bg:
                        normmats[i, :] = seq_background
                    if self.target_1d:
                        target_1ds[i, :] = seq_target_1ds
                else:
                    offset = 0
                    for s in seq:
                        sequences[i, offset: offset + s.shape[0], :] = s
                        offset += s.shape[0]
                    if self.target_1d:
                        offset = 0
                        for t in seq_target_1ds:
                            target_1ds[i, :, offset: offset + t.shape[1]] = t
                            offset += t.shape[1]
                    offsetx = 0
                    for row in seq_targets:
                        offsety = 0
                        for t in row:
                            if targets.ndim == 3:
                                targets[i, offsetx: offsetx + t.shape[0], offsety: offsety + t.shape[1]] = t
                            else:
                                targets[i, :, offsetx: offsetx + t.shape[-2], offsety: offsety + t.shape[-1]] = t
                            offsety += t.shape[-1]
                        offsetx += t.shape[-2]
                    assert offsetx == targets.shape[-2]
                    assert offsety == targets.shape[-1]
                    if self.bg:
                        offsetx = 0
                        for row in seq_background:
                            offsety = 0
                            for t in row:
                                if normmats.ndim == 3:
                                    normmats[i, offsetx: offsetx + t.shape[-2], offsety: offsety + t.shape[-1]] = t
                                else:
                                    normmats[i, :, offsetx: offsetx + t.shape[-2], offsety: offsety + t.shape[-1]] = t
                                offsety += t.shape[-1]
                            offsetx += t.shape[-2]
                        assert offsetx == normmats.shape[-2]
                        assert offsety == normmats.shape[-1]

        if coordinate_only:
            return allcoords
        if self.bg:
            return ((sequences, targets, normmats, target_1ds) if self.target_1d
                    else (sequences, targets, normmats))
        return (sequences, targets, target_1ds) if self.target_1d else (sequences, targets)


def make_sampler(genome, target, feat_name, core_bp, resolution,
                 holdout_test, holdout_val, atac_mm):
    """Build a RandomPositionsSamplerHiCWithATAC configured for the locked mESC run."""
    s = RandomPositionsSamplerHiCWithATAC(
        reference_sequence=genome,
        target=target,
        target_1d=None,                 # NUM_1D=0: no 1D auxiliary target
        features=[feat_name],
        test_holdout=holdout_test,
        validation_holdout=holdout_val,
        sequence_length=core_bp,
        position_resolution=resolution,
        random_shift=100,
        random_strand=False,
        cross_chromosome=False,
    )
    s.atac_mm = atac_mm                 # per-cell ATAC binding (used by _retrieve_multi)
    return s
