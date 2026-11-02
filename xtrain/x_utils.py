"""Genomic input readers and fixed-window sampling for XChroma training.

FASTA coordinates are zero-based and half-open. DNA jitter never changes the
contact or ATAC coordinates. The sampler draws fixed-length, positive-strand
windows from one chromosome at a time.
"""
from __future__ import annotations

import gzip
import os
from pathlib import Path
import random
import tempfile

import numpy as np


class MemmapGenome:
    """Read A/C/G/T encodings from indexed FASTA or a compatible raw cache.

    Caches are float32, C-order arrays of shape ``(4, total_bases)``; contigs
    are concatenated in lexical order. Existing caches must belong to this
    exact FASTA. A cache has no embedded sequence identity or freshness data.
    New caches are built atomically without loading the whole genome in RAM.
    """

    def __init__(self, input_path, memmapfile=None, blacklist_regions=None,
                 bases_order=None, init_unpicklable=False):
        if bases_order is not None and list(bases_order) != list("ACGT"):
            raise NotImplementedError("Training supports A,C,G,T channel order only")
        self.input_path = os.fspath(input_path)
        self.memmapfile = None if memmapfile is None else os.fspath(memmapfile)
        self.blacklist_regions = blacklist_regions
        self._fasta = self._cache = None
        self._pid = None
        self._lengths = self._offsets = self._blacklist = None
        if init_unpicklable:
            self._open()

    @staticmethod
    def _encode(sequence):
        bases = np.frombuffer(str(sequence).upper().encode("ascii"), dtype=np.uint8)
        encoded = np.full((len(bases), 4), 0.25, dtype=np.float32)
        for channel, base in enumerate(b"ACGT"):
            positions = bases == base
            encoded[positions] = 0
            encoded[positions, channel] = 1
        return encoded

    def _open(self):
        if self._pid == os.getpid():
            return
        from pyfaidx import Fasta
        self._fasta = Fasta(self.input_path, as_raw=True, sequence_always_upper=True)
        self._lengths = {name: len(self._fasta[name]) for name in sorted(self._fasta.keys())}
        self._offsets = {}
        total = 0
        for name, length in self._lengths.items():
            self._offsets[name] = total
            total += length
        self._blacklist = self._read_blacklist()
        if self.memmapfile is not None:
            path = Path(self.memmapfile)
            if not path.exists():
                self._build_cache(path, total)
            if path.stat().st_size != 4 * total * np.dtype(np.float32).itemsize:
                raise ValueError("Genome cache size does not match FASTA chromosome lengths")
            self._cache = np.memmap(path, dtype=np.float32, mode="r", shape=(4, total))
        self._pid = os.getpid()

    def _build_cache(self, path, total):
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
        os.close(handle)
        try:
            cache = np.memmap(temporary, dtype=np.float32, mode="w+", shape=(4, total))
            for chrom, length in self._lengths.items():
                offset = self._offsets[chrom]
                for start in range(0, length, 1_000_000):
                    end = min(length, start + 1_000_000)
                    cache[:, offset + start:offset + end] = self._encode(self._fasta[chrom][start:end]).T
            cache.flush()
            del cache
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _read_blacklist(self):
        if self.blacklist_regions is None:
            return {}
        path = os.fspath(self.blacklist_regions)
        if not os.path.isfile(path):
            raise ValueError("blacklist_regions must be a BED or gzip-compressed BED file")
        regions = {}
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as handle:
            for line in handle:
                if not line.strip() or line.startswith(("#", "track", "browser")):
                    continue
                chrom, start, end, *_ = line.split()
                regions.setdefault(chrom, []).append((int(start), int(end)))
        return regions

    def get_chr_lens(self):
        self._open()
        return list(self._lengths.items())

    def get_chrs(self):
        return [chrom for chrom, _ in self.get_chr_lens()]

    def coords_in_bounds(self, chrom, start, end):
        self._open()
        if chrom not in self._lengths or start < 0 or end <= start or end > self._lengths[chrom]:
            return False
        return not any(start < right and end > left for left, right in self._blacklist.get(chrom, ()))

    def get_encoding_from_coords(self, chrom, start, end, strand="+", pad=False):
        self._open()
        if strand not in ("+", "-", "."):
            raise ValueError("strand must be '+', '-' or '.'")
        if chrom not in self._lengths:
            raise KeyError(chrom)
        if end < start:
            raise ValueError("end must not precede start")
        length = self._lengths[chrom]
        if not pad and (start < 0 or end > length):
            raise ValueError("Requested coordinates exceed chromosome bounds; use pad=True")
        left, right = max(0, start), min(length, end)
        if pad:
            encoded = np.full((end - start, 4), 0.25, dtype=np.float64)
        else:
            encoded = np.empty((end - start, 4), dtype=np.float32)
        if right > left:
            if self._cache is None:
                section = self._encode(self._fasta[chrom][left:right])
            else:
                offset = self._offsets[chrom]
                section = self._cache[:, offset + left:offset + right].T
            encoded[left - start:right - start] = section
        return encoded[::-1, ::-1].copy() if strand == "-" else encoded

    def get_encoding_from_coords_check_unk(self, chrom, start, end, strand="+", pad=False):
        """Return the encoding and whether any position is ambiguous or padded."""
        encoded = self.get_encoding_from_coords(chrom, start, end, strand=strand, pad=pad)
        return encoded, bool(np.any(np.all(encoded == 0.25, axis=1)))

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_fasta=None, _cache=None, _pid=None)
        return state


class Genomic2DFeatures:
    """Fetch stored-weight balanced Cooler targets, retaining missing bins."""

    def __init__(self, input_paths, features, shape, cg=False, cuda=False):
        if cuda:
            raise NotImplementedError("Training target coarse-graining is CPU-only")
        self.input_paths = [input_paths] if isinstance(input_paths, (str, os.PathLike)) else list(input_paths)
        self.features = [features] if isinstance(features, str) else list(features)
        if not self.input_paths or len(self.input_paths) != len(self.features):
            raise ValueError("Provide one feature name for each Cooler input")
        self.shape = tuple(shape)
        self.cg = bool(cg)
        self._sources = None

    def _open(self):
        if self._sources is None:
            from cooler import Cooler
            self._sources = [Cooler(os.fspath(path)) for path in self.input_paths]

    @property
    def chromnames(self):
        self._open()
        common = set(self._sources[0].chromnames)
        for source in self._sources[1:]:
            common.intersection_update(source.chromnames)
        return sorted(common)

    def get_feature_data(self, chrom, start, end, chrom2=None, start2=None, end2=None):
        self._open()
        region = (chrom, start, end)
        second = (chrom2, start2, end2) if all(value is not None for value in (chrom2, start2, end2)) else region
        matrices = []
        for source in self._sources:
            balanced = source.matrix(balance=True).fetch(region, second)
            if self.cg:
                from cooltools.lib.numutils import adaptive_coarsegrain
                raw = source.matrix(balance=False).fetch(region, second)
                rows, cols = balanced.shape
                side = max(rows, cols, 9)
                padded = np.full((side, side), np.nan)
                counts = np.full((side, side), np.nan)
                padded[:rows, :cols], counts[:rows, :cols] = balanced, raw
                balanced = adaptive_coarsegrain(padded, counts, cutoff=5, max_levels=12, min_shape=8)[:rows, :cols]
            matrices.append(np.asarray(balanced, dtype=np.float32))
        return matrices[0] if len(matrices) == 1 else np.stack(matrices)


class FixedWindowSampler:
    """Sample one positive-strand chromosome window per example.

    The callback ``sequence_transform(chrom,start,end,dna,strand)`` appends
    the normalized ATAC channel. ``dna`` includes DNA-only random jitter;
    callback and target coordinates remain unshifted. Randomness intentionally
    follows NumPy's global MT19937 stream, including per-mode chromosome
    prefetches, for compatibility with the retained Selene worker initializer.
    """
    _PREFETCH = 200_000

    def __init__(self, reference_sequence, target, sequence_length=1_000_000,
                 position_resolution=1_000, validation_holdout=("chr8",),
                 test_holdout=("chr9", "chr10"), random_shift=100, seed=436,
                 sequence_transform=None, mode="train"):
        if sequence_length <= 0 or position_resolution <= 0 or sequence_length % position_resolution:
            raise ValueError("Window length must be a positive multiple of position resolution")
        if random_shift < 0:
            raise ValueError("random_shift must be nonnegative")
        self.reference_sequence, self.target = reference_sequence, target
        self.sequence_length, self.position_resolution = int(sequence_length), int(position_resolution)
        self.validation_holdout, self.test_holdout = tuple(validation_holdout), tuple(test_holdout)
        self.random_shift, self.seed = int(random_shift), int(seed)
        self.sequence_transform = sequence_transform
        self.atac_mm = None
        if self.test_holdout and not self.validation_holdout:
            raise ValueError("test_holdout requires a validation_holdout")
        self.modes = ["train", "validate"] + (["test"] if self.test_holdout else []) if self.validation_holdout else ["train"]
        self.set_mode(mode)
        self._partitions = None
        self._streams, self._positions = {}, {}
        np.random.seed(self.seed)
        random.seed(self.seed + 1)

    def set_mode(self, mode):
        if mode not in self.modes:
            raise ValueError(f"Unknown or disabled sampler mode: {mode}")
        self.mode = mode

    def _initialize(self):
        if self._partitions is not None:
            return
        partitions = {mode: [] for mode in self.modes}
        for chrom, length in sorted(self.reference_sequence.get_chr_lens()):
            mode = "validate" if chrom in self.validation_holdout else "test" if chrom in self.test_holdout else "train"
            if length > 0:
                partitions[mode].append((chrom, length))
        for mode, rows in partitions.items():
            if not rows or not any(length >= self.sequence_length for _, length in rows):
                raise ValueError(f"No chromosome can supply a full window in {mode!r} split")
            lengths = np.asarray([length for _, length in rows], dtype=float)
            probabilities = lengths / lengths.sum()
            keep = probabilities > 1e-10
            partitions[mode] = ([row for row, retain in zip(rows, keep) if retain], probabilities[keep] / probabilities[keep].sum())
        self._partitions = partitions
        for mode in self.modes:
            self._refill(mode)

    def _refill(self, mode):
        rows, probabilities = self._partitions[mode]
        self._streams[mode] = np.random.choice(len(rows), self._PREFETCH, p=probabilities)
        self._positions[mode] = 0

    def _window(self, mode):
        rows, _ = self._partitions[mode]
        for _ in range(100_000):
            if self._positions[mode] == self._PREFETCH:
                self._refill(mode)
            index = self._streams[mode][self._positions[mode]]
            self._positions[mode] += 1
            chrom, length = rows[index]
            start = np.random.randint(max(1, length - self.sequence_length))
            start = int(start // self.position_resolution * self.position_resolution)
            end = start + self.sequence_length
            if self.reference_sequence.coords_in_bounds(chrom, start, end):
                return chrom, start, end
        raise RuntimeError("Could not draw a valid window after 100000 attempts")

    def sample(self, batch_size=1, mode=None, coordinate_only=False):
        active = self.mode if mode is None else mode
        if active not in self.modes:
            raise ValueError(f"Unknown or disabled sampler mode: {active}")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._initialize()
        windows = [self._window(active) for _ in range(batch_size)]
        if coordinate_only:
            return [[[chrom], [start], [end], ["+"]] for chrom, start, end in windows]
        if self.sequence_transform is None:
            raise ValueError("sequence_transform must append the normalized ATAC channel")
        sequences = np.empty((batch_size, self.sequence_length, 5), dtype=np.float64)
        targets = np.empty((batch_size, *self.target.shape), dtype=np.float64)
        for index, (chrom, start, end) in enumerate(windows):
            shift = np.random.randint(-self.random_shift, self.random_shift) if self.random_shift else 0
            dna = self.reference_sequence.get_encoding_from_coords(chrom, start + shift, end + shift, strand="+", pad=True)
            sequences[index] = self.sequence_transform(chrom, start, end, dna, "+")
            targets[index] = self.target.get_feature_data(chrom, start, end)
        return sequences, targets
