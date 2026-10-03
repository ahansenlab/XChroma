"""Lightweight, lossless readers for aligned DNA and native ATAC inputs."""
from pathlib import Path
import hashlib
import json
import os
import struct

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class ZAtac:
    """Random-access AZ1 reader; native float16/float32 values are unchanged."""
    def __init__(self, path):
        self.path = str(path)
        self.f = open(path, 'rb')
        try:
            if self.f.read(3) != b'AZ1':
                raise ValueError(f'Invalid AZ1 magic: {path}')
            self.n, self.CH, dnum = struct.unpack('<qqB', self.f.read(17))
            types = {np.dtype(t).num: np.dtype(t) for t in (np.float16, np.float32)}
            if dnum not in types:
                raise ValueError(f'Unsupported AZ dtype code: {dnum}')
            self.dtype = types[dnum]
            (nb,) = struct.unpack('<q', self.f.read(8))
            if self.n <= 0 or self.CH <= 0 or nb != (self.n + self.CH - 1) // self.CH:
                raise ValueError('Invalid AZ1 dimensions')
            self.offs = np.frombuffer(self.f.read((nb + 1) * 8), dtype='<i8')
            self.base = self.f.tell()
            if len(self.offs) != nb + 1 or self.offs[0] != 0 or np.any(np.diff(self.offs) <= 0):
                raise ValueError('Invalid AZ1 chunk offsets')
            if self.base + self.offs[-1] != os.fstat(self.f.fileno()).st_size:
                raise ValueError('AZ1 file size does not match its index')
            self.shape = (self.n,)
        except BaseException:
            self.f.close()
            raise

    def read(self, start, end):
        import zstandard
        start, end = max(0, start), min(self.n, end)
        if end <= start:
            return np.zeros(0, dtype=self.dtype)
        first, last = start // self.CH, (end - 1) // self.CH
        dec = zstandard.ZstdDecompressor()
        chunks = []
        for k in range(first, last + 1):
            size = int(self.offs[k + 1] - self.offs[k])
            compressed = os.pread(self.f.fileno(), size, self.base + int(self.offs[k]))
            expected = min(self.CH, self.n - k * self.CH)
            raw = dec.decompress(compressed, max_output_size=expected * self.dtype.itemsize)
            if len(raw) != expected * self.dtype.itemsize:
                raise ValueError(f'AZ1 decompressed chunk length mismatch: {self.path}')
            chunks.append(np.frombuffer(raw, dtype=self.dtype))
        values = np.concatenate(chunks)
        return values[start - first * self.CH:end - first * self.CH]

    def __getitem__(self, key):
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError('ZAtac supports contiguous slices only')
        start, end, _ = key.indices(self.n)
        return self.read(start, end)

    def close(self):
        self.f.close()


def read_fasta_region(path, chrom, start, end):
    """Read a 0-based half-open interval from an uncompressed indexed FASTA."""
    path = Path(path)
    entries = {}
    for line in Path(str(path) + '.fai').read_text().splitlines():
        fields = line.split('\t')
        entries[fields[0]] = tuple(map(int, fields[1:5]))
    if chrom not in entries:
        raise ValueError(f'{chrom} is absent from the FASTA index')
    length, offset, bases, width = entries[chrom]
    if not 0 <= start < end <= length:
        raise ValueError('Requested region is outside the FASTA chromosome')
    if bases <= 0 or width < bases:
        raise ValueError('Invalid FASTA index line width')
    left = offset + start // bases * width + start % bases
    right = offset + (end - 1) // bases * width + (end - 1) % bases + 1
    with path.open('rb') as stream:
        stream.seek(left)
        sequence = stream.read(right - left).replace(b'\n', b'').replace(b'\r', b'').upper()
    if len(sequence) != end - start:
        raise ValueError('FASTA and index disagree; rebuild the .fai index')
    return sequence, length


def encode_dna(sequence):
    """ACGT one-hot encoding; unknown bases have four zeros."""
    bases = np.frombuffer(sequence.upper(), dtype=np.uint8)
    result = np.zeros((4, len(bases)), dtype=np.float32)
    for i, base in enumerate(b'ACGT'):
        result[i, bases == base] = 1.0
    return result


def chromosome_lengths(manifest):
    source = manifest.get('chroms', manifest)
    result = {}
    for chrom, entry in source.items():
        if chrom.startswith('_'):
            continue
        if isinstance(entry, dict) and 'length' in entry:
            result[chrom] = int(entry['length'])
        elif isinstance(manifest.get('chroms'), dict) and isinstance(entry, (int, float)):
            result[chrom] = int(entry)
    return result


def read_atac_region(directory, chrom, start, end, expected_length=None):
    """Read a native log1p ATAC slice from AZ1 or float16 memmap storage."""
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    lengths = chromosome_lengths(manifest)
    if chrom not in lengths:
        raise ValueError(f'{chrom} is absent from the ATAC manifest')
    length = lengths[chrom]
    if expected_length is not None and length != expected_length:
        raise ValueError('FASTA and ATAC chromosome lengths differ; check the genome assembly')
    if not 0 <= start < end <= length:
        raise ValueError('Requested region is outside the ATAC chromosome')
    path = directory / f'{chrom}.atac.log1p.az'
    if path.is_file():
        reader = ZAtac(path)
        try:
            if reader.n != length:
                raise ValueError('AZ1 length differs from the ATAC manifest')
            values = reader.read(start, end).copy()
        finally:
            reader.close()
    else:
        path = directory / f'{chrom}.atac.log1p.f16.memmap'
        if path.stat().st_size != length * np.dtype(np.float16).itemsize:
            raise ValueError('ATAC memmap size differs from the manifest')
        values = np.memmap(path, dtype=np.float16, mode='r', shape=(length,))[start:end].copy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Native log1p ATAC must contain finite, nonnegative values')
    return values, path, manifest


def prepare_input(fasta, atac_dir, chrom, start, end, global_p99=None):
    """Compose the validated native input; divide log1p ATAC by one global p99."""
    sequence, length = read_fasta_region(fasta, chrom, start, end)
    native, atac_path, manifest = read_atac_region(atac_dir, chrom, start, end, length)
    cache = Path(atac_dir) / '.global_p99'
    normalization_source = 'explicit --global-p99'
    if global_p99 is None:
        if not cache.is_file():
            raise ValueError('Supply --global-p99 or the source ATAC directory .global_p99 file')
        global_p99 = float(cache.read_text().strip())
        normalization_source = str(cache.resolve())
    if not np.isfinite(global_p99) or global_p99 <= 1e-6:
        raise ValueError('The source-global p99 must be a finite positive value greater than 1e-6')
    dna = encode_dna(sequence)
    normalized = native.astype(np.float32) / np.float32(global_p99)
    result = np.concatenate((dna, normalized[None]), axis=0)
    receipt = {
        'fasta': str(Path(fasta).resolve()),
        'fasta_index_sha256': sha256(str(fasta) + '.fai'),
        'sequence_sha256': hashlib.sha256(sequence).hexdigest(),
        'atac_store': str(atac_path.resolve()),
        'atac_manifest_sha256': sha256(Path(atac_dir) / 'manifest.json'),
        'atac_region_native_sha256': hashlib.sha256(native.tobytes()).hexdigest(),
        'atac_region_native_dtype': str(native.dtype),
        'atac_source_metadata': manifest.get('_source', {}),
        'normalization': 'native log1p ATAC -> float32 / source-global p99; scale=1',
        'global_p99': float(global_p99),
        'normalization_source': normalization_source,
        'dna_channel_order': 'ACGT',
        'unknown_dna_bases': int((dna.sum(axis=0) == 0).sum()),
        'input_sha256': hashlib.sha256(result.tobytes()).hexdigest(),
    }
    return result, receipt
