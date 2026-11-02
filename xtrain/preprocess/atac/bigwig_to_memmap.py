#!/usr/bin/env python
"""Convert a nonnegative per-base bigWig signal to native log1p float16 ATAC stores.
Usage: bigwig_to_memmap.py <chrom.sizes> <input.bw> <outdir>.
The converter preserves the input coverage units and records the source file.
"""
import json
import os
import sys

import numpy as np
import pyBigWig

# Read 32 million bases per chunk to bound memory use.
CHUNK = 32_000_000


def main_chroms(path):
    """Read chr<number>, chrX and chrY entries, preserving chrom.sizes order."""
    out = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            name, length = parts[0], parts[1]
            if name.startswith("chr") and (name[3:].isdigit() or name[3:] in ("X", "Y")):
                out[name] = int(length)
    return out


def convert(chrom_sizes, bw_path, outdir, chunk=CHUNK):
    os.makedirs(outdir, exist_ok=True)
    sizes = main_chroms(chrom_sizes)
    if not sizes:
        raise SystemExit("[fatal] no chr1..chrY entries in %s" % chrom_sizes)
    bw = pyBigWig.open(bw_path)
    have = bw.chroms()
    manifest = {}
    for chrom, length in sizes.items():
        if chrom not in have:
            print("[skip] %s not in the bigWig" % chrom, flush=True)
            continue
        if have[chrom] != length:
            # Require the same chromosome length in both reference descriptions.
            raise SystemExit("[fatal] %s: bigWig says %d, chrom.sizes says %d -- assembly mismatch"
                             % (chrom, have[chrom], length))
        path = os.path.join(outdir, "%s.atac.log1p.f16.memmap" % chrom)
        mm = np.memmap(path, dtype=np.float16, mode="w+", shape=(length,))
        # Request NumPy arrays directly and convert one chunk at a time.
        for start in range(0, length, chunk):
            end = min(start + chunk, length)
            sig = bw.values(chrom, start, end, numpy=True).astype(np.float32)
            np.nan_to_num(sig, copy=False)          # unmapped bases come back as NaN
            mm[start:end] = np.log1p(sig).astype(np.float16)
        mm.flush()
        del mm
        manifest[chrom] = {"length": length, "dtype": "float16",
                           "encoding": "log1p(bigwig_signal)"}
        print("[out] %s len=%d" % (chrom, length), flush=True)
    bw.close()
    if not manifest:
        raise SystemExit("[fatal] the bigWig had none of the expected chromosomes")
    # Record the source signal and reference chromosome sizes.
    manifest["_source"] = {"bigwig": os.path.abspath(bw_path),
                           "chrom_sizes": os.path.abspath(chrom_sizes)}
    with open(os.path.join(outdir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("DONE %s (%d chromosomes)" % (outdir, len(manifest) - 1), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: bigwig_to_memmap.py <chrom.sizes> <in.bw> <outdir>")
    convert(sys.argv[1], sys.argv[2], sys.argv[3])
