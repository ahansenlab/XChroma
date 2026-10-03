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

# 32 Mb of float32 per read (128 MB). Whole-chromosome reads are what the recorded version did and
# they cost several GB on chr1: bw.values() without numpy=True returns a PYTHON LIST of ~195
# million floats (~28 bytes each) before np.array copies it.
CHUNK = 32_000_000


def main_chroms(path):
    """chr1..chr19, chrX, chrY from a chrom.sizes file, in file order."""
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
            # A length disagreement means the bigWig and the chrom.sizes are different assemblies
            # or patch levels. Writing anyway would silently shift every coordinate downstream.
            raise SystemExit("[fatal] %s: bigWig says %d, chrom.sizes says %d -- assembly mismatch"
                             % (chrom, have[chrom], length))
        path = os.path.join(outdir, "%s.atac.log1p.f16.memmap" % chrom)
        mm = np.memmap(path, dtype=np.float16, mode="w+", shape=(length,))
        # CHANGED (1): chunked, and numpy=True. The recorded version read each whole chromosome as
        # a Python list. Same output, bounded memory.
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
    # CHANGED (2): the manifest records which bigWig this came from. Without it a memmap directory
    # carries no trace of its input, and the RPGC-versus-count question above cannot be answered
    # after the fact -- which is exactly the situation the germline directories are in now.
    manifest["_source"] = {"bigwig": os.path.abspath(bw_path),
                           "chrom_sizes": os.path.abspath(chrom_sizes)}
    with open(os.path.join(outdir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("DONE %s (%d chromosomes)" % (outdir, len(manifest) - 1), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: bigwig_to_memmap.py <chrom.sizes> <in.bw> <outdir>")
    convert(sys.argv[1], sys.argv[2], sys.argv[3])
