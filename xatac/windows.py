"""FASTA + verified site bigWig + split BED → compact, shift-ready training records."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyBigWig
import pysam

from .io import encode_bf16, encode_dna, sha256, write_json, write_npz


def read_windows(path, seq_len, flank):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        chrom, start, end, subset = line.split()
        start, end, subset = int(start), int(end), subset.upper()
        if end - start != seq_len or start < flank or subset not in {"TRAIN", "VALID", "TEST"}:
            raise ValueError("BED rows need chrom/start/end/TRAIN|VALID|TEST and seq_len bp plus flanks")
        rows.append(dict(chrom=chrom, start=start - flank, end=end + flank, subset=subset))
    if not rows or len({(r['chrom'], r['start'], r['end']) for r in rows}) != len(rows):
        raise ValueError("Empty or duplicated windows")
    # No DNA-context overlap between different splits. Same-split overlaps are allowed.
    furthest = {}
    for row in sorted(rows, key=lambda r: (r["chrom"], r["start"], r["end"])):
        for split in ("TRAIN", "VALID", "TEST"):
            if split != row["subset"] and row["start"] < furthest.get((row["chrom"], split), -1):
                raise ValueError("Different splits overlap after adding shift flanks")
        key = row["chrom"], row["subset"]
        furthest[key] = max(furthest.get(key, -1), row["end"])
    return rows


def build_windows(fasta, sites, bed, output, seq_len=1048576, flank=2048):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty dataset directory")
    if seq_len < 1024 or seq_len % 1024 or flank < 0:
        raise ValueError("seq_len must be a multiple of 1024, with nonnegative flanks")
    site_meta = json.loads(Path(str(sites) + ".json").read_text())
    if site_meta.get("status") != "PASS" or site_meta.get("signal") != "tn5_insertion_sites":
        raise ValueError("Supply a verified insertion-site track, not fragment coverage")
    if sha256(sites) != site_meta["sha256"]:
        raise ValueError("Site bigWig changed after preparation")
    rows = read_windows(bed, seq_len, flank)
    output.mkdir(parents=True, exist_ok=True)
    with pysam.FastaFile(str(fasta)) as fa, pyBigWig.open(str(sites)) as bw:
        sizes = dict(zip(fa.references, fa.lengths))
        for chrom, length in bw.chroms().items():
            if sizes.get(chrom) != length:
                raise ValueError(f"FASTA/site assembly mismatch: {chrom}")
        for i, row in enumerate(rows):
            chrom, start, end = row["chrom"], row["start"], row["end"]
            if chrom not in bw.chroms() or end > sizes.get(chrom, 0):
                raise ValueError(f"Window outside the prepared reference: {row}")
            code = encode_dna(fa.fetch(chrom, start, end))
            signal = bw.values(chrom, start, end, numpy=True)
            if np.isinf(signal).any() or (signal[np.isfinite(signal)] < 0).any():
                raise ValueError("Invalid site bigWig values")
            # sites.py writes a sparse count track: absent entries mean zero counts.
            signal = np.nan_to_num(signal, nan=0.)
            path = output / "records" / row["subset"] / f"{i:06d}.npz"
            write_npz(path, sequence_code=code, atac_bf16=encode_bf16(signal))
            row.update(path=str(path.relative_to(output)), length=len(code), target_length=len(code),
                       target_offset=0, non_acgt_bases=int((code == 4).sum()),
                       bytes=path.stat().st_size, sha256=sha256(path))
            print(f"[windows] {i + 1}/{len(rows)} {chrom}:{start}-{end} {row['subset']}", flush=True)
    write_json(output / "records.json", rows)
    metadata = dict(schema_version=1, status="PASS", assembly=site_meta["assembly"],
                    signal="tn5_insertion_sites", seq_len=seq_len, flank_bp=flank,
                    track_mean=site_meta["track_mean"], records=len(rows),
                    records_sha256=sha256(output / "records.json"),
                    fasta_sha256=sha256(fasta), sites_sha256=site_meta["sha256"],
                    split_bed_sha256=sha256(bed), sites_metadata=site_meta,
                    storage="uint8 DNA code and uint16 round-to-nearest-even bfloat16 target bits",
                    subsets={s: sum(r["subset"] == s for r in rows) for s in ("TRAIN", "VALID", "TEST")},
                    split_policy="User-supplied BED; no DNA-context overlap across subsets")
    write_json(output / "dataset.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True, help="Indexed matching reference FASTA")
    parser.add_argument("--sites", type=Path, required=True, help="sites.py output with its .json sidecar")
    parser.add_argument("--windows", type=Path, required=True, help="Four-column BED: chrom start end subset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=1048576)
    parser.add_argument("--flank", type=int, default=2048)
    args = parser.parse_args()
    build_windows(args.fasta, args.sites, args.windows, args.output, args.seq_len, args.flank)


if __name__ == "__main__":
    main()
