"""Deduplicated, indexed BAM libraries → normalized 1-bp insertion-site bigWig."""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile

import numpy as np
import pyBigWig
import pysam

from .io import sha256, write_json


def site_position(read, input_plus_shift=0, input_minus_shift=0):
    """ChromBPNet convention: shift BED endpoints, then count their 5' base."""
    if read.is_reverse:
        return read.reference_end - 1 - 4 - input_minus_shift
    return read.reference_start + 4 - input_plus_shift


def count_chrom(bams, chrom, length, mapq, shifts):
    counts = np.zeros(length, dtype=np.uint32)
    stats = []
    for bam in bams:
        tally = Counter()
        if chrom not in bam.references or bam.get_reference_length(chrom) != length:
            raise ValueError(f"BAM/reference chromosome mismatch: {chrom}")
        for read in bam.fetch(chrom):
            tally["examined"] += 1
            # samtools view -q 30 -f 2 -F 3852, one insertion for each retained read.
            if not read.is_proper_pair or read.flag & 3852 or read.mapping_quality < mapq:
                tally["filtered"] += 1
                continue
            position = site_position(read, *shifts)
            if not 0 <= position < length:
                tally["outside_reference"] += 1
                continue
            if counts[position] == np.iinfo(np.uint32).max:
                raise OverflowError(f"Site counter overflow at {chrom}:{position}")
            counts[position] += 1
            tally["retained_sites"] += 1
        stats.append(dict(tally))
    return counts, stats


def generate_sites(bam_paths, chrom_sizes, output, assembly, input_shifts,
                   normalization_total=100000000.0, mapq=30, tmpdir=None):
    output = Path(output)
    sidecar = Path(str(output) + ".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("Refusing to replace an existing site track")
    if normalization_total <= 0 or not np.isfinite(normalization_total) or mapq < 0:
        raise ValueError("Positive finite normalization total and nonnegative MAPQ required")
    chroms = {}
    for line in Path(chrom_sizes).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        chrom, size = line.split()
        if chrom in chroms or int(size) <= 0:
            raise ValueError("Invalid or repeated chromosome")
        chroms[chrom] = int(size)
    bam_paths = [Path(path).resolve() for path in bam_paths]
    if not chroms or not bam_paths or len(set(bam_paths)) != len(bam_paths):
        raise ValueError("Supply nonempty chromosomes and distinct per-library BAMs")
    output.parent.mkdir(parents=True, exist_ok=True)
    qc, raw_total, nonzero = {}, 0, 0
    with ExitStack() as stack, tempfile.TemporaryDirectory(dir=tmpdir, prefix="xatac-sites-") as tmp:
        bams = [stack.enter_context(pysam.AlignmentFile(str(path), "rb")) for path in bam_paths]
        for bam in bams:
            bam.check_index()
        for i, (chrom, length) in enumerate(chroms.items()):
            counts, qc[chrom] = count_chrom(bams, chrom, length, mapq, input_shifts)
            raw_total += int(counts.sum(dtype=np.uint64))
            nonzero += int(np.count_nonzero(counts))
            np.save(Path(tmp) / f"{i}.npy", counts)
            print(f"[sites] {chrom}: {int(counts.sum(dtype=np.uint64))} insertions", flush=True)
            del counts
        if raw_total == 0:
            raise ValueError("No retained insertion sites")
        scale, actual_sum = normalization_total / raw_total, 0.0
        partial = output.with_name(output.name + ".partial")
        with pyBigWig.open(str(partial), "w") as bw:
            bw.addHeader(list(chroms.items()))
            for i, (chrom, length) in enumerate(chroms.items()):
                counts = np.load(Path(tmp) / f"{i}.npy", mmap_mode="r")
                for start in range(0, length, 1000000):
                    values = (counts[start:start + 1000000].astype(np.float64) * scale).astype(np.float32)
                    actual_sum += float(values.sum(dtype=np.float64))
                    positions = np.flatnonzero(values)
                    if len(positions):
                        starts = positions + start
                        bw.addEntries([chrom] * len(starts), starts.tolist(),
                                      ends=(starts + 1).tolist(), values=values[positions].tolist())
        if abs(actual_sum / normalization_total - 1) > 1e-6:
            raise RuntimeError("Insertion normalization did not conserve the requested total")
        partial.replace(output)
    metadata = dict(schema_version=1, status="PASS", assembly=assembly,
                    signal="tn5_insertion_sites", resolution_bp=1, coordinate_convention="0-based half-open",
                    path=output.name, sha256=sha256(output), chromosomes=chroms,
                    sources=[dict(path=str(Path(p).resolve()), bytes=Path(p).stat().st_size,
                                  mtime_ns=Path(p).stat().st_mtime_ns) for p in bam_paths],
                    source_identity="BAM paths/sizes/mtimes; BAM content hashes not computed",
                    input_shifts=list(input_shifts), plus_site="reference_start + 4 - input_plus_shift",
                    minus_site="reference_end - 5 - input_minus_shift", mapq=mapq,
                    require_flags=2, exclude_flags=3852,
                    deduplication="Input libraries must be independently duplicate-marked or duplicate-removed",
                    raw_site_count=raw_total, normalization_total=normalization_total,
                    sum_float32=actual_sum, nonzero_bases=nonzero, track_mean=actual_sum / nonzero,
                    aggregation="Sum raw per-library sites then apply one global scale; equivalent to mean then scale",
                    chrom_sizes_sha256=sha256(chrom_sizes), qc=qc)
    write_json(sidecar, metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam", nargs="+", type=Path, required=True)
    parser.add_argument("--chrom-sizes", type=Path, required=True, help="Selected nuclear chromosomes, two columns")
    parser.add_argument("--assembly", required=True)
    parser.add_argument("--input-shifts", nargs=2, type=int, required=True, metavar=("PLUS", "MINUS"),
                        help="Previously applied shifts (0 0 for verified unshifted BAMs); never guessed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normalization-total", type=float, default=100000000.)
    parser.add_argument("--mapq", type=int, default=30)
    parser.add_argument("--tmpdir", type=Path)
    args = parser.parse_args()
    result = generate_sites(args.bam, args.chrom_sizes, args.output, args.assembly, args.input_shifts,
                            args.normalization_total, args.mapq, args.tmpdir)
    print(json.dumps({k: result[k] for k in ("raw_site_count", "track_mean", "sha256")}, indent=2))


if __name__ == "__main__":
    main()
