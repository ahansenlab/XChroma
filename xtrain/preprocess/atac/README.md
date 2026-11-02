# Accessibility input preparation

XChroma reads one per-base accessibility track per chromosome, stored as `log1p(signal)` in float16 memmaps or losslessly compressed AZ1 files. The model receives this signal divided by the source-specific global p99.

## Convert a bigWig

```bash
python -m pip install numpy pyBigWig
python xtrain/preprocess/atac/bigwig_to_memmap.py \
  /path/to/reference.chrom.sizes /path/to/accessibility.bw /path/to/cell_atac
```

The converter writes `{chrom}.atac.log1p.f16.memmap` files and `manifest.json`. It reads in chunks, represents missing bigWig positions as zero and checks chromosome lengths against the provided reference sizes. The source bigWig must contain nonnegative coverage before log1p; do not supply an already logged track.

## Prepare coverage from paired FASTQ

`scatac_pipeline.sh` aligns reads with bwa-mem2, processes mates, removes duplicates with samtools, and creates a 1-bp coverage track with deepTools before memmap conversion. Paths are configured through `ROOT`, `FQ`, `REF`, `CHROMSIZES`, `OUT` and `MEMMAP`. The reference defaults to mm39; set `REF`, `CHROMSIZES` and, for RPGC, `EFFECTIVE_GENOME` for another assembly. Intermediate BAM and bigWig names retain the script's `.mm39` suffix regardless of the reference supplied.

```bash
ROOT=/path/to/atac_project NORMALIZE=none \
  bash xtrain/preprocess/atac/scatac_pipeline.sh CellA CellB
```

Inputs are `FQ/CellA.R1.fastq.gz` and `FQ/CellA.R2.fastq.gz`. `NORMALIZE` must be selected explicitly: `none` produces unscaled coverage, while `RPGC` uses effective-genome-size normalization and requires the matching `EFFECTIVE_GENOME`. Coverage units affect log1p values; preserve the signal-generation recipe and global-p99 constant associated with an existing model input. The converter records the source file but does not infer its original normalization.

Required command-line tools: bwa-mem2, samtools and bamCoverage. A bwa-mem2 index must already exist beside the reference FASTA.
