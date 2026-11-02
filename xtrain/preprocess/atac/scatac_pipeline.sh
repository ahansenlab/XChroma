#!/usr/bin/env bash
# Paired FASTQ to per-base accessibility inputs; see README.md.
set -euo pipefail

ROOT=${ROOT:?Set ROOT to your accessibility data project directory}
FQ=${FQ:-$ROOT/incoming}                        # <celltype>.R1.fastq.gz / .R2.fastq.gz
REF=${REF:-$ROOT/ref/mm39.fa}                   # mm39 fasta WITH a bwa-mem2 index beside it
CHROMSIZES=${CHROMSIZES:-$ROOT/ref/mm39.chrom.sizes}
OUT=${OUT:-$ROOT/processed}                     # bam / bigwig
MEMMAP=${MEMMAP:-$ROOT/memmap}                  # one <celltype>_atac subdirectory each
THREADS=${THREADS:-16}
SORT_MEM=${SORT_MEM:-2G}                        # per sort thread; 4 threads -> 4x this
NORMALIZE=${NORMALIZE:?Set NORMALIZE to none or RPGC}                    # RPGC | none
EFFECTIVE_GENOME=${EFFECTIVE_GENOME:-2654621783}   # mm39, deeptools table
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Default cell names; command-line arguments select an explicit list.
CELLTYPES=(Endoderm d1PGCLC early_PGCLC PGCLC_Soma1_early_Soma PGCLC_Soma2_Late_Soma
           d2PGCLC d2EpiLC GSC d1EpiLC ESC cPGCLC)

[[ $NORMALIZE == none || $NORMALIZE == RPGC ]] || { echo "NORMALIZE must be none or RPGC" >&2; exit 1; }

for tool in bwa-mem2 samtools bamCoverage; do
  command -v "$tool" >/dev/null || { echo "[fatal] $tool not on PATH" >&2; exit 1; }
done
python -c 'import pyBigWig, numpy' || { echo "[fatal] pyBigWig/numpy missing" >&2; exit 1; }
[[ -f $REF ]] || { echo "[fatal] no reference at $REF" >&2; exit 1; }
# Check the reference index before alignment.
[[ -f ${REF}.bwt.2bit.64 ]] || { echo "[fatal] no bwa-mem2 index for $REF (run: bwa-mem2 index $REF)" >&2; exit 1; }
[[ -f $CHROMSIZES ]] || { echo "[fatal] no chrom.sizes at $CHROMSIZES" >&2; exit 1; }
mkdir -p "$OUT" "$MEMMAP"

process_one () {
  local C=$1
  local R1=$FQ/$C.R1.fastq.gz R2=$FQ/$C.R2.fastq.gz
  local BAM=$OUT/$C.mm39.sorted.bam
  local FIXM=$OUT/$C.mm39.fixmate.bam
  local DEDUP=$OUT/$C.mm39.dedup.bam
  local BW=$OUT/$C.mm39.atac.bw
  echo "=== [$C] start $(date) ==="
  [[ -f $R1 && -f $R2 ]] || { echo "[skip] $C: fastq missing" >&2; return 0; }

  # align, coordinate-sort
  bwa-mem2 mem -t "$THREADS" "$REF" "$R1" "$R2" \
    | samtools sort -@ 4 -m "$SORT_MEM" -o "$BAM" -
  samtools index -@ 4 "$BAM"

  # Name-sort before fixmate, then coordinate-sort for duplicate removal.
  samtools sort -n -@ 4 -m "$SORT_MEM" "$BAM" \
    | samtools fixmate -m -@ 4 - - \
    | samtools sort -@ 4 -m "$SORT_MEM" -o "$FIXM" -
  samtools markdup -r -@ 4 "$FIXM" "$DEDUP"
  samtools index -@ 4 "$DEDUP"
  rm -f "$FIXM"

  # Generate 1-bp coverage with paired-fragment extension.
  local norm=(--normalizeUsing "$NORMALIZE" --effectiveGenomeSize "$EFFECTIVE_GENOME")
  [[ $NORMALIZE == none ]] && norm=(--normalizeUsing None)
  bamCoverage -b "$DEDUP" -o "$BW" \
    --binSize 1 "${norm[@]}" \
    --extendReads -p "$THREADS" \
    --ignoreForNormalization chrX chrY chrM
  echo "=== [$C] bigWig done $(date) ==="

  python "$HERE/bigwig_to_memmap.py" "$CHROMSIZES" "$BW" "$MEMMAP/${C}_atac"
  echo "=== [$C] memmap done $(date) ==="
}

targets=("${@:-}")
[[ -z ${targets[0]:-} ]] && targets=("${CELLTYPES[@]}")
for C in "${targets[@]}"; do
  process_one "$C"
done
echo "ALL DONE $(date)"

