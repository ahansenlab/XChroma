#!/bin/bash
# Epiphany signal recipe: RPKM, 10-bp bins, chrX excluded from normalization.
# Usage: EPIPHANY_WORK=/path/to/work bash build_tracks_cell.sh CELL MARK
set -euo pipefail
: "${EPIPHANY_WORK:?Set a working directory containing bam_urls_CELL.tsv}"
CELL="${1:?Provide cell name}"
MARK="${2:?Provide track name}"
D="$EPIPHANY_WORK"
URLS="$D/bam_urls_${CELL}.tsv"
[[ -s "$URLS" ]] || { echo "Missing $URLS" >&2; exit 1; }
mkdir -p "$D/bam/$CELL" "$D/bw_$CELL"
OUT="$D/bw_${CELL}/${MARK}.bw"
[[ ! -e "$OUT" ]] || { echo "Output exists: $OUT" >&2; exit 1; }
BAMS=()
while IFS=$'\t' read -r name acc url; do
  [[ "$name" == "$MARK" ]] || continue
  f="$D/bam/$CELL/${acc}.bam"
  [[ -s "$f" ]] || curl --fail --location --retry 5 --output "$f" "$url"
  BAMS+=("$f")
done < "$URLS"
[[ ${#BAMS[@]} -gt 0 ]] || { echo "No BAMs for $CELL/$MARK" >&2; exit 1; }
M="$D/bam/$CELL/${MARK}_merged.bam"
S="$D/bam/$CELL/${MARK}_sorted.bam"
[[ ! -e "$M" && ! -e "$S" ]] || { echo "Intermediate output exists" >&2; exit 1; }
samtools merge -f -@ 8 -o "$M" "${BAMS[@]}"
samtools sort -@ 8 -o "$S" "$M"
samtools index -@ 8 "$S"
bamCoverage --bam "$S" -o "$OUT" --normalizeUsing RPKM --binSize 10 --ignoreForNormalization chrX -p 8
