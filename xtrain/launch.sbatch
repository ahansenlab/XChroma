#!/bin/bash
#SBATCH --job-name=xchrom_train
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail
export PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

RES="${XCHROM_RES:?Set XCHROM_RES to your training resource directory}"
export GENOME_FA="${GENOME_FA:-${RES}/Mus_musculus.GRCm39.main.fa}"
export GENOME_MMAP="${GENOME_MMAP:-${RES}/Mus_musculus.GRCm39.main.fa.mmap}"
export COOL="${COOL:-${RES}/mESC_1000bp.cool}"
export EXPECTED="${EXPECTED:-${RES}/mesc_expected_cis_1000_rebal.tsv.npy}"
export ATAC_DIR="${ATAC_DIR:-${RES}/scatac_mesc_GSE227320/bulk_mm39}"

export RESOLUTION=1000
export WINDOW_BP=1000000

export BATCH_SIZE=4
export NUM_WORKERS=2
export MAX_STEPS=150000
export LR=0.0005
export DIAG_BAND=2
export VAL_EVERY=500
export SAVE_EVERY=250
export SKIP_INITIAL_VAL=1
export BLACKLIST=""
export HOLDOUT_TEST="chr9,chr10"
export HOLDOUT_VAL="chr8"
export VAL_COORD="chr8:85000000-86000000"

export ATAC_NORM=global
export ATAC_NORM_SCALE=1.0


RUNDIR="${OUTDIR:-${HERE}/runs/xchrom}"
mkdir -p "${RUNDIR}/models" logs
export OUTDIR="${RUNDIR}"

LATEST="${RUNDIR}/models/xchrom_1mb_1kb.latest.ckpt.pt"
if [ -s "$LATEST" ]; then export INIT_FROM="$LATEST"; echo "[resume] $LATEST"; else echo "[fresh] from scratch"; fi

python train.py
