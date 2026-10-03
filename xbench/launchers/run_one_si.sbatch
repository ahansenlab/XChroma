#!/bin/bash
#SBATCH --job-name=xbsi
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
XCHROM_REPO_ROOT="${XCHROM_REPO_ROOT:-${XCHROM_CODE_ROOT:-}}"
: "${XCHROM_REPO_ROOT:?Set the absolute repository root}"
export PYTHONPATH="${XCHROM_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "$XCHROM_REPO_ROOT"
XCHROM_PYTHON="${XCHROM_PYTHON:-$(command -v python)}"
: "${OUT:?Set a fresh scoring output directory}"
: "${LINES:?Set explicit model lines, e.g. epcot:intacthic}"
export PERWIN=1
export ONLY_METRIC=ssim,insulation
exec "$XCHROM_PYTHON" -m xbench.run_filtered
