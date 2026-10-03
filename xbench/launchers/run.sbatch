#!/bin/bash
#SBATCH --job-name=xbench
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
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
exec "$XCHROM_PYTHON" -m xbench.run_filtered
