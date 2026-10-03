#!/bin/bash
#SBATCH --job-name=xbagg
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
XCHROM_REPO_ROOT="${XCHROM_REPO_ROOT:-${XCHROM_CODE_ROOT:-}}"
: "${XCHROM_REPO_ROOT:?Set the absolute repository root}"
export PYTHONPATH="${XCHROM_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "$XCHROM_REPO_ROOT"
XCHROM_PYTHON="${XCHROM_PYTHON:-$(command -v python)}"
: "${XBENCH_BATCH_ROOT:?Set the batch root}"
exec "$XCHROM_PYTHON" -m xbench.launchers.batch merge --out "$XBENCH_BATCH_ROOT"
