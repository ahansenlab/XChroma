#!/bin/bash
#SBATCH --job-name=xb1
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
: "${XBENCH_BATCH_ROOT:?Set the batch root}"
: "${XBENCH_TASK:?Set the task key}"
exec "$XCHROM_PYTHON" -m xbench.launchers.batch task --out "$XBENCH_BATCH_ROOT" --task "$XBENCH_TASK"
