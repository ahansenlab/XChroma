#!/bin/bash
#SBATCH --job-name=xbench_all
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
XCHROM_REPO_ROOT="${XCHROM_REPO_ROOT:-${XCHROM_CODE_ROOT:-}}"
: "${XCHROM_REPO_ROOT:?Set the absolute repository root}"
export PYTHONPATH="${XCHROM_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "$XCHROM_REPO_ROOT"
XCHROM_PYTHON="${XCHROM_PYTHON:-$(command -v python)}"
if [[ -n "${OUT:-}" && -n "${OUT_BASE:-}" && "$OUT" != "$OUT_BASE" ]]; then
  echo 'OUT and OUT_BASE differ; set one explicit fresh batch directory.' >&2
  exit 2
fi
export OUT="${OUT:-${OUT_BASE:-}}"
: "${OUT:?Set a fresh batch output directory (OUT or OUT_BASE)}"
exec "$XCHROM_PYTHON" -m xbench.launchers.batch serial --out "$OUT"
