#!/bin/bash
set -euo pipefail
XCHROM_REPO_ROOT="${XCHROM_REPO_ROOT:-${XCHROM_CODE_ROOT:-}}"
: "${XCHROM_REPO_ROOT:?Set the absolute repository root}"
if [[ -n "${OUT:-}" && -n "${OUT_BASE:-}" && "$OUT" != "$OUT_BASE" ]]; then
  echo 'OUT and OUT_BASE differ; set one explicit fresh batch directory.' >&2
  exit 2
fi
export OUT="${OUT:-${OUT_BASE:-}}"
: "${OUT:?Set a fresh batch output directory (OUT or OUT_BASE)}"
export PYTHONPATH="${XCHROM_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "$XCHROM_REPO_ROOT"
mode=submit
if [[ "${DRY_RUN:-0}" == 1 ]]; then mode=plan; fi
exec "${XCHROM_PYTHON:-python}" -m xbench.launchers.batch "$mode" --out "$OUT"
