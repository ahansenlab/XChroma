# xbench

Score XChrom and comparison-model predictions on matched contact-map grids.
The benchmark includes **whole map**, **fine structure (DoG)** and
**GT-defined focal loop strength** comparisons. All commands below run from the
repository root and consume explicitly selected input files.

## Installation

Use a separate CPU scoring environment:

```bash
python -m venv .venv-bench
source .venv-bench/bin/activate
python -m pip install -r xbench/requirements.txt
python -m unittest discover -s xbench/tests -v
```

The scoring environment is independent of comparator inference environments.
Prepared prediction caches, genomic inputs and comparator weights are supplied
separately. See [comparators](comparators/README.md) for model sources and formats.

## Whole map and fine structure

```bash
export OURS_PRED_BASE=/path/to/xchrom_predictions
export XCHROM_CACHE_MANIFEST=/path/to/cache_manifest.json
export GT_TILES=/path/to/gt_tiles
export EPCOT_INTACTHIC_DIR=/path/to/epcot_intacthic_predictions
export OUT=/path/to/new_scores
export LINES=epcot:intacthic
export PERWIN=1
python -m xbench.run_filtered
```

`OUT` must be empty. `LINES` selects `epcot:intacthic`, `epcot:microc`,
`epcot:rcmc`, `alphagenome`, `akita`, `corigami` or `epiphany` (comma-separated).
Set `ONLY_CELL`, `ONLY_CHROM` and `LIMIT` to restrict a run. `ONLY_METRIC` accepts
comma-separated metric names. `INSUL_BINS=5` selects a five-bin insulation window;
by default the driver evaluates multiple physical scales up to 250 kb.

For whole-map-only scoring:

```bash
export MODELS=epcot
export EPCOT_HEAD=intacthic
python -m xbench.run
```

Use a fresh `OUT` for this second command. Its selectors are `MODELS`, `METRICS`,
`CELLS`, `CHROMS` and `LIMIT`. Both drivers write per-window scores and matched
comparison summaries; the DoG driver writes `summary.tsv` and `perwin.tsv`.
The serialized filter names are `raw` (whole map) and `dog` (fine structure).

### XChrom cache identity

`XCHROM_CACHE_MANIFEST` is a JSON file with this structure:

```json
{
  "prediction_root": "/path/to/xchrom_predictions",
  "release_id": "xchrom_avg10_present_30500_35000",
  "members": [
    {"step": 30500, "sha256": "<checkpoint SHA256>", "weight": 0.1}
  ],
  "ensemble": {
    "operator": "arithmetic_mean_outputs",
    "space": "natural_log_observed_expected"
  }
}
```

Include all ten members, in the exact order and with the hashes in
[`current_model.json`](../xtrain/checkpoints/current_model.json). The cache must
have been generated from those checkpoints with the matching xtrain model.
`prediction_root` may be relative to the manifest file. Each cell directory
contains `chrN.pred.manifest.json` with a `starts` list and the corresponding
`chrN.pred.f16.memmap`, shaped `(len(starts), 1000, 1000)` in natural-log O/E.
Chromosome starts use zero-based genomic coordinates in bp.

`GT_TILES` has the same cell/chromosome organization. Files are
`chrN.w1000k.s1000k.manifest.json` and `chrN.w1000k.s1000k.f16.memmap`, shaped
`(len(starts), 1000, 1000)` and containing observed counts at 1 kb. Comparator
formats and required environment variables are listed in
[comparators/README.md](comparators/README.md). Inputs within a comparison must
use the same genome assembly, coordinates and retained windows.

Each run records model declarations, adapter/source hashes, configured input
paths, software versions and scoring options alongside the scores.

## Metrics

| Name | Quantity |
|---|---|
| `spearman` | Rank correlation of contact pixels |
| `pearson` | Linear correlation of contact pixels |
| `scc` | Stratum-adjusted contact correlation |
| `mse` | Mean squared pixel error; lower is better |
| `ssim` | Structural similarity, with GT-derived data range |
| `insulation` | Spearman correlation of cooltools diamond-insulation tracks |
| `distance_enrichment` | Agreement of contacts summarized by genomic separation |
| `eigenvector` | Agreement of the first eigenvector tracks |
| `contact_directionality` | Agreement of upstream/downstream contact-direction tracks |

The preserved [Pollard source](metrics/vendor/pollard/NOTICE.md) supplies seven
metric definitions. Pearson uses SciPy; insulation uses
`cooltools.api.insulation._insul_diamond_dense`. SSIM is evaluated on whole maps
and fine structure; MSE is whole-map only. Means use windows where both model
scores are finite. Compare models within their matched adapter grids.

DoG uses mask-normalized Gaussian smoothing with `sigma1=3.2 kb`,
`sigma2/sigma1=2^(1/10)` and nearest boundaries. NaN and exact-zero masked
locations remain missing. EPCOT, AlphaGenome and Akita use DoG at their 1/2-kb
grids. C.Origami and Epiphany use whole-map scoring at 10 kb. Missing values are
preserved by default (`KEEP_MISSING=1`). The input transforms and numerical
scoring operators are independent of figure colour scales.

## GT-defined loop strength

Each 7×7 DoG patch is summarized as its central 3×3 mean minus the mean of its
four seven-pixel edges (corners count twice). A patch needs at least 80% finite
pixels. GT, XChrom and the comparison model are evaluated at shared GT-defined
upper-triangle loop sites, then compared using Pearson correlation.

Create a manifest and one NPZ per window:

```json
{
  "input_space": "clipped_natural_log_observed_expected",
  "windows": [
    {"npz": "window.npz", "sha256": "<file SHA256>",
     "cell": "GM12878", "line": "epcot_intacthic", "res_kb": 1,
     "chrom": "chr1", "window_start": 1000000}
  ]
}
```

The NPZ contains matching square `gt`, `ours`, `opp` arrays, transformed to the
comparison grid and clipped ln(O/E) range [−2, 2], plus integer vectors `ia` and
`ib` with `ia < ib`. Paths are relative to the manifest. File hashes are checked.

```bash
python -m xbench.loop_strength maps --manifest /path/to/manifest.json --out /path/to/new_loop_scores
python -m xbench.loop_strength replay --input-dir /path/to/new_loop_scores
```

Outputs include retained sites, strengths, input hashes and grouped Pearson
correlations. Replay accepts NPZs containing `gt`, `ours`, `opp` scalar vectors,
and scalar `cell` and `line` labels.

## Batch scoring

The full plan includes four non-EPCOT comparisons and three EPCOT heads across
four cells. Configure each input directory explicitly; multi-head jobs use
`EPCOT_MICROC_DIR`, `EPCOT_INTACTHIC_DIR` and `EPCOT_RCMC_DIR`. A global
`EPCOT_DIR` is reserved for one-head runs.

```bash
python -m xbench.launchers.batch plan --out /path/to/new_batch
python -m xbench.launchers.batch serial --out /path/to/new_batch
# Alternatively, in an activated environment on a Slurm cluster:
python -m xbench.launchers.batch submit --out /path/to/new_batch
```

Serial and Slurm are alternatives and require different fresh output paths.
Slurm templates use site defaults for partition/account; set those through your
Slurm configuration. `XCHROM_PYTHON` may select an absolute interpreter.
Merging verifies task completion, file hashes, matched identities and unique
rows. Summarize the merged table with:

```bash
python -m xbench.aggregate_full10 --summary /path/to/new_batch/merged/summary.tsv
```
