# Contact-map metrics

Evaluate prepared NumPy arrays with the same genomic interval, bin size and
matrix shape. Inputs are numerical contact maps, independent of figure colour
scales. Dataset preparation and model inference are separate from these metrics.

## Whole map

```python
from xbench import metrics

# pred and experimental: square NumPy arrays on the same comparison grid
scores = {
    name: metrics.load(name).score(pred, experimental)
    for name in metrics.all_names() if name != "insulation"
}
name, insulation = metrics.load("insulation").make(window_bins=5, res_kb=1)
scores[name] = insulation(pred, experimental)
```

The legacy API argument name `gt` denotes the experimental comparison map.
It is retained for compatibility with existing keyword calls.

| Module | Quantity | Better |
|---|---|---|
| `spearman` | Spearman correlation of paired contact pixels | Higher |
| `pearson` | Pearson correlation of paired contact pixels | Higher |
| `scc` | HiCRep stratum-adjusted correlation | Higher |
| `mse` | Mean squared pixel error; whole maps only | Lower |
| `ssim` | Structural similarity, with a data range derived from the experimental map | Higher |
| `insulation` | Spearman correlation of cooltools diamond-insulation tracks | Higher |
| `distance_enrichment` | Spearman correlation of distance-enrichment tracks | Higher |
| `eigenvector` | Spearman correlation of first-principal-component tracks | Higher |
| `contact_directionality` | Spearman correlation of contact-directionality tracks | Higher |

The XChroma benchmark uses comparison-grid natural-log observed/expected maps,
clipped to [−2, 2], with the comparison's selected diagonal mask already applied.
Supply `NaN` for missing pixels and finite values elsewhere. These functions do
not convert counts to O/E, resize maps or select windows. Apply the comparison's
selected diagonal mask before scoring. Insulation additionally excludes
separations below two bins; SCC uses the strict upper triangle.
The insulation function exponentiates its input before computing diamond scores,
so it specifically expects natural-log O/E rather than linear O/E or log10.

Missingness follows each metric's preserved numerical operator. Pearson and
Spearman score paired valid pixels; distance-enrichment, directionality and
PCA synchronize missing positions between the maps. SCC, SSIM and PCA fill the
shared missing positions with zero internally. Distance enrichment omits zero
values when summarizing each diagonal; directionality uses NaN-skipping column
sums. Keep NaNs in the input so each operator can apply its intended paired mask.

SSIM passes the experimental map first to the underlying operator and obtains
`data_range` from that map after applying the
pair-specific missing mask. For a model comparison, summarize scores over the
same windows with finite values for every model. Use this paired-window rule
for **both whole-map and fine-structure SSIM**. MSE is evaluated on whole maps.

## Fine structure (DoG)

```python
from xbench.metrics.fine_structure import dog
from xbench.metrics import spearman, ssim

pred_fine = dog(pred, res_kb=1)
experimental_fine = dog(experimental, res_kb=1)
rho = spearman.score(pred_fine, experimental_fine)
structural_similarity = ssim.score(pred_fine, experimental_fine)
```

Difference-of-Gaussians uses mask-normalized Gaussian smoothing with physical
widths `sigma1 = 3.2 kb` and `sigma2 = sigma1 × 2^(1/10)`, with nearest boundary
handling. Apply it on the 1- or 2-kb comparison grid. NaN and exact-zero input
locations remain NaN in the filtered result, matching the current benchmark's
mask convention; exact zero is therefore excluded even when it represents O/E=1.
MSE remains a whole-map metric.

## Multiscale insulation

```python
from xbench.metrics import insulation

# Example: four physical scales on a 10-kb grid.
res_kb = 10
insulation_scores = {}
for window_kb in (50, 100, 150, 250):
    name, metric = insulation.make(window_bins=window_kb // res_kb, res_kb=res_kb)
    insulation_scores[name] = metric(pred, experimental)
```

`window_bins` must be an integer. The scale equals `window_bins × res_kb`; five bins are 5 kb on a 1-kb grid and
50 kb on a 10-kb grid. Scores use cooltools 0.7.1 diamond insulation with
`ignore_diags=2` and median normalization, followed by Spearman correlation of
the finite paired tracks. Windows with too few valid positions or constant
tracks return NaN.

## Focal loop strength

```python
import numpy as np
from xbench.metrics import loop_strength

# Upper-triangle bin indices of independently supplied, experimentally defined loop sites.
ia = np.array([10, 20, 30], dtype=int)
ib = np.array([30, 40, 50], dtype=int)
r = loop_strength.score(pred, experimental, ia, ib, res_kb=1)

# For a matched three-map comparison, retain sites jointly across all maps.
values = loop_strength.score_maps(experimental, pred, comparison, ia, ib, res_kb=1)
summary = loop_strength.correlations(values["gt"], values["ours"], values["opp"])
```

The legacy return key `values["gt"]` contains strengths from the experimental
map and is retained for API compatibility.

Maps must already be on the 1- or 2-kb scoring grid in clipped ln(O/E) [−2, 2].
DoG is applied internally. Each 7×7 patch is its central 3×3 mean minus the mean
of its four seven-pixel edges (corners count twice). At least 80% of the patch
must be finite. All compared maps use the same jointly retained experimental loop sites.
Strength vectors are cast to float32 before Pearson correlation, as in the
benchmark. `score_maps` returns these vectors and their retained coordinates.
This module evaluates supplied loop sites; it does not call loops.

## Sources

[Pollard contact-map scoring](vendor/pollard/NOTICE.md) supplies the retained
Spearman, SCC, MSE, SSIM, distance-enrichment and directionality operators.
The PCA metric uses the existing scikit-learn implementation; Pearson uses
SciPy, and insulation uses cooltools. The vendored Pollard source files are
retained byte-for-byte with attribution.
