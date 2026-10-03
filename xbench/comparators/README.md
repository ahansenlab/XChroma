# Comparator inputs

The scoring adapters read prepared caches; inference runs in the corresponding
upstream model environment. They preserve the grid and contact-space conversion
used for each comparison.

| Model | Required cache setting | Stored arrays | Scoring grid |
|---|---|---|---|
| EPCOT | `EPCOT_MICROC_DIR`, `EPCOT_INTACTHIC_DIR`, `EPCOT_RCMC_DIR` (or one `EPCOT_DIR`) | `CELL/chrN.epcot.f16.memmap`, `(n,1000,1000)`, log2(O/E); starts follow GT manifest | Two 500-kb blocks per 1-Mb tile, 1 kb |
| AlphaGenome | `AG_MAP_CACHE` | `CELL/chrN_START.npy`, `(500,500)`, clipped ln(O/E) | 1 Mb, 2 kb |
| Akita | `AKITA_MAP_CACHE` | `CELL/chrN_START.npy`, `(500,500)`, resampled and clipped ln(O/E) | 1 Mb, 2 kb |
| C.Origami | `CO_STORE`, `CO_MODEL`, `CO_CTCF_PROTOCOL` | `CELL_chrN.maps.npz`, `starts` and `corigami` `(n,200,200)` in ln(O/E) | Aligned 1-Mb submatrices, 10 kb |
| Epiphany | `EPIPHANY_DIR` | `CELL/chrN.epiphany.f16.memmap` `(n,100,100)`, linear O/E, plus `chrN.epiphany.manifest.json` | 1 Mb, 10 kb |

All caches must use the coordinates and genome assembly of `GT_TILES`. EPCOT
memmap window order must exactly match the GT manifest. AlphaGenome/Akita
resampling is performed during cache preparation, not during scoring. XChrom
is coarsened in linear O/E space, then per-diagonal renormalized and clipped.

EPCOT and Epiphany use full matrices. AlphaGenome, Akita and C.Origami exclude
|i−j|≤2. GT receives window-level ICE before aggregation. Model-specific
transformations are implemented in `xbench/models/`.

## Model software

- [EPCOT](https://github.com/liu-bioinfo-lab/EPCOT): use the upstream implementation
  and weights corresponding to the selected head. Human and mouse heads are
  distinct; cached maps must retain their generating head identity.
- [AlphaGenome](https://github.com/google-deepmind/alphagenome): export the
  appropriate cell-specific contact prediction before scoring.
- [Akita](https://github.com/calico/basenji/tree/master/manuscripts/akita): use its
  official preprocessing and model output grid before resampling the cache.
- [C.Origami](https://github.com/tanjimin/C.Origami): see [cache preparation](corigami/README.md).
- [Epiphany](https://github.com/arnavmdas/epiphany): see [prediction helper](epiphany/README.md).

Comparator environments and weights are external to this repository. Current
cache adapters and reusable prediction helpers are supplied here; upstream
packages retain their own licensing and installation requirements.
