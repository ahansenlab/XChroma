# C.Origami

## Score a prepared cache

```bash
export CO_STORE=/path/to/corigami_cache
export CO_MODEL=released
export CO_CTCF_PROTOCOL=humanctcf30M
export LINES=corigami
python -m xbench.comparators.corigami.stores
python -m xbench.run_filtered
```

Also configure the common XChrom/GT/output settings in [xbench](../../README.md).
`CO_MODEL` is `released` or `retrained_53p5b`; these remain distinct output keys.
`CO_CTCF_PROTOCOL=humanctcf30M` identifies the approximately 30-million-read
human CTCF recipe. `unsampled_bam` identifies the earlier unsampled input.
`CO_GM12878_STORE` optionally supplies a separate matching-weight cache for the
original official GM12878 inputs. Each selected cell needs
`CELL_chrN.maps.npz` with `starts` and `corigami` arrays.

## Generate maps

Install [C.Origami](https://github.com/tanjimin/C.Origami) in its own environment.
The prepared-input helper preserves the benchmark output transformation:

```bash
python -m xbench.comparators.corigami.predict   --cell HCT116 --checkpoint /path/to/model.ckpt   --dna /path/to/hg38/dna_sequence --ctcf /path/to/ctcf_log2fc.bw   --atac /path/to/atac.bw --windows /path/to/reference_cache   --out /path/to/new_corigami_cache
```

Only the `starts` arrays are read from the reference cache. For each aligned
2-Mb window, the helper predicts the 2,097,152-bp input, applies nonnegative
`expm1`, removes six bins at each edge, resizes the resulting 244×244 matrix to
200×200, and computes per-diagonal ln(O/E). The scorer extracts aligned 1-Mb
submatrices at 10 kb.

CTCF preprocessing follows the official RPKM/bigwigCompare signal recipe.
For the `humanctcf30M` caches, aligned IP/control BAMs were subsampled with
samtools seed 2021 when above the target depth, retaining lower-depth inputs;
signal preparation used 1-bp `bamCoverage --normalizeUsing RPKM` followed by
`bigwigCompare`. The predictor accepts the resulting explicit bigWig paths;
it does not select input experiments or normalize raw reads.
