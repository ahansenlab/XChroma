# Getting started

## Install

Use Python 3.10 or newer and a PyTorch build appropriate for your CPU or CUDA hardware:

```bash
python -m pip install -r xtrain/requirements-inference.txt
```

Training additionally requires `xtrain/requirements.txt`. For cell-specific DNA-to-ATAC training and inference, see [xatac](../xatac/README.md).

## Inputs

Provide an uncompressed FASTA with its matching `.fai` index and an ATAC directory from the same genome assembly. Coordinates are 0-based and half-open. The prediction interval spans exactly 1,000,000 bp and starts on a 1-kb boundary.

An ATAC directory contains:

```text
cell_atac/
  manifest.json
  .global_p99
  chr1.atac.log1p.az
  chr2.atac.log1p.az
  ...
```

The equivalent `{chrom}.atac.log1p.f16.memmap` files are also supported. The manifest can use either `{"chr1": {"length": N}, ...}` or `{"chroms": {"chr1": N, ...}}`. The native files store per-base `log1p` signal before p99 normalization. AZ compression preserves the stored float16/float32 values exactly. The FASTA and ATAC chromosome lengths must match.

The ATAC input is converted to float32 and divided once by the source-specific genome-wide p99 constant in `.global_p99`. An explicit `--global-p99` overrides the cache. Use the constant belonging to that ATAC source; the inference command does not estimate it from the requested window. Do not apply log1p again to a store that already contains log1p signal. DNA channels follow A, C, G, T, with four zeros for unknown bases.

For a new store, the training helper can estimate its global p99 using the same fixed-stride sampling procedure as the data pipeline (approximately one million bases per chromosome):

```bash
ATAC_DIR=/path/to/cell_atac PYTHONPATH=xtrain python -c \
  'import os, data; data.gp99_for_dir(os.environ["ATAC_DIR"])'
```

This uses the inference dependencies and writes `.global_p99`. Existing source-provided constants should be preserved when reproducing a prediction. [ATAC preparation](preprocess/atac/README.md) documents the signal-generation inputs.

## Weights and inference

```bash
gh release download main-model-avg10-30500-35000 \
  --repo ahansenlab/XChroma --pattern '*.ckpt.pt' --dir weights
python xtrain/checkpoints/verify.py --weights-dir weights
python xtrain/predict.py \
  --genome-fa /path/to/hg38.fa --atac-dir /path/to/cell_atac \
  --assembly hg38 --region chr2:182000000-183000000 \
  --weights-dir weights --device cpu --threads 8 --out results/example
```

All ten files are size- and SHA256-checked before inference. Each state dictionary loads strictly into the packaged architecture. Model forwards use float32 without autocast or TF32, and the ten native ln(O/E) output maps are accumulated and averaged in float64. The weights themselves are not averaged.

A full 1-Mb forward has substantial memory requirements on both CPU and GPU. `--check-only` performs weight and input validation without running the network. It writes only a validation manifest, and its output directory must differ from a later inference directory.

## DNA-only prediction

Export cell-specific [xatac predictions](../xatac/README.md#4-predict-atac) with
the fixed training-derived `--xchroma-scale` and matching `--gp99`, then supply
that NPZ instead of an experimental ATAC directory:

```bash
python -m xtrain.predict \
  --genome-fa genome.fa --assembly GRCh38 \
  --region chr8:127000000-128000000 --atac-npz MYC.atac.npz \
  --weights-dir weights --device cuda --out results/MYC_DNA_only
```

The `.npz.json` sidecar must remain beside the NPZ. The command checks the
reference checksum, assembly, interval, one-base coverage and input conversion.
`xchroma_input` is already normalized and is used unchanged as the fifth input
channel; `--global-p99` is therefore not accepted with `--atac-npz`.

## Read the prediction

```python
import numpy as np
with np.load("results/example/prediction.npz", allow_pickle=False) as result:
    contact_logoe = result["prediction"]  # float64, shape (1000, 1000)
```

The array is the native ln(O/E) prediction. It has no display contrast transformation or additional self-normalization. Coordinates, assembly, bin size and model ID are stored alongside it; the JSON manifest records checksums and runtime details.
