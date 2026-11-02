# XChroma training and inference

XChroma predicts a 1-Mb contact map at 1-kb resolution from DNA sequence and chromatin accessibility. The pretrained model combines ten output maps with equal weights. Its exact checkpoints are listed in [current_model.json](checkpoints/current_model.json).

## Predict a region

Place the ten local checkpoints listed in `checkpoints/current_model.json` in
`weights/`. Checkpoint files are not distributed in this repository.

```bash
python -m pip install -r xtrain/requirements-inference.txt
python xtrain/checkpoints/verify.py --weights-dir weights
python xtrain/predict.py \
  --genome-fa /path/to/hg38.fa \
  --atac-dir /path/to/cell_atac \
  --assembly hg38 --region chr2:182000000-183000000 \
  --weights-dir weights --device cuda --out results/example
```

Use `--device cpu` to run without a GPU. Input genomes and accessibility datasets are supplied separately. See [Getting started](GETTING_STARTED.md) for the input contract and normalization.

Each output directory contains `prediction.npz` and a `manifest.json` with input checksums, normalization, model members and runtime settings. Predictions are native natural-log observed/expected values. The inference entry does not apply a second O/E normalization, clipping or display scaling. `--check-only` validates inputs and all ten weight files without a model forward pass.

## Architecture

| Component | Configuration |
|---|---|
| Input | Four ACGT one-hot channels and one aligned ATAC channel at 1-bp resolution |
| Encoder | Kernel size 9; widths 64, 96, 128, 128, 128, 128, 128; two convolutions in each downsampling path and two in its residual branch |
| Pooling | Fixed anti-aliased max pooling, factors 5, 5, 5, 4, 2, 1 |
| Bridge | Pairwise sum of the 128-channel sequence representation |
| Decoder | 21 depthwise DCLS blocks; 64 channels; 7 × 7 kernel extent; nine learned sampling locations per channel |
| Dilation | 1, 2, 4, 8, 16, 32, 64 repeated three times |
| Output | Symmetric 1000 × 1000 contact map |

`model.py:build_net()` is the checkpoint-matched architecture. DCLS tap positions use a separate optimizer group with five times the base learning rate, no weight decay, and post-update position clamping.

## Training

Training requires PyTorch 2.3 or newer. Install the dependencies with `python -m pip install -r xtrain/requirements.txt` and configure:

| Variable | Input |
|---|---|
| `GENOME_FA` | Indexed reference FASTA |
| `GENOME_MMAP` | Optional raw float32 genome cache (ACGT channels; lexical chromosome order) |
| `ATAC_DIR` | Per-base log1p ATAC stores, manifest and `.global_p99` |
| `COOL` | Balanced 1-kb contact target in Cooler format |
| `EXPECTED` | NumPy array of natural-log distance-expected contact values |
| `OUTDIR` | Directory for checkpoints and training records |

Run `python xtrain/train.py` in a CUDA environment. Default holdout chromosomes are chr8 for validation and chr9/chr10 for the recorded held-out score. The default training configuration uses float32, global ATAC normalization and no reverse-complement augmentation. The model input window and output grid remain 1 Mb and 1 kb. Checkpoints and numerical training records are written to `OUTDIR`.

Training reads indexed FASTA and balanced Cooler targets through `x_utils.py` and uses Selene’s batch loader. Unknown or padded training bases have 0.25 in each DNA channel. DNA-only jitter leaves the ATAC and target coordinates fixed. A genome cache must match the exact FASTA. The inference encoding is specified in [Getting started](GETTING_STARTED.md).

## Additional workflows

- [Cell-specific DNA-to-ATAC prediction](../xatac/README.md)
- [ATAC preparation](preprocess/atac/README.md)
- [Benchmarking](../xbench/README.md)
- [External inputs and dependencies](EXTERNAL_DEPENDENCIES.md)
