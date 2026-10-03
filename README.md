# XChrom

High-resolution, cell-type-specific 3D genome prediction from DNA sequence and chromatin accessibility.

XChrom predicts a 1 Mb contact map at 1 kb resolution. This repository contains the model, training and inference code in **[xtrain](xtrain/)**, and whole-map, fine-structure and loop-strength evaluation in **[xbench](xbench/)**.

## Predict a contact map

Install the inference dependencies and download the [model weights](https://github.com/YanXu2077/X-Chrom/releases/tag/main-model-avg10-30500-35000):

```bash
python -m pip install -r xtrain/requirements-inference.txt
gh release download main-model-avg10-30500-35000 \
  --repo YanXu2077/X-Chrom --pattern '*.ckpt.pt' --dir weights
python -m xtrain.checkpoints.verify --weights-dir weights
```

Supply a matching reference genome and chromatin-accessibility input:

```bash
python -m xtrain.predict \
  --genome-fa /path/to/hg38.fa \
  --atac-dir /path/to/cell_atac \
  --assembly hg38 \
  --region chr2:182000000-183000000 \
  --weights-dir weights \
  --device cuda \
  --out outputs/example
```

CPU inference is available with `--device cpu`. ATAC inputs use native log1p fragment counts in an AZ or float16 memmap store, normalized by the cell's source-global p99. See [input preparation and inference](xtrain/GETTING_STARTED.md) for the file formats, normalization and output arrays.

## Evaluate predictions

```bash
python -m pip install -r xbench/requirements.txt
python -m xbench.loop_strength --help
```

[xbench](xbench/README.md) evaluates prepared predictions against measured maps, including whole-map scores, fine structure extracted with difference-of-Gaussians filtering, multiscale insulation and GT-defined loop strengths. Its adapters support EPCOT, AlphaGenome, C.Origami, Epiphany and Akita comparisons.

For model training, ATAC preparation and target-specific distillation, see [xtrain](xtrain/README.md).
