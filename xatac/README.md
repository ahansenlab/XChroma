# xatac

Cell-specific DNA-to-ATAC prediction for XChroma. This package provides a
convolutional model trained on experimental ATAC-seq, with site-label
preparation, training and inference. Train one model per cell type.

## Install

Use Python 3.10 or later on Linux or macOS. For GPU training, install a PyTorch
build matching your CUDA installation. Run from the repository root:

```bash
python -m pip install -r xatac/requirements.txt
```

A fresh installation of the xtrain and xatac inference dependencies took about 82 s on
macOS 26.1 arm64 with Python 3.12.2, excluding model and reference downloads.

To run the tests:

```bash
python -m pip install pytest
python -m pytest xatac/tests -q
```

## Pretrained ensembles

[Download the pretrained xATAC weights](https://github.com/ahansenlab/XChroma/releases/tag/xatac-avg10-20261102).
The release contains 120 checkpoint files: ten each for GM12878, HCT116, HepG2,
K562, HAP1, mESC, MDP, CDP, precDC1, cDC1, THP-1 and d2PGCLC. The complete set is
17.53 GB; one cell's ten-member ensemble is about 1.46 GB. Authenticated access
is required while the repository is private.

For example, download and verify the GM12878 ensemble. Use a new directory
for each cell so the checkpoint pattern selects exactly its ten members:

```bash
gh release download xatac-avg10-20261102 --repo ahansenlab/XChroma \
  --pattern 'XChroma_xATAC_gm12878_step*.pt' \
  --pattern xatac_models.json --pattern SHA256SUMS --dir weights/xatac/gm12878
(
  cd weights/xatac/gm12878
  awk '$2 == "xatac_models.json" || $2 ~ /^XChroma_xATAC_gm12878_step/ {print}' SHA256SUMS | shasum -a 256 -c -
)
```

`xatac_models.json` lists every filename, checksum, byte size, ensemble and
calibration. Each checkpoint contains the model configuration, including its
track mean. Pass all ten members of one ensemble in the listed order. Shell
expansion of the step-number filenames gives that order.

The following example uses the fixed GM12878 training calibration and a
matching indexed hg38 reference FASTA:

```bash
python -m xatac.predict \
  --checkpoint weights/xatac/gm12878/XChroma_xATAC_gm12878_step*.pt \
  --fasta genome.fa --assembly GRCh38 --chrom chr8 \
  --start 127000000 --end 128000000 \
  --xchroma-scale 1062.9903291441547 --gp99 6.207031 \
  --device cuda --output MYC.atac.npz

python -m xtrain.predict --genome-fa genome.fa --assembly GRCh38 \
  --region chr8:127000000-128000000 --atac-npz MYC.atac.npz \
  --weights-dir weights --device cuda --out results/MYC_DNA_only
```

Download the separate [contact-model weights](../xtrain/checkpoints/README.md)
into `weights` before running the second command. For CPU inference, replace
`--device cuda` with `--device cpu --threads 8` in both commands. Verified 1-Mb
examples took 6.0–6.4 min on Linux with eight CPU threads and PyTorch 2.5.1,
using ten checkpoints for each model after reference-context preparation.
Keep the ATAC NPZ and its JSON sidecar together.

| Human ensemble | Fixed TRAIN scale | gp99 |
|---|---:|---:|
| GM12878 | 1062.9903291441547 | 6.207031 |
| HCT116 | 1020.3265055229913 | 6.191406 |
| HepG2 | 1030.0842872846365 | 6.132812 |
| K562 | 1215.6612872453968 | 6.054688 |

The release manifest also records the study-specific WT calibrations for HAP1,
Sox2, synEP/middle CTCF, Irf8, THP-1 and d2PGCLC. These scalars apply to the stated
locus and use. Sox2 and synEP share mESC weights but use different scalars. Keep
each WT scalar fixed across its edits. For accessibility output in insertion-site signal units,
omit both conversion arguments.

## 1. Generate insertion-site ATAC

Start with coordinate-sorted, indexed BAMs aligned to the same reference. Bulk
libraries must be independently duplicate-marked or duplicate-removed before
pooling. For example, apply this sequence separately to each aligned library:

```bash
samtools sort -n -o name.bam aligned.bam
samtools fixmate -m name.bam fixmate.bam
samtools sort -o coordinate.bam fixmate.bam
samtools markdup -r coordinate.bam rep1.dedup.bam
samtools index rep1.dedup.bam
samtools faidx genome.fa
awk '$1 ~ /^chr([0-9]+|X|Y)$/ {print $1 "\t" $2}' genome.fa.fai > nuclear.chrom.sizes

python -m xatac.sites \
  --bam rep1.dedup.bam rep2.dedup.bam \
  --chrom-sizes nuclear.chrom.sizes --assembly GRCh38 \
  --input-shifts 0 0 --output cell.sites.bw --tmpdir /path/to/scratch
```

The chromosome file explicitly selects the reference contigs to include. For
single-cell pseudobulk, retain the upstream barcode-aware duplicate handling;
do not deduplicate independent cells together using the bulk example.

`--input-shifts 0 0` means **verified unshifted input BAMs**. For shifted inputs,
supply their previously applied shifts using the ChromBPNet endpoint convention.
The script does not infer an unknown input shift. Its output sites are
`reference_start + 4 - input_plus_shift` on the forward strand and
`reference_end - 5 - input_minus_shift` on the reverse strand, where BAM
`reference_end` is exclusive. This matches shifting BED endpoints followed by
5′-base counting in [ChromBPNet's preprocessing](https://github.com/kundajelab/chrombpnet/blob/master/chrombpnet/helpers/preprocessing/reads_to_bigwig.py).

Default filtering matches `samtools view -q 30 -f 2 -F 3852`. Each retained read
contributes one site; a retained pair normally contributes two. Raw library
tracks are summed and globally scaled to 100 million insertions over the selected
chromosomes. The intervening fragment is not filled. No spatial smoothing is
applied. A JSON sidecar records filters, shifts, counts, coordinates, reference
lengths and output SHA256. `track_mean` is the mean over nonzero normalized bases.

Preparation holds one chromosome of uint32 counts in RAM (about 1 GB for human
chr1) and uses about 4 bytes per reference base in temporary disk space. Put
`--tmpdir` on scratch/shared storage. BAM source paths, sizes and mtimes are
recorded; archive your input checksums separately if required.

## 2. Pack fixed windows

Supply a four-column, 0-based half-open BED: `chrom start end TRAIN|VALID|TEST`.
Each interval spans exactly 1,048,576 bp by default. For example:

```text
chr1 2000000 3048576 TRAIN
chr8 2000000 3048576 VALID
chr9 2000000 3048576 TEST
```

These illustrate the format, not the study's full training split. Use your
predefined genome-wide split; chromosome holdouts are convenient. The packer
adds 2,048 bp of real DNA/labels on each side to support training shifts and
rejects overlap between different splits, including those flanks.

```bash
python -m xatac.windows \
  --fasta genome.fa --sites cell.sites.bw \
  --windows windows.bed --output /path/to/cell_dataset
```

The site JSON sidecar is required so fragment-coverage tracks are not silently
treated as site labels. Sparse gaps in this generated count bigWig mean zero.
Records store DNA as uint8 A/C/G/T/N codes and targets as round-to-nearest-even
bfloat16 bits. Coordinates, data splits and hashes are recorded in `records.json`
and `dataset.json`. An incomplete preparation has no passing dataset manifest.

Existing verified record caches with `READY.json` are also supported when their
metadata identifies experimental insertion-site ATAC. This BAM workflow builds
normalized site targets from the libraries you supply. To reproduce an existing
training dataset exactly, use its original records and split manifest.

## 3. Train and resume

```bash
python -m xatac.train --data /path/to/cell_dataset \
  --run /path/to/training_run --config xatac/config.json

python -m xatac.train --data /path/to/cell_dataset \
  --run /path/to/training_run --config xatac/config.json --resume
```

The example uses batch 8, 50,000 optimizer steps, AdamW (betas 0.9/0.999,
weight decay 0.01), 250 warmup steps, and cosine decay from 2e-4 to 2e-5.
Global gradient norm is clipped to 1.0. Training uses BF16 with FP32 loss
arithmetic, activation checkpointing, paired uniform ±1,000 bp shifts and 50%
reverse complements. `track_mean` comes from the dataset and must match the model.
Adjust batch size for your GPU; hardware memory requirements depend on the config.

Records are sampled deterministically by seed and optimizer step. Four workers
prefetch batches and cache compact records. Each loaded record is checked against
its manifest hash when present; datasets generated by `xatac.windows` include
hashes for every record.
Startup reads manifests without stat-ing every record on shared storage.
`train.tsv` records loss components, LR, gradient norm and data/compute time.
Every 500 steps, fixed VALID windows produce loss and 1-bp/1-kb ATAC correlations;
TEST records are not used. Short centered VALID targets are scored only where
observed. `excess_loss` subtracts the loss obtained when prediction equals target.

`models/latest.pt` saves the optimizer, completed step and RNG state every 100
steps. Validation also saves inference checkpoints. SIGUSR1/SIGTERM requests a
checkpoint after the current optimizer step. `--stop-after N` permits a planned
pause without changing the LR schedule. Resume checks the dataset and recipe,
restores the original LR position, and removes uncommitted steps from the active
TSV while retaining its pre-resume copy. Training stops at `max_steps`.

## 4. Predict ATAC

```bash
python -m xatac.predict --checkpoint /path/to/training_run/models/step50000.pt \
  --fasta genome.fa --assembly GRCh38 --chrom chr8 \
  --start 127000000 --end 128000000 --output MYC.atac.npz
```

The requested interval is centered in the model's full native context. This
example uses chr8:[126975712,128024288) and exports the central 1,000,000 positions.
Intervals that do not fit a complete reference context are rejected. The NPZ
contains linear `signal`, its finite-value `valid` mask and coordinates; the JSON records
context offsets, checkpoint identities and units. Inference defaults to FP32.
`--device cpu` supports small inference/verification jobs. Load only trusted
PyTorch checkpoints.

For an ensemble, pass checkpoints for the same cell type trained with the same
target units after `--checkpoint`. Predictions are averaged in inverse-scaled
site-signal units. For XChroma input export, add `--xchroma-scale S --gp99 P`. For the pretrained
ensembles, use the release manifest's exact constants and scope. For a newly
trained model, establish two fixed calibration inputs:

- `S`: over predefined, equal-length TRAIN intervals, divide the sum of mean
  experimental **linear fragment coverage** by the sum of mean predicted
  **inverse-scaled insertion-site signal**. Include every base in each mean and
  use the same checkpoint or ensemble as inference. This is a ratio of summed
  means, not a mean of per-window ratios. Recover linear coverage with `expm1`
  when reading an experimental log1p store.
- `P`: the matching experimental ATAC store's `.global_p99` value, calculated
  in log1p signal units as described in [xtrain](../xtrain/GETTING_STARTED.md#inputs).

These constants are supplied separately; the trainer does not calculate them.
Keep them fixed for subsequent predictions with that model and input source.
Export adds `xchroma_input = log1p(signal * S) / P` without fitting a scale to the
requested prediction window. Site targets and experimental fragment coverage
have distinct units; `S` calibrates their mean signal levels.
Use the exported NPZ directly with the frozen contact model:

```bash
python -m xtrain.predict --genome-fa genome.fa --assembly GRCh38 \
  --region chr8:127000000-128000000 --atac-npz MYC.atac.npz \
  --weights-dir weights --device cuda --out results/MYC_DNA_only
```

Keep the `.npz.json` sidecar beside the NPZ. XChroma validates the reference,
coordinates and recorded conversion, then uses `xchroma_input` directly without
another log transform or p99 division. See [xtrain](../xtrain/GETTING_STARTED.md)
for contact-model weights and output conventions.

## Model and loss

The model has a convolutional encoder, GroupNorm, a 125-bp trunk with 13 dilated
residual blocks (dilations 1…4096), and a three-stage skip-connected decoder back
to 1 bp. Input length is 1,048,576; 49 zeros are added on the right internally to
make the trunk divisible by 125, then removed from the output. No labels shift.
The positive output is `softplus(head) * softplus(per_track_scale)`.

Targets are divided by their nonzero-base track mean and soft-clipped above 10
as `2*sqrt(10*x)-10`. Inverse scaling uses `(x+10)^2/40` above 10. The auxiliary
128-bp prediction sums inverse-scaled 1-bp predictions before scaling back;
it is not an independent learned head. Both resolutions use eight segments and
mean-reduced total-count Poisson plus 5×positional loss. See
[`loss.py`](loss.py) for the scaling and reduction formulas.

`load_weights` accepts compatible CNN inference checkpoints and strictly checks
all active weights. Optimizer resume supports checkpoints created by this
`xatac-v1` trainer. Pretrained weight files are distributed as the separate
release assets described above.

Files: `sites.py` prepares labels; `windows.py` packs data; `data.py` loads and
augments; `model.py` defines the CNN; `loss.py` defines the objective; `train.py`
trains/resumes; `predict.py` exports signal. The tests cover site coordinates,
filtering, loss reduction, paired augmentation, split separation, and complete
training/resume/inference on synthetic data.
