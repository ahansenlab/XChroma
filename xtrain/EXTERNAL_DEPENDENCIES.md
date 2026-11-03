# Inputs and environments

The main inference entry requires PyTorch, NumPy and zstandard, plus an indexed reference FASTA, matching per-base accessibility stores and the ten [local checkpoints](checkpoints/README.md). All input and output paths are supplied by the command line.

## Tested environments

The recorded configurations below use Python 3.12; see the requirement files for installation bounds and choose the PyTorch build for your CPU or CUDA device.

| Check | Operating system and Python | Runtime packages |
|---|---|---|
| Regression and synthetic-data tests: 61 passed | macOS 26.1 arm64; Python 3.12.2 | PyTorch 2.3.1, NumPy 1.26.4, zstandard 0.25.0, pysam 0.23.3, pyBigWig 0.3.29 |
| Full 1-Mb HAP1 WT/deletion CPU replay, ten members per model, eight threads | Linux kernel 5.15.0-139-generic x86_64, glibc 2.31; Python 3.12.11 | PyTorch 2.5.1+cpu, NumPy 2.2.6, pysam 0.23.3 |

Additional versions recorded in the regression environment for metrics and training readers: SciPy 1.17.1, pandas 2.2.2, scikit-image 0.26.0, scikit-learn 1.5.1, cooltools 0.7.1, Cooler 0.10.4, pyfaidx 0.9.0.3 and selene-sdk 0.6.0. Inspection of that environment on 2 November 2026 additionally recorded Matplotlib 3.10.9, pytabix 0.1 and setuptools 75.1.0.

## Optional training and input preparation

Training uses Selene’s batch loader, Cooler, cooltools and pyfaidx. Tabix is installed for Selene compatibility; the XChroma BED reader accepts plain or gzip-compressed intervals. It needs a balanced contact target and a distance-expected array for the same assembly and resolution. These datasets are not bundled with the source code.

ATAC FASTQ preprocessing uses bwa-mem2, samtools and deepTools. BigWig conversion additionally uses pyBigWig. Record the coverage units, normalization and reference assembly when preparing a new input store.

[xatac](../xatac/README.md) provides cell-specific DNA-to-ATAC training and inference with its own checkpoints.

The external FASTQ-preprocessing executables are not needed when using supplied accessibility stores or predicted-ATAC NPZ inputs.
