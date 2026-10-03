# Inputs and environments

The main inference entry requires PyTorch, NumPy and zstandard, plus an indexed reference FASTA, matching per-base accessibility stores and the ten [released checkpoints](checkpoints/README.md). All input and output paths are supplied by the command line.

Training additionally uses Selene, Cooler, cooltools, pyfaidx and tabix. It needs a balanced contact target and a distance-expected array for the same assembly and resolution. These datasets are not bundled with the source code.

ATAC FASTQ preprocessing uses bwa-mem2, samtools and deepTools. BigWig conversion additionally uses pyBigWig. Record the coverage units, normalization and reference assembly when preparing a new input store.

[Distillation](distillation/README.md) has separate target-specific weights and frozen-contact-model bindings. Its optional study-specific evaluation callbacks (`infer_per_cell`, `koska_val`) belong to the corresponding target analysis environment; the core predictor definitions and distillation loop are included here.
