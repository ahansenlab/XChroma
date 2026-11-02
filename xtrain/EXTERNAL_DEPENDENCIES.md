# Inputs and environments

The main inference entry requires PyTorch, NumPy and zstandard, plus an indexed reference FASTA, matching per-base accessibility stores and the ten [released checkpoints](checkpoints/README.md). All input and output paths are supplied by the command line.

Training uses Selene’s batch loader, Cooler, cooltools and pyfaidx. Tabix is installed for Selene compatibility; the XChroma BED reader accepts plain or gzip-compressed intervals. It needs a balanced contact target and a distance-expected array for the same assembly and resolution. These datasets are not bundled with the source code.

ATAC FASTQ preprocessing uses bwa-mem2, samtools and deepTools. BigWig conversion additionally uses pyBigWig. Record the coverage units, normalization and reference assembly when preparing a new input store.

[xatac](../xatac/README.md) provides cell-specific DNA-to-ATAC training and inference with its own checkpoints.
