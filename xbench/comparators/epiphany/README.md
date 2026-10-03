# Epiphany

Use the [official Epiphany runtime](https://github.com/arnavmdas/epiphany) and
10-kb weights in their own PyTorch environment. The predictor requires
`pyBigWig` and reads five tracks in this order:
DNase, H3K27ac, H3K4me3, H3K27me3, CTCF.

```bash
export EPIPHANY_UPSTREAM=/path/to/epiphany_checkout
export EPIPHANY_CHECKPOINT=/path/to/pretrained_10kb.pt_model
export BWDIR=/path/to/five_bigwigs
export GT_TILES=/path/to/gt_tiles
export CELL=K562
export CHROM=chr1
export OUTDIR=/path/to/new_predictions/K562
python -m xbench.comparators.epiphany.predict
```

`BWDIR` contains `DNase.bw`, `H3K27ac.bw`, `H3K4me3.bw`, `H3K27me3.bw` and
`CTCF.bw`. Input signal preparation follows 10-bp RPKM coverage with chrX
excluded from normalization. The predictor averages inputs into 100-bp bins,
uses the original model's 3.4-Mb receptive interval, and assembles its output
into 1-Mb/10-kb cache windows matching the GT tile starts. Existing chromosome outputs are never overwritten. Other chromosomes may be
written to the same cell directory.

Optional ENCODE preparation helpers:

```bash
python -m xbench.comparators.epiphany.fetch_bam_urls --cells K562 --out /path/to/new_work
EPIPHANY_WORK=/path/to/new_work bash xbench/comparators/epiphany/build_tracks_cell.sh K562 DNase
```

Inspect and retain the generated URL list before building tracks; live ENCODE
queries select the latest released GRCh38 alignment processing and may differ
from a previously frozen input list. Repeat track preparation for each mark.
GM12878 scoring uses chr3, chr11 and chr17; other benchmark cells use all
available chromosomes.
