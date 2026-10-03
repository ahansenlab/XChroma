# EPCOT

Select one output with `LINES=epcot:intacthic`, `epcot:microc` or `epcot:rcmc`.
Provide `GT_TILES`, the XChrom cache identity, and the matching
`EPCOT_INTACTHIC_DIR`, `EPCOT_MICROC_DIR` or `EPCOT_RCMC_DIR`.
A single-head invocation can instead set `EPCOT_DIR`.

The adapter expects 1-Mb/1-kb maps in native log2(O/E), ordered exactly like the
GT tile manifest. It converts outputs to linear O/E, self-normalizes to ln(O/E),
and scores matched 500-kb submatrices. `EPCOT_CONV` can explicitly select `log2`
(default), `ln` or `linear` when preparing a separately identified input source.

Generate predictions with the [official EPCOT runtime](https://github.com/liu-bioinfo-lab/EPCOT)
and the weights for the relevant organism/head. Do not substitute an older
runtime architecture when using a newer checkpoint.
