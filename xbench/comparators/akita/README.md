# Akita

Use the [official Akita implementation](https://github.com/calico/basenji/tree/master/manuscripts/akita)
and model head for GM12878 or HCT116. The xbench adapter reads prepared
`CELL/chrN_START.npy` files from `AKITA_MAP_CACHE`.

The 448-bin, 2,048-bp output is resampled during cache preparation to the
500×500 comparison grid and clipped to ln(O/E) in [−2,2]. Scoring applies the
same 2-kb GT/XChrom transforms as the AlphaGenome adapter and excludes
|i−j|≤2. Keep the original output and resampling parameters with the cache.
