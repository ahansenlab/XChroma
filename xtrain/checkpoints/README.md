# Main-model weights

The current XChrom model is the equal-weight mean of ten predictions from steps 30,500 through 35,000 at 500-step intervals. The exact filenames, byte counts and SHA256 values are in [current_model.json](current_model.json).

```bash
gh release download main-model-avg10-30500-35000 \
  --repo YanXu2077/X-Chrom --pattern '*.ckpt.pt' --dir weights
python xtrain/checkpoints/verify.py --weights-dir weights
```

[Download the weight assets](https://github.com/YanXu2077/X-Chrom/releases/tag/main-model-avg10-30500-35000). Authenticated access is required while the repository is private.

The inference command verifies every member and averages native natural-log observed/expected outputs, using float32 forwards and float64 accumulation. Model parameters are not averaged. Target-specific distillation uses its own frozen-trunk and student checkpoints, documented with each target system.
