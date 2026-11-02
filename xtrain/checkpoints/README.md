# Main-model weights

The current XChroma model is the equal-weight mean of ten predictions from steps 30,500 through 35,000 at 500-step intervals. The exact filenames, byte counts and SHA256 values are in [current_model.json](current_model.json).

```bash
gh release download main-model-avg10-30500-35000 \
  --repo ahansenlab/XChroma --pattern '*.ckpt.pt' --dir weights
python xtrain/checkpoints/verify.py --weights-dir weights
```

[Download the weight assets](https://github.com/ahansenlab/XChroma/releases/tag/main-model-avg10-30500-35000). Authenticated access is required while the repository is private.

The inference command verifies every member and averages native natural-log observed/expected outputs, using float32 forwards and float64 accumulation. Model parameters are not averaged. Cell-specific accessibility models use separate checkpoints; see [xatac](../../xatac/README.md).
