# Main-model weights

The current XChroma model is the equal-weight mean of ten predictions from steps 30,500 through 35,000 at 500-step intervals. The exact filenames, byte counts and SHA256 values are in [current_model.json](current_model.json).

```bash
python xtrain/checkpoints/verify.py --weights-dir weights
```

Place these ten local checkpoint files in `weights/` before running inference.
Checkpoint files are not distributed in this repository.

The inference command verifies every member and averages native natural-log observed/expected outputs, using float32 forwards and float64 accumulation. Model parameters are not averaged. Cell-specific accessibility models use separate checkpoints; see [xatac](../../xatac/README.md).
