# Target-specific distillation

`dna2atac_consist.py` contains two DNA-to-accessibility predictor architectures (`Dna2Atac` and `Dna2AtacCBP`) and the `run(net)` training interface. Gradients pass through a frozen contact network to the accessibility predictor. Depending on the configuration, contact targets come from the teacher or from measured contact maps.

Each student is paired with its own frozen contact architecture, checkpoint and ATAC normalization. Those bindings are independent of the main ten-checkpoint ensemble. Preserve the paired trunk and student when reproducing a target-system result.

```python
from xtrain.distillation.dna2atac_consist import run

# Construct and strictly load the contact network specified for this student.
# `frozen_contact_net` must be on CUDA and use its checkpoint-matched architecture.
run(frozen_contact_net)
```

Supply the matching network through the Python interface; this module does not silently substitute the main model. Student and independently paired trunk weight files are supplied separately.

## Configuration

Install `requirements.txt` in this directory. Set `D2A_GENOME_FA`, `D2A_ATAC_DIR` and `OUTDIR`. Configure the target's original `D2A_ATAC_NORM` and, for global normalization, its `D2A_GP99` constant. `D2A_GT_TARGET=1` uses measured contact supervision from `D2A_COOL`; teacher supervision uses `D2A_GT_TARGET=0`.

The optional HAP1 capture-contact evaluation requires `D2A_HAP1_HIC_DIR` when `D2A_HAP1_HIC_VAL=1`. Study-specific evaluation hooks that call `infer_per_cell` or `koska_val` additionally need those analysis modules. They are separate from the included predictor and training interfaces.

The default accessibility-predictor width, depth, loss and training budget remain configurable in the source. Match the configuration recorded for the student of interest.
