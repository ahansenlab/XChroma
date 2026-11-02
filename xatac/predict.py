"""Predict a genomic interval with one checkpoint or a mean of per-base signals."""
import argparse
from pathlib import Path

import numpy as np
import pysam
import torch

from .io import DNA_LUT, encode_dna, sha256, write_json, write_npz
from .loss import predictions_scaling
from .model import DNA2ATAC, load_weights


def _resolve_scale(xchroma_scale, xchrom_scale):
    if xchroma_scale is not None and xchrom_scale is not None and xchroma_scale != xchrom_scale:
        raise ValueError("Conflicting values for --xchroma-scale and its legacy alias")
    return xchroma_scale if xchroma_scale is not None else xchrom_scale


def predict(checkpoints, fasta, chrom, start, end, output, device_name="cuda",
            assembly=None, xchroma_scale=None, gp99=None, *, xchrom_scale=None):
    xchroma_scale = _resolve_scale(xchroma_scale, xchrom_scale)
    if start < 0 or end <= start:
        raise ValueError("Use a nonempty 0-based half-open interval")
    output = Path(output)
    if output.exists() or Path(str(output) + ".json").exists():
        raise FileExistsError(output)
    if (xchroma_scale is None) != (gp99 is None):
        raise ValueError("Provide both a fixed training-derived XChroma scale and gp99, or neither")
    if xchroma_scale is not None and (not np.isfinite([xchroma_scale, gp99]).all() or min(xchroma_scale, gp99) <= 0):
        raise ValueError("Input conversion parameters must be finite and positive")
    device = torch.device(device_name)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    total, members, context = None, [], None
    with pysam.FastaFile(str(fasta)) as fa:
        for path in checkpoints:
            state = torch.load(path, map_location="cpu", weights_only=False)
            config = dict(state["config"]["model"])
            config["grad_checkpoint"] = False
            model = DNA2ATAC(**config)
            discarded = load_weights(model, state["model"])
            model = model.eval().to(device)
            seq_len = model.seq_len
            if end - start > seq_len:
                raise ValueError("Requested interval exceeds native model context; no implicit tiling")
            left = start - (seq_len - (end - start)) // 2
            right = left + seq_len
            if left < 0 or chrom not in fa.references or right > fa.get_reference_length(chrom):
                raise ValueError("Full model context extends beyond the reference")
            current = dict(chrom=chrom, start=left, end=right, display_offset=start - left)
            if context is not None and current != context:
                raise ValueError("Ensemble members must share the same native DNA context")
            context = current
            code = encode_dna(fa.fetch(chrom, left, right))
            with torch.inference_mode():
                native, _ = model(torch.from_numpy(DNA_LUT[code][None]).to(device))
                counts = predictions_scaling(native[0, :, 0], model.track_mean).cpu().numpy()
            if not np.isfinite(counts).all() or (counts < 0).any():
                raise FloatingPointError(f"Invalid inverse-scaled prediction from {path}")
            signal = counts[start - left:end - left].astype(np.float64)
            total = signal if total is None else total + signal
            members.append(dict(path=str(Path(path)), sha256=sha256(path), step=state.get("step"),
                                model_config=config, legacy_unused_head_keys_discarded=discarded))
            del model, state, native, counts
    if not members:
        raise ValueError("At least one checkpoint is required")
    signal = (total / len(members)).astype(np.float32)
    arrays = dict(signal=signal, valid=np.isfinite(signal), start=np.int64(start), end=np.int64(end))
    if xchroma_scale is not None:
        arrays["xchroma_input"] = (np.log1p(signal.astype(np.float64) * xchroma_scale) / gp99).astype(np.float32)
    write_npz(output, **arrays)
    write_json(Path(str(output) + ".json"), dict(
        status="PASS", assembly=assembly, chrom=chrom, start=start, end=end,
        coordinate_convention="0-based half-open", resolution_bp=1, context=context,
        units="insertion-site target signal after inverse model scaling",
        ensemble="Arithmetic mean of inverse-scaled per-base signals, not weights or scaled head values",
        precision="FP32 forward, FP64 ensemble accumulation, TF32 off", members=members,
        fasta=str(fasta), fasta_sha256=sha256(fasta), output_sha256=sha256(output),
        xchroma_conversion=None if xchroma_scale is None else dict(
            formula="log1p(signal * fixed_training_scale) / gp99", fixed_training_scale=xchroma_scale,
            gp99=gp99, signal_definition="Scaled predicted sites; not measured fragment coverage")))
    return signal


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--assembly", required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--xchroma-scale", type=float)
    parser.add_argument("--xchrom-scale", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--gp99", type=float)
    args = parser.parse_args(argv)
    try:
        scale = _resolve_scale(args.xchroma_scale, args.xchrom_scale)
    except ValueError as exc:
        parser.error(str(exc))
    torch.set_num_threads(args.threads)
    predict(args.checkpoint, args.fasta, args.chrom, args.start, args.end, args.output,
            args.device, args.assembly, scale, args.gp99)


if __name__ == "__main__":
    main()
