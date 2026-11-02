"""Compact site-label records and deterministic paired shift/RC augmentation."""
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from .io import DNA_LUT, decode_bf16, fingerprint, sha256


def read_dataset(root):
    root = Path(root).resolve()
    meta_path = root / "dataset.json"
    if not meta_path.exists():
        meta_path = root / "READY.json"  # Original experimental-target record caches.
    metadata = json.loads(meta_path.read_text())
    if metadata.get("status") != "PASS" or metadata.get("fixture", False):
        raise ValueError("Dataset is incomplete or marked as a fixture")
    known_sites = (metadata.get("signal") == "tn5_insertion_sites"
                   or metadata.get("data_source") == "official_experimental_ATAC_TFRecords"
                   or "ChromBPNet insertion counts" in metadata.get("label_source", ""))
    if not known_sites:
        raise ValueError("Dataset metadata does not identify insertion-site ATAC labels")
    if not np.isfinite(metadata["track_mean"]) or metadata["track_mean"] <= 0:
        raise ValueError("Dataset requires a positive nonzero-base track mean")
    records_path = root / "records.json"
    records_hash = sha256(records_path)
    if records_hash != metadata["records_sha256"]:
        raise ValueError("Dataset record manifest changed")
    rows = json.loads(records_path.read_text())
    for row in rows:
        path = PurePosixPath(row["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Invalid record path: {path}")
        if row["subset"] not in {"TRAIN", "VALID", "TEST"}:
            raise ValueError("Unknown data split")
        length, target_length = row["length"], row["target_length"]
        if not 0 < target_length <= length or (length - target_length) % 2:
            raise ValueError("Targets must be centered within the DNA context")
        if row["target_offset"] != (length - target_length) // 2:
            raise ValueError("Incorrect target offset")
    dataset_id = fingerprint({"metadata": metadata, "records_sha256": records_hash})
    return metadata, rows, dataset_id


@lru_cache(maxsize=48)
def read_record(path, expected_bytes=None, expected_sha256=None):
    payload = Path(path).read_bytes()
    if expected_bytes is not None and len(payload) != expected_bytes:
        raise ValueError(f"Truncated or replaced record: {path}")
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"Record content changed: {path}")
    with np.load(io.BytesIO(payload), allow_pickle=False) as record:
        code, bits = record["sequence_code"], record["atac_bf16"]
    if code.dtype != np.uint8 or code.ndim != 1 or (code > 4).any():
        raise ValueError("DNA encoding must be A=0,C=1,G=2,T=3,unknown=4")
    if bits.dtype != np.uint16 or bits.ndim != 1:
        raise ValueError("ATAC must be a one-dimensional bfloat16 bit array")
    return code, bits


def example(root, row, seq_len, shift=0, reverse=False):
    code, bits = read_record(str(Path(root) / row["path"]), row.get("bytes"), row.get("sha256"))
    if len(code) != row["length"] or len(bits) != row["target_length"]:
        raise ValueError("Record dimensions differ from manifest")
    crop = (len(code) - seq_len) // 2 + shift
    end = crop + seq_len
    if crop < 0 or end > len(code):
        raise ValueError("Requested shift exceeds the stored DNA flanks")
    offset = row["target_offset"]
    left, right = max(crop, offset), min(end, offset + len(bits))
    if left >= right:
        raise ValueError("Crop does not intersect observed targets")
    dna = DNA_LUT[code[crop:end]].copy()
    counts = decode_bf16(bits[left - offset:right - offset])[:, None]
    target_slice = (left - crop, right - crop)
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Invalid ATAC target values")
    if reverse:
        dna, counts = dna[::-1, ::-1].copy(), counts[::-1].copy()
        target_slice = (seq_len - target_slice[1], seq_len - target_slice[0])
    return dna, counts, target_slice


class TrainingBatches(torch.utils.data.Dataset):
    """One optimizer step per item; sample order survives resume and worker changes."""
    def __init__(self, root, rows, config, first_step, last_step):
        self.root, self.rows, self.config = str(root), rows, config
        self.first_step, self.last_step = first_step, last_step
        n, shift = config["model"]["seq_len"], config["shift_bp"]
        if not rows or any(row["subset"] != "TRAIN" for row in rows):
            raise ValueError("Training requires TRAIN records")
        for row in rows:
            if row["target_length"] < n + 2 * shift:
                raise ValueError("TRAIN labels lack sufficient flanks for the requested shift")

    def __len__(self):
        return max(0, self.last_step - self.first_step + 1)

    def __getitem__(self, index):
        step = self.first_step + index
        c = self.config
        rng = np.random.default_rng(np.random.SeedSequence([c["seed"], step]))
        dna, counts, indices, shifts, reverses = [], [], [], [], []
        for _ in range(c["batch_size"]):
            ri = int(rng.integers(len(self.rows)))
            shift = int(rng.integers(-c["shift_bp"], c["shift_bp"] + 1))
            reverse = rng.random() < c["reverse_complement_probability"]
            x, y, observed = example(self.root, self.rows[ri], c["model"]["seq_len"], shift, reverse)
            if observed != (0, c["model"]["seq_len"]):
                raise ValueError("TRAIN crop is not fully supervised")
            dna.append(x); counts.append(y)
            indices.append(ri); shifts.append(shift); reverses.append(reverse)
        return dict(step=step, x=torch.from_numpy(np.stack(dna)),
                    raw=torch.from_numpy(np.stack(counts)), indices=indices,
                    shifts=shifts, reverses=reverses)
