"""Sequence encoding, checksums and atomic output writes."""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(path)


def encode_dna(sequence):
    table = np.full(256, 4, np.uint8)
    for i, base in enumerate(b"ACGT"):
        table[base] = i
    return table[np.frombuffer(sequence.upper().encode(), np.uint8)]


DNA_LUT = np.vstack((np.eye(4, dtype=np.float32), np.zeros((1, 4), np.float32)))


def encode_bf16(values):
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Site labels must be finite and nonnegative")
    bits = values.view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def decode_bf16(bits):
    if bits.dtype != np.uint16:
        raise ValueError("Expected uint16 bfloat16 bit patterns")
    return (bits.astype(np.uint32) << 16).view(np.float32)
