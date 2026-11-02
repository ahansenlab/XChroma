"""Accessibility scaling and mean-reduced count/profile losses.

Tensors use batch, sequence, track order.
"""

from collections.abc import Sequence

import torch
from torch import Tensor


def targets_scaling(counts: Tensor, track_mean: Tensor) -> Tensor:
    """Normalize counts and smoothly compress normalized values above ten."""
    normalized = counts.float() / track_mean
    # Clamping only the inactive curved branch avoids sqrt'(0) in backprop.
    curved = 2 * torch.sqrt(10 * normalized.clamp_min(10)) - 10
    return torch.where(normalized <= 10, normalized, curved)


def predictions_scaling(values: Tensor, track_mean: Tensor) -> Tensor:
    """Invert the target transform and restore the count scale."""
    values = values.float()
    normalized = torch.where(values <= 10, values, (values + 10).square() / 40)
    return normalized * track_mean


def pool128(counts: Tensor) -> Tensor:
    """Sum successive, non-overlapping groups of 128 sequence positions."""
    if counts.ndim != 3 or counts.shape[1] % 128:
        raise ValueError("counts must have shape (batch, length, tracks), with length divisible by 128")
    batch, length, tracks = counts.shape
    return counts.reshape(batch, length // 128, 128, tracks).sum(dim=2)


def multinomial_loss(
    prediction: Tensor, target: Tensor, segment_length: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Return combined, mass, and profile losses for contiguous segments.

    The mass term is a Poisson deviance without its factor of two, divided by
    segment length. The profile term is the average negative target-weighted
    log probability. Both inputs enter the calculation as float32 tensors;
    target values below zero are treated as zero.
    """
    if prediction.ndim != 3 or prediction.shape != target.shape:
        raise ValueError("prediction and target must share shape (batch, length, tracks)")
    if not isinstance(segment_length, int) or segment_length <= 0:
        raise ValueError("segment_length must be a positive integer")
    batch, length, tracks = prediction.shape
    if length == 0 or length % segment_length:
        raise ValueError("sequence length must be nonzero and divisible by segment_length")

    shape = (batch, length // segment_length, segment_length, tracks)
    predicted = prediction.float().reshape(shape)
    observed = target.float().clamp_min(0).reshape(shape)
    predicted_mass = predicted.sum(dim=2, keepdim=True)
    observed_mass = observed.sum(dim=2, keepdim=True)
    epsilon = 1e-7

    total = (
        predicted_mass
        - observed_mass * torch.log(predicted_mass + epsilon)
        - (observed_mass - observed_mass * torch.log(observed_mass + epsilon))
    ).mean() / segment_length
    positional = -(observed * torch.log(predicted / (predicted_mass + epsilon) + epsilon)).mean()
    return total + 5 * positional, total, positional


def atac_loss(
    predictions: Sequence[Tensor],
    counts: Tensor,
    track_mean: Tensor,
    target_slice: tuple[int, int] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Combine full-resolution and 128-base accessibility losses.

    ``counts`` contains the cropped target. ``target_slice`` identifies that
    crop within both prediction tensors in full-resolution coordinates. Each
    resolution uses eight sequence segments over the cropped interval.
    """
    if len(predictions) != 2:
        raise ValueError("predictions must contain full-resolution and 128-base tensors")
    if counts.ndim != 3 or any(value.ndim != 3 for value in predictions):
        raise ValueError("counts and predictions must have shape (batch, length, tracks)")
    left, right = (0, predictions[0].shape[1]) if target_slice is None else target_slice
    if not isinstance(left, int) or not isinstance(right, int):
        raise ValueError("target_slice endpoints must be integer base coordinates")
    length = right - left
    if left < 0 or right > predictions[0].shape[1] or length <= 0:
        raise ValueError("target_slice must identify a nonempty interval within the predictions")
    if left % 128 or right % 128 or length % 1024:
        raise ValueError("slice endpoints must be divisible by 128 and cropped length by 1024")
    if counts.shape[1] != length:
        raise ValueError("counts length must equal the selected prediction interval")

    selected = (predictions[0][:, left:right], predictions[1][:, left // 128:right // 128])
    targets = (targets_scaling(counts, track_mean), targets_scaling(pool128(counts), track_mean * 128))
    result: dict[str, Tensor] = {}
    losses = []
    for resolution, predicted, observed in zip((1, 128), selected, targets):
        loss, total, positional = multinomial_loss(predicted, observed, length // 8 // resolution)
        result[f"loss_{resolution}bp"] = loss
        result[f"total_{resolution}bp"] = total
        result[f"positional_{resolution}bp"] = positional
        losses.append(loss)
    return losses[0] + losses[1], result
