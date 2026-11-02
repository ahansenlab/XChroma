"""Independent checks against scalar equations and elementary invariants."""

import math

import pytest
import torch

from xatac.loss import atac_loss, multinomial_loss, pool128, predictions_scaling, targets_scaling


def scalar_loss(prediction, target, segment_length):
    """Small Python-loop oracle, independent of tensor reduction layout."""
    batch, length, tracks = prediction.shape
    mass_terms = []
    profile_terms = []
    for sample in range(batch):
        for start in range(0, length, segment_length):
            for track in range(tracks):
                p = [float(prediction[sample, i, track]) for i in range(start, start + segment_length)]
                t = [max(0.0, float(target[sample, i, track])) for i in range(start, start + segment_length)]
                mass_p, mass_t = sum(p), sum(t)
                mass_terms.append(mass_p - mass_t * math.log(mass_p + 1e-7) - (mass_t - mass_t * math.log(mass_t + 1e-7)))
                profile_terms.extend(-v * math.log(u / (mass_p + 1e-7) + 1e-7) for u, v in zip(p, t))
    total = sum(mass_terms) / len(mass_terms) / segment_length
    positional = sum(profile_terms) / len(profile_terms)
    return total + 5 * positional, total, positional


def test_scaling_equations_and_inverse():
    counts = torch.tensor([[[0.0, 2.0, 20.0], [10.0, 40.0, 100.0]]])
    mean = torch.tensor([1.0, 2.0, 2.0])
    actual = targets_scaling(counts, mean)
    normalized = counts / mean
    expected = torch.tensor([v if v <= 10 else 2 * math.sqrt(10 * v) - 10 for v in normalized.flatten()]).reshape_as(counts)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(predictions_scaling(actual, mean), counts)


def test_transform_gradients_at_zero_threshold_and_above():
    counts = torch.tensor([[[0.0], [10.0], [40.0]]], requires_grad=True)
    actual = targets_scaling(counts, torch.ones(1))
    actual.sum().backward()
    torch.testing.assert_close(counts.grad, torch.tensor([[[1.0], [1.0], [0.5]]]))
    values = actual.detach().requires_grad_()
    predictions_scaling(values, torch.ones(1)).sum().backward()
    torch.testing.assert_close(values.grad, torch.tensor([[[1.0], [1.0], [2.0]]]))


def test_pool_sums_and_gradients():
    counts = torch.arange(2 * 256 * 3, dtype=torch.float32).reshape(2, 256, 3).requires_grad_()
    pooled = pool128(counts)
    expected = torch.stack((counts[:, :128].sum(1), counts[:, 128:].sum(1)), dim=1)
    torch.testing.assert_close(pooled, expected)
    pooled.sum().backward()
    torch.testing.assert_close(counts.grad, torch.ones_like(counts))


@pytest.mark.parametrize("zero_target", [False, True])
def test_loss_against_scalar_equations(zero_target):
    generator = torch.Generator().manual_seed(2718)
    prediction = torch.rand(2, 8, 3, generator=generator) + 0.3
    target = torch.rand(2, 8, 3, generator=generator) * 4 - 0.2
    if zero_target:
        target.zero_()
    for actual, expected in zip(multinomial_loss(prediction, target, 4), scalar_loss(prediction, target, 4)):
        assert actual.item() == pytest.approx(expected, abs=1e-6, rel=1e-6)


@pytest.mark.parametrize("zero_prediction,zero_target", [(False, False), (False, True), (True, False), (True, True)])
def test_loss_gradients_are_finite(zero_prediction, zero_target):
    prediction = torch.full((1, 8, 2), 0.0 if zero_prediction else 0.6, requires_grad=True)
    target = torch.full_like(prediction, 0.0 if zero_target else 2.0)
    loss, _, _ = multinomial_loss(prediction, target, 4)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
    if zero_target:
        torch.testing.assert_close(prediction.grad, torch.full_like(prediction, 1 / prediction.numel()))


def test_positive_gradient_matches_closed_form():
    prediction = torch.tensor([[[0.4], [1.2], [0.9], [0.5]]], requires_grad=True)
    target = torch.tensor([[[1.0], [0.0], [2.0], [0.5]]])
    loss, _, _ = multinomial_loss(prediction, target, 4)
    loss.backward()
    p, t = prediction.detach().flatten().double(), target.flatten().double()
    denominator = p.sum() + 1e-7
    probability = p / denominator + 1e-7
    expected = (1 - t.sum() / denominator) / 4
    expected = expected + 5 / 4 * (-t / (probability * denominator) + (t * p / probability).sum() / denominator.square())
    torch.testing.assert_close(prediction.grad.flatten().double(), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("cropped", [False, True])
def test_combined_loss_matches_explicit_resolution_sum(cropped):
    generator = torch.Generator().manual_seed(99)
    counts = torch.rand(2, 1024, 3, generator=generator) * 50
    length = 1280 if cropped else 1024
    prediction = [torch.rand(2, length, 3, generator=generator).requires_grad_(), torch.rand(2, length // 128, 3, generator=generator).requires_grad_()]
    means = torch.tensor([1.0, 2.0, 3.0])
    bounds = (128, 1152) if cropped else None
    result, details = atac_loss(prediction, counts, means, bounds)
    left, right = bounds or (0, length)
    expected_1 = multinomial_loss(prediction[0][:, left:right], targets_scaling(counts, means), 128)
    expected_128 = multinomial_loss(prediction[1][:, left // 128:right // 128], targets_scaling(pool128(counts), means * 128), 1)
    torch.testing.assert_close(result, expected_1[0] + expected_128[0])
    assert set(details) == {f"{name}_{resolution}bp" for name in ("loss", "total", "positional") for resolution in (1, 128)}
    result.backward()
    assert all(torch.isfinite(value.grad).all() for value in prediction)
    if cropped:
        assert torch.count_nonzero(prediction[0].grad[:, :left]) == 0
        assert torch.count_nonzero(prediction[0].grad[:, right:]) == 0


@pytest.mark.parametrize("bounds,target_length", [((1, 1025), 1024), ((0, 512), 512), ((0, 1024), 512), ((-128, 896), 1024), ((128, 2176), 2048)])
def test_invalid_crop(bounds, target_length):
    prediction = [torch.ones(1, 2048, 1), torch.ones(1, 16, 1)]
    with pytest.raises(ValueError):
        atac_loss(prediction, torch.ones(1, target_length, 1), torch.ones(1), bounds)


def test_invalid_segment_and_shapes():
    with pytest.raises(ValueError):
        multinomial_loss(torch.ones(1, 8, 1), torch.ones(1, 7, 1), 4)
    with pytest.raises(ValueError):
        multinomial_loss(torch.ones(1, 8, 1), torch.ones(1, 8, 1), 3)
    with pytest.raises(ValueError):
        pool128(torch.ones(1, 127, 1))
