"""Convolutional DNA→ATAC model, with the original 125-bp trunk and 1-bp decoder."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .loss import pool128, predictions_scaling, targets_scaling


def group_norm(channels, groups=8):
    groups = next(g for g in (groups, 4, 2, 1) if channels % g == 0)
    return nn.GroupNorm(groups, channels)


class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.net = nn.Sequential(
            group_norm(channels), nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            group_norm(channels), nn.GELU(), nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        return x + self.net(x)


class ConvNet(nn.Module):
    def __init__(self, width, n_dilated, dilation_cycles, pool_factors, up_factors, up_channels):
        super().__init__()
        if len(pool_factors) != 6 or len(up_factors) != len(up_channels):
            raise ValueError("Six encoder stages and matching decoder factors/channels are required")
        if any(v < 1 for v in (*pool_factors, *up_factors, *up_channels)):
            raise ValueError("Pooling factors and channel counts must be positive")
        self.trunk_bin = math.prod(pool_factors)
        if self.trunk_bin != math.prod(up_factors):
            raise ValueError("Decoder must return the trunk to 1-bp resolution")
        self.pool_factors, self.up_factors = tuple(pool_factors), tuple(up_factors)
        self.lconv1 = nn.Sequential(
            group_norm(4, 1), nn.GELU(), nn.Conv1d(4, 64, 9, padding=4),
            group_norm(64), nn.GELU(), nn.Conv1d(64, 64, 9, padding=4))
        self.stages = nn.ModuleList()
        channels, resolution, skips = 64, 1, {1: 64}
        for out, factor in zip((96, width, width, width, width, width), pool_factors):
            self.stages.append(nn.Sequential(
                nn.MaxPool1d(factor, factor) if factor > 1 else nn.Identity(),
                group_norm(channels), nn.GELU(), nn.Conv1d(channels, out, 9, padding=4),
                group_norm(out), nn.GELU(), nn.Conv1d(out, out, 9, padding=4)))
            resolution *= factor
            channels, skips[resolution] = out, out
        self.dil_tower = nn.ModuleList(
            DilatedBlock(width, 2 ** i) for _ in range(dilation_cycles) for i in range(n_dilated))
        self.upres, self.up_skip, self.up_post = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for factor, out in zip(up_factors, up_channels):
            resolution //= factor
            if resolution not in skips:
                raise ValueError(f"No encoder skip at {resolution} bp")
            self.upres.append(nn.Sequential(group_norm(channels), nn.GELU(),
                                           nn.Conv1d(channels, out, 9, padding=4)))
            self.up_skip.append(nn.Sequential(group_norm(skips[resolution]), nn.GELU(),
                                             nn.Conv1d(skips[resolution], out, 1)))
            self.up_post.append(nn.Sequential(group_norm(out), nn.GELU(),
                                             nn.Conv1d(out, out, 9, padding=4)))
            channels = out
        self.residual_scale = nn.Parameter(torch.full((len(up_factors),), 0.9))
        self.head_ag = nn.Sequential(group_norm(channels), nn.GELU(), nn.Conv1d(channels, channels, 1),
                                     nn.GELU(), nn.Conv1d(channels, 1, 1))
        self.ag_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, use_checkpoint=False):
        def run(block, values):
            if use_checkpoint and self.training and torch.is_grad_enabled():
                return checkpoint(block, values, use_reentrant=False, preserve_rng_state=True)
            return block(values)

        x = run(self.lconv1, x)
        skips, resolution = {1: x}, 1
        for stage, factor in zip(self.stages, self.pool_factors):
            x = run(stage, x)
            resolution *= factor
            skips[resolution] = x
        for block in self.dil_tower:
            x = run(block, x)
        for i, (factor, up, skip, post) in enumerate(zip(
                self.up_factors, self.upres, self.up_skip, self.up_post)):
            resolution //= factor
            y = run(up, x).repeat_interleave(factor, dim=2) * self.residual_scale[i]
            y = y + run(skip, skips[resolution])
            x = y + run(post, y)
        return F.softplus(run(self.head_ag, x)) * F.softplus(self.ag_scale).view(1, -1, 1)


class DNA2ATAC(nn.Module):
    """One cell per model; input (B, L, 4), scaled outputs (B, L, 1) and (B, L/128, 1)."""
    def __init__(self, seq_len=1048576, width=512, n_dilated=13, dilation_cycles=1,
                 pool_factors=(5, 5, 5, 1, 1, 1), up_factors=(5, 5, 5),
                 up_channels=(128, 64, 32), grad_checkpoint=True, track_mean=0.37652507):
        super().__init__()
        if seq_len < 1024 or seq_len % 1024 or not math.isfinite(track_mean) or track_mean <= 0:
            raise ValueError("seq_len must be a positive multiple of 1024; track_mean must be positive")
        self.seq_len, self.track_mean, self.grad_checkpoint = seq_len, track_mean, grad_checkpoint
        self.cnn = ConvNet(width, n_dilated, dilation_cycles, pool_factors, up_factors, up_channels)
        self.right_pad = (-seq_len) % self.cnn.trunk_bin

    def forward(self, dna):
        if dna.ndim != 3 or dna.shape[1:] != (self.seq_len, 4):
            raise ValueError(f"Expected (batch, {self.seq_len}, 4), got {tuple(dna.shape)}")
        x = F.pad(dna.transpose(1, 2), (0, self.right_pad))
        p1 = self.cnn(x, self.grad_checkpoint)[..., :self.seq_len].transpose(1, 2)
        p128 = targets_scaling(pool128(predictions_scaling(p1, self.track_mean)), self.track_mean * 128)
        return p1, p128


def load_weights(model, state):
    """Strict loading, allowing only the unused legacy head that had no forward path."""
    legacy = {"cnn.head.0.weight", "cnn.head.0.bias", "cnn.head.2.weight", "cnn.head.2.bias",
              "cnn.head.4.weight", "cnn.head.4.bias"}
    found = legacy & state.keys()
    if found and found != legacy:
        raise ValueError("Incomplete legacy unused-head state")
    model.load_state_dict({k: v for k, v in state.items() if k not in legacy}, strict=True)
    return sorted(found)
