"""Reusable depthwise DCLS decoder blocks with learnable sampling positions."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt

class DCLSDepthwise2d(nn.Module):
    """Depthwise convolution with bilinearly interpolated learnable tap positions."""

    def __init__(self, ch, m_taps=49, K=17, dilation=1):
        super().__init__()
        self.ch, self.m, self.K, self.d = ch, m_taps, K, dilation
        self.pad = (K // 2) * dilation
        self.lim = (K - 1) / 2.0

        bound = 1.0 / math.sqrt(m_taps)
        self.weight = nn.Parameter(torch.empty(ch, m_taps).uniform_(-bound, bound))

        g = int(round(math.sqrt(m_taps)))
        assert g * g == m_taps, f"m_taps={m_taps} 必须是完全平方数"
        step = (K - 1) / (g - 1) if g > 1 else 0.0
        base = torch.arange(g, dtype=torch.float32) * step - self.lim
        gy, gx = torch.meshgrid(base, base, indexing="ij")
        p0 = torch.stack([gy.reshape(-1), gx.reshape(-1)], dim=-1)
        p0 = p0.unsqueeze(0).repeat(ch, 1, 1)
        p0 = p0 + 0.05 * torch.randn_like(p0)
        self.pos = nn.Parameter(p0.clamp(-self.lim, self.lim))
        self.pos._dcls_position = True

    @torch.no_grad()
    def clamp_positions(self):
        """Clamp learnable tap positions to the kernel extent after optimizer updates."""
        self.pos.clamp_(-self.lim, self.lim)

    def build_kernel(self):
        """Scatter each learned tap to four neighboring kernel-grid locations."""
        K, C, m = self.K, self.ch, self.m
        p = self.pos.clamp(-self.lim, self.lim) + self.lim
        p0 = p.floor()
        frac = p - p0
        p0 = p0.long()

        ry, rx = p0[..., 0], p0[..., 1]
        fy, fx = frac[..., 0], frac[..., 1]

        ry1, rx1 = (ry + 1).clamp(max=K - 1), (rx + 1).clamp(max=K - 1)
        oy = (ry + 1 <= K - 1).to(p.dtype)
        ox = (rx + 1 <= K - 1).to(p.dtype)

        w = self.weight
        vals = torch.stack([
            w * (1 - fy) * (1 - fx),
            w * (1 - fy) * fx * ox,
            w * fy * ox.new_ones(()) * (1 - fx) * oy,
            w * fy * fx * oy * ox,
        ], dim=-1)
        idx = torch.stack([
            ry * K + rx, ry * K + rx1, ry1 * K + rx, ry1 * K + rx1,
        ], dim=-1)

        flat = torch.zeros(C, K * K, dtype=w.dtype, device=w.device)
        flat = flat.scatter_add(1, idx.reshape(C, -1), vals.reshape(C, -1))
        return flat.view(C, 1, K, K)

    def forward(self, x):
        return F.conv2d(x, self.build_kernel(), bias=None,
                        padding=self.pad, dilation=self.d, groups=self.ch)

class DCLSBlock2D(nn.Module):
    """Residual depthwise DCLS block with pointwise channel mixing."""

    def __init__(self, dim, m_taps=49, K=17, dilation=1, mlp_ratio=2.0, ls_init=1e-6):
        super().__init__()
        self.dwconv = DCLSDepthwise2d(dim, m_taps=m_taps, K=K, dilation=dilation)
        self.norm = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.pwconv1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden, dim)
        self.gamma = nn.Parameter(torch.full((dim,), ls_init))

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        x = self.gamma * x
        return shortcut + x.permute(0, 3, 1, 2).contiguous()

class DCLSDecoder2D(nn.Module):
    """Stack of DCLS blocks at the supplied dilation rates."""

    def __init__(self, dim=64, dilations=(1, 2, 4, 8, 16, 32, 64), m_taps=49, K=17,
                 mlp_ratio=2.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            DCLSBlock2D(dim, m_taps=m_taps, K=K, dilation=d, mlp_ratio=mlp_ratio)
            for d in dilations
        ])
        r = (K // 2) * sum(dilations)
        print(f"[dcls_dec] {len(self.blocks)} blocks K={K} taps={m_taps} dil={list(dilations)} "
              f"-> reach radius {r} bins", flush=True)

    def clamp_positions(self):
        for b in self.blocks:
            b.dwconv.clamp_positions()

    def forward(self, x):
        for blk in self.blocks:
            x = _ckpt(blk, x, use_reentrant=False)
        return x
