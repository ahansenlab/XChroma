"""XChroma contact-model architecture. DNA and normalized ATAC at 1-bp resolution
produce a symmetric 1000-by-1000 native ln(O/E) map for a 1-Mb interval.
The current output-ensemble weights are listed in checkpoints/current_model.json.
Architecture: anti-aliased encoder, sum bridge, and 21 DCLS decoder blocks
with nine learnable sampling positions per channel."""
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
        assert g * g == m_taps, f"m_taps={m_taps} must be a perfect square"
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

ENC_KERNEL = 9
POOL_FACTORS = [5, 5, 5, 4, 2, 1]
ENC_W_END = 128
ENC_WIDTHS = [64, 96, 128, 128, 128, 128, 128]

DCLS_DIM = 64
DCLS_K = 7
DCLS_TAPS = 9
DCLS_MLP_RATIO = 2.0
DCLS_DIL = [1, 2, 4, 8, 16, 32, 64] * 3

DCLS_POS_LR_MULT = 5.0

class MaxBlurPool1d(nn.Module):
    """Max pooling followed by a fixed binomial low-pass filter and subsampling."""

    def __init__(self, channels, p):
        super().__init__()
        self.p = int(p)
        self.channels = int(channels)
        self.maxpool = nn.MaxPool1d(kernel_size=self.p, stride=1, padding=(self.p - 1) // 2)
        K = 2 * self.p - 1
        coef = torch.tensor([math.comb(K - 1, i) for i in range(K)], dtype=torch.float32)
        coef = coef / coef.sum()
        self.register_buffer("blur", coef.view(1, 1, K).repeat(self.channels, 1, 1))
        self.bpad = (K - 1) // 2

    def forward(self, x):
        x = self.maxpool(x)
        return F.conv1d(x, self.blur, stride=self.p, padding=self.bpad, groups=self.channels)

def _gn(c):
    return nn.GroupNorm(8 if c % 8 == 0 else 1, c)

class XChromaNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(ENC_WIDTHS[0], ENC_WIDTHS[0], kernel_size=ENC_KERNEL, padding=ENC_KERNEL // 2),
            nn.BatchNorm1d(ENC_WIDTHS[0]), nn.ReLU(inplace=True),
            nn.Conv1d(ENC_WIDTHS[0], ENC_WIDTHS[0], kernel_size=ENC_KERNEL, padding=ENC_KERNEL // 2),
            nn.BatchNorm1d(ENC_WIDTHS[0]), nn.ReLU(inplace=True))
        for i, c in enumerate(ENC_WIDTHS[1:], start=2):
            setattr(self, f"conv{i}", nn.Sequential(
                _gn(c), nn.GELU(),
                nn.Conv1d(c, c, kernel_size=ENC_KERNEL, padding=ENC_KERNEL // 2),
                _gn(c), nn.GELU(),
                nn.Conv1d(c, c, kernel_size=ENC_KERNEL, padding=ENC_KERNEL // 2)))

        self.final = nn.Sequential(
            nn.Conv2d(64, 5, kernel_size=1, padding=0),
            nn.BatchNorm2d(5), nn.ReLU(inplace=True),
            nn.Conv2d(5, 1, kernel_size=1, padding=0))

        if hasattr(self, "lconvtwos"):
            del self.lconvtwos
        if hasattr(self, "convtwos"):
            del self.convtwos
        self.convnext_decoder_2d = DCLSDecoder2D(
            dim=DCLS_DIM, dilations=tuple(DCLS_DIL), m_taps=DCLS_TAPS, K=DCLS_K,
            mlp_ratio=DCLS_MLP_RATIO)

        self.cnxt_stem = (nn.Identity() if DCLS_DIM == ENC_W_END else
                          nn.Sequential(nn.Conv2d(ENC_W_END, DCLS_DIM, 1),
                                        nn.GroupNorm(8, DCLS_DIM), nn.GELU()))

        self.cnxt_head = (nn.Identity() if DCLS_DIM == 64 else
                          nn.Sequential(nn.Conv2d(DCLS_DIM, 64, 1),
                                        nn.GroupNorm(8, 64), nn.GELU()))
        _n = sum(p.numel() for p in self.convnext_decoder_2d.parameters())
        print(f"[arch] DECODER=dcls_e1  dim={DCLS_DIM} K={DCLS_K} taps={DCLS_TAPS} "
              f"mlp={DCLS_MLP_RATIO} -> {_n/1e6:.2f}M params", flush=True)

        self.lconv1 = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=ENC_KERNEL, padding=ENC_KERNEL // 2),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=ENC_KERNEL, padding=ENC_KERNEL // 2),
            nn.BatchNorm1d(64),
        )

        layers = [
            (64, 96, "lconv2"),
            (96, 128, "lconv3"),
            (128, 128, "lconv4"),
            (128, 128, "lconv5"),
            (128, 128, "lconv6"),
            (128, 128, "lconv7"),
        ]

        for i, (in_ch, out_ch, name) in enumerate(layers):
            p = POOL_FACTORS[i]
            pool = nn.Identity() if p <= 1 else MaxBlurPool1d(in_ch, p)
            setattr(self, name, nn.Sequential(
                pool,
                _gn(in_ch), nn.GELU(),
                nn.Conv1d(in_ch, out_ch, kernel_size=ENC_KERNEL, padding=ENC_KERNEL // 2),
                _gn(out_ch), nn.GELU(),
                nn.Conv1d(out_ch, out_ch, kernel_size=ENC_KERNEL, padding=ENC_KERNEL // 2),
            ))
        print(f"[arch] POOL_TYPE=blur  POOL_FACTORS={','.join(map(str, POOL_FACTORS))}", flush=True)
        print("[arch] BRIDGE=sum", flush=True)

    def forward(self, x):

        def run0(x):
            lout1 = self.lconv1(x);             out1 = self.conv1(lout1)
            lout2 = self.lconv2(out1 + lout1);  out2 = self.conv2(lout2)
            lout3 = self.lconv3(out2 + lout2);  out3 = self.conv3(lout3)
            lout4 = self.lconv4(out3 + lout3);  out4 = self.conv4(lout4)
            lout5 = self.lconv5(out4 + lout4);  out5 = self.conv5(lout5)
            lout6 = self.lconv6(out5 + lout5);  out6 = self.conv6(lout6)
            lout7 = self.lconv7(out6 + lout6);  out7 = self.conv7(lout7)

            return out7[:, :, :, None] + out7[:, :, None, :]

        cur = _ckpt(run0, x, use_reentrant=False)

        def run1(cur):
            cur = self.cnxt_stem(cur)
            cur = self.convnext_decoder_2d(cur)
            cur = self.cnxt_head(cur)
            cur = self.final(cur)
            return 0.5 * cur + 0.5 * cur.transpose(2, 3)

        return run1(cur)

    def clamp_positions(self):
        """Clamp learnable tap positions to the kernel extent after optimizer updates."""
        self.convnext_decoder_2d.clamp_positions()

# Retain the original import name for existing callers.
XChromNet = XChromaNet

def _replace_bn_with_gn(module):
    """Replace BatchNorm with GroupNorm using eight groups where possible."""
    def _groups(channels):
        return 8 if channels % 8 == 0 else 1
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
            setattr(module, name, nn.GroupNorm(_groups(child.num_features), child.num_features))
        else:
            _replace_bn_with_gn(child)
    return module

def dcls_param_groups(net, lr, pos_mult=DCLS_POS_LR_MULT):
    """Separate learnable positions with a multiplied learning rate and zero weight decay."""
    params = [p for p in net.parameters() if p.requires_grad]
    pos = [p for p in params if getattr(p, "_dcls_position", False)]
    if not pos:
        return params
    ids = {id(p) for p in pos}
    rest = [p for p in params if id(p) not in ids]
    print(f"[dcls] {len(pos)} position tensors in a separate group: lr={lr}*{pos_mult}, weight_decay=0", flush=True)
    return [{"params": rest, "lr": lr},
            {"params": pos, "lr": lr * pos_mult, "weight_decay": 0.0}]

def build_net():
    net = XChromaNet()
    net = _replace_bn_with_gn(net)
    print("[arch] Replaced BatchNorm with GroupNorm", flush=True)
    return net
