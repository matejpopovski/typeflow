"""The velocity field: a U-Net conditioned by FiLM, and the encoder that reads a typeface's style.

Style reaches the U-Net through one global vector and nothing else. Spatial cross-attention to the
reference glyphs tends to transfer detail better, but a single vector gives interpolation a plain
meaning: a straight line between two typefaces. That property is the point of the project.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .render import CHARSET


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 128
    # The heavy levels run at image_size / patch on patch^2 channels, which keeps full output
    # resolution at the cost of a network working on a quarter of the pixels.
    patch: int = 2
    # Parameters sit in the deep levels and compute in the shallow ones, so capping the deep
    # widths keeps the weights small enough to commit without giving up high-resolution capacity.
    base: int = 40
    mults: tuple[int, ...] = (1, 2, 3, 3)
    blocks: int = 2
    attention_at: tuple[int, ...] = (16, 8)
    heads: int = 4
    cond_dim: int = 192
    style_dim: int = 192
    style_channels: tuple[int, ...] = (16, 32, 64, 128, 160)
    chars: int = len(CHARSET)
    groups: int = 8

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> ModelConfig:
        tuples = ("mults", "attention_at", "style_channels")
        return cls(**{k: tuple(v) if k in tuples else v for k, v in values.items()})


def tiny_config(image_size: int = 32) -> ModelConfig:
    """A model small enough to train for a few steps on a CI runner's CPU."""
    return ModelConfig(
        image_size=image_size,
        patch=2,
        base=8,
        mults=(1, 2),
        blocks=1,
        attention_at=(8,),
        heads=2,
        cond_dim=32,
        style_dim=32,
        style_channels=(8, 16),
        groups=4,
    )


def timestep_embedding(t: Tensor, dim: int) -> Tensor:
    half = dim // 2
    scale = math.log(10_000.0) / (half - 1)
    frequencies = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype) * -scale)
    angles = t[:, None] * 1_000.0 * frequencies[None]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class FiLM(nn.Module):
    """Predict a per-channel scale and shift from the conditioning vector.

    The projection starts at zero, so a fresh block is the identity and the conditioning path
    earns its influence during training instead of injecting noise from the first step.
    """

    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.project = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, h: Tensor, cond: Tensor) -> Tensor:
        scale, shift = self.project(cond)[:, :, None, None].chunk(2, dim=1)
        return h * (1 + scale) + shift


class ResBlock(nn.Module):
    def __init__(self, channels: int, out: int, cond_dim: int, groups: int) -> None:
        super().__init__()
        self.norm_in = nn.GroupNorm(groups, channels)
        self.conv_in = nn.Conv2d(channels, out, 3, padding=1)
        self.norm_out = nn.GroupNorm(groups, out)
        self.film = FiLM(cond_dim, out)
        self.conv_out = nn.Conv2d(out, out, 3, padding=1)
        self.skip = nn.Conv2d(channels, out, 1) if channels != out else nn.Identity()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h = self.conv_in(F.silu(self.norm_in(x)))
        h = self.film(self.norm_out(h), cond)
        return self.skip(x) + self.conv_out(F.silu(h))


class SelfAttention(nn.Module):
    """Whole-glyph context, which convolutions at low levels reach only slowly."""

    def __init__(self, channels: int, heads: int, groups: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(groups, channels)
        self.keys = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor) -> Tensor:
        b, c, height, width = x.shape
        q, k, v = (
            part.reshape(b, self.heads, c // self.heads, -1).transpose(-1, -2)
            for part in self.keys(self.norm(x)).chunk(3, dim=1)
        )
        attended = F.scaled_dot_product_attention(q, k, v)
        return x + self.out(attended.transpose(-1, -2).reshape(b, c, height, width))


class GlyphUNet(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        channels = [cfg.base * m for m in cfg.mults]
        self.stem = nn.Conv2d(cfg.patch**2, channels[0], 3, padding=1)

        resolution = cfg.image_size // cfg.patch
        block = lambda a, b: ResBlock(a, b, cfg.cond_dim, cfg.groups)  # noqa: E731
        attention = lambda c: SelfAttention(c, cfg.heads, cfg.groups)  # noqa: E731

        self.down = nn.ModuleList()
        skips = [channels[0]]
        current = channels[0]
        for level, width in enumerate(channels):
            for _ in range(cfg.blocks):
                layers = nn.ModuleList([block(current, width)])
                if resolution in cfg.attention_at:
                    layers.append(attention(width))
                self.down.append(layers)
                current = width
                skips.append(current)
            if level < len(channels) - 1:
                self.down.append(nn.Conv2d(current, current, 3, stride=2, padding=1))
                skips.append(current)
                resolution //= 2

        self.mid_in = block(current, current)
        self.mid_attention = attention(current)
        self.mid_out = block(current, current)

        self.up = nn.ModuleList()
        for level, width in reversed(list(enumerate(channels))):
            for _ in range(cfg.blocks + 1):
                layers = nn.ModuleList([block(current + skips.pop(), width)])
                if resolution in cfg.attention_at:
                    layers.append(attention(width))
                self.up.append(layers)
                current = width
            if level > 0:
                self.up.append(nn.Conv2d(current, current, 3, padding=1))
                resolution *= 2

        self.norm_out = nn.GroupNorm(cfg.groups, current)
        self.head = nn.Conv2d(current, cfg.patch**2, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h = self.stem(F.pixel_unshuffle(x, self.cfg.patch))
        skips = [h]
        for module in self.down:
            h = module(h) if isinstance(module, nn.Conv2d) else _apply(module, h, cond)
            skips.append(h)

        h = self.mid_out(self.mid_attention(self.mid_in(h, cond)), cond)

        for module in self.up:
            if isinstance(module, nn.Conv2d):
                h = module(F.interpolate(h, scale_factor=2.0, mode="nearest"))
            else:
                h = _apply(module, torch.cat([h, skips.pop()], dim=1), cond)
        return F.pixel_shuffle(self.head(F.silu(self.norm_out(h))), self.cfg.patch)


def _apply(layers: nn.ModuleList, h: Tensor, cond: Tensor) -> Tensor:
    h = layers[0](h, cond)
    return layers[1](h) if len(layers) > 1 else h


class StyleEncoder(nn.Module):
    """Turn K reference glyphs into one style vector, in any order and for any K.

    Each reference is encoded knowing which letter it is, since the same stroke means different
    things in an "l" and an "o". Features are pooled from every stage instead of only the last,
    because stroke contrast and serif shape live in the early, high-resolution stages.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.char = nn.Embedding(cfg.chars, cfg.cond_dim)
        widths = cfg.style_channels
        self.stem = nn.Conv2d(1, widths[0], 3, stride=2, padding=1)
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        for a, b in zip(widths, (*widths[1:], widths[-1])):
            self.blocks.append(ResBlock(a, b, cfg.cond_dim, cfg.groups))
            self.pools.append(nn.Conv2d(b, b, 3, stride=2, padding=1))
        self.pools = self.pools[:-1]
        self.project = nn.Sequential(
            nn.Linear(sum((*widths[1:], widths[-1])), cfg.style_dim * 2),
            nn.SiLU(),
            nn.Linear(cfg.style_dim * 2, cfg.style_dim),
        )

    def forward(self, refs: Tensor, ref_chars: Tensor) -> Tensor:
        """``refs`` has shape ``(B, K, H, W)`` and ``ref_chars`` has shape ``(B, K)``."""
        b, k, height, width = refs.shape
        cond = self.char(ref_chars.reshape(b * k))
        h = self.stem(refs.reshape(b * k, 1, height, width))
        pooled = []
        for stage, block in enumerate(self.blocks):
            h = block(h, cond)
            pooled.append(h.mean(dim=(2, 3)))
            if stage < len(self.pools):
                h = self.pools[stage](h)
        per_reference = torch.cat(pooled, dim=1).reshape(b, k, -1)
        return self.project(per_reference.mean(dim=1))


class TypeFlow(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.style_encoder = StyleEncoder(cfg)
        self.unet = GlyphUNet(cfg)
        self.char = nn.Embedding(cfg.chars, cfg.cond_dim)
        self.null_style = nn.Parameter(torch.zeros(cfg.style_dim))
        self.style_in = nn.Linear(cfg.style_dim, cfg.cond_dim)
        self.time = nn.Sequential(
            nn.Linear(cfg.cond_dim, cfg.cond_dim), nn.SiLU(), nn.Linear(cfg.cond_dim, cfg.cond_dim)
        )
        self.cond_out = nn.Sequential(nn.SiLU(), nn.Linear(cfg.cond_dim, cfg.cond_dim))

    def encode_style(self, refs: Tensor, ref_chars: Tensor) -> Tensor:
        return self.style_encoder(refs, ref_chars)

    def forward(self, x: Tensor, t: Tensor, chars: Tensor, style: Tensor) -> Tensor:
        time = self.time(timestep_embedding(t, self.cfg.cond_dim))
        cond = self.cond_out(time + self.char(chars) + self.style_in(style))
        return self.unet(x, cond)
