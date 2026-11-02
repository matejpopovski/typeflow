"""Inference: load weights, read a style from references, draw glyphs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .data import to_unit
from .flow import guided, integrate
from .model import ModelConfig, TypeFlow


@dataclass(frozen=True)
class SamplerConfig:
    """Defaults are the validation winner: lowest perceptual distance within 32 velocity calls."""

    steps: int = 8
    method: str = "euler"
    guidance: float = 1.25
    batch: int = 96


def load_model(path: Path, device: torch.device) -> TypeFlow:
    """Load either exported weights or a training checkpoint's EMA."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = TypeFlow(ModelConfig.from_dict(saved["model_config"]))
    state = saved.get("ema", saved.get("weights"))
    model.load_state_dict({k: v.float() for k, v in state.items()})
    return model.to(device).eval().requires_grad_(False)


def export_weights(checkpoint: Path, out: Path) -> int:
    """Strip a checkpoint to half-precision EMA weights so that it fits in the repository."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": saved["model_config"],
            "weights": {k: v.half() for k, v in saved["ema"].items()},
            "step": saved["step"],
        },
        out,
    )
    return out.stat().st_size


@torch.no_grad()
def encode_styles(
    model: TypeFlow, refs: np.ndarray, ref_chars: np.ndarray, batch: int = 64
) -> Tensor:
    """Style vectors for uint8 references shaped ``(F, K, H, W)``, with ``ref_chars`` ``(F, K)``."""
    device = next(model.parameters()).device
    out = []
    for start in range(0, len(refs), batch):
        images = to_unit(refs[start : start + batch]).to(device)
        chars = torch.from_numpy(ref_chars[start : start + batch]).to(device)
        out.append(model.encode_style(images, chars))
    return torch.cat(out)


def noise_for(shape: tuple[int, ...], seed: int) -> Tensor:
    """Noise from a CPU generator, so one seed means the same glyph on any device."""
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def sample_glyphs(
    model: TypeFlow, style: Tensor, chars: Tensor, noise: Tensor, cfg: SamplerConfig
) -> np.ndarray:
    """Draw one glyph per row of ``style`` and ``chars``, returned as uint8 with ink at 255."""
    device = next(model.parameters()).device
    out = []
    for start in range(0, len(style), cfg.batch):
        stop = start + cfg.batch
        s, c = style[start:stop].to(device), chars[start:stop].to(device)
        null = model.null_style[None].expand(len(s), -1)

        def conditional(x: Tensor, t: Tensor, s: Tensor = s, c: Tensor = c) -> Tensor:
            return model(x, t, c, s)

        def unconditional(x: Tensor, t: Tensor, c: Tensor = c, null: Tensor = null) -> Tensor:
            return model(x, t, c, null)

        velocity = guided(conditional, unconditional, cfg.guidance)
        x = integrate(
            velocity, noise[start:stop].to(device), cfg.steps, cfg.method, skip_endpoint=True
        )
        out.append(x.cpu())
    glyphs = torch.cat(out)[:, 0]
    return ((glyphs.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).numpy()
