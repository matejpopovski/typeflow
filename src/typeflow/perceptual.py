"""A perceptual distance built for glyphs.

LPIPS would mean depending on an ImageNet network that has never seen a serif. This classifier
learns to name the font and the character from one glyph, using training fonts only, so that its
features are sensitive to exactly what a typeface comparison should care about. It shares no
weights with the generative model, which keeps the model from marking its own work. The character
head doubles as a legibility check on generated glyphs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .data import to_unit


class GlyphNet(nn.Module):
    def __init__(self, fonts: int, chars: int, widths: tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        self.stages = nn.ModuleList()
        current = 1
        for width in widths:
            self.stages.append(
                nn.Sequential(
                    nn.Conv2d(current, width, 3, stride=2, padding=1),
                    nn.GroupNorm(8, width),
                    nn.SiLU(),
                    nn.Conv2d(width, width, 3, padding=1),
                    nn.GroupNorm(8, width),
                    nn.SiLU(),
                )
            )
            current = width
        self.font_head = nn.Linear(current, fonts)
        self.char_head = nn.Linear(current, chars)
        self.fonts, self.chars, self.widths = fonts, chars, widths

    def features(self, x: Tensor) -> list[Tensor]:
        out = []
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        return out

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        pooled = self.features(x)[-1].mean(dim=(2, 3))
        return self.font_head(pooled), self.char_head(pooled)


def feature_distance(net: GlyphNet, a: Tensor, b: Tensor) -> Tensor:
    """LPIPS-style distance between batches of images in [-1, 1], shaped ``(N, 1, H, W)``."""
    total = torch.zeros(a.shape[0], device=a.device)
    for x, y in zip(net.features(a), net.features(b)):
        difference = F.normalize(x, dim=1) - F.normalize(y, dim=1)
        total = total + difference.square().sum(dim=1).mean(dim=(1, 2))
    return total / len(net.widths)


def held_out_mask(styles: int, chars: int, share: float = 0.1, seed: int = 7) -> np.ndarray:
    """Mark (style, char) pairs kept from training, to check that font naming generalises."""
    return np.random.default_rng(seed).random((styles, chars)) < share


def train_perceptual(
    glyphs: np.ndarray,
    style_ids: np.ndarray,
    device: torch.device,
    out: Path,
    steps: int = 4000,
    batch: int = 96,
    lr: float = 1e-3,
    seed: int = 0,
    eval_batches: int = 40,
) -> dict[str, float]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_chars = glyphs.shape[1]
    net = GlyphNet(len(style_ids), n_chars).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=steps)
    mask = held_out_mask(len(style_ids), n_chars)

    def draw(held_out: bool) -> tuple[Tensor, Tensor, Tensor]:
        fonts, chars = np.nonzero(mask == held_out)
        pick = rng.integers(0, len(fonts), size=batch)
        rows, columns = fonts[pick], chars[pick]
        images = to_unit(glyphs[style_ids[rows], columns])[:, None].to(device)
        return images, torch.from_numpy(rows).to(device), torch.from_numpy(columns).to(device)

    started = time.monotonic()
    net.train()
    for _ in range(steps):
        images, fonts, chars = draw(held_out=False)
        font_logits, char_logits = net(images)
        loss = F.cross_entropy(font_logits, fonts) + F.cross_entropy(char_logits, chars)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()

    net.eval()
    font_hits = char_hits = seen = 0
    with torch.no_grad():
        for _ in range(eval_batches):
            images, fonts, chars = draw(held_out=True)
            font_logits, char_logits = net(images)
            font_hits += int((font_logits.argmax(1) == fonts).sum())
            char_hits += int((char_logits.argmax(1) == chars).sum())
            seen += len(fonts)
    report = {
        "steps": steps,
        "seconds": round(time.monotonic() - started, 1),
        "font_top1_on_held_out_glyphs": round(font_hits / seen, 4),
        "char_top1_on_held_out_glyphs": round(char_hits / seen, 4),
        "font_classes": len(style_ids),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().half() for k, v in net.state_dict().items()}
    torch.save({"fonts": net.fonts, "chars": net.chars, "state": state}, out)
    return report


def load_perceptual(path: Path, device: torch.device) -> GlyphNet:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    net = GlyphNet(saved["fonts"], saved["chars"])
    net.load_state_dict({k: v.float() for k, v in saved["state"].items()})
    return net.to(device).eval().requires_grad_(False)


def write_report(report: dict[str, float], path: Path) -> None:
    path.write_text(json.dumps(report, indent=2) + "\n")
