"""Draw training batches directly from the glyph store."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


def to_unit(glyphs: np.ndarray) -> Tensor:
    """Map uint8 ink to [-1, 1] with paper at -1."""
    return torch.from_numpy(np.asarray(glyphs, dtype=np.float32) / 127.5 - 1.0)


@dataclass
class Batch:
    target: Tensor
    chars: Tensor
    refs: Tensor
    ref_chars: Tensor

    def to(self, device: torch.device) -> Batch:
        return Batch(
            self.target.to(device),
            self.chars.to(device),
            self.refs.to(device),
            self.ref_chars.to(device),
        )


def sample_batch(
    glyphs: np.ndarray,
    style_ids: np.ndarray,
    fonts: int,
    per_font: int,
    k_range: tuple[int, int],
    rng: np.random.Generator,
) -> Batch:
    """Draw ``per_font`` targets and K references from each of ``fonts`` styles.

    Targets from one font share a single pass through the style encoder, which is the larger half
    of the cost of a step when K is high. References never include a target, and K is shared across
    the batch so that references stack into one tensor.
    """
    n_chars = glyphs.shape[1]
    styles = rng.choice(style_ids, size=fonts)
    order = np.argsort(rng.random((fonts, n_chars)), axis=1)
    k = int(rng.integers(k_range[0], k_range[1] + 1))
    chars = order[:, :per_font]
    refs = order[:, per_font : per_font + k]
    return Batch(
        to_unit(glyphs[styles[:, None], chars]).reshape(fonts * per_font, 1, *glyphs.shape[2:]),
        torch.from_numpy(chars.reshape(-1)),
        to_unit(glyphs[styles[:, None], refs]),
        torch.from_numpy(refs),
    )
