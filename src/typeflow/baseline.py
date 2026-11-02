"""The nearest-neighbour font baseline.

Given the reference letters of an unseen font, find the training font whose same letters are
closest in pixels, then copy that font's glyphs. It needs no training and its output is perfectly
sharp, which makes it a hard opponent on a pixel metric: a confident copy of a similar font can
outscore a generated glyph whose strokes sit one pixel off.
"""

from __future__ import annotations

import numpy as np
import torch


def glyph_distances(corpus: np.ndarray, queries: np.ndarray, chunk: int = 256) -> torch.Tensor:
    """L1 distance per character between query and corpus styles.

    ``corpus`` has shape ``(C, chars, H, W)`` and ``queries`` has shape ``(Q, chars, H, W)``. The
    result has shape ``(chars, Q, C)``, so a lookup for any subset of letters is a sum of slices.
    Memory stays bounded because only one character's images are in play at a time.
    """
    n_chars = corpus.shape[1]
    out = torch.empty(n_chars, len(queries), len(corpus))
    for char in range(n_chars):
        references = torch.from_numpy(
            np.asarray(corpus[:, char], dtype=np.float32).reshape(len(corpus), -1) / 255.0
        )
        for start in range(0, len(queries), chunk):
            block = torch.from_numpy(
                np.asarray(queries[start : start + chunk, char], dtype=np.float32).reshape(
                    -1, references.shape[1]
                )
                / 255.0
            )
            out[char, start : start + chunk] = torch.cdist(block, references, p=1)
    return out / (corpus.shape[2] * corpus.shape[3])


def nearest_fonts(distances: torch.Tensor, chars: list[int]) -> torch.Tensor:
    """Pick the corpus style closest to each query over ``chars``."""
    return distances[chars].sum(dim=0).argmin(dim=1)
