"""The nearest-neighbour font baseline.

Given the reference letters of an unseen font, find the training font whose same letters are
closest in pixels, then copy that font's glyphs. It needs no training and its output is perfectly
sharp, which makes it a hard opponent on a pixel metric: a confident copy of a similar font can
outscore a generated glyph whose strokes sit one pixel off.
"""

from __future__ import annotations

import numpy as np
import torch


def glyph_distances(
    glyphs: np.ndarray, corpus_ids: np.ndarray, query_ids: np.ndarray, chunk: int = 256
) -> torch.Tensor:
    """L1 distance per character between query and corpus styles of ``glyphs``.

    The result has shape ``(chars, Q, C)``, so a lookup for any subset of letters is a sum of
    slices. Only one character's images are read at a time, which keeps this far below the size of
    the corpus even when ``glyphs`` is a memory map.
    """
    n_chars = glyphs.shape[1]
    size = glyphs.shape[2] * glyphs.shape[3]
    out = torch.empty(n_chars, len(query_ids), len(corpus_ids))
    for char in range(n_chars):
        corpus = np.asarray(glyphs[corpus_ids, char], dtype=np.float32) / 255.0
        references = torch.from_numpy(corpus.reshape(len(corpus_ids), size))
        for start in range(0, len(query_ids), chunk):
            ids = query_ids[start : start + chunk]
            queries = np.asarray(glyphs[ids, char], dtype=np.float32) / 255.0
            block = torch.from_numpy(queries.reshape(len(ids), size))
            out[char, start : start + chunk] = torch.cdist(block, references, p=1)
    return out / size


def nearest_fonts(distances: torch.Tensor, chars: list[int]) -> torch.Tensor:
    """Pick the corpus style closest to each query over ``chars``."""
    return distances[chars].sum(dim=0).argmin(dim=1)
