"""Hold out typefaces, not files.

Splitting on the google/fonts directory is not enough. "Hind Madurai" and "Hind Siliguri" are
separate directories that share one Latin design, and many script-specific families reuse an
existing Latin outright. A test font with a twin in the training set would flatter the model and,
worse, hand the nearest-neighbour baseline an exact copy. Directories are therefore merged into
clusters before splitting, by shared name stem and by measured glyph similarity.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
import torch


class UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def name_stem(family: str) -> str:
    return family.split()[0].lower()


def glyph_votes(signatures: np.ndarray, glyph_distance: float) -> torch.Tensor:
    """Count, for each pair of styles, the glyphs drawn alike.

    Distance is L1 over total ink, 0 for identical shapes and 1 for disjoint ones. Raw L1 would call
    any two hairline faces alike simply because neither has much ink. Votes accumulate one glyph at
    a time because a full pairwise tensor would need tens of gigabytes.
    """
    styles, glyphs, _ = signatures.shape
    votes = torch.zeros((styles, styles), dtype=torch.int16)
    for glyph in range(glyphs):
        column = torch.from_numpy(signatures[:, glyph].astype(np.float32))
        ink = column.sum(-1)
        distance = torch.cdist(column, column, p=1) / (ink[:, None] + ink[None]).clamp_min(1.0)
        votes += (distance < glyph_distance).short()
    return votes


def twin_pairs(
    signatures: np.ndarray,
    owners: Sequence[str],
    glyph_distance: float = 0.05,
    glyph_share: float = 0.5,
) -> list[tuple[int, int, float]]:
    """Find style pairs from different families that share at least a share of their drawings.

    ``signatures`` has shape ``(styles, glyphs, pixels)``. Comparing whole fonts would miss a family
    that borrows only another's capitals, so each glyph votes separately. Fonts drawn independently
    almost never get this close, and a loose threshold is worse than none: similarity chains through
    the crowd of neutral sans faces and collapses them into a single cluster.
    """
    owner = np.unique(np.asarray(owners), return_inverse=True)[1]
    share = glyph_votes(signatures, glyph_distance).float() / signatures.shape[1]
    rows, cols = torch.nonzero(torch.triu(share >= glyph_share, diagonal=1), as_tuple=True)
    return [
        (i, j, float(share[i, j]))
        for i, j in zip(rows.tolist(), cols.tolist())
        if owner[i] != owner[j]
    ]


def assign_splits(
    cluster_sizes: dict[str, int],
    test_share: float = 0.10,
    val_share: float = 0.05,
    max_held_out_cluster: int = 30,
    salt: str = "typeflow-split-v1",
) -> dict[str, str]:
    """Assign clusters to splits by a salted hash, so that membership is stable under edits.

    Very large clusters such as Noto stay in training; held out, one design would dominate the test
    score.
    """
    total = sum(cluster_sizes.values())
    ordered = sorted(
        cluster_sizes,
        key=lambda c: hashlib.sha256(f"{salt}:{c}".encode()).hexdigest(),
    )
    assignment: dict[str, str] = {}
    budget = Counter(test=round(total * test_share), val=round(total * val_share))
    for cluster in ordered:
        size = cluster_sizes[cluster]
        split = "train"
        if size <= max_held_out_cluster:
            for candidate in ("test", "val"):
                if budget[candidate] >= size:
                    budget[candidate] -= size
                    split = candidate
                    break
        assignment[cluster] = split
    return assignment
