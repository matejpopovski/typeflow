"""Check the held-out split for surviving near-copies of training fonts.

A strong nearest-neighbour score could mean the corpus is dense or could mean the split leaks. This
measures which, at full resolution and over every character, using the same ink-normalised distance
as the survey: how far is each test font from its closest training font?
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from typeflow.store import load_store

CHUNK = 64


def main() -> None:
    store = load_store(Path("data"), 128)
    train = np.sort(store.split_indices("train"))
    test = np.sort(store.split_indices("test"))
    ink = store.glyphs[:, :, ::2, ::2]
    corpus = np.asarray(ink[train], dtype=np.float32).reshape(len(train), -1)
    corpus_ink = corpus.sum(axis=1)

    closest = np.empty(len(test))
    partner = np.empty(len(test), dtype=np.int64)
    for start in range(0, len(test), CHUNK):
        block = np.asarray(ink[test[start : start + CHUNK]], dtype=np.float32)
        block = block.reshape(len(block), -1)
        for i, row in enumerate(block):
            distance = np.abs(corpus - row).sum(axis=1) / (corpus_ink + row.sum())
            partner[start + i] = int(distance.argmin())
            closest[start + i] = float(distance.min())

    order = np.argsort(closest)[:8]
    report = {
        "test_fonts": len(test),
        "distance_scale": "0 identical, 1 disjoint, over all characters",
        "quantiles": {
            q: float(np.quantile(closest, float(q)))
            for q in ("0.0", "0.01", "0.05", "0.25", "0.5", "0.75", "1.0")
        },
        "fonts_below_0.05": int((closest < 0.05).sum()),
        "fonts_below_0.10": int((closest < 0.10).sum()),
        "fonts_below_0.15": int((closest < 0.15).sum()),
        "closest_pairs": [
            {
                "test": store.rows[test[i]].key,
                "train": store.rows[train[partner[i]]].key,
                "distance": round(float(closest[i]), 4),
            }
            for i in order
        ],
    }
    Path("results/leak_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
