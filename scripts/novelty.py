"""Split test fonts by how far they sit from anything in training, and compare methods per group.

This analysis was designed after the first results, so it is exploratory: it explains the headline
comparison and does not replace it. A font's novelty is the distance from its reference letters to
those of its nearest training font, which is known without looking at any target glyph.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from typeflow.baseline import nearest_fonts
from typeflow.evaluate import reference_chars

K = 4
GROUPS = 4
METRICS = ("l1", "ssim", "ink_iou", "perceptual", "legibility")


def main() -> None:
    report = json.loads(Path("results/test.json").read_text())
    distances = torch.from_numpy(np.load("data/outputs/distances.npy"))
    refs = reference_chars(K)
    nearest = nearest_fonts(distances, refs)
    summed = distances[refs].sum(dim=0)
    novelty = (summed[torch.arange(len(nearest)), nearest] / K).numpy()

    model = report["per_font"][f"model_k{K}"]
    copied = report["per_font"][f"nearest_k{K}"]
    edges = np.quantile(novelty, np.linspace(0, 1, GROUPS + 1))
    rows = []
    for group in range(GROUPS):
        mask = (novelty >= edges[group]) & (novelty <= edges[group + 1])
        row = {
            "group": group + 1,
            "fonts": int(mask.sum()),
            "novelty_range": [float(edges[group]), float(edges[group + 1])],
        }
        for metric in METRICS:
            row[f"model_{metric}"] = float(np.array(model[metric])[mask].mean())
            row[f"nearest_{metric}"] = float(np.array(copied[metric])[mask].mean())
        rows.append(row)

    wins = {}
    for metric in METRICS:
        ours, theirs = np.array(model[metric]), np.array(copied[metric])
        better = ours < theirs if metric in ("l1", "perceptual") else ours > theirs
        wins[metric] = float(better.mean())
    out = {"k": K, "groups": rows, "share_of_fonts_where_model_beats_nearest": wins}
    Path("results/novelty.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
