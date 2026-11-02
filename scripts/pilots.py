"""Rescore the eight-minute pilot runs on one protocol, for the comparison in the README.

The pilots differ in the fixes between them, and were run before those fixes were committed, so
their checkpoints are local and this script documents the measurement rather than a public rerun.
Pilot 1 predates the character maps, so it loads with an empty map.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from typeflow import evaluate as ev
from typeflow.generate import SamplerConfig
from typeflow.model import ModelConfig, TypeFlow
from typeflow.perceptual import load_perceptual
from typeflow.store import load_store
from typeflow.train import pick_device

RUNS = (
    ("global conditioning only, fixed EMA decay", "runs/pilot/latest.pt", "ema"),
    ("global conditioning only, live weights", "runs/pilot/latest.pt", "model"),
    ("character maps from random, EMA ramp", "runs/pilot2/latest.pt", "ema"),
    ("character maps seeded from mean glyphs, EMA ramp", "runs/pilot3/latest.pt", "ema"),
)
SAMPLER = SamplerConfig(steps=32, method="euler", guidance=1.0)


def load(path: str, key: str, device: torch.device) -> TypeFlow:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ModelConfig.from_dict(saved["model_config"])
    state = saved[key]
    if "content" not in state:
        cfg = replace(cfg, content=0)
    model = TypeFlow(cfg)
    model.load_state_dict(state, strict="content" in state)
    return model.to(device).eval().requires_grad_(False)


def main() -> None:
    device = pick_device()
    store = load_store(Path("data"), 128)
    net = load_perceptual(Path("weights/perceptual.pt"), device)
    val = store.split_indices("val")
    picks = np.sort(np.random.default_rng(0).choice(val, size=60, replace=False))
    split = ev.make_split(store, picks, ev.TARGET_CHARS[::4][:14])
    rows = []
    for label, path, key in RUNS:
        model = load(path, key, device)
        output = ev.run_model(model, store, split, 4, SAMPLER)
        scores = ev.score_glyphs(
            split.flatten(output), split.flatten(split.truth), split.flat_chars, net, device
        )
        steps = torch.load(path, map_location="cpu", weights_only=True)["step"]
        row = {"run": label, "steps": steps, **{k: float(v.mean()) for k, v in scores.items()}}
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path("results/pilots.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
