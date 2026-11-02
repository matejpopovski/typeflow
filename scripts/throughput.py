"""Compare training throughput of the 64x64 grid design with the 32x32 grid that was kept."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from typeflow.data import sample_batch
from typeflow.model import ModelConfig, TypeFlow
from typeflow.store import load_store
from typeflow.train import TrainConfig, flow_loss, pick_device

STEPS, SKIP = 40, 8
DESIGNS = (
    ("64x64 grid", ModelConfig(patch=2, base=40, mults=(1, 2, 3, 3), blocks=2, refine=4)),
    ("32x32 grid", ModelConfig()),
)


def measure(cfg: ModelConfig, glyphs: np.ndarray, ids: np.ndarray, device: torch.device) -> dict:
    train = TrainConfig()
    model = TypeFlow(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    rng = np.random.default_rng(0)
    generator = torch.Generator().manual_seed(0)
    timings = []
    for _ in range(STEPS):
        batch = sample_batch(glyphs, ids, train.fonts, train.per_font, (1, 8), rng).to(device)
        started = time.monotonic()
        loss = flow_loss(model, batch, generator, 0.1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()
        timings.append(time.monotonic() - started)
    seconds = float(np.median(timings[SKIP:]))
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "milliseconds_per_step": round(seconds * 1000, 1),
        "glyphs_per_second": round(train.fonts * train.per_font / seconds, 1),
    }


def main() -> None:
    device = pick_device()
    store = load_store(Path("data"), 128)
    ids = store.split_indices("train")
    report = {name: measure(cfg, store.glyphs, ids, device) for name, cfg in DESIGNS}
    report["device"] = str(device)
    report["glyph_share_of_paper"] = float(1 - np.asarray(store.glyphs[::40]).mean() / 255)
    Path("results/throughput.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
