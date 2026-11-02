"""Time sampling on each available device, to decide where evaluation should run."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import torch

from typeflow.flow import velocity_calls
from typeflow.generate import SamplerConfig, load_model, noise_for, sample_glyphs

WEIGHTS = Path("weights/typeflow.pt")
GLYPHS = 96
SAMPLER = SamplerConfig(steps=16, method="euler", guidance=1.0, batch=GLYPHS)


def measure(device: torch.device) -> dict[str, float]:
    model = load_model(WEIGHTS, device)
    size = model.cfg.image_size
    style = torch.zeros(GLYPHS, model.cfg.style_dim)
    chars = torch.arange(GLYPHS) % model.cfg.chars
    noise = noise_for((GLYPHS, 1, size, size), 0)
    runs = []
    for _ in range(4):
        started = time.monotonic()
        sample_glyphs(model, style, chars, noise, SAMPLER)
        runs.append(time.monotonic() - started)
    best = min(runs[1:])
    calls = velocity_calls(SAMPLER.method, SAMPLER.steps)
    return {
        "seconds_per_batch": round(best, 3),
        "glyphs_per_second": round(GLYPHS / best, 1),
        "velocity_calls_per_glyph": calls,
    }


def main() -> None:
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    report = {
        "machine": platform.machine(),
        "torch_threads": torch.get_num_threads(),
        "batch": GLYPHS,
        **{name: measure(torch.device(name)) for name in devices},
    }
    Path("results/decode_speed.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
