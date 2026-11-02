"""Show why a trained flow must not be evaluated at t=1.

Reports how well the network still recovers the noise as t approaches 1, then scores Heun and RK4
on validation fonts with and without the final stage at t=1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from typeflow import evaluate as ev
from typeflow.data import to_unit
from typeflow.flow import flow_pair, guided, integrate
from typeflow.generate import encode_styles, load_model, noise_for
from typeflow.perceptual import load_perceptual
from typeflow.store import load_store
from typeflow.train import pick_device

TIMES = (0.5, 0.9, 0.99, 0.999, 1.0)
SOLVERS = (("heun", 16), ("rk4", 8))
FONTS, CHARS, SEED = 60, 12, 0


def noise_recovery(model, store, device: torch.device) -> list[dict[str, float]]:
    ids = np.sort(store.split_indices("val"))[:32]
    refs = np.tile(np.array(ev.reference_chars(4)), (32, 1))
    style = encode_styles(model, np.asarray(store.glyphs[ids[:, None], refs]), refs)
    char = ev.TARGET_CHARS[0]
    x1 = to_unit(store.glyphs[ids, char])[:, None].to(device)
    chars = torch.full((32,), char, device=device)
    noise = noise_for(x1.shape, SEED).to(device)
    paper = x1 < -0.99
    rows = []
    with torch.no_grad():
        for value in TIMES:
            t = torch.full((32,), value, device=device)
            x, target = flow_pair(x1, noise, t)
            predicted = model(x, t, chars, style)
            pair = torch.stack([predicted[paper], -noise[paper]])
            rows.append(
                {
                    "t": value,
                    "velocity_rmse": float((predicted - target).square().mean().sqrt()),
                    "correlation_with_true_noise_on_paper": float(torch.corrcoef(pair)[0, 1]),
                }
            )
    return rows


def score_solver(model, net, store, split, device, method: str, steps: int, skip: bool) -> dict:
    refs = ev.reference_chars(4)
    images = np.asarray(store.glyphs[split.ids][:, refs])
    style = encode_styles(model, images, np.tile(np.array(refs), (len(split.ids), 1)))
    m = len(split.target_chars)
    styles = style.repeat_interleave(m, dim=0)
    chars = torch.from_numpy(split.flat_chars).to(device)
    noise = noise_for((len(styles), 1, store.size, store.size), SEED).to(device)
    out = []
    for start in range(0, len(styles), 96):
        s, c = styles[start : start + 96], chars[start : start + 96]

        def field(x, t, s=s, c=c):
            return model(x, t, c, s)

        with torch.no_grad():
            x = integrate(guided(field, field, 1.0), noise[start : start + 96], steps, method, skip)
        out.append(x.cpu())
    glyphs = ((torch.cat(out)[:, 0].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).numpy()
    scores = ev.score_glyphs(glyphs, split.flatten(split.truth), split.flat_chars, net, device)
    return {
        "method": method,
        "steps": steps,
        "skip_endpoint": skip,
        **{k: float(v.mean()) for k, v in scores.items()},
    }


def main() -> None:
    device = pick_device()
    store = load_store(Path("data"), 128)
    model = load_model(Path("weights/typeflow.pt"), device)
    net = load_perceptual(Path("weights/perceptual.pt"), device)
    val = store.split_indices("val")
    picks = np.sort(np.random.default_rng(SEED).choice(val, size=FONTS, replace=False))
    targets = ev.TARGET_CHARS[:: len(ev.TARGET_CHARS) // CHARS][:CHARS]
    split = ev.make_split(store, picks, targets)
    report = {
        "noise_recovery": noise_recovery(model, store, device),
        "solvers": [
            score_solver(model, net, store, split, device, method, steps, skip)
            for method, steps in SOLVERS
            for skip in (False, True)
        ],
    }
    Path("results/endpoint.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
