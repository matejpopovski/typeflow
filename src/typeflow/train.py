"""The training loop."""

from __future__ import annotations

import faulthandler
import json
import math
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from .data import Batch, sample_batch
from .flow import flow_pair, sample_time
from .model import ModelConfig, TypeFlow
from .runtime import atomic_save, beat, synchronize, to_cpu

CHECKPOINT = "latest.pt"


@dataclass(frozen=True)
class TrainConfig:
    fonts: int = 8
    per_font: int = 6
    lr: float = 2e-4
    weight_decay: float = 1e-4
    ema: float = 0.9995
    grad_clip: float = 1.0
    warmup_steps: int = 500
    max_steps: int = 100_000
    budget_seconds: float = 3600.0
    k_min: int = 1
    k_max: int = 8
    style_dropout: float = 0.1
    seed: int = 0
    log_every: int = 50
    checkpoint_seconds: float = 300.0
    # Keeps Metal's unified allocations under the 10 GB this project is allowed.
    mps_memory_fraction: float = 0.33


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def learning_rate(cfg: TrainConfig, step: int, elapsed: float) -> float:
    """Cosine decay driven by whichever of steps or wall clock is further along.

    On a machine shared with other jobs the step rate is unknown in advance. Driving the schedule
    from elapsed time as well means a slow run still finishes at a low learning rate when the budget
    ends, instead of being cut off near the peak.
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = max(step / cfg.max_steps, elapsed / cfg.budget_seconds)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def flow_loss(
    model: TypeFlow, batch: Batch, generator: torch.Generator, style_dropout: float
) -> Tensor:
    b = batch.target.shape[0]
    device = batch.target.device
    per_font = b // batch.refs.shape[0]
    t = sample_time(b, generator).to(device)
    noise = torch.randn(batch.target.shape, generator=generator).to(device)
    x_t, target = flow_pair(batch.target, noise, t)
    style = model.encode_style(batch.refs, batch.ref_chars).repeat_interleave(per_font, dim=0)
    drop = (torch.rand(b, generator=generator) < style_dropout).to(device)
    style = torch.where(drop[:, None], model.null_style[None], style)
    return (model(x_t, t, batch.chars, style) - target).square().mean()


@torch.no_grad()
def update_ema(ema: TypeFlow, model: TypeFlow, decay: float) -> None:
    for tracked, live in zip(ema.parameters(), model.parameters()):
        tracked.lerp_(live, 1 - decay)


def build(
    model_cfg: ModelConfig, cfg: TrainConfig, device: torch.device
) -> tuple[TypeFlow, TypeFlow, torch.optim.Optimizer]:
    torch.manual_seed(cfg.seed)
    model = TypeFlow(model_cfg).to(device)
    ema = TypeFlow(model_cfg).to(device)
    ema.load_state_dict(model.state_dict())
    ema.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return model, ema, optimizer


def save_checkpoint(
    run_dir: Path,
    model: TypeFlow,
    ema: TypeFlow,
    optimizer: torch.optim.Optimizer,
    step: int,
    elapsed: float,
    rng: np.random.Generator,
    generator: torch.Generator,
) -> None:
    device = next(model.parameters()).device
    synchronize(device)
    state = {
        "model_config": model.cfg.to_dict(),
        "model": to_cpu(model.state_dict()),
        "ema": to_cpu(ema.state_dict()),
        "optimizer": to_cpu(optimizer.state_dict()),
        "step": step,
        "elapsed": elapsed,
        "numpy_rng": rng.bit_generator.state,
        "torch_rng": generator.get_state(),
    }
    atomic_save(state, run_dir / CHECKPOINT)


def train(
    glyphs: np.ndarray,
    style_ids: np.ndarray,
    model_cfg: ModelConfig,
    cfg: TrainConfig,
    run_dir: Path,
    device: torch.device,
    resume: bool = True,
    stop_after: int | None = None,
) -> TypeFlow:
    """Train, resuming from ``run_dir`` when a checkpoint exists, and return the EMA model."""
    run_dir.mkdir(parents=True, exist_ok=True)
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(cfg.mps_memory_fraction)

    model, ema, optimizer = build(model_cfg, cfg, device)
    rng = np.random.default_rng(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    step, elapsed = 0, 0.0

    checkpoint = run_dir / CHECKPOINT
    if resume and checkpoint.exists():
        state: dict[str, Any] = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        step, elapsed = state["step"], state["elapsed"]
        rng.bit_generator.state = state["numpy_rng"]
        generator.set_state(state["torch_rng"])
    else:
        (run_dir / "config.json").write_text(
            json.dumps({"model": model_cfg.to_dict(), "train": asdict(cfg)}, indent=2)
        )

    log = (run_dir / "log.jsonl").open("a")
    started = time.monotonic() - elapsed
    last_save = time.monotonic()
    window: list[Tensor] = []
    window_start = time.monotonic()

    def persist() -> None:
        save_checkpoint(
            run_dir, model, ema, optimizer, step, time.monotonic() - started, rng, generator
        )

    model.train()
    limit = cfg.max_steps if stop_after is None else min(cfg.max_steps, stop_after)
    while step < limit and time.monotonic() - started < cfg.budget_seconds:
        lr = learning_rate(cfg, step, time.monotonic() - started)
        for group in optimizer.param_groups:
            group["lr"] = lr
        k_range = (cfg.k_min, cfg.k_max)
        batch = sample_batch(glyphs, style_ids, cfg.fonts, cfg.per_font, k_range, rng).to(device)
        loss = flow_loss(model, batch, generator, cfg.style_dropout)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        update_ema(ema, model, cfg.ema)
        step += 1
        window.append(loss.detach())
        if step % cfg.log_every == 0:
            mean = float(torch.stack(window).mean())
            if not math.isfinite(mean):
                raise FloatingPointError(f"loss became {mean} at step {step}")
            now = time.monotonic()
            log.write(
                json.dumps(
                    {
                        "step": step,
                        "loss": round(mean, 5),
                        "lr": lr,
                        "elapsed": round(now - started, 1),
                        "steps_per_second": round(len(window) / (now - window_start), 2),
                    }
                )
                + "\n"
            )
            log.flush()
            window, window_start = [], now
            beat(run_dir)
        if time.monotonic() - last_save > cfg.checkpoint_seconds:
            persist()
            last_save = time.monotonic()
            window_start = last_save
    persist()
    log.close()
    (run_dir / "done").write_text(json.dumps({"step": step, "elapsed": time.monotonic() - started}))
    return ema
