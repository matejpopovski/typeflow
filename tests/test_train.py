import json
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import synthetic_store

from typeflow.data import sample_batch, to_unit
from typeflow.generate import export_weights, load_model
from typeflow.model import ModelConfig, TypeFlow, tiny_config
from typeflow.train import (
    TrainConfig,
    ema_decay,
    flow_loss,
    learning_rate,
    seed_character_maps,
    train,
)

CPU = torch.device("cpu")
GLYPHS = synthetic_store(fonts=8)
IDS = np.arange(8)


def config(**overrides: object) -> TrainConfig:
    values = dict(fonts=4, per_font=2, warmup_steps=2, max_steps=50, log_every=2, lr=1e-3)
    return TrainConfig(**{**values, **overrides})


def test_to_unit_maps_paper_and_ink_to_the_ends() -> None:
    out = to_unit(np.array([0, 255], dtype=np.uint8))
    assert out.tolist() == [-1.0, 1.0]


def test_batch_shapes_group_targets_by_font() -> None:
    batch = sample_batch(GLYPHS, IDS, 4, 3, (2, 2), np.random.default_rng(0))
    assert batch.target.shape == (12, 1, 32, 32)
    assert batch.chars.shape == (12,)
    assert batch.refs.shape == (4, 2, 32, 32)


def test_references_never_include_a_target() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        batch = sample_batch(GLYPHS, IDS, 4, 6, (8, 8), rng)
        chars = batch.chars.reshape(4, 6).numpy()
        for targets, refs in zip(chars, batch.ref_chars.numpy()):
            assert not set(targets) & set(refs)
            assert len(set(refs)) == 8


def test_targets_and_references_come_from_the_same_font() -> None:
    glyphs = np.zeros((3, 65, 8, 8), dtype=np.uint8)
    for font in range(3):
        glyphs[font] = font * 100
    batch = sample_batch(glyphs, np.arange(3), 8, 2, (3, 3), np.random.default_rng(0))
    targets = batch.target[:, 0, 0, 0].reshape(8, 2)
    assert torch.equal(targets[:, 0], targets[:, 1])
    assert torch.equal(batch.refs[:, 0, 0, 0], targets[:, 0])


def test_k_varies_within_its_range() -> None:
    rng = np.random.default_rng(2)
    seen = {sample_batch(GLYPHS, IDS, 2, 1, (1, 4), rng).refs.shape[1] for _ in range(60)}
    assert seen == {1, 2, 3, 4}


def test_learning_rate_warms_up_then_decays() -> None:
    cfg = TrainConfig(warmup_steps=10, max_steps=100, lr=1.0, budget_seconds=1e9)
    assert learning_rate(cfg, 0, 0) == pytest.approx(0.1)
    assert learning_rate(cfg, 10, 0) > learning_rate(cfg, 50, 0) > learning_rate(cfg, 99, 0)
    assert learning_rate(cfg, 100, 0) == pytest.approx(0)


def test_a_slow_run_still_finishes_decayed() -> None:
    cfg = TrainConfig(warmup_steps=10, max_steps=1000, lr=1.0, budget_seconds=100.0)
    assert learning_rate(cfg, 50, 100.0) == pytest.approx(0)


def test_character_maps_start_from_each_letters_mean_shape() -> None:
    glyphs = np.zeros((4, 65, 32, 32), dtype=np.uint8)
    glyphs[:, 0, :8, :] = 255
    glyphs[:, 1, -8:, :] = 255
    model = TypeFlow(tiny_config())
    before = model.content[:, 1:].clone()
    seed_character_maps(model, glyphs, np.arange(4))
    first, second = model.content[0, 0], model.content[1, 0]
    assert first[:2].mean() > first[-2:].mean()
    assert second[:2].mean() < second[-2:].mean()
    assert torch.equal(model.content[:, 1:], before)


def test_ema_decay_starts_at_zero_influence_of_the_past_and_ramps_to_target() -> None:
    assert ema_decay(0.999, 0) == pytest.approx(0.1)
    assert ema_decay(0.999, 100) < ema_decay(0.999, 1000)
    assert ema_decay(0.999, 1_000_000) == 0.999


@pytest.mark.slow
def test_ema_tracks_training_early_on(tmp_path: Path) -> None:
    """A fixed high decay scored ten times worse than the live weights after 3000 steps."""
    cfg = config(max_steps=40, ema=0.9995)
    ema = train(GLYPHS, IDS, tiny_config(), cfg, tmp_path, CPU)
    live = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)["model"]
    untrained = TypeFlow(tiny_config()).state_dict()
    near, far = 0.0, 0.0
    for name, value in ema.state_dict().items():
        near += (value - live[name]).norm().item()
        far += (untrained[name] - live[name]).norm().item()
    assert near < 0.5 * far


def test_style_dropout_of_one_ignores_the_references() -> None:
    torch.manual_seed(0)
    model = TypeFlow(tiny_config())
    with torch.no_grad():
        model.unet.refine_out.weight.normal_(std=0.1)
    batch = sample_batch(GLYPHS, IDS, 2, 2, (2, 2), np.random.default_rng(0))
    other = sample_batch(GLYPHS, IDS, 2, 2, (2, 2), np.random.default_rng(0))
    other.refs = torch.randn_like(other.refs)
    a = flow_loss(model, batch, torch.Generator().manual_seed(0), 1.0)
    b = flow_loss(model, other, torch.Generator().manual_seed(0), 1.0)
    assert a.item() == pytest.approx(b.item())


@pytest.mark.slow
def test_loss_falls_on_a_tiny_model(tmp_path: Path) -> None:
    train(GLYPHS, IDS, tiny_config(), config(max_steps=60, ema=0.9), tmp_path, CPU)
    losses = [json.loads(line)["loss"] for line in (tmp_path / "log.jsonl").open()]
    # Flow matching has a large irreducible term, because x1 - noise is unknowable from x_t alone.
    assert np.mean(losses[-5:]) < 0.75 * np.mean(losses[:5])


@pytest.mark.slow
def test_resuming_reproduces_an_uninterrupted_run(tmp_path: Path) -> None:
    cfg = config(max_steps=6)
    straight = train(GLYPHS, IDS, tiny_config(), cfg, tmp_path / "a", CPU)
    train(GLYPHS, IDS, tiny_config(), cfg, tmp_path / "b", CPU, stop_after=3)
    resumed = train(GLYPHS, IDS, tiny_config(), cfg, tmp_path / "b", CPU)
    for x, y in zip(straight.state_dict().values(), resumed.state_dict().values()):
        assert torch.equal(x, y)


def test_non_finite_loss_stops_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object) -> torch.Tensor:
        return torch.tensor(float("nan"), requires_grad=True)

    monkeypatch.setattr("typeflow.train.flow_loss", broken)
    with pytest.raises(FloatingPointError):
        train(GLYPHS, IDS, tiny_config(), config(max_steps=4), tmp_path, CPU)


@pytest.mark.slow
def test_exported_weights_are_half_precision_and_load(tmp_path: Path) -> None:
    train(GLYPHS, IDS, tiny_config(), config(max_steps=2), tmp_path, CPU)
    out = tmp_path / "weights.pt"
    assert export_weights(tmp_path / "latest.pt", out) > 0
    saved = torch.load(out, weights_only=False)
    assert all(v.dtype == torch.float16 for v in saved["weights"].values() if v.is_floating_point())
    model = load_model(out, CPU)
    assert isinstance(model.cfg, ModelConfig) and not model.training


def test_committed_weights_load_and_draw_when_present() -> None:
    path = Path(__file__).parents[1] / "weights" / "typeflow.pt"
    if not path.exists():
        pytest.skip("weights not exported yet")
    model = load_model(path, CPU)
    size = model.cfg.image_size
    refs = torch.zeros(1, 2, size, size) - 1
    style = model.encode_style(refs, torch.tensor([[0, 1]]))
    with torch.no_grad():
        out = model(torch.randn(1, 1, size, size), torch.zeros(1), torch.tensor([2]), style)
    assert out.shape == (1, 1, size, size) and torch.isfinite(out).all()
