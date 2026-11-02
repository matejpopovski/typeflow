from pathlib import Path

import numpy as np
import torch
from conftest import synthetic_store

from typeflow.evaluate import (
    REFERENCE_ORDER,
    TARGET_CHARS,
    Split,
    evaluate_test,
    interpolation_study,
    make_split,
    reference_chars,
    run_model,
    score_glyphs,
    sweep_sampler,
)
from typeflow.generate import SamplerConfig, noise_for, sample_glyphs
from typeflow.model import TypeFlow, tiny_config
from typeflow.perceptual import GlyphNet
from typeflow.render import CHARSET
from typeflow.store import GlyphStore, StyleRow

CPU = torch.device("cpu")
FAST = SamplerConfig(steps=2, method="euler", batch=16)


def store() -> GlyphStore:
    glyphs = synthetic_store(fonts=12)
    splits = ["train"] * 6 + ["val"] * 2 + ["test"] * 4
    rows = [
        StyleRow(f"Font {i}", f"Font {i}", "SERIF" if i % 2 else "DISPLAY", s, f"c{i // 2}")
        for i, s in enumerate(splits)
    ]
    return GlyphStore(glyphs, rows)


def net() -> GlyphNet:
    torch.manual_seed(0)
    return GlyphNet(6, 65, widths=(8, 16)).eval()


def model() -> TypeFlow:
    torch.manual_seed(0)
    return TypeFlow(tiny_config()).eval()


def test_references_are_never_scored() -> None:
    assert not {CHARSET[i] for i in TARGET_CHARS} & set(REFERENCE_ORDER)
    assert len(TARGET_CHARS) == len(CHARSET) - len(REFERENCE_ORDER)


def test_references_grow_as_prefixes() -> None:
    assert reference_chars(4)[:2] == reference_chars(2)


def test_perfect_prediction_scores_perfectly() -> None:
    truth = synthetic_store(fonts=2, chars=4)[0]
    scores = score_glyphs(truth, truth, np.arange(4), net(), CPU)
    assert np.allclose(scores["l1"], 0)
    assert np.allclose(scores["ssim"], 1)
    assert np.allclose(scores["perceptual"], 0, atol=1e-5)


def test_split_flatten_keeps_font_then_char_order() -> None:
    split = Split(np.arange(2), np.zeros((2, 3, 4, 4)), [7, 8, 9])
    assert split.flat_chars.tolist() == [7, 8, 9, 7, 8, 9]


def test_one_seed_gives_one_output_and_ignores_batching() -> None:
    m = model()
    style = torch.randn(6, m.cfg.style_dim)
    chars = torch.arange(6)
    noise = noise_for((6, 1, 32, 32), 3)
    with torch.no_grad():
        m.unet.head.weight.normal_(std=0.05)
    a = sample_glyphs(m, style, chars, noise, SamplerConfig(steps=2, method="euler", batch=6))
    b = sample_glyphs(m, style, chars, noise, SamplerConfig(steps=2, method="euler", batch=2))
    assert np.abs(a.astype(int) - b.astype(int)).max() <= 1


def test_run_model_returns_a_glyph_per_font_and_target() -> None:
    s = store()
    split = make_split(s, s.split_indices("test"), TARGET_CHARS[:3])
    assert run_model(model(), s, split, 2, FAST).shape == (4, 3, 32, 32)


def test_sampler_sweep_stays_on_validation_fonts(monkeypatch) -> None:
    s = store()
    seen: list[np.ndarray] = []
    real = run_model

    def spy(m, st, split, k, cfg, seed):
        seen.append(split.ids)
        return real(m, st, split, k, cfg, seed)

    monkeypatch.setattr("typeflow.evaluate.run_model", spy)
    rows = sweep_sampler(model(), net(), s, CPU, fonts=2, chars_per_font=2)
    val = set(s.split_indices("val").tolist())
    assert all(set(ids.tolist()) <= val for ids in seen)
    assert {r["method"] for r in rows} == {"euler", "heun", "rk4"}


def test_full_evaluation_runs_and_reports_paired_differences(tmp_path: Path) -> None:
    s = store()
    report = evaluate_test(model(), net(), s, FAST, CPU, tmp_path, counts=(1, 4), every_nth_char=30)
    assert report["test_fonts"] == 4 and report["train_fonts_searched"] == 6
    assert {"oracle_font", "nearest_k4", "model_k4", "model_minus_nearest_k4"} <= set(
        report["summary"]
    )
    assert np.load(tmp_path / "model_k4.npy").shape == (4, report["target_chars"], 32, 32)
    assert set(report["by_category"]) == {"SERIF", "DISPLAY"}


def test_baseline_only_searches_training_fonts(tmp_path: Path) -> None:
    """A test font must not be able to find itself."""
    s = store()
    report = evaluate_test(model(), net(), s, FAST, CPU, tmp_path, counts=(1,), every_nth_char=30)
    assert report["summary"]["oracle_font"]["l1"]["mean"] > 0


def test_interpolation_study_reports_both_methods() -> None:
    s = store()
    report = interpolation_study(model(), net(), s, FAST, CPU, pairs=2, frames=3)
    assert {"flow_legibility", "crossfade_legibility", "flow_grey_share"} <= set(report)
