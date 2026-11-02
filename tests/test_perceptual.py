from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import synthetic_store

from typeflow.perceptual import (
    GlyphNet,
    feature_distance,
    held_out_mask,
    load_perceptual,
    train_perceptual,
)


def net() -> GlyphNet:
    torch.manual_seed(0)
    return GlyphNet(4, 65, widths=(8, 16)).eval()


def test_distance_is_zero_for_identical_images() -> None:
    x = torch.randn(2, 1, 32, 32)
    assert feature_distance(net(), x, x).abs().max() < 1e-5


def test_distance_is_symmetric_and_positive() -> None:
    x, y = torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32)
    model = net()
    a, b = feature_distance(model, x, y), feature_distance(model, y, x)
    assert torch.allclose(a, b, atol=1e-5) and (a > 0).all()


def test_held_out_mask_is_deterministic_and_roughly_sized() -> None:
    mask = held_out_mask(100, 65)
    assert np.array_equal(mask, held_out_mask(100, 65))
    assert 0.08 < mask.mean() < 0.12


@pytest.mark.slow
def test_training_writes_a_loadable_network(tmp_path: Path) -> None:
    glyphs = synthetic_store(fonts=6)
    out = tmp_path / "net.pt"
    report = train_perceptual(
        glyphs, np.arange(6), torch.device("cpu"), out, steps=8, batch=8, eval_batches=2
    )
    assert report["font_classes"] == 6
    assert load_perceptual(out, torch.device("cpu")).font_head.out_features == 6
