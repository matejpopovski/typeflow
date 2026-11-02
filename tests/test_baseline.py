import numpy as np
import pytest
import torch
from conftest import synthetic_store

from typeflow.baseline import glyph_distances, nearest_fonts


def test_distance_to_an_identical_font_is_zero() -> None:
    glyphs = synthetic_store(fonts=6, chars=5)
    distances = glyph_distances(glyphs, np.arange(6), np.array([2]))
    assert distances.shape == (5, 1, 6)
    assert distances[:, 0, 2].abs().max() == 0


def test_distances_match_a_direct_computation() -> None:
    glyphs = synthetic_store(fonts=6, chars=3)
    distances = glyph_distances(glyphs, np.array([0, 1, 2, 3]), np.array([4, 5]))
    direct = np.abs(glyphs[5, 2].astype(float) - glyphs[3, 2].astype(float)).mean() / 255
    assert distances[2, 1, 3].item() == pytest.approx(direct, abs=1e-5)


def test_nearest_font_recovers_the_right_font_from_few_letters() -> None:
    glyphs = synthetic_store(fonts=8, chars=10, seed=1)
    picks = nearest_fonts(glyph_distances(glyphs, np.arange(8), np.array([5, 2])), [0, 3])
    assert picks.tolist() == [5, 2]


def test_lookup_only_uses_the_requested_letters() -> None:
    distances = torch.zeros(3, 1, 2)
    distances[0, 0, 0] = 1
    distances[1, 0, 1] = 1
    assert nearest_fonts(distances, [0]).item() == 1
    assert nearest_fonts(distances, [1]).item() == 0
