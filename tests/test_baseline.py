import numpy as np
import torch
from conftest import synthetic_store

from typeflow.baseline import glyph_distances, nearest_fonts


def test_distance_to_an_identical_font_is_zero() -> None:
    corpus = synthetic_store(fonts=6, chars=5)
    distances = glyph_distances(corpus, corpus[[2]])
    assert distances.shape == (5, 1, 6)
    assert distances[:, 0, 2].abs().max() == 0


def test_distances_match_a_direct_computation() -> None:
    corpus = synthetic_store(fonts=4, chars=3)
    query = synthetic_store(fonts=2, chars=3, seed=9)
    distances = glyph_distances(corpus, query)
    direct = np.abs(query[1, 2].astype(float) - corpus[3, 2].astype(float)).mean() / 255
    assert distances[2, 1, 3].item() == np.float32(direct)


def test_nearest_font_recovers_the_right_font_from_few_letters() -> None:
    corpus = synthetic_store(fonts=8, chars=10, seed=1)
    query = corpus[[5, 2]]
    picks = nearest_fonts(glyph_distances(corpus, query), [0, 3])
    assert picks.tolist() == [5, 2]


def test_lookup_only_uses_the_requested_letters() -> None:
    distances = torch.zeros(3, 1, 2)
    distances[0, 0, 0] = 1
    distances[1, 0, 1] = 1
    assert nearest_fonts(distances, [0]).item() == 1
    assert nearest_fonts(distances, [1]).item() == 0
