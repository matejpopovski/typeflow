import json
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import synthetic_store
from PIL import Image

from typeflow import figures
from typeflow.evaluate import TARGET_CHARS
from typeflow.generate import SamplerConfig
from typeflow.model import TypeFlow, tiny_config
from typeflow.specimen import Typefaces
from typeflow.store import GlyphStore, StyleRow

CATEGORIES = ["SERIF", "SANS_SERIF", "DISPLAY", "HANDWRITING"]
FAST = SamplerConfig(steps=1, method="euler", batch=8)
FACES = Typefaces(Path("missing"), Path("missing"), Path("missing"))


@pytest.fixture
def scored(tmp_path: Path) -> figures.Scores:
    glyphs = synthetic_store(fonts=8, size=128)
    rows = [StyleRow(f"Font {i}", "f", CATEGORIES[i % 4], "test", str(i)) for i in range(8)]
    store = GlyphStore(glyphs, rows)
    distances = [0.8, 0.1, 0.5, 0.3, 0.7, 0.2, 0.6, 0.4]
    report = {
        "fonts": [r.key for r in rows],
        "per_font": {"model_k4": {"perceptual": distances}},
    }
    results = tmp_path / "test.json"
    results.write_text(json.dumps(report))
    output = glyphs[:, TARGET_CHARS]
    np.save(tmp_path / "model_k4.npy", output)
    scores = figures.Scores.load(store, results, tmp_path)
    scores.store = store
    return scores


def model() -> TypeFlow:
    torch.manual_seed(0)
    return TypeFlow(tiny_config(128)).eval()


def test_median_and_ranks_follow_the_scores(scored: figures.Scores) -> None:
    assert scored.at_rank(0.0) == 1
    assert scored.at_rank(1.0) == 0
    assert scored.median_of("SERIF") == 0
    assert scored.categories[scored.median_of("DISPLAY")] == "DISPLAY"


def test_gallery_reports_the_fonts_at_its_ranks(scored: figures.Scores, tmp_path: Path) -> None:
    names = figures.gallery(scored.store, scored, FACES, tmp_path / "g.png")
    assert names[0] == "best: Font 1" and names[-1] == "worst: Font 0"
    assert Image.open(tmp_path / "g.png").width > 0


def test_extension_draws_only_what_was_asked() -> None:
    glyphs = synthetic_store(fonts=1, size=128)[0]
    assert figures.extend_font(model(), glyphs, [0, 1], [5, 6, 7], FAST).shape == (3, 128, 128)


def test_hero_morph_and_steps_write_files(scored: figures.Scores, tmp_path: Path) -> None:
    m = model()
    figures.hero(m, scored.store, scored, FAST, FACES, tmp_path / "s.png")
    names = figures.morph(m, scored.store, scored, FAST, FACES, tmp_path / "m.png", frames=3)
    figures.sampler_steps(m, scored.store, scored, FACES, tmp_path / "t.png")
    assert len(names) == 3
    assert all((tmp_path / n).exists() for n in ("s.png", "m.png", "t.png"))
