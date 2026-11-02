from pathlib import Path

import numpy as np
import pytest
from conftest import build_font

from typeflow.render import (
    CHARSET,
    Frame,
    RenderError,
    covers_charset,
    letterform_spread,
    list_styles,
    pick_styles,
    render_glyphs,
)


def ink_rows(glyph: np.ndarray) -> tuple[int, int]:
    rows = np.nonzero((glyph > 127).any(axis=1))[0]
    return int(rows[0]), int(rows[-1])


def test_cap_height_and_baseline_are_normalised_across_fonts(tmp_path: Path) -> None:
    frame = Frame()
    small = build_font(tmp_path / "A.ttf", upem=1000, cap_height=600)
    large = build_font(tmp_path / "B.ttf", upem=2048, cap_height=1500)
    h = CHARSET.index("H")
    for path in (small, large):
        top, bottom = ink_rows(render_glyphs(path, frame=frame)[h])
        assert bottom + 1 == pytest.approx(frame.baseline * frame.size, abs=1)
        assert bottom + 1 - top == pytest.approx(frame.cap_height * frame.size, abs=1)


def test_x_height_ratio_survives_normalisation(box_font: Path) -> None:
    glyphs = render_glyphs(box_font)
    cap_top, base = ink_rows(glyphs[CHARSET.index("H")])
    x_top, _ = ink_rows(glyphs[CHARSET.index("n")])
    assert (base - x_top) / (base - cap_top) == pytest.approx(500 / 700, abs=0.03)


def test_glyphs_are_centred_horizontally(box_font: Path) -> None:
    columns = np.nonzero((render_glyphs(box_font)[0] > 127).any(axis=0))[0]
    assert columns[0] + columns[-1] + 1 == pytest.approx(128, abs=2)


def test_output_shape_and_type(box_font: Path) -> None:
    glyphs = render_glyphs(box_font, frame=Frame(size=64), chars="Hn")
    assert glyphs.shape == (2, 64, 64) and glyphs.dtype == np.uint8


def test_overflowing_style_is_rejected(tmp_path: Path) -> None:
    wide = build_font(tmp_path / "Wide.ttf", width=2500, cap_height=700)
    with pytest.raises(RenderError, match="overflows"):
        render_glyphs(wide)


def test_coverage_requires_every_character(tmp_path: Path, box_font: Path) -> None:
    assert covers_charset(box_font)
    assert not covers_charset(build_font(tmp_path / "Caps.ttf", chars="ABCH"))


def test_static_font_style_comes_from_the_name_table(tmp_path: Path) -> None:
    path = build_font(tmp_path / "NoHyphenBold.ttf", style="Bold Italic")
    assert list_styles(path) == [("BoldItalic", "")]


def test_pick_styles_drops_rows_that_name_the_same_style() -> None:
    rows = [("a", "Regular", ""), ("b", "Regular", "x"), ("c", "Bold", "")]
    assert [r[1] for r in pick_styles(rows, 5)] == ["Regular", "Bold"]


def test_pick_styles_prefers_regular_and_respects_cap() -> None:
    rows = [("f", s, "") for s in ("Thin", "Bold", "Regular", "Italic", "Black", "Light")]
    assert [r[1] for r in pick_styles(rows, 3)] == ["Regular", "Bold", "Italic"]


def test_letterform_spread_separates_identical_from_distinct_shapes() -> None:
    same = np.zeros((8, 32, 32), dtype=np.uint8)
    same[:, 8:24, 8:24] = 255
    distinct = np.zeros((8, 32, 32), dtype=np.uint8)
    for i in range(8):
        distinct[i, 4 * i : 4 * i + 4, :] = 255
    assert letterform_spread(same) == 0
    assert letterform_spread(distinct) > 0.8


def test_spread_does_not_penalise_thin_strokes() -> None:
    thin = np.zeros((8, 64, 64), dtype=np.uint8)
    for i in range(8):
        thin[i, 8 * i, :] = 255
    assert letterform_spread(thin) > 0.8
