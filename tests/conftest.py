"""Shared fixtures. Fonts here are built on the fly, so no test touches the real corpus."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

from typeflow.render import CHARSET


def _box(width: int, height: int, inset: int) -> object:
    pen = TTGlyphPen(None)
    pen.moveTo((inset, 0))
    pen.lineTo((inset, height))
    pen.lineTo((width - inset, height))
    pen.lineTo((width - inset, 0))
    pen.closePath()
    return pen.glyph()


def build_font(
    path: Path,
    chars: str = CHARSET,
    upem: int = 1000,
    cap_height: int = 700,
    x_height: int = 500,
    width: int = 600,
    style: str = "Regular",
) -> Path:
    """Write a TTF of filled rectangles with controllable metrics."""
    names = [".notdef"] + [f"g{ord(c)}" for c in chars]
    builder = FontBuilder(upem, isTTF=True)
    builder.setupGlyphOrder(names)
    builder.setupCharacterMap({ord(c): f"g{ord(c)}" for c in chars})
    glyphs = {".notdef": _box(width, cap_height, 50)}
    for i, char in enumerate(chars):
        height = cap_height if char.isupper() or char.isdigit() else x_height
        # A character-dependent inset gives each glyph a shape of its own.
        glyphs[f"g{ord(char)}"] = _box(width, height, 20 + (i * 7) % 180)
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics({n: (width, 0) for n in names})
    builder.setupHorizontalHeader(ascent=int(upem * 0.9), descent=-int(upem * 0.25))
    builder.setupNameTable({"familyName": "Test", "styleName": style})
    builder.setupOS2(sTypoAscender=int(upem * 0.9), sTypoDescender=-int(upem * 0.25))
    builder.setupPost()
    builder.save(path)
    return path


@pytest.fixture
def box_font(tmp_path: Path) -> Path:
    return build_font(tmp_path / "Box-Regular.ttf")


def synthetic_store(fonts: int = 12, chars: int = 65, size: int = 32, seed: int = 0) -> np.ndarray:
    """Glyph-like uint8 images in which each font has its own stroke weight and slant."""
    rng = np.random.default_rng(seed)
    out = np.zeros((fonts, chars, size, size), dtype=np.uint8)
    for font in range(fonts):
        weight = 1 + font % 4
        slant = rng.integers(-3, 4)
        for char in range(chars):
            x = 6 + (char * 5) % (size - 14)
            y = 4 + (char * 3) % (size - 16)
            out[font, char, y : y + 12, x : x + weight + 2] = 255
            out[font, char, y + 10 : y + 10 + weight, x : x + 8] = 255
            out[font, char] = np.roll(out[font, char], slant, axis=1)
    return out
