"""Rasterise fonts into a shared frame so that every glyph sits on the same baseline.

Fonts disagree wildly about how large an em is, so scaling by point size would hide most of the
real variation behind a nuisance one. The frame instead normalises each style by the measured
height of its capital H. Baseline, x-height and descender depth stay meaningful across fonts and
the model never has to learn per-font vertical offsets.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont, TTLibError
from PIL import Image, ImageDraw, ImageFont

CHARSET = string.ascii_uppercase + string.ascii_lowercase + string.digits + "&?!"
CHAR_TO_INDEX = {c: i for i, c in enumerate(CHARSET)}

# Hinting at low sizes moves outlines by whole pixels, so glyphs are drawn large and box-filtered.
SUPERSAMPLE = 4
# Share of a frame's height a glyph may overflow before the style is rejected.
CLIP_TOLERANCE = 0.01

STYLE_PRIORITY = ("Regular", "Bold", "Italic", "Black", "Light", "BoldItalic", "Medium", "SemiBold")


@dataclass(frozen=True)
class Frame:
    size: int = 128
    baseline: float = 0.70
    cap_height: float = 0.46


class RenderError(ValueError):
    """A style that cannot be represented in the frame; the message is the recorded reason."""


def covers_charset(path: Path, chars: str = CHARSET) -> bool:
    try:
        with TTFont(path, lazy=True, fontNumber=0) as font:
            cmap = font.getBestCmap() or {}
    except (TTLibError, OSError, KeyError, AssertionError):
        return False
    return all(ord(c) in cmap for c in chars)


def is_color_font(path: Path) -> bool:
    """Colour fonts carry their design in layers that a greyscale outline render throws away."""
    with TTFont(path, lazy=True) as font:
        return any(tag in font for tag in ("COLR", "CBDT", "sbix", "SVG "))


def _variation_names(font: ImageFont.FreeTypeFont) -> list[str]:
    try:
        return [n.decode() if isinstance(n, bytes) else n for n in font.get_variation_names()]
    except OSError:
        return []


def list_styles(path: Path) -> list[tuple[str, str]]:
    """Return ``(style, instance)`` pairs; instance is empty for a static font.

    Static styles come from the font's own name table, because file names follow no one convention.
    Named instances of an italic variable file are often called plain "Bold", so the file name has
    to supply the slant.
    """
    font = ImageFont.truetype(str(path), 32, layout_engine=ImageFont.Layout.BASIC)
    names = _variation_names(font)
    if not names:
        return [((font.getname()[1] or "Regular").replace(" ", ""), "")]
    italic = "Italic" in Path(path).stem
    styles = []
    for name in dict.fromkeys(names):
        style = name.replace(" ", "")
        if italic and "Italic" not in style:
            style = "Italic" if style == "Regular" else f"{style}Italic"
        styles.append((style, name))
    return styles


def pick_styles(styles: list[tuple[str, str, str]], cap: int) -> list[tuple[str, str, str]]:
    """Keep at most ``cap`` of a family's ``(file, style, instance)`` rows.

    Large superfamilies would otherwise outweigh single-style display faces by nine to one, and
    their adjacent weights add little that a model can learn from.
    """

    def rank(row: tuple[str, str, str]) -> tuple[int, str]:
        style = row[1]
        return (
            (STYLE_PRIORITY.index(style), style)
            if style in STYLE_PRIORITY
            else (len(STYLE_PRIORITY), style)
        )

    unique: dict[str, tuple[str, str, str]] = {}
    for row in sorted(styles, key=rank):
        unique.setdefault(row[1], row)
    return list(unique.values())[:cap]


def _load(path: Path, px: float, instance: str) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(str(path), px, layout_engine=ImageFont.Layout.BASIC)
    if instance:
        font.set_variation_by_name(instance)
    return font


def render_glyphs(
    path: Path, instance: str = "", frame: Frame | None = None, chars: str = CHARSET
) -> np.ndarray:
    """Render ``chars`` to a ``(len(chars), size, size)`` uint8 array with 255 as ink."""
    frame = frame or Frame()
    probe_px = 400.0
    probe = _load(path, probe_px, instance)
    left, top, right, _ = probe.getbbox("H", anchor="ls")
    measured_cap = -top
    if measured_cap <= 0 or right <= left:
        raise RenderError("no usable capital H")

    canvas = frame.size * SUPERSAMPLE
    px = probe_px * frame.cap_height * canvas / measured_cap
    font = _load(path, px, instance)
    baseline = round(frame.baseline * canvas)
    tolerance = CLIP_TOLERANCE * canvas
    # Glyphs are drawn on an oversized sheet and measured there. getbbox() includes the advance
    # width, so it cannot say where the ink ends or whether a glyph would overflow the frame.
    pad = canvas // 2
    sheet = canvas + 2 * pad

    out = np.zeros((len(chars), frame.size, frame.size), dtype=np.uint8)
    for i, char in enumerate(chars):
        image = Image.new("L", (sheet, sheet), 0)
        left, _, right, _ = font.getbbox(char, anchor="ls")
        origin = (sheet - (right - left)) // 2 - left
        ImageDraw.Draw(image).text((origin, pad + baseline), char, 255, font=font, anchor="ls")
        ink = image.getbbox()
        if ink is None:
            raise RenderError(f"empty glyph {char!r}")
        x0, y0, x1, y1 = ink
        if x1 - x0 > canvas + tolerance or y0 < pad - tolerance or y1 > pad + canvas + tolerance:
            raise RenderError(f"glyph {char!r} overflows the frame")
        start = (x0 + x1 - canvas) // 2
        framed = image.crop((start, pad, start + canvas, pad + canvas))
        out[i] = np.asarray(framed.reduce(SUPERSAMPLE), dtype=np.uint8)
        if out[i].max() < 128:
            raise RenderError(f"blank glyph {char!r}")
    return out


def letterform_spread(glyphs: np.ndarray) -> float:
    """How different a style's glyphs are from one another: 0 if identical, 1 if disjoint.

    Normalised by ink so that a hairline face is not mistaken for a font of near-identical shapes.
    """
    small = glyphs[:, ::4, ::4].astype(np.float32) / 255.0
    flat = small.reshape(len(small), -1)
    return float(np.abs(flat[:, None] - flat[None]).mean() / (2 * flat.mean()))
