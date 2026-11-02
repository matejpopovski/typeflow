"""Type-specimen sheets, composed directly from glyph arrays.

Generated glyphs are placed pixel for pixel. Enlarging them would hide what the model produced,
so no sheet scales a glyph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .render import CHAR_TO_INDEX

PAPER = (247, 243, 233)
INK = (28, 26, 24)
RUBRIC = (168, 40, 32)
MUTED = (120, 112, 100)
RULE = (196, 188, 172)

MARGIN = 56


@dataclass
class Typefaces:
    """Faces used for captions, loaded from the fetched corpus."""

    title: Path
    body: Path
    italic: Path
    cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = field(default_factory=dict)

    def get(self, kind: str, size: int) -> ImageFont.FreeTypeFont:
        key = (kind, size)
        if key not in self.cache:
            path: Path = getattr(self, kind)
            self.cache[key] = (
                ImageFont.truetype(str(path), size)
                if path.exists()
                else ImageFont.load_default(size=size)
            )
        return self.cache[key]


def default_typefaces(data_dir: Path) -> Typefaces:
    base = data_dir / "upstream" / "ofl"
    return Typefaces(
        base / "playfairdisplaysc" / "PlayfairDisplaySC-Regular.ttf",
        base / "ebgaramond" / "EBGaramond[wght].ttf",
        base / "ebgaramond" / "EBGaramond-Italic[wght].ttf",
    )


class Sheet:
    def __init__(self, width: int, height: int, faces: Typefaces) -> None:
        self.image = Image.new("RGB", (width, height), PAPER)
        self.draw = ImageDraw.Draw(self.image)
        self.faces = faces
        self.width, self.height = width, height

    def text(
        self,
        xy: tuple[int, int],
        text: str,
        kind: str = "body",
        size: int = 22,
        fill: tuple[int, int, int] = INK,
        anchor: str = "ls",
    ) -> None:
        self.draw.text(xy, text, font=self.faces.get(kind, size), fill=fill, anchor=anchor)

    def rule(self, y: int, weight: int = 1, fill: tuple[int, int, int] = RULE) -> None:
        self.draw.rectangle((MARGIN, y, self.width - MARGIN, y + weight - 1), fill=fill)

    def header(self, title: str, subtitle: str) -> int:
        self.text((MARGIN, 88), title, "title", 54)
        self.rule(106, 3, INK)
        self.text((MARGIN, 140), subtitle, "italic", 24, MUTED)
        return 172

    def glyph(
        self, xy: tuple[int, int], glyph: np.ndarray, color: tuple[int, int, int] = INK
    ) -> None:
        """Lay a glyph onto the paper as ink, so grey levels stay exactly what the model drew."""
        x, y = xy
        h, w = glyph.shape
        region = np.asarray(self.image.crop((x, y, x + w, y + h)), dtype=np.float32)
        alpha = (np.asarray(glyph, dtype=np.float32) / 255.0)[..., None]
        blended = region * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
        self.image.paste(Image.fromarray(blended.round().astype(np.uint8)), (x, y))

    def row(
        self,
        x: int,
        y: int,
        glyphs: np.ndarray,
        advance: int,
        colors: list[tuple[int, int, int]] | None = None,
    ) -> None:
        for i, glyph in enumerate(glyphs):
            self.glyph((x + i * advance, y), glyph, colors[i] if colors else INK)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(path, optimize=True)


def set_line(glyphs: dict[str, np.ndarray], text: str, gap: int = 6, space: int = 26) -> np.ndarray:
    """Set text from loose glyphs using ink bounds, since the model does not predict spacing."""
    height = next(iter(glyphs.values())).shape[0]
    pieces: list[np.ndarray] = []
    for char in text:
        if char == " ":
            pieces.append(np.zeros((height, space), dtype=np.uint8))
            continue
        glyph = glyphs[char]
        columns = np.nonzero((glyph > 40).any(axis=0))[0]
        left, right = (columns[0], columns[-1] + 1) if len(columns) else (0, glyph.shape[1] // 3)
        pieces.append(glyph[:, left:right])
        pieces.append(np.zeros((height, gap), dtype=np.uint8))
    return np.concatenate(pieces, axis=1)


def trim(glyphs: np.ndarray, top: int, bottom: int) -> np.ndarray:
    """Drop empty rows so that lines of type sit at a natural leading."""
    return glyphs[..., top:bottom, :]


@dataclass
class Extension:
    """One font extended from references: what the model drew and what the font really has."""

    name: str
    note: str
    references: str
    chars: str
    generated: np.ndarray
    truth: np.ndarray
    reference_glyphs: np.ndarray


TOP, BOTTOM = 8, 128


def _chunks(chars: str, size: int) -> list[str]:
    return [chars[i : i + size] for i in range(0, len(chars), size)]


def extension_sheet(
    items: list[Extension], title: str, subtitle: str, faces: Typefaces, per_row: int = 13
) -> Sheet:
    """Each block: the references in red, then generated glyphs above the true ones."""
    advance = 88
    lines = len(_chunks(items[0].chars, per_row))
    block = 66 + 120 + (120 * 2 + 16) * lines
    width = MARGIN * 2 + advance * per_row + 40
    sheet = Sheet(width, 190 + block * len(items) + 40, faces)
    y = sheet.header(title, subtitle)
    for item in items:
        sheet.text((MARGIN, y + 34), item.name, "body", 34)
        sheet.text((width - MARGIN, y + 34), item.note, "italic", 22, MUTED, "rs")
        sheet.rule(y + 46)
        y += 56
        sheet.text((MARGIN, y + 72), "given", "italic", 22, RUBRIC)
        sheet.row(MARGIN + 96, y, trim(item.reference_glyphs, TOP, BOTTOM), advance, [RUBRIC] * 8)
        y += 124
        for line, chunk in enumerate(_chunks(item.chars, per_row)):
            indices = [item.chars.index(c) for c in chunk]
            sheet.row(MARGIN, y, trim(item.generated[indices], TOP, BOTTOM), advance)
            sheet.text((width - MARGIN + 20, y + 70), "drawn", "italic", 16, RUBRIC, "ms")
            y += 120
            faint = [MUTED] * len(indices)
            sheet.row(MARGIN, y, trim(item.truth[indices], TOP, BOTTOM), advance, faint)
            sheet.text((width - MARGIN + 20, y + 70), "true", "italic", 16, MUTED, "ms")
            y += 136 if line + 1 < lines else 120
        y += 30
    return sheet


def morph_sheet(
    rows: list[tuple[str, str, np.ndarray]],
    chars: str,
    title: str,
    subtitle: str,
    faces: Typefaces,
) -> Sheet:
    """Rows are ``(name_a, name_b, frames)`` with frames shaped ``(frames, chars, H, W)``."""
    frames = rows[0][2].shape[0]
    advance = 92
    block = 60 + 120 * len(chars) + 30
    width = MARGIN * 2 + advance * frames
    sheet = Sheet(width, 190 + block * len(rows) + 30, faces)
    y = sheet.header(title, subtitle)
    for name_a, name_b, glyphs in rows:
        sheet.text((MARGIN, y + 34), name_a, "body", 28)
        sheet.text((width - MARGIN, y + 34), name_b, "body", 28, INK, "rs")
        sheet.rule(y + 46)
        y += 56
        for c in range(len(chars)):
            sheet.row(MARGIN, y, trim(glyphs[:, c], TOP, BOTTOM), advance)
            y += 120
        y += 34
    return sheet


def gallery_sheet(
    rows: list[tuple[str, str, np.ndarray, np.ndarray]],
    title: str,
    subtitle: str,
    faces: Typefaces,
) -> Sheet:
    """Rows are ``(label, name, generated, truth)`` with one line of glyphs each."""
    columns = rows[0][2].shape[0]
    advance = 92
    block = 46 + 120 * 2 + 30
    width = MARGIN * 2 + advance * columns
    sheet = Sheet(width, 190 + block * len(rows) + 30, faces)
    y = sheet.header(title, subtitle)
    for label, name, generated, truth in rows:
        sheet.text((MARGIN, y + 24), label, "italic", 22, RUBRIC)
        sheet.text((width - MARGIN, y + 24), name, "body", 24, INK, "rs")
        sheet.rule(y + 34)
        y += 42
        sheet.row(MARGIN, y, trim(generated, TOP, BOTTOM), advance)
        y += 120
        sheet.row(MARGIN, y, trim(truth, TOP, BOTTOM), advance, [MUTED] * columns)
        y += 150
    return sheet


def steps_sheet(
    columns: list[tuple[str, np.ndarray]],
    chars: str,
    title: str,
    subtitle: str,
    faces: Typefaces,
) -> Sheet:
    """Columns are ``(label, glyphs)`` with glyphs shaped ``(chars, H, W)``."""
    advance = 110
    width = MARGIN * 2 + advance * len(columns)
    sheet = Sheet(width, 240 + 120 * len(chars) + 40, faces)
    y = sheet.header(title, subtitle)
    for i, (label, _) in enumerate(columns):
        x = MARGIN + i * advance + 64
        sheet.text((x, y + 24), label, "italic", 20, RUBRIC, "ms")
    y += 36
    for c in range(len(chars)):
        line = np.stack([glyphs[c] for _, glyphs in columns])
        sheet.row(MARGIN, y, trim(line, TOP, BOTTOM), advance)
        y += 120
    return sheet


def to_indices(chars: str) -> list[int]:
    return [CHAR_TO_INDEX[c] for c in chars]
