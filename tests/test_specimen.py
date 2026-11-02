from pathlib import Path

import numpy as np
from conftest import build_font

from typeflow.cli import main
from typeflow.generate import export_weights
from typeflow.model import tiny_config
from typeflow.specimen import (
    INK,
    PAPER,
    Extension,
    Sheet,
    extension_sheet,
    gallery_sheet,
    morph_sheet,
    set_line,
    steps_sheet,
    trim,
)
from typeflow.specimen import Typefaces as Faces
from typeflow.train import TrainConfig, train


def faces() -> Faces:
    missing = Path("missing.ttf")
    return Faces(missing, missing, missing)


def test_full_ink_lands_exactly_on_the_ink_colour() -> None:
    sheet = Sheet(40, 40, faces())
    sheet.glyph((10, 10), np.full((4, 4), 255, dtype=np.uint8))
    pixels = np.asarray(sheet.image)
    assert tuple(pixels[11, 11]) == INK
    assert tuple(pixels[0, 0]) == PAPER


def test_grey_levels_are_kept_rather_than_thresholded() -> None:
    sheet = Sheet(4, 4, faces())
    sheet.glyph((0, 0), np.full((2, 2), 128, dtype=np.uint8))
    value = np.asarray(sheet.image)[0, 0, 0]
    assert INK[0] < value < PAPER[0]


def test_set_line_tightens_to_ink_and_keeps_height() -> None:
    glyph = np.zeros((16, 16), dtype=np.uint8)
    glyph[4:12, 6:10] = 255
    line = set_line({"a": glyph}, "a a", gap=2, space=5)
    assert line.shape == (16, 4 + 2 + 5 + 4 + 2)


def test_trim_cuts_rows_only() -> None:
    assert trim(np.zeros((3, 128, 128)), 8, 128).shape == (3, 120, 128)


def glyphs(n: int) -> np.ndarray:
    return np.zeros((n, 128, 128), dtype=np.uint8)


def test_each_sheet_builds_without_overflowing_its_canvas() -> None:
    item = Extension("Name", "note", "nH", "ABCD", glyphs(4), glyphs(4), glyphs(2))
    assert extension_sheet([item], "T", "s", faces(), per_row=3).image.width > 0
    frames = np.zeros((5, 3, 128, 128), dtype=np.uint8)
    assert morph_sheet([("a", "b", frames)], "Rag", "T", "s", faces()).image.height > 0
    assert gallery_sheet([("best", "n", glyphs(6), glyphs(6))], "T", "s", faces()).image
    assert steps_sheet([("1 step", glyphs(2))], "Qg", "T", "s", faces()).image


def tiny_weights(tmp_path: Path) -> Path:
    from conftest import synthetic_store

    store = synthetic_store(fonts=4)
    cfg = TrainConfig(fonts=2, per_font=1, max_steps=1, warmup_steps=1)
    train(store, np.arange(4), tiny_config(128), cfg, tmp_path / "run", _cpu())
    out = tmp_path / "tiny.pt"
    export_weights(tmp_path / "run" / "latest.pt", out)
    return out


def _cpu():
    import torch

    return torch.device("cpu")


def test_extend_and_morph_commands_write_sheets(tmp_path: Path) -> None:
    weights = tiny_weights(tmp_path)
    font = build_font(tmp_path / "Box-Regular.ttf")
    common = ["--weights", str(weights), "--device", "cpu", "--steps", "1", "--method", "euler"]
    extended = tmp_path / "extended.png"
    code = main(
        ["--data", str(tmp_path), "extend", "--font", str(font), "--letters", "nH",
         "--chars", "abc", "--out", str(extended), *common]
    )  # fmt: skip
    assert code == 0 and extended.exists()
    morphed = tmp_path / "morphed.png"
    code = main(
        ["--data", str(tmp_path), "morph", str(font), str(font), "--letters", "nH",
         "--chars", "R", "--frames", "2", "--out", str(morphed), *common]
    )  # fmt: skip
    assert code == 0 and morphed.exists()
