from pathlib import Path

import numpy as np
import pytest
from conftest import build_font

from typeflow import corpus, store
from typeflow.corpus import Family, Style
from typeflow.render import CHARSET, Frame
from typeflow.survey import SIGNATURE_BLOCK, FamilyTask, inspect_family, signature

SIZE = 32


def layout(root: Path, directory: str, names: list[str], **font: object) -> dict[str, str]:
    base = root / "ofl" / directory
    base.mkdir(parents=True)
    for name in names:
        build_font(base / name, **font)
    return {name: "0" * 40 for name in names}


def task(root: Path, files: dict[str, str], directory: str = "foo") -> FamilyTask:
    return FamilyTask("Foo", directory, root, files, Frame(size=128))


def test_signature_averages_blocks() -> None:
    glyphs = np.zeros((2, 128, 128), dtype=np.uint8)
    glyphs[0, :4, :4] = 200
    out = signature(glyphs)
    assert out.shape == (2, (128 // SIGNATURE_BLOCK) ** 2)
    assert out[0, 0] == 200 and out[1].max() == 0


def test_inspect_family_accepts_a_complete_font(tmp_path: Path) -> None:
    files = layout(tmp_path, "foo", ["Foo-Regular.ttf"])
    result = inspect_family(task(tmp_path, files))
    assert [s.style for s in result.styles] == ["Regular"]
    assert result.styles[0].path == "ofl/foo/Foo-Regular.ttf"
    assert result.signatures[0].shape[0] == len(CHARSET)
    assert result.rejected == []


def test_inspect_family_records_why_a_font_was_refused(tmp_path: Path) -> None:
    files = layout(tmp_path, "foo", ["Foo-Regular.ttf"], chars="ABCH")
    result = inspect_family(task(tmp_path, files))
    assert result.styles == []
    assert result.rejected == [("Foo-Regular.ttf", "incomplete Latin coverage")]


def test_inspect_family_drops_a_style_identical_to_another(tmp_path: Path) -> None:
    files = layout(tmp_path, "foo", ["Foo-Regular.ttf"])
    build_font(tmp_path / "ofl/foo/Foo-Bold.ttf", style="Bold")
    files["Foo-Bold.ttf"] = "0" * 40
    result = inspect_family(task(tmp_path, files))
    assert len(result.styles) == 1
    assert "identical" in result.rejected[0][1]


@pytest.fixture
def small_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "fonts"
    root = tmp_path / "upstream"
    layout(root, "foo", ["Foo-Regular.ttf"])
    layout(root, "bar", ["Bar-Regular.ttf"], x_height=350)
    families = [
        Family("Bar", "bar", "SERIF", "OFL-1.1", "0" * 40, "Bar", "test"),
        Family("Foo", "foo", "DISPLAY", "OFL-1.1", "0" * 40, "Foo", "train"),
    ]
    styles = [
        Style("Bar", "ofl/bar/Bar-Regular.ttf", "0" * 40, "Regular", ""),
        Style("Foo", "ofl/foo/Foo-Regular.ttf", "0" * 40, "Regular", ""),
    ]
    corpus.save_manifest(families, styles, directory)
    monkeypatch.setattr(corpus.load_styles, "__defaults__", (directory,))
    monkeypatch.setattr(corpus.load_families, "__defaults__", (directory,))
    return tmp_path


@pytest.mark.slow
def test_store_round_trip_keeps_manifest_order(small_manifest: Path) -> None:
    store.build_store(small_manifest, SIZE, workers=1)
    loaded = store.load_store(small_manifest, SIZE)
    assert loaded.glyphs.shape == (2, len(CHARSET), SIZE, SIZE)
    assert [r.key for r in loaded.rows] == ["Bar Regular", "Foo Regular"]
    assert loaded.rows[0].split == "test" and loaded.rows[1].category == "DISPLAY"
    assert loaded.split_indices("train").tolist() == [1]
    assert loaded.index_of("Foo Regular") == 1
    assert not np.array_equal(loaded.glyphs[0], loaded.glyphs[1])


@pytest.mark.slow
def test_a_font_missing_from_disk_stops_the_build(small_manifest: Path) -> None:
    (small_manifest / "upstream/ofl/foo/Foo-Regular.ttf").unlink()
    with pytest.raises(RuntimeError, match="Foo Regular"):
        store.build_store(small_manifest, SIZE, workers=1)
    assert not store.store_path(small_manifest, SIZE).exists()
