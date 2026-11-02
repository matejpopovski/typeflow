"""The rendered corpus as one flat uint8 file, read through a memory map.

Every batch is a handful of slices out of that map, taken in the main process. There are no
DataLoader workers, so there is nothing that can hang between the data and the training loop, and
the file's pages stay evictable when the machine is short of memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import corpus
from .pool import Failure, map_with_timeout
from .render import CHARSET, Frame, render_glyphs


@dataclass(frozen=True)
class StyleRow:
    key: str
    family: str
    category: str
    split: str
    cluster: str


@dataclass
class GlyphStore:
    """Glyphs of shape ``(styles, chars, size, size)`` and one metadata row per style."""

    glyphs: np.ndarray
    rows: list[StyleRow]

    @property
    def size(self) -> int:
        return self.glyphs.shape[-1]

    def split_indices(self, split: str) -> np.ndarray:
        return np.array([i for i, r in enumerate(self.rows) if r.split == split], dtype=np.int64)

    def index_of(self, key: str) -> int:
        return next(i for i, r in enumerate(self.rows) if r.key == key)


def _render_task(task: tuple[str, str, int]) -> np.ndarray:
    path, instance, size = task
    return render_glyphs(Path(path), instance, Frame(size=size))


def store_path(data_dir: Path, size: int) -> Path:
    return data_dir / f"glyphs-{size}.bin"


def build_store(data_dir: Path, size: int, workers: int = 8) -> Path:
    """Render every manifest style into the store. A style failing here is a hard error."""
    styles = corpus.load_styles()
    root = data_dir / "upstream"
    tasks = [(str(root / s.path), s.instance, size) for s in styles]
    target = store_path(data_dir, size)
    temp = target.with_suffix(".part")
    shape = (len(styles), len(CHARSET), size, size)
    glyphs = np.memmap(temp, dtype=np.uint8, mode="w+", shape=shape)
    for index, result in map_with_timeout(_render_task, tasks, workers, timeout=120):
        if isinstance(result, Failure):
            raise RuntimeError(f"{styles[index].key}: {result.reason}")
        glyphs[index] = result
    glyphs.flush()
    del glyphs
    temp.replace(target)
    target.with_suffix(".json").write_text(json.dumps({"shape": shape}))
    return target


def load_store(data_dir: Path, size: int) -> GlyphStore:
    target = store_path(data_dir, size)
    shape = tuple(json.loads(target.with_suffix(".json").read_text())["shape"])
    glyphs = np.memmap(target, dtype=np.uint8, mode="r", shape=shape)
    families = {f.family: f for f in corpus.load_families()}
    rows = [
        StyleRow(
            s.key,
            s.family,
            families[s.family].category,
            families[s.family].split,
            families[s.family].cluster,
        )
        for s in corpus.load_styles()
    ]
    return GlyphStore(glyphs, rows)
