"""Build the README sheets from real outputs.

Fonts are chosen by rank in the saved test scores, never by eye. A sheet labelled "median" shows
the median font, and the worst font is shown as readily as the best.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .evaluate import REFERENCE_ORDER, TARGET_CHARS, reference_chars
from .generate import SamplerConfig, encode_styles, noise_for, sample_glyphs
from .model import TypeFlow
from .render import CHARSET
from .specimen import (
    Extension,
    Typefaces,
    extension_sheet,
    gallery_sheet,
    morph_sheet,
    steps_sheet,
    to_indices,
)
from .store import GlyphStore

HEADLINE_K = 4
HERO_CHARS = "ABGKMQRbdefhkpqtuy38&?"[:26]
GALLERY_CHARS = "ABGQRbdefhpy3&"


@dataclass
class Scores:
    """Saved per-font test scores, joined to the store's rows."""

    ids: np.ndarray
    perceptual: np.ndarray
    categories: list[str]
    output: np.ndarray

    @classmethod
    def load(cls, store: GlyphStore, results: Path, outputs: Path) -> Scores:
        report = json.loads(results.read_text())
        ids = np.array([store.index_of(key) for key in report["fonts"]])
        perceptual = np.array(report["per_font"][f"model_k{HEADLINE_K}"]["perceptual"])
        return cls(
            ids,
            perceptual,
            [store.rows[i].category for i in ids],
            np.load(outputs / f"model_k{HEADLINE_K}.npy", mmap_mode="r"),
        )

    def median_of(self, category: str) -> int:
        positions = [i for i, c in enumerate(self.categories) if c == category]
        ordered = sorted(positions, key=lambda i: self.perceptual[i])
        return ordered[len(ordered) // 2]

    def at_rank(self, fraction: float) -> int:
        ordered = np.argsort(self.perceptual)
        return int(ordered[round(fraction * (len(ordered) - 1))])


def extend_font(
    model: TypeFlow,
    glyphs: np.ndarray,
    refs: list[int],
    chars: list[int],
    sampler: SamplerConfig,
    seed: int = 0,
) -> np.ndarray:
    """Draw ``chars`` for one font of shape ``(chars, H, W)`` given reference indices."""
    style = encode_styles(model, np.asarray(glyphs[refs])[None], np.array([refs]))
    noise = noise_for((len(chars), 1, glyphs.shape[-2], glyphs.shape[-1]), seed)
    indices = torch.tensor(chars, device=style.device)
    return sample_glyphs(model, style.expand(len(chars), -1), indices, noise, sampler)


def hero(
    model: TypeFlow,
    store: GlyphStore,
    scores: Scores,
    sampler: SamplerConfig,
    faces: Typefaces,
    path: Path,
) -> list[str]:
    items, names = [], []
    refs = reference_chars(HEADLINE_K)
    chars = to_indices(HERO_CHARS)
    for category in ("SERIF", "SANS_SERIF", "DISPLAY"):
        position = scores.median_of(category)
        style = int(scores.ids[position])
        glyphs = store.glyphs[style]
        items.append(
            Extension(
                store.rows[style].key,
                f"held-out family, median {category.lower().replace('_', ' ')}",
                REFERENCE_ORDER[:HEADLINE_K],
                HERO_CHARS,
                extend_font(model, glyphs, refs, chars, sampler),
                np.asarray(glyphs[chars]),
                np.asarray(glyphs[refs]),
            )
        )
        names.append(store.rows[style].key)
    subtitle = f"Each face is extended from the {HEADLINE_K} letters in red."
    extension_sheet(items, "Typeflow Specimen", subtitle, faces).save(path)
    return names


def gallery(store: GlyphStore, scores: Scores, faces: Typefaces, path: Path) -> list[str]:
    """Best to worst by perceptual distance, at fixed ranks."""
    targets = [CHARSET[i] for i in TARGET_CHARS]
    picks = [t for t in GALLERY_CHARS if t in targets]
    columns = [targets.index(c) for c in picks]
    rows, names = [], []
    for label, fraction in (
        ("best", 0.0),
        ("25th percentile", 0.25),
        ("median", 0.5),
        ("75th percentile", 0.75),
        ("95th percentile", 0.95),
        ("worst", 1.0),
    ):
        position = scores.at_rank(fraction)
        style = int(scores.ids[position])
        generated = np.asarray(scores.output[position][columns])
        truth = np.asarray(store.glyphs[style][[TARGET_CHARS[c] for c in columns]])
        rows.append((label, store.rows[style].key, generated, truth))
        names.append(f"{label}: {store.rows[style].key}")
    subtitle = "Held-out fonts at fixed ranks. Upper line drawn, lower line true."
    gallery_sheet(rows, "From Best to Worst", subtitle, faces).save(path)
    return names


def morph(
    model: TypeFlow,
    store: GlyphStore,
    scores: Scores,
    sampler: SamplerConfig,
    faces: Typefaces,
    path: Path,
    chars: str = "Rag",
    frames: int = 9,
) -> list[str]:
    pairs = (("SERIF", "DISPLAY"), ("SANS_SERIF", "SERIF"), ("HANDWRITING", "SANS_SERIF"))
    refs = reference_chars(8)
    indices = to_indices(chars)
    rows, names = [], []
    device = next(model.parameters()).device
    for first, second in pairs:
        a = int(scores.ids[scores.median_of(first)])
        b = int(scores.ids[scores.median_of(second)])
        style_a = encode_styles(model, np.asarray(store.glyphs[a][refs])[None], np.array([refs]))
        style_b = encode_styles(model, np.asarray(store.glyphs[b][refs])[None], np.array([refs]))
        weights = torch.linspace(0, 1, frames, device=device)[:, None]
        styles = torch.lerp(style_a, style_b, weights).repeat_interleave(len(indices), dim=0)
        char_ids = torch.tensor(indices * frames, device=device)
        noise = noise_for((len(indices), 1, store.size, store.size), 0).repeat(frames, 1, 1, 1)
        glyphs = sample_glyphs(model, styles, char_ids, noise, sampler).reshape(
            frames, len(indices), store.size, store.size
        )
        rows.append((store.rows[a].key, store.rows[b].key, glyphs))
        names.append(f"{store.rows[a].key} to {store.rows[b].key}")
    subtitle = "Both ends are held-out fonts. The noise is fixed, only the style vector moves."
    morph_sheet(rows, chars, "Between Two Faces", subtitle, faces).save(path)
    return names


def sampler_steps(
    model: TypeFlow, store: GlyphStore, scores: Scores, faces: Typefaces, path: Path
) -> None:
    style = int(scores.ids[scores.median_of("SERIF")])
    glyphs = store.glyphs[style]
    chars = "Qg&"
    indices = to_indices(chars)
    columns = []
    for steps in (1, 2, 4, 8, 16, 32):
        cfg = SamplerConfig(steps=steps, method="euler")
        columns.append(
            (
                f"{steps} step" + ("s" if steps > 1 else ""),
                extend_font(model, glyphs, reference_chars(HEADLINE_K), indices, cfg),
            )
        )
    subtitle = "One Euler step gives the blurred average of every glyph the model finds plausible."
    steps_sheet(columns, chars, "Sampler Steps", subtitle, faces).save(path)
