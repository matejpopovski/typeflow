"""Held-out evaluation: the model against nearest-neighbour fonts on unseen families.

Every protocol choice below is fixed before looking at test results. Sampler settings are chosen
on the validation split, and the test split is then scored once with those settings. Scores are
averaged per font and resampled by cluster, because styles of one design are not independent.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from . import metrics
from .baseline import glyph_distances, nearest_fonts
from .data import to_unit
from .generate import SamplerConfig, encode_styles, noise_for, sample_glyphs
from .model import TypeFlow
from .perceptual import GlyphNet, feature_distance
from .render import CHAR_TO_INDEX, CHARSET
from .store import GlyphStore

# The first K of these are the references. Round, straight, ascending and descending shapes come
# early, so that small K still sees the features a type designer would start from.
REFERENCE_ORDER = "nHoEagsR"
REFERENCE_COUNTS = (1, 2, 4, 8)
# Every K is scored on the same glyphs, which excludes all eight possible references.
TARGET_CHARS = [i for i, c in enumerate(CHARSET) if c not in REFERENCE_ORDER]
METRIC_DIRECTIONS = {
    "l1": "lower",
    "ssim": "higher",
    "ink_iou": "higher",
    "perceptual": "lower",
    "legibility": "higher",
    "grey_share": "lower",
}


def reference_chars(k: int) -> list[int]:
    return [CHAR_TO_INDEX[c] for c in REFERENCE_ORDER[:k]]


@torch.no_grad()
def score_glyphs(
    predicted: np.ndarray,
    truth: np.ndarray,
    chars: np.ndarray,
    net: GlyphNet,
    device: torch.device,
    batch: int = 128,
) -> dict[str, np.ndarray]:
    """Score uint8 glyph rows ``(N, H, W)``, returning one value per glyph for each metric."""
    columns: dict[str, list[Tensor]] = defaultdict(list)
    for start in range(0, len(predicted), batch):
        p = torch.from_numpy(np.asarray(predicted[start : start + batch])).float() / 255
        t = torch.from_numpy(np.asarray(truth[start : start + batch])).float() / 255
        columns["l1"].append(metrics.l1(p, t))
        columns["ssim"].append(metrics.ssim(p, t))
        columns["ink_iou"].append(metrics.ink_iou(p, t))
        columns["grey_share"].append(metrics.grey_share(p))
        x, y = (p[:, None] * 2 - 1).to(device), (t[:, None] * 2 - 1).to(device)
        columns["perceptual"].append(feature_distance(net, x, y).cpu())
        wanted = torch.from_numpy(chars[start : start + batch]).to(device)
        columns["legibility"].append((net(x)[1].argmax(1) == wanted).float().cpu())
    return {name: torch.cat(values).numpy() for name, values in columns.items()}


def per_font(scores: dict[str, np.ndarray], fonts: int) -> dict[str, np.ndarray]:
    return {k: v.reshape(fonts, -1).mean(axis=1) for k, v in scores.items()}


def cluster_bootstrap(
    values: np.ndarray, clusters: list[str], draws: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% interval for a mean, resampling whole clusters."""
    names = sorted(set(clusters))
    groups = np.array([names.index(c) for c in clusters])
    totals = np.bincount(groups, weights=values, minlength=len(names))
    counts = np.bincount(groups, minlength=len(names))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(names), size=(draws, len(names)))
    means = totals[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


@dataclass(frozen=True)
class Split:
    """Everything needed to score one set of fonts on a fixed set of target characters."""

    ids: np.ndarray
    truth: np.ndarray
    target_chars: list[int]

    @property
    def flat_chars(self) -> np.ndarray:
        return np.tile(np.array(self.target_chars), len(self.ids))

    def flatten(self, glyphs: np.ndarray) -> np.ndarray:
        return np.asarray(glyphs).reshape(-1, *glyphs.shape[2:])


def make_split(store: GlyphStore, ids: np.ndarray, target_chars: list[int]) -> Split:
    return Split(ids, np.asarray(store.glyphs[ids][:, target_chars]), target_chars)


def run_model(
    model: TypeFlow,
    store: GlyphStore,
    split: Split,
    k: int,
    sampler: SamplerConfig,
    seed: int = 0,
) -> np.ndarray:
    refs = reference_chars(k)
    ref_images = np.asarray(store.glyphs[split.ids][:, refs])
    ref_chars = np.tile(np.array(refs), (len(split.ids), 1))
    style = encode_styles(model, ref_images, ref_chars)
    n, m = len(split.ids), len(split.target_chars)
    device = style.device
    repeated = style.repeat_interleave(m, dim=0)
    chars = torch.from_numpy(split.flat_chars).to(device)
    noise = noise_for((n * m, 1, store.size, store.size), seed)
    return sample_glyphs(model, repeated, chars, noise, sampler).reshape(
        n, m, store.size, store.size
    )


def summarize(scores: dict[str, np.ndarray], clusters: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for name, values in scores.items():
        low, high = cluster_bootstrap(values, clusters)
        out[name] = {"mean": float(values.mean()), "ci95": [low, high]}
    return out


def sweep_sampler(
    model: TypeFlow,
    net: GlyphNet,
    store: GlyphStore,
    device: torch.device,
    fonts: int = 60,
    chars_per_font: int = 12,
    k: int = 4,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Measure sampler settings on validation fonts. The test split is not touched here."""
    val = store.split_indices("val")
    picks = np.random.default_rng(seed).choice(val, size=min(fonts, len(val)), replace=False)
    targets = TARGET_CHARS[:: max(1, len(TARGET_CHARS) // chars_per_font)][:chars_per_font]
    split = make_split(store, np.sort(picks), targets)
    rows = []
    grid = [("euler", s, 1.0) for s in (1, 2, 4, 8, 16, 32, 64)]
    grid += [("heun", s, 1.0) for s in (1, 2, 4, 8, 16, 32)]
    grid += [("rk4", s, 1.0) for s in (1, 2, 4, 8)]
    grid += [("heun", 16, g) for g in (1.25, 1.5, 2.0, 3.0)]
    for method, steps, guidance in grid:
        cfg = SamplerConfig(steps=steps, method=method, guidance=guidance)
        output = run_model(model, store, split, k, cfg, seed)
        scores = score_glyphs(
            split.flatten(output), split.flatten(split.truth), split.flat_chars, net, device
        )
        row: dict[str, object] = {
            "method": method,
            "steps": steps,
            "guidance": guidance,
            "velocity_calls": steps * {"euler": 1, "heun": 2, "rk4": 4}[method],
        }
        row.update({name: float(values.mean()) for name, values in scores.items()})
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def baseline_outputs(
    store: GlyphStore, split: Split, train_ids: np.ndarray
) -> tuple[torch.Tensor, np.ndarray]:
    """Return per-character distances to the training fonts, and those fonts' target glyphs."""
    corpus = store.glyphs[np.sort(train_ids)]
    queries = store.glyphs[split.ids]
    return glyph_distances(corpus, queries), np.asarray(corpus[:, split.target_chars])


def evaluate_test(
    model: TypeFlow,
    net: GlyphNet,
    store: GlyphStore,
    sampler: SamplerConfig,
    device: torch.device,
    out_dir: Path,
    seed: int = 0,
    counts: tuple[int, ...] = REFERENCE_COUNTS,
    every_nth_char: int = 1,
) -> dict[str, object]:
    """Score the model and baselines on the test split, and keep outputs for the sheets."""
    ids = store.split_indices("test")
    targets = TARGET_CHARS[::every_nth_char]
    split = make_split(store, ids, targets)
    train_ids = np.sort(store.split_indices("train"))
    distances, corpus_targets = baseline_outputs(store, split, train_ids)
    clusters = [store.rows[i].cluster for i in ids]
    categories = [store.rows[i].category for i in ids]
    flat_truth = split.flatten(split.truth)
    flat_chars = split.flat_chars
    fonts = len(ids)

    def score(glyphs: np.ndarray) -> dict[str, np.ndarray]:
        return per_font(
            score_glyphs(split.flatten(glyphs), flat_truth, flat_chars, net, device), fonts
        )

    rows: dict[str, dict[str, object]] = {}
    per_font_scores: dict[str, dict[str, list[float]]] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    target_positions = torch.tensor(targets)
    oracle = distances[target_positions].sum(dim=0).argmin(dim=1)
    oracle_scores = score(corpus_targets[oracle.numpy()])
    rows["oracle_font"] = summarize(oracle_scores, clusters)

    for k in counts:
        nearest = nearest_fonts(distances, reference_chars(k)).numpy()
        nn_output = corpus_targets[nearest]
        nn_scores = score(nn_output)
        rows[f"nearest_k{k}"] = summarize(nn_scores, clusters)
        output = run_model(model, store, split, k, sampler, seed)
        model_scores = score(output)
        rows[f"model_k{k}"] = summarize(model_scores, clusters)
        np.save(out_dir / f"model_k{k}.npy", output)
        np.save(out_dir / f"nearest_k{k}.npy", nn_output)
        differences = {}
        for name in model_scores:
            gap = model_scores[name] - nn_scores[name]
            low, high = cluster_bootstrap(gap, clusters)
            differences[name] = {"mean": float(gap.mean()), "ci95": [low, high]}
        rows[f"model_minus_nearest_k{k}"] = differences
        per_font_scores[f"model_k{k}"] = {n: v.tolist() for n, v in model_scores.items()}
        per_font_scores[f"nearest_k{k}"] = {n: v.tolist() for n, v in nn_scores.items()}
        print(json.dumps({k: rows[f"model_k{k}"], "nn": rows[f"nearest_k{k}"]}), flush=True)

    by_category: dict[str, dict[str, object]] = {}
    headline_k = 4 if 4 in counts else counts[0]
    for category in sorted(set(categories)):
        mask = np.array([c == category for c in categories])
        entry: dict[str, object] = {"fonts": int(mask.sum())}
        for label in (f"model_k{headline_k}", f"nearest_k{headline_k}"):
            entry[label] = {
                n: float(np.array(v)[mask].mean()) for n, v in per_font_scores[label].items()
            }
        by_category[category] = entry

    return {
        "sampler": asdict(sampler),
        "seed": seed,
        "test_fonts": fonts,
        "test_clusters": len(set(clusters)),
        "target_chars": len(targets),
        "reference_order": REFERENCE_ORDER,
        "train_fonts_searched": len(train_ids),
        "summary": rows,
        "by_category": by_category,
        "fonts": [store.rows[i].key for i in ids],
        "per_font": per_font_scores,
    }


@torch.no_grad()
def interpolation_study(
    model: TypeFlow,
    net: GlyphNet,
    store: GlyphStore,
    sampler: SamplerConfig,
    device: torch.device,
    pairs: int = 40,
    frames: int = 9,
    seed: int = 0,
    k: int = 8,
) -> dict[str, object]:
    """Compare style-space interpolation with a pixel crossfade on random held-out font pairs.

    The crossfade is the straightforward alternative. Its midpoint is two translucent glyphs on top
    of each other, so grey share and legibility at the midpoint show whether the model is doing
    something that a crossfade cannot.
    """
    rng = np.random.default_rng(seed)
    test = store.split_indices("test")
    clusters = np.array([store.rows[i].cluster for i in test])
    chosen: list[tuple[int, int]] = []
    while len(chosen) < pairs:
        a, b = rng.choice(len(test), size=2, replace=False)
        if clusters[a] != clusters[b]:
            chosen.append((int(test[a]), int(test[b])))
    targets = np.array(TARGET_CHARS[:: len(TARGET_CHARS) // 8][:8])
    refs = reference_chars(k)
    ref_chars = np.array([refs])
    weights = torch.linspace(0, 1, frames, device=device)
    middle = frames // 2

    evenness, rows = [], defaultdict(list)
    for index, (a, b) in enumerate(chosen):
        style_a = encode_styles(model, np.asarray(store.glyphs[[a]][:, refs]), ref_chars)
        style_b = encode_styles(model, np.asarray(store.glyphs[[b]][:, refs]), ref_chars)
        path = torch.lerp(style_a, style_b, weights[:, None])
        styles = path.repeat_interleave(len(targets), dim=0)
        chars = torch.from_numpy(np.tile(targets, frames)).to(device)
        base = noise_for((len(targets), 1, store.size, store.size), seed + index)
        noise = base.repeat(frames, 1, 1, 1)
        sheet = sample_glyphs(model, styles, chars, noise, sampler).reshape(
            frames, len(targets), store.size, store.size
        )
        tensor = torch.from_numpy(sheet).float() / 255
        steps = (tensor[1:] - tensor[:-1]).abs().mean(dim=(1, 2, 3))
        evenness.append(float(steps.max() / steps.mean().clamp_min(1e-6)))

        first, last = to_unit(store.glyphs[a][targets]), to_unit(store.glyphs[b][targets])
        crossfade = ((first + last) / 2 + 1) * 0.5
        generated = tensor[middle]
        wanted = torch.from_numpy(targets).to(device)
        for name, midpoint in (("flow", generated), ("crossfade", crossfade)):
            logits = net((midpoint[:, None] * 2 - 1).to(device))[1]
            rows[f"{name}_legibility"].append(float((logits.argmax(1) == wanted).float().mean()))
            rows[f"{name}_grey_share"].append(float(metrics.grey_share(midpoint).mean()))
    return {
        "pairs": pairs,
        "frames": frames,
        "max_over_mean_frame_step": float(np.mean(evenness)),
        **{name: float(np.mean(values)) for name, values in rows.items()},
    }


def write_results(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
