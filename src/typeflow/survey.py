"""Decide which fonts enter the corpus and write the committed manifest.

This is the only step that needs the whole of google/fonts. Its output, ``fonts/*.csv``, is what
everyone else fetches, so a rejected font stays visible with the reason it was rejected.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .corpus import (
    MANIFEST_DIR,
    MAX_FONT_BYTES,
    REJECTED_CSV,
    Family,
    Style,
    confirm_ofl,
    fetch_all,
    group_by_directory,
    list_upstream,
    parse_metadata,
    save_manifest,
)
from .pool import Failure, map_with_timeout
from .render import (
    CHARSET,
    Frame,
    RenderError,
    covers_charset,
    is_color_font,
    letterform_spread,
    list_styles,
    pick_styles,
    render_glyphs,
)
from .split import UnionFind, assign_splits, name_stem, twin_pairs

# Below this a font's glyphs are near-identical shapes, as in redaction blocks.
MIN_LETTERFORM_SPREAD = 0.12
# Fonts whose Latin code points hold something other than letters, in a way spread cannot detect.
NOT_LETTERFORMS = re.compile(r"Barcode|Redacted|Keyboard|Guides?\b")
SIGNATURE_BLOCK = 4
STYLES_PER_FAMILY = 4


@dataclass
class FamilyTask:
    name: str
    directory: str
    root: Path
    files: dict[str, str]
    frame: Frame


@dataclass
class FamilyResult:
    styles: list[Style] = field(default_factory=list)
    signatures: list[np.ndarray] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)


def signature(glyphs: np.ndarray) -> np.ndarray:
    n, h, w = glyphs.shape
    b = SIGNATURE_BLOCK
    blocks = glyphs[:, : h - h % b, : w - w % b].reshape(n, h // b, b, w // b, b)
    return blocks.mean(axis=(2, 4)).astype(np.uint8).reshape(n, -1)


def inspect_family(task: FamilyTask) -> FamilyResult:
    result = FamilyResult()
    candidates: list[tuple[str, str, str]] = []
    for name in sorted(task.files):
        path = task.root / "ofl" / task.directory / name
        if not covers_charset(path):
            result.rejected.append((name, "incomplete Latin coverage"))
        elif is_color_font(path):
            result.rejected.append((name, "colour font"))
        else:
            candidates += [(name, style, inst) for style, inst in list_styles(path)]

    seen: set[str] = set()
    for name, style, instance in pick_styles(candidates, STYLES_PER_FAMILY):
        path = task.root / "ofl" / task.directory / name
        label = f"{name}:{style}"
        try:
            glyphs = render_glyphs(path, instance, task.frame)
        except (RenderError, OSError) as exc:
            result.rejected.append((label, str(exc)))
            continue
        digest = hashlib.sha256(glyphs.tobytes()).hexdigest()
        if digest in seen:
            result.rejected.append((label, "identical to another style of the family"))
            continue
        seen.add(digest)
        if letterform_spread(glyphs) < MIN_LETTERFORM_SPREAD:
            result.rejected.append((label, "glyphs are not distinct letterforms"))
            continue
        result.styles.append(
            Style(task.name, f"ofl/{task.directory}/{name}", task.files[name], style, instance)
        )
        result.signatures.append(signature(glyphs))
    return result


def survey(root: Path, manifest_dir: Path = MANIFEST_DIR, workers: int = 8) -> dict[str, int]:
    frame = Frame()
    grouped = group_by_directory(list_upstream())
    fetch_all(
        [
            (entry["path"], entry["sha"])
            for files in grouped.values()
            for key, entry in files.items()
            if key in ("OFL.txt", "METADATA.pb")
        ],
        root,
    )

    rejected: list[tuple[str, str, str]] = []
    drafts: dict[str, Family] = {}
    tasks: list[FamilyTask] = []
    for directory, files in sorted(grouped.items()):
        if "OFL.txt" not in files or "METADATA.pb" not in files:
            rejected.append((directory, "", "no licence or metadata file"))
            continue
        base = root / "ofl" / directory
        meta = parse_metadata((base / "METADATA.pb").read_text(errors="replace"))
        licence = (base / "OFL.txt").read_text(errors="replace")
        if meta["license"] != "OFL" or not confirm_ofl(licence):
            rejected.append((directory, "", "licence text is not SIL OFL 1.1"))
            continue
        if "latin" not in meta["subsets"]:
            rejected.append((directory, "", "no Latin subset"))
            continue
        if NOT_LETTERFORMS.search(str(meta["name"])):
            rejected.append((directory, "", "font does not contain letterforms"))
            continue
        fonts = {k: e for k, e in files.items() if k.endswith(".ttf") and "/" not in k}
        small = {k: e["sha"] for k, e in fonts.items() if int(e["size"]) <= MAX_FONT_BYTES}
        rejected += [
            (directory, k, "font file over the size limit") for k in fonts.keys() - small.keys()
        ]
        if not small:
            continue
        name = str(meta["name"])
        drafts[name] = Family(
            name, directory, str(meta["category"]), "OFL-1.1", files["OFL.txt"]["sha"]
        )
        tasks.append(FamilyTask(name, directory, root, small, frame))

    failures = fetch_all(
        [(f"ofl/{t.directory}/{n}", b) for t in tasks for n, b in t.files.items()], root
    )
    if failures:
        raise RuntimeError(f"{len(failures)} fonts failed to download: {list(failures)[:3]}")

    styles: list[Style] = []
    signatures: list[np.ndarray] = []
    for index, outcome in map_with_timeout(inspect_family, tasks, workers, timeout=240):
        task = tasks[index]
        if isinstance(outcome, Failure):
            rejected.append((task.directory, "", outcome.reason))
            continue
        rejected += [(task.directory, label, why) for label, why in outcome.rejected]
        styles += outcome.styles
        signatures += outcome.signatures

    order = sorted(range(len(styles)), key=lambda i: (styles[i].family, styles[i].style))
    styles = [styles[i] for i in order]
    stack = np.stack([signatures[i] for i in order])

    kept = sorted({s.family for s in styles})
    union = UnionFind(kept)
    categories = {name: drafts[name].category for name in kept}
    by_stem: dict[str, str] = {}
    for family in kept:
        union.union(family, by_stem.setdefault(name_stem(family), family))
    twins = twin_pairs(stack, [s.family for s in styles])
    for i, j, _ in twins:
        union.union(styles[i].family, styles[j].family)

    different_stems = {
        tuple(sorted((styles[i].family, styles[j].family)))
        for i, j, _ in twins
        if name_stem(styles[i].family) != name_stem(styles[j].family)
    }
    sizes = Counter(union.find(s.family) for s in styles)
    splits = assign_splits(dict(sizes))
    families = [
        Family(
            f.family,
            f.directory,
            f.category,
            f.licence,
            f.ofl_blob,
            union.find(f.family),
            splits[union.find(f.family)],
        )
        for f in (drafts[name] for name in kept)
    ]
    save_manifest(families, styles, manifest_dir)
    with (manifest_dir / REJECTED_CSV).open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("directory", "item", "reason"))
        writer.writerows(sorted(rejected))

    return {
        "upstream_directories": len(grouped),
        "families": len(families),
        "styles": len(styles),
        "clusters": len(sizes),
        "largest_cluster_styles": max(sizes.values()),
        "twin_style_pairs": len(twins),
        "twin_family_pairs_with_different_stems": len(different_stems),
        "chars": len(CHARSET),
        "styles_by_split": dict(Counter(splits[union.find(s.family)] for s in styles)),
        "styles_by_category": dict(Counter(categories[s.family] for s in styles)),
    }
