"""Turn the JSON results into the Markdown tables quoted in the README."""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("results")
COLUMNS = (
    ("l1", "L1", 4),
    ("ssim", "SSIM", 3),
    ("ink_iou", "Ink IoU", 3),
    ("perceptual", "Perceptual", 3),
    ("legibility", "Legible", 3),
    ("grey_share", "Grey share", 3),
)
METHODS = (
    ("mean_glyph", "Mean glyph (no references)"),
    ("nearest_k1", "Nearest font, K=1"),
    ("model_k1", "typeflow, K=1"),
    ("nearest_k2", "Nearest font, K=2"),
    ("model_k2", "typeflow, K=2"),
    ("nearest_k4", "Nearest font, K=4"),
    ("model_k4", "typeflow, K=4"),
    ("nearest_k8", "Nearest font, K=8"),
    ("model_k8", "typeflow, K=8"),
    ("oracle_font", "Oracle font (sees the targets)"),
)


def header(first: str) -> list[str]:
    names = [name for _, name, _ in COLUMNS]
    return [f"| {first} | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]


def row(label: str, scores: dict) -> str:
    cells = [f"{scores[m]['mean']:.{d}f}" for m, _, d in COLUMNS]
    return f"| {label} | " + " | ".join(cells) + " |"


def main_table(summary: dict, secondary: dict) -> list[str]:
    lines = header("Method")
    for key, label in METHODS:
        lines.append(row(label, summary[key]))
        if key == "model_k4":
            lines.append(
                row("typeflow, K=4, 32 Euler steps (picked by eye)", secondary["model_k4"])
            )
    return lines


def difference_table(summary: dict) -> list[str]:
    lines = ["| K | Metric | typeflow minus nearest | 95% interval |", "|---|---|---:|---|"]
    for k in (1, 2, 4, 8):
        gaps = summary[f"model_minus_nearest_k{k}"]
        for metric, name, digits in COLUMNS[:5]:
            gap = gaps[metric]
            low, high = gap["ci95"]
            interval = f"{low:+.{digits}f} to {high:+.{digits}f}"
            lines.append(f"| {k} | {name} | {gap['mean']:+.{digits}f} | {interval} |")
    return lines


def category_table(by_category: dict) -> list[str]:
    lines = [
        "| Category | Fonts | L1 typeflow | L1 nearest | Perceptual typeflow | Perceptual nearest |"
        " Legible typeflow |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category, entry in by_category.items():
        model, nearest = entry["model_k4"], entry["nearest_k4"]
        lines.append(
            f"| {category.replace('_', ' ').title()} | {entry['fonts']} | {model['l1']:.4f} "
            f"| {nearest['l1']:.4f} | {model['perceptual']:.3f} | {nearest['perceptual']:.3f} "
            f"| {model['legibility']:.3f} |"
        )
    return lines


def sampler_table(rows: list[dict]) -> list[str]:
    lines = header("Solver, steps")
    lines[0] = lines[0].replace("| Solver, steps |", "| Solver, steps | Calls |")
    lines[1] = "|---|---:|" + "---:|" * len(COLUMNS)
    for row in rows:
        guidance = f", guidance {row['guidance']}" if row["guidance"] != 1.0 else ""
        cells = [f"{row[m]:.{d}f}" for m, _, d in COLUMNS]
        label = f"{row['method']}, {row['steps']}{guidance}"
        lines.append(f"| {label} | {row['velocity_calls']} | " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    report = json.loads((RESULTS / "test.json").read_text())
    sampler = json.loads((RESULTS / "sampler_val.json").read_text())
    secondary = json.loads((RESULTS / "test_euler32.json").read_text())
    sections = [
        ("Test split, mean per font", main_table(report["summary"], secondary["summary"])),
        ("Paired difference against the nearest font", difference_table(report["summary"])),
        ("By category at K=4", category_table(report["by_category"])),
        ("Sampler sweep on validation fonts at K=4", sampler_table(sampler)),
    ]
    text = []
    for title, lines in sections:
        text += [f"## {title}", "", *lines, ""]
    (RESULTS / "tables.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
