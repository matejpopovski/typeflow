"""Command line entry point."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .generate import SamplerConfig

DEFAULT_WEIGHTS = Path("weights/typeflow.pt")
PERCEPTUAL = Path("weights/perceptual.pt")


def _train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", type=Path, default=Path("runs/main"))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--minutes", type=float, default=70.0)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--fonts", type=int, default=16)
    parser.add_argument("--per-font", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-seconds", type=float, default=300.0)


def _sampler_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--steps", type=int, default=SamplerConfig.steps)
    parser.add_argument("--method", default=SamplerConfig.method)
    parser.add_argument("--guidance", type=float, default=SamplerConfig.guidance)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)


def _sampler(args: argparse.Namespace) -> SamplerConfig:
    return SamplerConfig(steps=args.steps, method=args.method, guidance=args.guidance)


def _train(args: argparse.Namespace) -> None:
    from .model import ModelConfig
    from .store import load_store
    from .train import TrainConfig, pick_device, train

    store = load_store(args.data, args.size)
    cfg = TrainConfig(
        fonts=args.fonts,
        per_font=args.per_font,
        max_steps=args.steps,
        budget_seconds=args.minutes * 60,
        seed=args.seed,
        checkpoint_seconds=args.checkpoint_seconds,
    )
    model = ModelConfig(image_size=args.size)
    train(
        store.glyphs, store.split_indices("train"), model, cfg, args.run, pick_device(args.device)
    )


def _supervise(args: argparse.Namespace) -> int:
    from .runtime import supervise, trainer_command

    command = trainer_command(
        [
            "--data", str(args.data),
            "train",
            "--run", str(args.run),
            "--size", str(args.size),
            "--minutes", str(args.minutes),
            "--steps", str(args.steps),
            "--fonts", str(args.fonts),
            "--per-font", str(args.per_font),
            "--seed", str(args.seed),
            "--device", args.device,
            "--checkpoint-seconds", str(args.checkpoint_seconds),
        ]
    )  # fmt: skip
    result = supervise(command, args.run, args.stall_seconds, log=args.run / "trainer.log")
    print(json.dumps(asdict(result)))
    return result.code


def _reference_glyphs(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    """Read references from a font file, or from a folder of ``<letter>.png`` images."""
    from PIL import Image

    from .render import render_glyphs

    if args.font:
        return render_glyphs(args.font, args.instance, chars=args.letters), args.letters
    images = [
        255 - np.asarray(Image.open(args.glyphs / f"{c}.png").convert("L"), dtype=np.uint8)
        for c in args.letters
    ]
    return np.stack(images), args.letters


def _extend(args: argparse.Namespace) -> None:
    from .generate import encode_styles, load_model, noise_for, sample_glyphs
    from .specimen import Sheet, default_typefaces, to_indices, trim
    from .train import pick_device

    model = load_model(args.weights, pick_device(args.device))
    refs, letters = _reference_glyphs(args)
    style = encode_styles(model, refs[None], np.array([to_indices(letters)]))
    wanted = args.chars
    size = model.cfg.image_size
    import torch

    chars = torch.tensor(to_indices(wanted), device=style.device)
    noise = noise_for((len(wanted), 1, size, size), args.seed)
    glyphs = sample_glyphs(model, style.expand(len(wanted), -1), chars, noise, _sampler(args))

    advance, per_row = int(size * 0.72), 13
    lines = [glyphs[i : i + per_row] for i in range(0, len(glyphs), per_row)]
    height = int(size * 0.95)
    sheet_height = 220 + height * (len(lines) + 1)
    sheet = Sheet(advance * per_row + 112, sheet_height, default_typefaces(args.data))
    y = sheet.header("Extension", f"Drawn from {' '.join(letters)}.")
    colors = [(168, 40, 32)] * len(refs)
    sheet.row(56, y, trim(refs, 0, height), advance, colors)
    y += height
    for line in lines:
        sheet.row(56, y, trim(line, 0, height), advance)
        y += height
    sheet.save(args.out)
    print(args.out)


def _morph(args: argparse.Namespace) -> None:
    import torch

    from .generate import encode_styles, load_model, noise_for, sample_glyphs
    from .render import render_glyphs
    from .specimen import default_typefaces, morph_sheet, to_indices
    from .train import pick_device

    model = load_model(args.weights, pick_device(args.device))
    letters = args.letters
    ref_chars = np.array([to_indices(letters)])
    a = encode_styles(model, render_glyphs(args.a, chars=letters)[None], ref_chars)
    b = encode_styles(model, render_glyphs(args.b, chars=letters)[None], ref_chars)
    size = model.cfg.image_size
    weights = torch.linspace(0, 1, args.frames, device=a.device)[:, None]
    styles = torch.lerp(a, b, weights).repeat_interleave(len(args.chars), dim=0)
    chars = torch.tensor(to_indices(args.chars) * args.frames, device=a.device)
    noise = noise_for((len(args.chars), 1, size, size), args.seed).repeat(args.frames, 1, 1, 1)
    glyphs = sample_glyphs(model, styles, chars, noise, _sampler(args)).reshape(
        args.frames, len(args.chars), size, size
    )
    rows = [(args.a.stem, args.b.stem, glyphs)]
    subtitle = "The noise is fixed, only the style vector moves."
    faces = default_typefaces(args.data)
    morph_sheet(rows, args.chars, "Between Two Faces", subtitle, faces).save(args.out)
    print(args.out)


def _experiment(args: argparse.Namespace) -> None:
    from . import evaluate as ev
    from .generate import export_weights, load_model
    from .perceptual import load_perceptual, train_perceptual, write_report
    from .store import load_store
    from .train import pick_device

    device = pick_device(args.device)
    if args.command == "perceptual":
        store = load_store(args.data, args.size)
        train_ids = np.sort(store.split_indices("train"))
        report = train_perceptual(store.glyphs, train_ids, device, PERCEPTUAL, args.steps)
        write_report(report, Path("results/perceptual.json"))
        print(json.dumps(report, indent=2))
        return
    if args.command == "export":
        print(export_weights(args.run / "latest.pt", DEFAULT_WEIGHTS))
        return

    store = load_store(args.data, args.size)
    model = load_model(args.weights, device)
    net = load_perceptual(PERCEPTUAL, device)
    if args.command == "sweep":
        rows = ev.sweep_sampler(model, net, store, device)
        ev.write_results(Path("results/sampler_val.json"), rows)
    elif args.command == "evaluate":
        report = ev.evaluate_test(
            model, net, store, _sampler(args), device, args.outputs, args.seed,
            tuple(args.reference_counts), args.every_nth_char,
        )  # fmt: skip
        ev.write_results(args.results, report)
        print(json.dumps(report["summary"], indent=2))
    elif args.command == "interpolate":
        report = ev.interpolation_study(model, net, store, _sampler(args), device)
        ev.write_results(Path("results/interpolation.json"), report)
        print(json.dumps(report, indent=2))
    elif args.command == "figures":
        from . import figures
        from .specimen import default_typefaces

        faces = default_typefaces(args.data)
        scores = figures.Scores.load(store, args.results, args.outputs)
        sampler = _sampler(args)
        shown = {
            "specimen": figures.hero(
                model, store, scores, sampler, faces, Path("figures/specimen.png")
            ),
            "gallery": figures.gallery(store, scores, faces, Path("figures/gallery.png")),
            "morph": figures.morph(model, store, scores, sampler, faces, Path("figures/morph.png")),
        }
        figures.sampler_steps(model, store, scores, faces, Path("figures/steps.png"))
        ev.write_results(Path("results/figures.json"), shown)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="typeflow", description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("survey", help="rebuild fonts/*.csv from google/fonts")
    commands.add_parser("fetch", help="download the fonts listed in fonts/*.csv")
    render = commands.add_parser("render", help="rasterise the manifest into a glyph store")
    render.add_argument("--size", type=int, default=128)

    _train_arguments(commands.add_parser("train", help="train in this process"))
    watched = commands.add_parser("supervise", help="train under the stall watchdog")
    _train_arguments(watched)
    watched.add_argument("--stall-seconds", type=float, default=180.0)

    perceptual = commands.add_parser("perceptual", help="train the metric network")
    perceptual.add_argument("--size", type=int, default=128)
    perceptual.add_argument("--steps", type=int, default=12_000)
    perceptual.add_argument("--device", default="auto")

    export = commands.add_parser("export", help="write half-precision EMA weights")
    export.add_argument("--run", type=Path, default=Path("runs/main"))
    export.add_argument("--device", default="auto")

    for name, description in (
        ("sweep", "measure sampler settings on validation fonts"),
        ("evaluate", "score the test split against the baselines"),
        ("interpolate", "measure interpolation against a pixel crossfade"),
        ("figures", "build the specimen sheets"),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
        command.add_argument("--size", type=int, default=128)
        command.add_argument("--every-nth-char", type=int, default=1)
        command.add_argument("--reference-counts", type=int, nargs="+", default=[1, 2, 4, 8])
        command.add_argument("--results", type=Path, default=Path("results/test.json"))
        command.add_argument("--outputs", type=Path, default=Path("data/outputs"))
        _sampler_arguments(command)

    extend = commands.add_parser("extend", help="draw a whole character set from a few letters")
    extend.add_argument("--font", type=Path, help="take the reference letters from this font")
    extend.add_argument("--instance", default="", help="named instance of a variable font")
    extend.add_argument("--glyphs", type=Path, help="or from <letter>.png files in this folder")
    extend.add_argument("--letters", default="nHoE")
    extend.add_argument("--chars", default="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    extend.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    extend.add_argument("--out", type=Path, default=Path("extension.png"))
    _sampler_arguments(extend)

    morph = commands.add_parser("morph", help="interpolate between two font files")
    morph.add_argument("a", type=Path)
    morph.add_argument("b", type=Path)
    morph.add_argument("--letters", default="nHoEagsR")
    morph.add_argument("--chars", default="Rag")
    morph.add_argument("--frames", type=int, default=9)
    morph.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    morph.add_argument("--out", type=Path, default=Path("morph.png"))
    _sampler_arguments(morph)

    args = parser.parse_args(argv)

    if args.command == "survey":
        from .survey import survey

        summary = json.dumps(survey(args.data / "upstream"), indent=2)
        Path("results").mkdir(exist_ok=True)
        Path("results/corpus.json").write_text(summary + "\n")
        print(summary)
    elif args.command == "fetch":
        from .corpus import fetch_manifest

        failures = fetch_manifest(args.data / "upstream")
        for path, error in failures.items():
            print(f"{path}: {error}")
        return 1 if failures else 0
    elif args.command == "render":
        from .store import build_store

        print(build_store(args.data, args.size))
    elif args.command == "train":
        _train(args)
    elif args.command == "supervise":
        return _supervise(args)
    elif args.command == "extend":
        _extend(args)
    elif args.command == "morph":
        _morph(args)
    else:
        _experiment(args)
    return 0
