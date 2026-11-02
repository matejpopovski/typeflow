"""The font corpus: what google/fonts contains, what the licences say, and how it is fetched.

Everything is pinned to one upstream commit. Files are addressed by git blob hash, so a fetch
either reproduces the corpus byte for byte or fails loudly.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

UPSTREAM_COMMIT = "6c1dbe27dcf7d817d4b62cec033ba77fa8e2a5ec"
TREE_URL = "https://api.github.com/repos/google/fonts/git/trees/{commit}?recursive=1"
RAW_URL = "https://raw.githubusercontent.com/google/fonts/{commit}/{path}"

# Files this large are CJK or other wide-coverage fonts whose Latin is usually borrowed.
MAX_FONT_BYTES = 3_000_000

MANIFEST_DIR = Path(__file__).resolve().parents[2] / "fonts"
FAMILIES_CSV = "families.csv"
STYLES_CSV = "styles.csv"
REJECTED_CSV = "rejected.csv"

FAMILY_FIELDS = ("family", "directory", "category", "licence", "ofl_blob", "cluster", "split")
STYLE_FIELDS = ("family", "path", "blob", "style", "instance")


@dataclass(frozen=True)
class Family:
    family: str
    directory: str
    category: str
    licence: str
    ofl_blob: str
    cluster: str = ""
    split: str = ""


@dataclass(frozen=True)
class Style:
    family: str
    path: str
    blob: str
    style: str
    instance: str

    @property
    def key(self) -> str:
        return f"{self.family} {self.style}"


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def parse_metadata(text: str) -> dict[str, object]:
    """Read the fields used here from a METADATA.pb without a protobuf dependency."""

    def field(name: str) -> str:
        match = re.search(rf'^{name}: "([^"]*)"', text, re.M)
        return match.group(1) if match else ""

    return {
        "name": field("name"),
        "category": field("category"),
        "license": field("license"),
        "subsets": re.findall(r'^subsets: "([^"]+)"', text, re.M),
    }


def confirm_ofl(text: str) -> bool:
    """Check the licence text itself. The directory name and METADATA are only claims about it."""
    head = " ".join(text.upper().split())
    return "SIL OPEN FONT LICENSE" in head and "VERSION 1.1" in head


def _get(url: str, attempts: int = 5, timeout: float = 30.0) -> bytes:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def fetch_blob(path: str, blob: str, root: Path, commit: str = UPSTREAM_COMMIT) -> Path:
    """Download one upstream file into ``root``, verifying it against its git blob hash."""
    target = root / path
    if target.exists() and git_blob_sha(target.read_bytes()) == blob:
        return target
    data = _get(RAW_URL.format(commit=commit, path=path))
    if git_blob_sha(data) != blob:
        raise ValueError(f"hash mismatch for {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".part")
    temp.write_bytes(data)
    temp.replace(target)
    return target


def fetch_all(
    items: Iterable[tuple[str, str]], root: Path, workers: int = 12
) -> dict[str, Exception]:
    """Fetch ``(path, blob)`` pairs concurrently and return any failures by path."""
    failures: dict[str, Exception] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_blob, p, b, root): p for p, b in items}
        for future in as_completed(futures):
            exc = future.exception()
            if exc is not None:
                failures[futures[future]] = exc
    return failures


def list_upstream(commit: str = UPSTREAM_COMMIT) -> list[dict[str, object]]:
    tree = json.loads(_get(TREE_URL.format(commit=commit)))
    if tree["truncated"]:
        raise RuntimeError("GitHub truncated the tree listing")
    return [e for e in tree["tree"] if e["type"] == "blob" and e["path"].startswith("ofl/")]


def group_by_directory(entries: Iterable[dict[str, object]]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = {}
    for entry in entries:
        parts = str(entry["path"]).split("/")
        grouped.setdefault(parts[1], {})["/".join(parts[2:])] = entry
    return grouped


def _write(path: Path, fields: tuple[str, ...], rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({f: getattr(row, f) for f in fields})


def save_manifest(families: list[Family], styles: list[Style], directory: Path) -> None:
    _write(directory / FAMILIES_CSV, FAMILY_FIELDS, families)
    _write(directory / STYLES_CSV, STYLE_FIELDS, styles)


def load_families(directory: Path = MANIFEST_DIR) -> list[Family]:
    with (directory / FAMILIES_CSV).open(newline="") as handle:
        return [Family(**row) for row in csv.DictReader(handle)]


def load_styles(directory: Path = MANIFEST_DIR) -> list[Style]:
    """Styles in manifest order, which is also their row order in the rendered glyph store."""
    with (directory / STYLES_CSV).open(newline="") as handle:
        return [Style(**row) for row in csv.DictReader(handle)]


def fetch_manifest(root: Path, directory: Path = MANIFEST_DIR) -> dict[str, Exception]:
    """Fetch every font in the manifest and each family's licence, verifying both."""
    families = load_families(directory)
    items = {(s.path, s.blob) for s in load_styles(directory)}
    items |= {(f"ofl/{f.directory}/OFL.txt", f.ofl_blob) for f in families}
    failures = fetch_all(sorted(items), root)
    for family in families:
        licence = root / "ofl" / family.directory / "OFL.txt"
        if licence.exists() and not confirm_ofl(licence.read_text(errors="replace")):
            failures[str(licence)] = ValueError("licence text is not SIL OFL 1.1")
    return failures
