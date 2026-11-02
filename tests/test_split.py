from collections import defaultdict

import numpy as np

from typeflow import corpus
from typeflow.split import UnionFind, assign_splits, name_stem, twin_pairs


def test_union_find_merges_transitively() -> None:
    union = UnionFind("abcd")
    union.union("a", "b")
    union.union("b", "c")
    assert union.find("a") == union.find("c") != union.find("d")


def test_name_stem_groups_siblings() -> None:
    assert name_stem("Hind Madurai") == name_stem("Hind Siliguri") == "hind"


def signature(seed: int, ink: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return ((rng.random((20, 1024)) < 0.1) * 255 * ink).astype(np.uint8)


def test_twins_are_found_even_when_only_some_glyphs_match() -> None:
    original = signature(0)
    borrowed = signature(1)
    borrowed[:12] = original[:12]
    pairs = twin_pairs(np.stack([original, borrowed]), ["A", "B"])
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]


def test_sparse_unrelated_faces_are_not_twins() -> None:
    a, b = signature(0, ink=0.05), signature(1, ink=0.05)
    assert twin_pairs(np.stack([a, b]), ["A", "B"]) == []


def test_styles_of_one_family_are_not_reported() -> None:
    a = signature(0)
    assert twin_pairs(np.stack([a, a]), ["A", "A"]) == []


def test_assign_splits_is_stable_and_respects_budget() -> None:
    sizes = {f"c{i}": 4 for i in range(200)}
    first = assign_splits(sizes)
    assert first == assign_splits(dict(reversed(list(sizes.items()))))
    counts = {s: sum(v == s for v in first.values()) for s in ("train", "val", "test")}
    assert counts == {"train": 170, "val": 10, "test": 20}


def test_huge_clusters_stay_in_training() -> None:
    sizes = {"huge": 400, **{f"c{i}": 1 for i in range(100)}}
    assert assign_splits(sizes)["huge"] == "train"


class TestCommittedManifest:
    """The held-out split is a claim the repository makes, so these check the real lists."""

    families = corpus.load_families()
    styles = corpus.load_styles()

    def test_every_family_is_confirmed_ofl(self) -> None:
        assert {f.licence for f in self.families} == {"OFL-1.1"}
        assert all(len(f.ofl_blob) == 40 for f in self.families)

    def test_every_font_comes_from_the_ofl_folder_at_a_fixed_hash(self) -> None:
        assert all(s.path.startswith("ofl/") and len(s.blob) == 40 for s in self.styles)

    def test_every_style_belongs_to_a_listed_family(self) -> None:
        names = {f.family for f in self.families}
        assert {s.family for s in self.styles} == names

    def test_no_cluster_straddles_splits(self) -> None:
        seen: dict[str, set[str]] = defaultdict(set)
        for family in self.families:
            seen[family.cluster].add(family.split)
        assert all(len(splits) == 1 for splits in seen.values())

    def test_no_name_stem_straddles_splits(self) -> None:
        seen: dict[str, set[str]] = defaultdict(set)
        for family in self.families:
            seen[name_stem(family.family)].add(family.split)
        assert all(len(splits) == 1 for splits in seen.values())

    def test_a_directory_maps_to_one_split(self) -> None:
        directories = [f.directory for f in self.families]
        assert len(directories) == len(set(directories))

    def test_style_keys_are_unique(self) -> None:
        keys = [s.key for s in self.styles]
        assert len(keys) == len(set(keys))

    def test_held_out_sets_cover_every_category(self) -> None:
        split = {f.family: f.split for f in self.families}
        category = {f.family: f.category for f in self.families}
        held = {category[s.family] for s in self.styles if split[s.family] == "test"}
        assert held == set(category.values())

    def test_shares_are_near_their_targets(self) -> None:
        split = {f.family: f.split for f in self.families}
        counts = defaultdict(int)
        for style in self.styles:
            counts[split[style.family]] += 1
        assert abs(counts["test"] / len(self.styles) - 0.10) < 0.005
        assert abs(counts["val"] / len(self.styles) - 0.05) < 0.005
