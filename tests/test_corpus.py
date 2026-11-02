from pathlib import Path

import pytest

from typeflow import corpus


def test_git_blob_sha_matches_git() -> None:
    assert corpus.git_blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_parse_metadata_reads_the_fields_used() -> None:
    text = (
        'name: "Foo Sans"\ncategory: "SERIF"\nlicense: "OFL"\nsubsets: "latin"\nsubsets: "greek"\n'
    )
    meta = corpus.parse_metadata(text)
    assert meta == {
        "name": "Foo Sans",
        "category": "SERIF",
        "license": "OFL",
        "subsets": ["latin", "greek"],
    }


def test_confirm_ofl_reads_the_licence_text() -> None:
    assert corpus.confirm_ofl(
        "Copyright 2020\nThis Font Software is licensed under the SIL Open Font License, "
        "Version 1.1.\nSIL OPEN FONT LICENSE\nVersion 1.1 - 26 February 2007"
    )
    assert not corpus.confirm_ofl("Apache License\nVersion 2.0, January 2004")
    assert not corpus.confirm_ofl("SIL Open Font License, Version 1.0")


def test_fetch_blob_refuses_content_with_the_wrong_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(corpus, "_get", lambda url: b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        corpus.fetch_blob("ofl/a/A.ttf", corpus.git_blob_sha(b"original"), tmp_path)
    assert not (tmp_path / "ofl/a/A.ttf").exists()


def test_fetch_blob_skips_a_file_that_is_already_correct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "ofl/a/A.ttf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"data")

    def fail(url: str) -> bytes:
        raise AssertionError("should not download")

    monkeypatch.setattr(corpus, "_get", fail)
    assert corpus.fetch_blob("ofl/a/A.ttf", corpus.git_blob_sha(b"data"), tmp_path) == target


def test_group_by_directory_nests_static_subfolders() -> None:
    entries = [{"path": "ofl/foo/Foo.ttf"}, {"path": "ofl/foo/static/Foo-Bold.ttf"}]
    grouped = corpus.group_by_directory(entries)
    assert set(grouped["foo"]) == {"Foo.ttf", "static/Foo-Bold.ttf"}
