"""Filesystem and sequence contracts for Temporal Lab, independent of a model."""
from pathlib import Path
import os
import sys

import pytest
from PIL import Image

from sdfx.temporal_lab_io import OutputTransaction, read_sequence


def make_frame(path: Path, size=(8, 6), color=(20, 40, 60)):
    Image.new("RGB", size, color).save(path)
    return path


def test_sequence_is_naturally_sorted_and_retains_dimensions(tmp_path):
    for name in ("frame10.png", "frame2.png", "frame1.png"):
        make_frame(tmp_path / name)
    result = read_sequence(str(tmp_path))
    assert [Path(p).name for p in result.paths] == ["frame1.png", "frame2.png", "frame10.png"]
    assert result.count == 3
    assert result.dimensions == (8, 6)


def test_sequence_rejects_empty_and_invalid_directories(tmp_path):
    with pytest.raises(ValueError, match="(?i)(empty|no.*frame)"):
        read_sequence(str(tmp_path))
    with pytest.raises((ValueError, OSError)):
        read_sequence(str(tmp_path / "missing"))


def test_sequence_rejects_corrupt_and_mixed_dimensions(tmp_path):
    make_frame(tmp_path / "frame1.png")
    (tmp_path / "frame2.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="(?i)(corrupt|decode|image)"):
        read_sequence(str(tmp_path))
    make_frame(tmp_path / "frame2.png", size=(9, 6))
    with pytest.raises(ValueError, match="(?i)(dimension|size)"):
        read_sequence(str(tmp_path))


def test_sequence_rejects_ambiguous_duplicate_stems(tmp_path):
    make_frame(tmp_path / "frame1.png")
    make_frame(tmp_path / "frame1.jpg")
    with pytest.raises(ValueError, match="(?i)(duplicate|ambiguous)"):
        read_sequence(str(tmp_path))


def test_transaction_publishes_only_on_commit(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    original = make_frame(source / "001.png")
    output_parent = tmp_path / "output"
    output_parent.mkdir()
    with OutputTransaction(str(output_parent), "result", [str(source)]) as tx:
        stage = Path(tx.staging)
        assert stage.is_dir()
        assert not (output_parent / "result").exists()
        make_frame(stage / "001.png", color=(5, 6, 7))
        published = Path(tx.commit())
        assert published == output_parent / "result"
    assert (published / "001.png").is_file()
    assert original.is_file()
    assert not stage.exists()


def test_cancellation_or_failure_cleans_only_owned_staging(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    original = make_frame(source / "001.png")
    output_parent = tmp_path / "output"
    output_parent.mkdir()
    unrelated = output_parent / "keep.txt"
    unrelated.write_text("keep", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cancelled"):
        with OutputTransaction(str(output_parent), "result", [str(source)]) as tx:
            stage = Path(tx.staging)
            make_frame(stage / "partial.png")
            raise RuntimeError("cancelled")
    assert original.is_file()
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert not stage.exists()
    assert not (output_parent / "result").exists()


@pytest.mark.parametrize("name", ["", ".", "..", "../escape", "nested/name", "nested\\name"])
def test_transaction_rejects_unsafe_destination_names(tmp_path, name):
    with pytest.raises(ValueError):
        with OutputTransaction(str(tmp_path), name, []):
            pass


def test_transaction_rejects_input_overlap(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    original = make_frame(source / "001.png")
    with pytest.raises(ValueError, match="(?i)(overlap|input|source)"):
        with OutputTransaction(str(source), "result", [str(source)]):
            pass
    assert original.is_file()
    assert list(source.iterdir()) == [original]


def test_transaction_never_replaces_existing_destination(tmp_path):
    dest = tmp_path / "result"
    dest.mkdir()
    existing = dest / "keep.txt"
    existing.write_text("preexisting", encoding="utf-8")
    with pytest.raises((ValueError, FileExistsError)):
        with OutputTransaction(str(tmp_path), "result", []):
            pass
    assert existing.read_text(encoding="utf-8") == "preexisting"


def test_destination_created_after_staging_is_not_replaced(tmp_path):
    with OutputTransaction(str(tmp_path), "result", []) as tx:
        stage = Path(tx.staging)
        make_frame(stage / "001.png")
        destination = tmp_path / "result"
        destination.mkdir()
        (destination / "other.txt").write_text("other writer", encoding="utf-8")
        with pytest.raises(FileExistsError):
            tx.commit()
    assert (destination / "other.txt").read_text(encoding="utf-8") == "other writer"
    assert not stage.exists()


def test_symlinked_input_output_parent_is_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    make_frame(source / "001.png")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(source, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(ValueError):
        with OutputTransaction(str(alias), "result", [str(source)]):
            pass


def test_dotdot_does_not_hide_link_in_raw_input_or_output_path(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    make_frame(source / "001.png")
    target = tmp_path / "other" / "nested"
    target.mkdir(parents=True)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    raw = str(alias) + os.sep + ".."
    with pytest.raises(ValueError):
        read_sequence(raw)
    with pytest.raises(ValueError):
        with OutputTransaction(raw, "result", [str(source)]):
            pass
    assert not (tmp_path / "result").exists()
    assert not (tmp_path / "other" / "result").exists()


def test_directory_scan_limits_all_entries_including_non_images(tmp_path, monkeypatch):
    import sdfx.temporal_lab_io as io

    monkeypatch.setattr(io, "MAX_DIRECTORY_ENTRIES", 3)
    make_frame(tmp_path / "001.png")
    for i in range(3):
        (tmp_path / f"ignored-{i}.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)(entries|directory|limit)"):
        read_sequence(str(tmp_path))


@pytest.mark.skipif(sys.platform != "win32", reason="Windows short-name alias")
def test_short_name_alias_cannot_bypass_overlap_detection(tmp_path):
    import ctypes

    source = tmp_path / "long source directory for short alias"
    source.mkdir()
    make_frame(source / "001.png")
    child = source / "output child"
    child.mkdir()
    buf = ctypes.create_unicode_buffer(32768)
    n = ctypes.windll.kernel32.GetShortPathNameW(str(child), buf, len(buf))
    if not n or os.path.normcase(buf.value) == os.path.normcase(str(child)):
        pytest.skip("8.3 short names unavailable on this volume")
    with pytest.raises(ValueError, match="(?i)(overlap|input|source)"):
        with OutputTransaction(buf.value, "result", [str(source)]):
            pass
    assert not (child / "result").exists()
