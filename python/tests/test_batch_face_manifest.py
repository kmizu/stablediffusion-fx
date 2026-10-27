"""Read-only batch image/mask manifest safety contract; no model inference or writes."""
import hashlib
from pathlib import Path

import pytest
from PIL import Image

from sdfx.batch_face_manifest import build_manifest, verify_manifest


def img(path, size=(8, 6), color="white"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_detection_manifest_is_sorted_read_only_and_snapshots_identity(tmp_path):
    source = tmp_path / "images"
    a = img(source / "b.png")
    b = img(source / "a.jpg")
    before = {p: sha(p) for p in (a, b)}
    result = build_manifest(source, check_cancel=lambda: None)
    assert [item.image.path.name for item in result.items] == ["a.jpg", "b.png"]
    assert all(item.mask is None and item.image.dimensions == (8, 6) for item in result.items)
    assert all(len(item.image.sha256) == 64 for item in result.items)
    assert verify_manifest(result) is None
    assert {p: sha(p) for p in (a, b)} == before
    assert set(tmp_path.iterdir()) == {source}


def test_explicit_masks_pair_by_unique_relative_stem_not_directory_order(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    img(images / "sub/a.jpg")
    img(images / "sub/b.png")
    img(masks / "sub/b.jpg", color="black")
    img(masks / "sub/a.png", color="black")
    result = build_manifest(images, masks, recursive=True)
    assert [(x.image.path.name, x.mask.path.name) for x in result.items] == [("a.jpg", "a.png"), ("b.png", "b.jpg")]
    assert verify_manifest(result) is None


@pytest.mark.parametrize("scenario", ["missing", "extra", "duplicate_image", "duplicate_mask", "dimensions"])
def test_ambiguous_or_invalid_masks_rejected_without_changes(tmp_path, scenario):
    images, masks = tmp_path / "images", tmp_path / "masks"
    first = img(images / "face.png")
    second = img(masks / "face.png", size=(9, 6) if scenario == "dimensions" else (8, 6))
    if scenario == "missing": second.unlink()
    if scenario == "extra": img(masks / "extra.png")
    if scenario == "duplicate_image": img(images / "face.jpg")
    if scenario == "duplicate_mask": img(masks / "face.jpg")
    before = {p: sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        build_manifest(images, masks)
    assert {p: sha(p) for p in tmp_path.rglob("*") if p.is_file()} == before
    assert first.exists()


def test_exif_oriented_image_and_mask_must_have_matching_display_dimensions(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (8, 6), "white").save(images / "face.jpg", exif=exif)
    img(masks / "face.png", size=(6, 8), color="black")
    result = build_manifest(images, masks)
    assert result.items[0].image.dimensions == result.items[0].mask.dimensions == (6, 8)


def test_parent_traversal_and_overlapping_roots_rejected(tmp_path):
    images = tmp_path / "images"
    img(images / "face.png")
    masks = images / "masks"
    img(masks / "face.png")
    with pytest.raises(ValueError):
        build_manifest(images, masks)
    with pytest.raises(ValueError):
        build_manifest(images / ".." / "images")


def test_revalidate_detects_image_or_mask_change(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    image = img(images / "face.png")
    mask = img(masks / "face.png", color="black")
    result = build_manifest(images, masks)
    img(mask, color="red")
    with pytest.raises(ValueError, match="(?i)(changed|identity|hash)"):
        verify_manifest(result)
    assert image.exists() and mask.exists()


def test_cancellation_leaves_input_untouched(tmp_path):
    source = tmp_path / "images"
    path = img(source / "face.png")
    before = sha(path)
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        if calls >= 3: raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        build_manifest(source, check_cancel=cancel)
    assert sha(path) == before
    assert set(tmp_path.iterdir()) == {source}


def test_nested_symlink_is_rejected(tmp_path):
    source, outside = tmp_path / "images", tmp_path / "outside"
    source.mkdir()
    img(outside / "face.png")
    try:
        (source / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="(?i)(link|reparse)"):
        build_manifest(source, recursive=True)
    assert (outside / "face.png").exists()


def test_corrupt_image_and_entry_limit_fail_without_outputs(tmp_path, monkeypatch):
    import sdfx.batch_face_manifest as module
    source = tmp_path / "images"
    source.mkdir()
    (source / "bad.png").write_bytes(b"not a png")
    with pytest.raises(ValueError, match="(?i)(image|decode|corrupt)"):
        build_manifest(source)
    (source / "bad.png").unlink()
    img(source / "a.png")
    img(source / "b.png")
    monkeypatch.setattr(module, "MAX_FILES", 1)
    with pytest.raises(ValueError, match="(?i)(limit|many|exceed)"):
        build_manifest(source)
    assert len(list(source.iterdir())) == 2


def test_revalidate_detects_byte_change_even_when_size_and_mtime_restored(tmp_path):
    import os
    source = tmp_path / "images"
    file = img(source / "face.png")
    result = build_manifest(source)
    old = file.stat()
    data = bytearray(file.read_bytes())
    data[-1] ^= 1
    file.write_bytes(data)
    os.utime(file, ns=(old.st_atime_ns, old.st_mtime_ns))
    with pytest.raises(ValueError, match="(?i)(changed|hash)"):
        verify_manifest(result)


def test_same_inode_cannot_be_used_as_image_and_mask(tmp_path):
    import os
    images, masks = tmp_path / "images", tmp_path / "masks"
    image = img(images / "face.png")
    masks.mkdir()
    try:
        os.link(image, masks / "face.png")
    except OSError:
        pytest.skip("hard links unavailable")
    with pytest.raises(ValueError, match="(?i)(same|alias|identity)"):
        build_manifest(images, masks)
    assert image.exists() and (masks / "face.png").exists()


def test_overlapping_root_ancestors_rejected_by_identity_even_if_paths_look_distinct(tmp_path, monkeypatch):
    import sdfx.batch_face_manifest as module
    images = tmp_path / "images"
    nested = images / "mask_dir"
    img(images / "face.png")
    img(nested / "face.png")
    # Simulate path aliases bypassing string-based overlap detection without
    # making assumptions about a platform's short-name support.
    monkeypatch.setattr(module, "_inside", lambda parent, child: False)
    with pytest.raises(ValueError, match="(?i)(overlap|alias|same)"):
        build_manifest(images, nested)
    with pytest.raises(ValueError, match="(?i)(overlap|alias|same)"):
        build_manifest(nested, images)
    assert (images / "face.png").exists() and (nested / "face.png").exists()


def test_windows_short_name_root_alias_cannot_hide_nested_overlap(tmp_path):
    import os
    if os.name != "nt":
        pytest.skip("Windows short names only")
    import ctypes
    from ctypes import wintypes
    images = tmp_path / "long image directory for short name"
    nested = images / "nested mask directory"
    img(images / "face.png")
    img(nested / "face.png")
    call = ctypes.windll.kernel32.GetShortPathNameW
    call.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD)
    call.restype = wintypes.DWORD
    buffer = ctypes.create_unicode_buffer(32768)
    n = call(str(images), buffer, len(buffer))
    if not n or n >= len(buffer):
        pytest.skip("short name unavailable")
    alias = Path(buffer.value)
    if os.path.normcase(str(alias)) == os.path.normcase(str(images)):
        pytest.skip("filesystem did not produce a short alias")
    with pytest.raises(ValueError, match="(?i)(overlap|alias|same)"):
        build_manifest(alias, nested)
    assert (images / "face.png").exists() and (nested / "face.png").exists()
