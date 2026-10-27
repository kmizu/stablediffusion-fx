import hashlib
from pathlib import Path
from PIL import Image
import pytest

from sdfx.batch_png_metadata import parameters, write_parameters, MAX_METADATA_BYTES


def test_effective_per_image_settings_and_pinned_provenance():
    text = parameters({"params": {"prompt": "original", "seed": -1, "batch_size": 2,
        "n_iter": 3, "eta": .2, "img2img": {"resize_mode": 2, "denoising_strength": .4}}},
        {"positive": "猫, effective", "negative": "bad", "input": "/source.png"},
        {"seed": 123, "subseed": 456, "width": 64, "height": 80},
        {"model_name": "checkpoint", "model_hash": "abcdef012345", "scheduler": "Karras"},
        [{"backend": "wd_ext", "model_sha256": "a" * 64}])
    assert text.startswith("猫, effective\nNegative prompt: bad\n")
    for field in ("Seed: 123", "Size: 64x80", "Batch size: 2", "Batch count: 3",
                  "Eta: 0.2", "Resize mode: 2", "Denoising strength: 0.4", "Schedule type: Karras"):
        assert field in text
    assert "a" * 64 in text and "original" not in text


def test_png_roundtrip_preserves_pixels_and_source_chunks(tmp_path):
    path = tmp_path / "output.png"
    Image.new("RGBA", (8, 9), (12, 34, 56, 78)).save(path)
    original = path.read_bytes()
    write_parameters(path, "猫\nSteps: 2, Seed: 123, Size: 8x9", lambda: None)
    with Image.open(path) as image:
        assert image.info["parameters"].startswith("猫")
        assert image.getpixel((0, 0)) == (12, 34, 56, 78)
    # Every original chunk survives exactly, including compressed IDAT bytes.
    cursor = 8
    while cursor < len(original):
        length = int.from_bytes(original[cursor:cursor + 4], 'big') + 12
        assert original[cursor:cursor + length] in path.read_bytes()
        cursor += length
    write_parameters(path, "replacement", lambda: None)
    with Image.open(path) as image:
        assert image.info["parameters"] == "replacement"


def test_metadata_limits_and_cancel_leave_png_unchanged(tmp_path):
    path = tmp_path / "output.png"; Image.new("RGB", (2, 2)).save(path)
    original = path.read_bytes()
    with pytest.raises(ValueError):
        write_parameters(path, "x" * (MAX_METADATA_BYTES + 1), lambda: None)
    def cancel(): raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        write_parameters(path, "metadata", cancel)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
