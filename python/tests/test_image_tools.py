"""Non-diffusion image tools (sdfx.image_tools)."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
Image = pytest.importorskip("PIL.Image")

from sdfx import image_tools as it  # noqa: E402


def _img(seed=0, w=64, h=48):
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 255, w * h * 3).reshape(h, w, 3)
    return Image.fromarray(np.clip(base * 0.5 + rng.integers(0, 120, (h, w, 3)), 0, 255).astype(np.uint8))


def test_pixel_art_palette_and_blocks():
    out, info = it.pixel_art(_img(), pixel_size=8, colors=5, upscale=True)
    a = np.asarray(out)
    assert out.size == (64, 48) and info["palette"] == 5
    assert len({tuple(c) for c in a.reshape(-1, 3)}) <= 5
    assert (a[0:8, 0:8] == a[0, 0]).all()                            # one colour per 8x8 cell
    small, _ = it.pixel_art(_img(), pixel_size=8, colors=5, upscale=False)
    assert small.size == (8, 6)
    d, _ = it.pixel_art(_img(), pixel_size=4, colors=4, dither=1.0, outline=True)
    assert d.size == (64, 48)
    assert it.bayer(4).shape == (4, 4) and abs(float(it.bayer(4).mean())) < 1e-6


def test_color_enhance_raises_chroma_keeps_lightness():
    dull = Image.fromarray(np.full((16, 16, 3), (140, 120, 110), np.uint8))
    dull.putpixel((0, 0), (150, 110, 100))
    out, info = it.color_enhance(dull, 1.0)
    lab0 = cv2.cvtColor(np.asarray(dull).astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    lab1 = cv2.cvtColor(np.asarray(out).astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    c0, c1 = np.hypot(lab0[..., 1], lab0[..., 2]).mean(), np.hypot(lab1[..., 1], lab1[..., 2]).mean()
    assert c1 > c0 * 1.5 and info["chroma_gain"] > 1
    assert abs(float(lab1[..., 0].mean() - lab0[..., 0].mean())) < 3
    gray = Image.fromarray(np.full((8, 8, 3), 128, np.uint8))
    assert np.array_equal(np.asarray(it.color_enhance(gray)[0]), np.asarray(gray))


def test_cas_keeps_flat_areas_and_sharpens_edges():
    flat = np.full((8, 8, 3), 0.5, np.float32)
    assert np.allclose(it.cas(flat, 1.0), flat)
    edge = np.full((8, 8, 3), 0.3, np.float32)                     # mid tones: CAS backs off near black / white
    edge[:, 4:] = 0.7
    s = it.cas(edge, 1.0)
    assert s[:, 3].mean() < 0.3 and s[:, 4].mean() > 0.7           # contrast across the edge grows
    out, _ = it.cas_upscale(_img(), 2.0, 0.5)
    assert out.size == (128, 96)


def test_run_dispatch_and_errors():
    assert it.run("pixel_art", _img(), {"pixel_size": 2, "colors": 3})[0].size == (64, 48)
    with pytest.raises(ValueError):
        it.run("nope", _img(), {})
    with pytest.raises(ValueError):
        it.remove_background(_img(), model="bogus")


def test_pixel_art_accepts_full_generation_seed_range():
    a, _ = it.pixel_art(_img(), seed=4294967295)
    b, _ = it.pixel_art(_img(), seed=4294967295)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_pixel_art_scale_back_preserves_nondivisible_dimensions():
    out, _ = it.pixel_art(_img(w=65, h=49), pixel_size=8, upscale=True)
    assert out.size == (65, 49)


def test_image_tool_closes_source_file(tmp_path, monkeypatch):
    from sdfx.base import BaseEngine
    source = tmp_path / "source.png"
    _img().save(source)
    opened = []
    original_open = Image.open

    def tracked_open(*args, **kwargs):
        im = original_open(*args, **kwargs)
        opened.append(im)
        return im

    monkeypatch.setattr(Image, "open", tracked_open)
    BaseEngine().image_tool({"tool": "pixel_art", "image_path": str(source), "output_dir": str(tmp_path)}, None)
    assert opened and all(im.fp is None for im in opened)
