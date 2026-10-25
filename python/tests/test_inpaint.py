import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402

from sdfx import inpaint  # noqa: E402
from sdfx.images import next_output_path, png_size, resize_image, write_png_rgb  # noqa: E402


def _img(w, h, color=(200, 50, 50)):
    return Image.new("RGB", (w, h), color)


def _mask(w, h, box):
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rectangle(box, fill=255)
    return m


def test_crop_region_and_expand():
    m = _mask(400, 300, (100, 100, 149, 129))
    assert inpaint.get_crop_region(m, 0) == (100, 100, 150, 130)
    assert inpaint.get_crop_region(m, 32) == (68, 68, 182, 162)
    assert inpaint.get_crop_region(Image.new("L", (10, 10), 0), 4) is None
    x1, y1, x2, y2 = inpaint.expand_crop_region((68, 68, 182, 162), 512, 512, 400, 300)
    assert (x2 - x1) == (y2 - y1)


def test_resize_modes():
    im = _img(300, 100)
    for mode in (0, 1, 2):
        assert resize_image(mode, im, 128, 128).size == (128, 128)


def test_prepare_and_finalize_keeps_unmasked_original():
    init = _img(256, 256, (10, 20, 30))
    m = _mask(256, 256, (64, 64, 191, 191))
    job = inpaint.prepare(init, m, 256, 256, mask_blur=0, inpainting_fill=1)
    gen = _img(256, 256, (250, 250, 250))
    out = job.finalize(gen)
    assert out.getpixel((5, 5)) == (10, 20, 30)
    assert out.getpixel((128, 128)) == (250, 250, 250)


def test_full_res_paste_back():
    init = _img(400, 300, (10, 20, 30))
    m = _mask(400, 300, (100, 100, 149, 129))
    job = inpaint.prepare(init, m, 128, 128, mask_blur=0, full_res=True, full_res_padding=16)
    assert job.image.size == (128, 128) and job.paste_box is not None
    out = job.finalize(_img(128, 128, (255, 255, 255)))
    assert out.size == (400, 300)
    assert out.getpixel((0, 0)) == (10, 20, 30)
    assert out.getpixel((120, 115)) == (255, 255, 255)


@pytest.mark.parametrize("fill", [0, 2, 3])
def test_fill_modes(fill):
    init = _img(64, 64, (10, 20, 30))
    m = _mask(64, 64, (16, 16, 47, 47))
    job = inpaint.prepare(init, m, 64, 64, mask_blur=2, inpainting_fill=fill, invert=False, seed=1)
    assert job.image.size == (64, 64)
    assert job.image.getpixel((0, 0)) == (10, 20, 30)


def test_invert():
    init = _img(64, 64)
    m = _mask(64, 64, (0, 0, 31, 63))
    job = inpaint.prepare(init, m, 64, 64, mask_blur=0, invert=True)
    assert job.mask.getpixel((5, 5)) == 0 and job.mask.getpixel((60, 5)) == 255


def test_output_naming_and_png_writer(tmp_path):
    a = next_output_path(str(tmp_path), "123")
    b = next_output_path(str(tmp_path), "123")
    assert a != b and a.endswith("00000-123.png") and b.endswith("00001-123.png")
    p = str(tmp_path / "w.png")
    write_png_rgb(p, 3, 2, [b"\x00\x01\x02" * 3, b"\xff\xfe\xfd" * 3])
    assert png_size(p) == (3, 2)
    with Image.open(p) as im:
        assert im.size == (3, 2) and im.getpixel((0, 1)) == (255, 254, 253)
