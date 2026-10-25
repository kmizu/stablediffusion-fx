"""Image helpers (PIL), output naming, preview encoding, and a pure-python PNG writer fallback."""
from __future__ import annotations

import base64
import io
import os
import re
import struct
import zlib
from pathlib import Path
from typing import Optional

try:
    from PIL import Image, ImageFilter, ImageOps  # type: ignore
    HAVE_PIL = True
except Exception:  # pragma: no cover - exercised only without pillow
    Image = ImageFilter = ImageOps = None  # type: ignore
    HAVE_PIL = False


def require_pil():
    if not HAVE_PIL:
        raise RuntimeError("Pillow is required for this operation")


# --------------------------------------------------------------------------- output naming

_num_re = re.compile(r"^(\d+)-")


def next_output_path(output_dir: str, suffix: str = "", ext: str = ".png") -> str:
    """A1111-style '00012-<suffix>.png' in output_dir, created atomically (never overwrites)."""
    d = Path(output_dir or ".")
    d.mkdir(parents=True, exist_ok=True)
    mx = -1
    try:
        for fn in os.listdir(d):
            m = _num_re.match(fn)
            if m:
                mx = max(mx, int(m.group(1)))
    except OSError:
        pass
    n = mx + 1
    safe = re.sub(r"[^\w.\-]+", "_", str(suffix))[:80] if suffix != "" else "img"
    while True:
        p = d / f"{n:05d}-{safe}{ext}"
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return str(p)
        except FileExistsError:
            n += 1


def save_png(img, path: str) -> str:
    require_pil()
    try:
        img.save(path, format="PNG", compress_level=4)
    except BaseException:
        try:  # do not leave the 0-byte placeholder from next_output_path behind
            os.remove(path)
        except OSError:
            pass
        raise
    return path


def write_png_rgb(path: str, width: int, height: int, rows: list[bytes]) -> str:
    """Minimal RGB8 PNG writer (used when Pillow is unavailable)."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + r for r in rows)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) \
        + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)
    return path


def png_size(path: str) -> tuple[int, int]:
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    return struct.unpack(">II", head[16:24])


# --------------------------------------------------------------------------- loading / resizing

def load_rgb(path: str):
    require_pil()
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"image not found: {path}")
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if im.mode == "RGBA" or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        im = bg
    return im.convert("RGB")


def load_mask(path: str):
    """Mask as 'L'; white = area to repaint. Alpha channel masks use alpha."""
    require_pil()
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"mask not found: {path}")
    im = Image.open(path)
    if im.mode in ("RGBA", "LA"):
        alpha = im.getchannel("A")
        if alpha.getextrema() != (255, 255):
            # transparent-paint masks: painted strokes are opaque, background transparent
            return alpha
    return im.convert("L")


def resize_image(resize_mode: int, im, width: int, height: int, resample=None):
    """A1111 images.resize_image: 0 just resize, 1 crop and resize, 2 resize and fill."""
    require_pil()
    resample = Image.LANCZOS if resample is None else resample

    def rs(img, w, h):
        return img.resize((max(1, int(w)), max(1, int(h))), resample=resample)

    if resize_mode == 0 or (im.width == width and im.height == height):
        return rs(im, width, height)
    ratio = width / height
    src_ratio = im.width / im.height
    if resize_mode == 1:
        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width
        resized = rs(im, src_w, src_h)
        res = Image.new(im.mode, (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
        return res
    # resize and fill
    src_w = width if ratio < src_ratio else im.width * height // im.height
    src_h = height if ratio >= src_ratio else im.height * width // im.width
    resized = rs(im, src_w, src_h)
    res = Image.new(im.mode, (width, height))
    res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    if ratio < src_ratio:
        fill_height = height // 2 - src_h // 2
        if fill_height > 0:
            res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
            res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)),
                      box=(0, fill_height + src_h))
    elif ratio > src_ratio:
        fill_width = width // 2 - src_w // 2
        if fill_width > 0:
            res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
            res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)),
                      box=(fill_width + src_w, 0))
    return res


# --------------------------------------------------------------------------- previews

def to_jpeg_b64(img, max_side: int = 256, quality: int = 70) -> str:
    require_pil()
    im = img.convert("RGB")
    s = max(im.width, im.height)
    if s != max_side:
        f = max_side / s
        im = im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def rgb_array_to_image(arr):
    """uint8 HxWx3 numpy array -> PIL image."""
    require_pil()
    return Image.fromarray(arr)


def copy_or_none(path: Optional[str]) -> Optional[str]:
    return path if path and os.path.isfile(path) else None
