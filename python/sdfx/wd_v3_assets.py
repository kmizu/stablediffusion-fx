"""Offline WD-v3 asset binding and preprocessing; no model execution.

Pins must originate in a separately reviewed internal allowlist, never from a
worker request. Matching a digest does not prove publisher provenance, model
operator safety or license rights. Source and staging ancestors must not be
writable by a hostile concurrent actor, as with other path-based transactions.
The current batch registry flattens inputs to RGB and loses alpha before a WD
provider could use this preprocessing. No ONNX session is bound or executed.
"""
from contextlib import contextmanager
import csv
from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import re
import stat
import tempfile

from .temporal_lab_io import _absolute, _reject_links

MAX_MODEL_BYTES = 512 * 1024 * 1024
MAX_TAG_BYTES = 1024 * 1024
MAX_TAG_ROWS = 20_000
MAX_SQUARE_PIXELS = 8_000_000
MAX_TARGET = 2048
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_CHUNK = 1024 * 1024


class WdCleanupFailure(RuntimeError):
    """Staging cleanup failed and the enclosing job must stop."""
    _sdfx_cleanup_failed = True


@dataclass(frozen=True)
class AssetPin:
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True)
class BoundAssets:
    model: Path
    tags: Path


@dataclass(frozen=True)
class Tag:
    tag_id: int
    name: str
    category: int
    count: int


def _validate_pin(pin, limit):
    if (not isinstance(pin, AssetPin) or type(pin.size) is not int or not 0 < pin.size <= limit
            or not isinstance(pin.sha256, str) or not _HASH.fullmatch(pin.sha256)):
        raise ValueError("invalid reviewed asset pin")
    path = _absolute(pin.path)
    _reject_links(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not info.st_ino or info.st_size != pin.size:
        raise ValueError("asset is not the approved regular file")
    return path, info


def _signature(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _copy(pin, checked, target, ctx):
    path, before = checked
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as source, target.open("xb") as dest:
        if _signature(os.fstat(source.fileno())) != _signature(before):
            raise ValueError("asset changed before copying")
        count = 0
        digest = hashlib.sha256()
        while True:
            ctx.check_interrupt()
            chunk = source.read(min(_CHUNK, pin.size - count + 1))
            if not chunk:
                break
            count += len(chunk)
            if count > pin.size:
                raise ValueError("asset exceeds approved size")
            digest.update(chunk)
            dest.write(chunk)
        if count != pin.size or digest.hexdigest() != pin.sha256 or _signature(os.fstat(source.fileno())) != _signature(before):
            raise ValueError("asset does not match approved identity")
    _reject_links(path)
    if _signature(path.lstat()) != _signature(before):
        raise ValueError("asset changed during copying")
    ctx.check_interrupt()


@contextmanager
def owned_assets(model, tags, temp_parent, ctx):
    if not callable(getattr(ctx, "check_interrupt", None)):
        raise ValueError("cancellation context required")
    ctx.check_interrupt()
    model_checked = _validate_pin(model, MAX_MODEL_BYTES)
    tags_checked = _validate_pin(tags, MAX_TAG_BYTES)
    parent = _absolute(temp_parent)
    _reject_links(parent)
    if not parent.is_dir():
        raise ValueError("temporary parent is unavailable")
    work = Path(tempfile.mkdtemp(prefix="sdfx-wd-v3-", dir=parent))
    paths = BoundAssets(work / "model.onnx", work / "tags.csv")
    primary = None
    try:
        _copy(model, model_checked, paths.model, ctx)
        _copy(tags, tags_checked, paths.tags, ctx)
        yield paths
    except BaseException as exc:
        primary = exc
        raise
    finally:
        failures = []
        for path in (paths.model, paths.tags):
            try:
                if os.path.lexists(path):
                    _reject_links(path)
                    if not stat.S_ISREG(path.lstat().st_mode):
                        raise OSError("unexpected staging entry")
                    path.unlink()
            except Exception as cleanup:
                failures.append(f"{path}: {cleanup}")
        try:
            work.rmdir()
        except Exception as cleanup:
            failures.append(f"{work}: {cleanup}")
        if failures:
            message = f"WD staging cleanup failed at {work}: {'; '.join(failures)}"
            if primary is None:
                raise WdCleanupFailure(message)
            if isinstance(getattr(primary, "message", None), str):
                primary.message = f"{primary.message}; {message}"
            primary.args = (f"{primary}; {message}",)
            primary._sdfx_cleanup_failed = True


def parse_tags(data):
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_TAG_BYTES:
        raise ValueError("tag CSV exceeds byte limit")
    try:
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""), strict=True)
        if reader.fieldnames != ["tag_id", "name", "category", "count"]:
            raise ValueError("invalid tag CSV schema")
        rows = []
        ids = set()
        for row in reader:
            if len(rows) >= MAX_TAG_ROWS or None in row or any(value is None for value in row.values()):
                raise ValueError("invalid or excessive tag rows")
            name = row["name"]
            if not name or len(name) > 512 or any(ord(char) < 32 or ord(char) == 127 for char in name):
                raise ValueError("invalid tag name")
            fields = (row["tag_id"], row["category"], row["count"])
            if any(not value.isascii() or not value.isdecimal() or len(value) > 12 for value in fields):
                raise ValueError("invalid tag numeric field")
            ident, category, count = map(int, fields)
            if ident in ids or category not in (0, 4, 9):
                raise ValueError("invalid tag category or duplicate id")
            ids.add(ident)
            rows.append(Tag(ident, name, category, count))
        if not rows:
            raise ValueError("empty tag CSV")
        return tuple(rows)
    except (UnicodeError, csv.Error) as exc:
        raise ValueError(f"invalid tag CSV: {exc}") from exc


def prepare_image(image, target_size):
    """Return white-composited, square padded float32 NHWC BGR input."""
    from PIL import Image, ImageOps
    import numpy as np
    if not isinstance(image, Image.Image) or type(target_size) is not int or not 1 <= target_size <= MAX_TARGET:
        raise ValueError("invalid WD image or target size")
    width, height = image.size
    side = max(width, height)
    if width <= 0 or height <= 0 or side * side > MAX_SQUARE_PIXELS:
        raise ValueError("WD image dimensions exceed limit")
    with ImageOps.exif_transpose(image) as oriented:
        with oriented.convert("RGBA") as rgba:
            with Image.new("RGBA", rgba.size, (255, 255, 255, 255)) as background:
                background.alpha_composite(rgba)
                with background.convert("RGB") as rgb:
                    with Image.new("RGB", (side, side), "white") as square:
                        square.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
                        with square.resize((target_size, target_size), Image.Resampling.BICUBIC) as resized:
                            return np.asarray(resized, dtype=np.float32)[None, :, :, ::-1].copy()
