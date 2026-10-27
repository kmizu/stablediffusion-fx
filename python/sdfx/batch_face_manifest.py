"""Read-only, bounded batch-face image/mask discovery and identity snapshots.

This module never creates outputs or modifies inputs. Paths must be in private
locations that an untrusted concurrent actor cannot mutate; path-based checks
and digests are not a cross-process compare-and-swap guarantee. Revalidate
before processing and separately enforce a safe publication transaction.
"""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import unicodedata
import warnings

from PIL import Image, UnidentifiedImageError

from .temporal_lab_io import _absolute, _ancestor_identities, _identity, _reject_links

MAX_FILES = 10_000
MAX_ENTRIES = 20_000
MAX_DEPTH = 16
MAX_DIMENSION = 16_384
MAX_IMAGE_PIXELS = 64_000_000
MAX_TOTAL_PIXELS = 2_000_000_000
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
_CHUNK = 1024 * 1024
_REPARSE = 0x400


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    dimensions: tuple[int, int]
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class ManifestItem:
    image: FileSnapshot
    mask: FileSnapshot | None = None


@dataclass(frozen=True)
class Manifest:
    images_root: Path
    masks_root: Path | None
    items: tuple[ManifestItem, ...]


def _cancel(check):
    if check is not None:
        check()


def _is_link(info):
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & _REPARSE)


def _root(path):
    if not isinstance(path, (str, os.PathLike)) or not os.fspath(path):
        raise ValueError("input directory must be an absolute path")
    raw = Path(path)
    if not raw.is_absolute():
        raise ValueError("input directory must be an absolute path")
    root = _absolute(raw)
    _reject_links(root)
    try:
        info = root.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"input directory unavailable: {root}: {exc}") from exc
    if _is_link(info) or not stat.S_ISDIR(info.st_mode) or not info.st_ino:
        raise ValueError(f"input directory identity unavailable: {root}")
    return root


def _inside(parent, child):
    try:
        return os.path.commonpath((os.path.normcase(str(parent)), os.path.normcase(str(child)))) == os.path.normcase(str(parent))
    except ValueError:
        return False


def _key(root, path):
    return unicodedata.normalize("NFC", path.relative_to(root).with_suffix("").as_posix()).casefold()


def _signature(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _inspect(path, check, budget):
    _cancel(check)
    _reject_links(path)
    try:
        before = path.stat(follow_symlinks=False)
        if _is_link(before) or not stat.S_ISREG(before.st_mode) or not before.st_ino:
            raise ValueError(f"input file identity unavailable: {path}")
        if not 0 < before.st_size <= MAX_FILE_BYTES:
            raise ValueError(f"input file size exceeds limit: {path}")
        budget["bytes"] += before.st_size
        if budget["bytes"] > MAX_TOTAL_BYTES:
            raise ValueError("total input bytes exceed limit")
        digest = hashlib.sha256()
        observed = 0
        with path.open("rb") as source:
            while True:
                _cancel(check)
                chunk = source.read(_CHUNK)
                if not chunk:
                    break
                observed += len(chunk)
                if observed > before.st_size:
                    raise ValueError(f"input grew during inspection: {path}")
                digest.update(chunk)
        if observed != before.st_size:
            raise ValueError(f"input changed during inspection: {path}")
        _cancel(check)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                width, height = image.size
                if (width <= 0 or height <= 0 or max(width, height) > MAX_DIMENSION
                        or width * height > MAX_IMAGE_PIXELS):
                    raise ValueError(f"image size exceeds limit: {path}")
                try:
                    image.verify()
                except RuntimeError as exc:
                    raise ValueError(f"corrupt image: {path}: {exc}") from exc
            _cancel(check)
            # Some Pillow plugins require verify() immediately after open;
            # inspect EXIF and decode in a second bounded open.
            with Image.open(path) as image:
                if image.size != (width, height):
                    raise ValueError(f"image changed during inspection: {path}")
                orientation = image.getexif().get(274, 1)
                dimensions = (height, width) if orientation in (5, 6, 7, 8) else (width, height)
                try:
                    image.load()
                except RuntimeError as exc:
                    raise ValueError(f"corrupt image: {path}: {exc}") from exc
        _cancel(check)
        after = path.stat(follow_symlinks=False)
        if _is_link(after) or _signature(before) != _signature(after):
            raise ValueError(f"input changed during inspection: {path}")
        budget["pixels"] += width * height
        if budget["pixels"] > MAX_TOTAL_PIXELS:
            raise ValueError("total input pixels exceed limit")
        return FileSnapshot(path, dimensions, digest.hexdigest(), *_signature(after))
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError(f"corrupt or unreadable image: {path}: {exc}") from exc


def _scan(root, recursive, check, budget):
    paths = []
    pending = [(root, 0)]
    while pending:
        folder, depth = pending.pop()
        _cancel(check)
        _reject_links(folder)
        try:
            with os.scandir(folder) as stream:
                for entry in stream:
                    _cancel(check)
                    budget["entries"] += 1
                    if budget["entries"] > MAX_ENTRIES:
                        raise ValueError("directory entries exceed limit")
                    info = entry.stat(follow_symlinks=False)
                    item = folder / entry.name
                    if _is_link(info):
                        raise ValueError(f"symbolic link or reparse point not supported: {item}")
                    if stat.S_ISDIR(info.st_mode):
                        if recursive:
                            if depth >= MAX_DEPTH:
                                raise ValueError("directory nesting exceeds limit")
                            pending.append((item, depth + 1))
                    elif Path(entry.name).suffix.lower() in _SUFFIXES:
                        if not stat.S_ISREG(info.st_mode):
                            raise ValueError(f"input is not a regular file: {item}")
                        paths.append(item)
                        budget["files"] += 1
                        if budget["files"] > MAX_FILES:
                            raise ValueError("input file count exceeds limit")
        except OSError as exc:
            raise ValueError(f"cannot scan input directory: {folder}: {exc}") from exc
    if not paths:
        raise ValueError(f"no image files found: {root}")
    paths.sort(key=lambda path: (_key(root, path), path.relative_to(root).as_posix()))
    found = {}
    for path in paths:
        key = _key(root, path)
        if key in found:
            raise ValueError(f"ambiguous duplicate image stem: {path}")
        found[key] = path
    return found


def build_manifest(images_root, masks_root=None, *, recursive=False, check_cancel=None):
    """Snapshot supported images, optionally matching exactly one mask by relative stem."""
    if type(recursive) is not bool:
        raise ValueError("recursive must be boolean")
    _cancel(check_cancel)
    images = _root(images_root)
    masks = _root(masks_root) if masks_root is not None else None
    if masks is not None:
        # Text prefixes alone miss aliases such as Windows 8.3 directory names.
        # Compare each root's device/inode to every ancestor of the other.
        if (_inside(images, masks) or _inside(masks, images)
                or _identity(images) in _ancestor_identities(masks)
                or _identity(masks) in _ancestor_identities(images)):
            raise ValueError("image and mask input directories overlap or alias")
    budget = {"entries": 0, "files": 0, "bytes": 0, "pixels": 0}
    image_paths = _scan(images, recursive, check_cancel, budget)
    mask_paths = _scan(masks, recursive, check_cancel, budget) if masks is not None else None
    if mask_paths is not None and image_paths.keys() != mask_paths.keys():
        raise ValueError("image and mask relative stems do not match")
    snapshots = []
    identities = set()
    for key, path in image_paths.items():
        image = _inspect(path, check_cancel, budget)
        mask = _inspect(mask_paths[key], check_cancel, budget) if mask_paths is not None else None
        if mask is not None and image.dimensions != mask.dimensions:
            raise ValueError(f"image and mask dimensions do not match: {path}")
        for item in (image, mask):
            if item is not None:
                identity = (item.device, item.inode)
                if identity in identities:
                    raise ValueError(f"input files have the same identity or alias: {item.path}")
                identities.add(identity)
        snapshots.append(ManifestItem(image, mask))
    _cancel(check_cancel)
    return Manifest(images, masks, tuple(snapshots))


def verify_manifest(manifest, *, check_cancel=None):
    """Recheck identity and full content digest; raises if any input changed."""
    if not isinstance(manifest, Manifest):
        raise ValueError("invalid batch manifest")
    for item in manifest.items:
        for snap in (item.image, item.mask):
            if snap is None:
                continue
            _cancel(check_cancel)
            _reject_links(snap.path)
            try:
                before = snap.path.stat(follow_symlinks=False)
                expected = (snap.device, snap.inode, snap.size, snap.mtime_ns)
                if _is_link(before) or not stat.S_ISREG(before.st_mode) or _signature(before) != expected:
                    raise ValueError(f"input identity changed: {snap.path}")
                digest = hashlib.sha256()
                observed = 0
                with snap.path.open("rb") as source:
                    while True:
                        _cancel(check_cancel)
                        chunk = source.read(_CHUNK)
                        if not chunk:
                            break
                        observed += len(chunk)
                        if observed > snap.size:
                            raise ValueError(f"input grew during revalidation: {snap.path}")
                        digest.update(chunk)
                after = snap.path.stat(follow_symlinks=False)
                if (_is_link(after) or _signature(after) != expected or observed != snap.size
                        or digest.hexdigest() != snap.sha256):
                    raise ValueError(f"input changed or hash differs: {snap.path}")
            except OSError as exc:
                raise ValueError(f"input changed or became unavailable: {snap.path}: {exc}") from exc
    _cancel(check_cancel)
