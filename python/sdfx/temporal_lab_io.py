"""Bounded frame discovery and non-overwriting output staging.

These checks reject existing links/reparse points and detect common changes in
directory identity, but they do not defend against a hostile local process
swapping a parent between path-based OS calls. The caller must use directories
that other users/processes cannot modify during a job.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import os
from pathlib import Path
from typing import Callable
import re
import shutil
import stat
import sys
import tempfile
import warnings

from PIL import Image, UnidentifiedImageError

MAX_FRAMES = 10000
MAX_DIRECTORY_ENTRIES = 20000
MAX_DIMENSION = 16384
MAX_FRAME_PIXELS = 64_000_000
MAX_TOTAL_PIXELS = 2_000_000_000
FRAME_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})
_REPARSE_POINT = 0x400


def _absolute(path: str | os.PathLike) -> Path:
    # abspath would collapse `link/..` before _reject_links sees the link and
    # silently change filesystem traversal. Require an unambiguous path.
    if ".." in Path(path).parts:
        raise ValueError("parent traversal (..) is not supported in a path")
    return Path(os.path.abspath(os.fspath(path)))


def _reject_links(path: Path) -> None:
    """Reject all existing path components which are links or Windows reparse points."""
    for component in reversed((path, *path.parents)):
        try:
            info = os.lstat(component)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & _REPARSE_POINT:
            raise ValueError(f"symbolic link or reparse point is not supported: {component}")


def _identity(path: Path) -> tuple[int, int]:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"not a directory: {path}")
    if not info.st_ino:
        raise ValueError(f"cannot verify directory identity: {path}")
    return info.st_dev, info.st_ino


def _ancestor_identities(path: Path) -> set[tuple[int, int]]:
    # Compare actual filesystem identities as well as spelling: e.g. Windows
    # 8.3 names can address the same directory without a shared text prefix.
    return {_identity(component) for component in (path, *path.parents)}


def _normalized(path: Path) -> str:
    return os.path.normcase(os.path.normpath(os.fspath(path)))


def _contains(parent: Path, child: Path) -> bool:
    try:
        return os.path.commonpath((_normalized(parent), _normalized(child))) == _normalized(parent)
    except ValueError:  # Different Windows volumes.
        return False


def _natural_key(name: str):
    # Keep the entire key comparable even when digit and letter runs differ.
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                 for part in re.split(r"(\d+)", name))


@dataclass(frozen=True)
class FrameSequence:
    paths: tuple[Path, ...]
    dimensions: tuple[int, int]

    @property
    def count(self) -> int:
        return len(self.paths)


def read_sequence(path: str, check_interrupt: Callable[[], None] | None = None) -> FrameSequence:
    """Discover and inspect frames; check cancellation between filesystem operations.

    Individual codec calls are not interruptible; cancel is observed before the
    next entry or image inspection.
    """
    check = check_interrupt if check_interrupt is not None else lambda: None
    check()
    root = _absolute(path)
    _reject_links(root)
    if not root.is_dir():
        raise ValueError(f"frame directory does not exist: {root}")
    files = []
    with os.scandir(root) as entries:
        for entry_count, entry in enumerate(entries, 1):
            check()
            if entry_count > MAX_DIRECTORY_ENTRIES:
                raise ValueError(f"directory entries exceed {MAX_DIRECTORY_ENTRIES}")
            if Path(entry.name).suffix.lower() not in FRAME_SUFFIXES:
                continue
            item = root / entry.name
            _reject_links(item)
            if not entry.is_file(follow_symlinks=False):
                raise ValueError(f"frame is not a regular file: {item}")
            files.append(item)
            if len(files) > MAX_FRAMES:
                raise ValueError(f"frame count exceeds {MAX_FRAMES}")
    if not files:
        raise ValueError("no image frames found")
    stems = [p.stem.casefold() for p in files]
    if len(set(stems)) != len(stems):
        raise ValueError("ambiguous duplicate frame names")
    files.sort(key=lambda p: (_natural_key(p.name), p.name.casefold(), p.name))
    dimensions = None
    for item in files:
        check()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(item) as im:
                    size = im.size
                    if (size[0] <= 0 or size[1] <= 0 or max(size) > MAX_DIMENSION
                            or size[0] * size[1] > MAX_FRAME_PIXELS):
                        raise ValueError(f"frame size exceeds limits: {item}")
                    im.verify()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
                Image.DecompressionBombWarning) as exc:
            raise ValueError(f"corrupt or unreadable image: {item}") from exc
        if dimensions is not None and size != dimensions:
            raise ValueError(f"frame dimensions do not match: {item}")
        dimensions = size
    assert dimensions is not None
    if dimensions[0] * dimensions[1] * len(files) > MAX_TOTAL_PIXELS:
        raise ValueError("total frame pixels exceed the limit")
    return FrameSequence(tuple(files), dimensions)


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Publish atomically without replacement; fail closed if unavailable."""
    if sys.platform == "win32":
        # Windows os.rename fails when the destination exists.
        os.rename(source, destination)
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError(errno.ENOTSUP, "atomic no-replace rename is unavailable")
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), os.fspath(destination))
        return
    raise OSError(errno.ENOTSUP, "atomic no-replace rename is unsupported on this platform")


class OutputTransaction:
    """One new output directory; no writes to inputs or existing destinations."""

    def __init__(self, parent: str, destination_name: str, input_paths: list[str]):
        if (not isinstance(destination_name, str) or destination_name in ("", ".", "..")
                or "/" in destination_name or "\\" in destination_name or "\x00" in destination_name
                or ":" in destination_name or destination_name.rstrip(" .") != destination_name):
            raise ValueError("destination must be a single safe directory name")
        self.parent = _absolute(parent)
        self.destination = self.parent / destination_name
        self.inputs = tuple(_absolute(p) for p in input_paths)
        self.staging: Path | None = None
        self._parent_identity: tuple[int, int] | None = None
        self._stage_identity: tuple[int, int] | None = None
        self._committed = False

    def __enter__(self) -> OutputTransaction:
        _reject_links(self.parent)
        self._parent_identity = _identity(self.parent)
        parent_ancestors = _ancestor_identities(self.parent)
        for source in self.inputs:
            _reject_links(source)
            if not source.exists():
                raise ValueError(f"input does not exist: {source}")
            if (_contains(source, self.parent) or _contains(self.destination, source)
                    or (source.is_dir() and _identity(source) in parent_ancestors)):
                raise ValueError("output overlaps an input")
        if os.path.lexists(self.destination):
            raise FileExistsError(errno.EEXIST, "destination already exists", str(self.destination))
        self.staging = Path(tempfile.mkdtemp(prefix=".sdfx-temporal-", dir=self.parent))
        self._stage_identity = _identity(self.staging)
        return self

    def _check_identity(self) -> None:
        if self.staging is None or self._parent_identity is None or self._stage_identity is None:
            raise RuntimeError("transaction is not active")
        _reject_links(self.parent)
        _reject_links(self.staging)
        if _identity(self.parent) != self._parent_identity or _identity(self.staging) != self._stage_identity:
            raise RuntimeError("output directory identity changed; refusing operation")

    def commit(self) -> Path:
        if self._committed:
            raise RuntimeError("transaction has already committed")
        self._check_identity()
        assert self.staging is not None
        _rename_no_replace(self.staging, self.destination)
        self._committed = True
        return self.destination

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if not self._committed and self.staging is not None:
            self._check_identity()
            # os.walk does not descend through symlinks; refuse unexpected links
            # or reparse points instead of relying on platform-specific rmtree.
            for root, dirs, files in os.walk(self.staging, followlinks=False):
                for name in (*dirs, *files):
                    _reject_links(Path(root) / name)
            shutil.rmtree(self.staging)
        return False
