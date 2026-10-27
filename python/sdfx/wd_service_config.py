"""Explicit local WD candidate setup; never fetch or install model assets."""
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import stat


@dataclass(frozen=True)
class WdLocalConfig:
    model: str
    tags: str
    temp_parent: str


@dataclass(frozen=True)
class ConfigResult:
    available: bool
    reason: str


def _dependencies_available():
    try:
        return all(importlib.util.find_spec(name) is not None for name in ("onnx", "onnxruntime", "numpy", "PIL"))
    except (ImportError, ValueError, AttributeError):
        return False


def _initialize_dependencies():
    """Import native dependencies on the worker's startup thread, before hello."""
    for name in ("numpy", "onnx", "onnxruntime", "PIL.Image"):
        importlib.import_module(name)


def _path(value):
    from .temporal_lab_io import _absolute, _reject_links
    if not isinstance(value, str) or not value or len(value) > 4096 or not Path(value).is_absolute():
        raise ValueError("explicit absolute local paths are required")
    path = _absolute(value)
    _reject_links(path)
    return path


def _signature(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _pin(path, expected):
    from .wd_v3_assets import AssetPin
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or not info.st_ino or info.st_size != expected[0]:
        raise ValueError("local WD asset size or identity differs from candidate")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    count = 0
    with os.fdopen(os.open(path, flags), "rb") as stream:
        if _signature(os.fstat(stream.fileno())) != _signature(info):
            raise ValueError("local WD asset changed")
        while True:
            data = stream.read(min(1024 * 1024, expected[0] - count + 1))
            if not data: break
            count += len(data)
            if count > expected[0]: raise ValueError("local WD asset grew")
            digest.update(data)
        if _signature(os.fstat(stream.fileno())) != _signature(info):
            raise ValueError("local WD asset changed")
    if count != expected[0] or digest.hexdigest() != expected[1] or _signature(path.stat(follow_symlinks=False)) != _signature(info):
        raise ValueError("local WD asset does not match candidate digest")
    return AssetPin(path, count, digest.hexdigest()), _signature(info)


def configure_wd_service(engine, config):
    """Configure a trusted service from explicit local files, or leave disabled."""
    engine._batch_interrogate_service = None
    if config is None:
        return ConfigResult(False, "not configured")
    if getattr(engine, "dry_run", False):
        return ConfigResult(False, "dry-run has no WD inference")
    if not isinstance(config, WdLocalConfig) or not _dependencies_available():
        return ConfigResult(False, "configuration or dependencies unavailable")
    try:
        from .temporal_lab_io import _reject_links
        from .wd_v3_runtime import _MODEL_IDENTITY, _TAGS_IDENTITY
        from .wd_v3_provider import WdV3Provider
        from .batch_interrogate_backends import BackendRegistry
        from .batch_interrogate_worker import BatchInterrogateService
        model_path, tags_path, temp = _path(config.model), _path(config.tags), _path(config.temp_parent)
        temp_info = temp.stat(follow_symlinks=False)
        if not stat.S_ISDIR(temp_info.st_mode) or not temp_info.st_ino:
            raise ValueError("WD temporary directory is unavailable")
        model, model_identity = _pin(model_path, _MODEL_IDENTITY)
        tags, tags_identity = _pin(tags_path, _TAGS_IDENTITY)
        _initialize_dependencies()
        temp_identity = (temp_info.st_dev, temp_info.st_ino)
        def ready():
            try:
                if not _dependencies_available(): return False
                for path in (model_path, tags_path, temp): _reject_links(path)
                current = temp.stat(follow_symlinks=False)
                return (stat.S_ISDIR(current.st_mode) and (current.st_dev, current.st_ino) == temp_identity
                        and _signature(model_path.stat(follow_symlinks=False)) == model_identity
                        and _signature(tags_path.stat(follow_symlinks=False)) == tags_identity)
            except (OSError, ValueError):
                return False
        provider = WdV3Provider(model, tags, temp)
        engine._batch_interrogate_service = BatchInterrogateService(BackendRegistry({"wd_ext": provider}), ready=ready)
        return ConfigResult(True, "configured local WD v3 candidate")
    except (OSError, ValueError, ImportError) as exc:
        return ConfigResult(False, f"local WD configuration unavailable: {type(exc).__name__}")
