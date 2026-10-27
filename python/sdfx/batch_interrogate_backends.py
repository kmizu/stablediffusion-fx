"""Offline, internal interrogator dispatch; no model adapters are registered.

Injected providers are for internal integration and tests, not public worker
input. A provider owns its model lifecycle and must not perform network access.
The owned scope unloads providers on success, error and cancellation. Native
decode/provider calls cannot be interrupted while they execute. Providers must
make unload safe after partial or failed loading.
Availability here is not a public capability or proof of model parity.
"""
from dataclasses import dataclass
from contextlib import contextmanager
import os
import re
import math
import warnings

from .batch_interrogate import MAX_TEXT
from .protocol import EngineError

_NAMES = ("clip", "deepbooru", "clip_ext", "wd_ext")
_MAX_SELECTION = 32
_MAX_BYTES = 128 * 1024 * 1024
_MAX_DIMENSION = 16_384
_MAX_PIXELS = 64_000_000


@dataclass(frozen=True)
class BackendStatus:
    name: str
    available: bool
    reason: str


class BackendRegistry:
    def __init__(self, providers=None):
        providers = dict(providers or {})
        for name, provider in providers.items():
            if (name not in _NAMES or not callable(getattr(provider, "interrogate", None))
                    or not callable(getattr(provider, "unload", None))
                    or getattr(provider, "input_mode", "RGB") not in ("RGB", "RGBA")):
                raise ValueError("invalid internal backend")
        self._providers = providers

    def available_backends(self):
        return [BackendStatus(name, name in self._providers,
                              "" if name in self._providers else "No reviewed local adapter and model are available")
                for name in _NAMES]

    def provenance(self, selection):
        providers = self._selection(selection)
        result = []
        for name, provider in zip(selection, providers):
            record = {"backend": name}
            describe = getattr(provider, "provenance", None)
            if describe is not None:
                if not callable(describe):
                    raise ValueError("invalid backend provenance")
                details = describe()
                if (not isinstance(details, dict) or set(details) != {"model_id", "model_sha256", "tags_sha256", "threshold",
                        "rating_threshold", "remove_underscores", "append_ratings", "preprocessing"}
                        or not isinstance(details["model_id"], str) or not 1 <= len(details["model_id"]) <= 512
                        or any(not isinstance(details[key], str) or not re.fullmatch(r"[0-9a-f]{64}", details[key])
                               for key in ("model_sha256", "tags_sha256"))
                        or any(type(details[key]) not in (int, float) or not 0 <= details[key] <= 1 or not math.isfinite(details[key])
                               for key in ("threshold", "rating_threshold"))
                        or any(type(details[key]) is not bool for key in ("remove_underscores", "append_ratings"))
                        or not isinstance(details["preprocessing"], str) or not 1 <= len(details["preprocessing"]) <= 128):
                    raise ValueError("invalid backend provenance")
                record.update(details)
            result.append(record)
        return result

    def _selection(self, selection):
        if not isinstance(selection, (tuple, list)) or not 1 <= len(selection) <= _MAX_SELECTION:
            raise ValueError("backend selection must be bounded and nonempty")
        if any(not isinstance(name, str) or name not in self._providers for name in selection):
            raise ValueError("selected interrogator is unavailable")
        return tuple(self._providers[name] for name in selection)

    def interrogate(self, image, selection, ctx):
        providers = self._selection(selection)
        modes = tuple(getattr(provider, "input_mode", "RGB") for provider in providers)
        if any(mode not in ("RGB", "RGBA") for mode in modes):
            raise ValueError("unsupported backend input mode")
        if not callable(getattr(ctx, "check_interrupt", None)):
            raise ValueError("cancellation context required")
        ctx.check_interrupt()
        if not isinstance(image, (str, os.PathLike)) or not 0 < os.stat(image).st_size <= _MAX_BYTES:
            raise ValueError("input image size exceeds limit")
        # Lazy import keeps unavailable optional imaging dependencies from
        # breaking the ordinary worker and performs no model loading.
        from PIL import Image, ImageOps
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(image) as source:
                width, height = source.size
                if width <= 0 or height <= 0 or max(width, height) > _MAX_DIMENSION or width * height > _MAX_PIXELS:
                    raise ValueError("input image dimensions exceed limit")
                source.load()
                with ImageOps.exif_transpose(source) as oriented:
                    results = []
                    total = 0
                    for provider, mode in zip(providers, modes):
                        ctx.check_interrupt()
                        with oriented.convert(mode) as pixels:
                            value = provider.interrogate(pixels, ctx)
                        ctx.check_interrupt()
                        if not isinstance(value, str):
                            raise ValueError("backend result must be text")
                        total += len(value) + bool(results)
                        if total > MAX_TEXT:
                            raise ValueError("backend result exceeds limit")
                        results.append(value)
                    return results

    def unload(self, selection):
        self._unload_providers(self._selection(selection))

    @staticmethod
    def _unload_providers(providers):
        errors = []
        seen = set()
        for provider in providers:
            if id(provider) in seen:
                continue
            seen.add(id(provider))
            try:
                provider.unload()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("backend unload failed: " + "; ".join(errors))

    @contextmanager
    def owned(self, selection):
        """Release each selected provider even when work fails or is cancelled."""
        providers = self._selection(selection)
        try:
            yield self
        except BaseException as primary:
            try:
                self._unload_providers(providers)
            except Exception as cleanup:
                if isinstance(primary, EngineError):
                    primary.message += f"; backend cleanup failed: {cleanup}"
                primary.args = (f"{primary}; backend cleanup failed: {cleanup}",)
            raise
        else:
            self._unload_providers(providers)
