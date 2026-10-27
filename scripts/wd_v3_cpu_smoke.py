"""Explicit local WD v3 CPU smoke. JSON goes to stdout; no download or UI.

Example: python scripts/wd_v3_cpu_smoke.py --model LOCAL_BLOB --tags LOCAL_BLOB
    --image LOCAL_IMAGE --temp-parent EXISTING_PRIVATE_DIR > report.json

The input files and their parents must not be modified by concurrent hostile
writers. Native allocation/reclamation is not proven by Python weakrefs.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import weakref

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from PIL import Image
from sdfx.temporal_lab_io import _absolute, _reject_links
from sdfx.wd_v3_assets import AssetPin, MAX_MODEL_BYTES, MAX_TAG_BYTES, MAX_SQUARE_PIXELS
from sdfx.wd_v3_runtime import run_internal

MAX_IMAGE_BYTES = 128 * 1024 * 1024


class _Context:
    def check_interrupt(self): pass


class _TrackedOrt:
    def __init__(self, module):
        self.module = module
        self.references = []

    def __getattr__(self, name): return getattr(self.module, name)

    def InferenceSession(self, *args, **kwargs):
        if kwargs.get("providers") != ["CPUExecutionProvider"]:
            raise ValueError("CPU-only provider required")
        session = self.module.InferenceSession(*args, **kwargs)
        try:
            self.references.append(weakref.ref(session))
        except TypeError:
            self.references.append(None)
        return session


def _pin(path, limit):
    path = _absolute(path)
    _reject_links(path)
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit or not before.st_ino:
        raise ValueError("input must be a bounded regular local file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    total = 0
    signature = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        if signature(os.fstat(stream.fileno())) != signature(before):
            raise ValueError("input changed before reading")
        while True:
            chunk = stream.read(min(1024 * 1024, before.st_size - total + 1))
            if not chunk: break
            total += len(chunk)
            if total > before.st_size: raise ValueError("input grew while reading")
            digest.update(chunk)
        if total != before.st_size or signature(os.fstat(stream.fileno())) != signature(before):
            raise ValueError("input changed while reading")
    _reject_links(path)
    if signature(path.stat(follow_symlinks=False)) != signature(before):
        raise ValueError("input changed while reading")
    return AssetPin(path, total, digest.hexdigest())


def run_smoke(model, tags, image, temp_parent, *, _runner=run_internal, _ort_module=None):
    """Inspect a caller-selected image with the fixed local model candidate."""
    if _ort_module is None:
        import onnxruntime as _ort_module
    tracker = _TrackedOrt(_ort_module)
    model_pin = _pin(model, MAX_MODEL_BYTES)
    tags_pin = _pin(tags, MAX_TAG_BYTES)
    image_pin = _pin(image, MAX_IMAGE_BYTES)
    with Image.open(image_pin.path) as source:
        width, height = source.size
        if width <= 0 or height <= 0 or max(width, height) ** 2 > MAX_SQUARE_PIXELS:
            raise ValueError("input image dimensions exceed limit")
        source.load()
        scores = _runner(model_pin, tags_pin, temp_parent, source, _Context(), _ort=tracker)
    if _pin(image_pin.path, MAX_IMAGE_BYTES) != image_pin:
        raise ValueError("input image changed during smoke")
    if _pin(model_pin.path, MAX_MODEL_BYTES) != model_pin or _pin(tags_pin.path, MAX_TAG_BYTES) != tags_pin:
        raise ValueError("model or tags changed during smoke")
    gc.collect()
    released = None if not tracker.references or any(ref is None for ref in tracker.references) else all(ref() is None for ref in tracker.references)
    ordered = [{"name": x.name, "category": x.category, "score": x.score} for x in scores]
    encoded = json.dumps(ordered, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    top = sorted((row for row in ordered if row["category"] == 0), key=lambda row: -row["score"])[:20]
    return {"kind": "local-wd-v3-cpu-smoke", "provider": "CPUExecutionProvider",
            "input": {"path": str(image_pin.path), "size": image_pin.size, "sha256": image_pin.sha256,
                      "dimensions": [width, height]},
            "model": {"path": str(model_pin.path), "size": model_pin.size, "sha256": model_pin.sha256},
            "tags": {"path": str(tags_pin.path), "size": tags_pin.size, "sha256": tags_pin.sha256},
            "score_count": len(ordered), "scores_sha256": hashlib.sha256(encoded).hexdigest(),
            "top_tags": top, "session_released": released,
            "release_evidence": "Python weak reference only; native memory release is not measured"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("model", "tags", "image", "temp-parent"):
        parser.add_argument("--" + option, required=True)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.model, args.tags, args.image, args.temp_parent), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
