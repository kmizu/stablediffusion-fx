"""Local WD v3 -> owned batch smoke with a synthetic, non-SD generator.

Explicit local files and an unused output directory name are required. No
model is fetched. The generated PNG is synthetic and is not SD inference.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from PIL import Image, ImageOps, PngImagePlugin
from sdfx.batch_face_manifest import FileSnapshot
from sdfx.batch_interrogate import BatchInterrogateOptions
from sdfx.batch_interrogate_backends import BackendRegistry
from sdfx.batch_interrogate_runner import BatchInput, run_registered_batch
from sdfx.wd_v3_assets import MAX_MODEL_BYTES, MAX_TAG_BYTES, MAX_SQUARE_PIXELS
from sdfx.wd_v3_provider import WdV3Provider
from wd_v3_cpu_smoke import MAX_IMAGE_BYTES, _pin


class _Context:
    def check_interrupt(self): pass
    def publish(self, action): return action()


class _TracedProvider:
    input_mode = "RGBA"
    def __init__(self, wrapped): self.wrapped, self.captions = wrapped, []
    def interrogate(self, image, ctx):
        result = self.wrapped.interrogate(image, ctx)
        self.captions.append(result)
        return result
    def unload(self): return self.wrapped.unload()


class _SyntheticGenerator:
    def __init__(self): self.prompts = []
    def img2img(self, request, ctx):
        ctx.check_interrupt()
        prompt = request["params"]["prompt"]
        self.prompts.append(prompt)
        path = Path(request["output_dir"]) / "synthetic-not-sd.png"
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("Description", "Synthetic test image; no Stable Diffusion inference")
        metadata.add_text("sdfx_prompt", prompt)
        with Image.new("RGB", (64, 64), (72, 126, 169)) as frame:
            frame.save(path, format="PNG", pnginfo=metadata)
        return {"images": [{"path": str(path)}], "generator": "synthetic"}


def _stages(parent, prefix):
    return {p.name for p in Path(parent).iterdir() if p.name.startswith(prefix)}


def run_batch_smoke(model, tags, image, temp_parent, output_parent, output_name, prompt):
    model_pin, tags_pin = _pin(model, MAX_MODEL_BYTES), _pin(tags, MAX_TAG_BYTES)
    image_pin = _pin(image, MAX_IMAGE_BYTES)
    info = image_pin.path.stat(follow_symlinks=False)
    with Image.open(image_pin.path) as source:
        width, height = source.size
        if width <= 0 or height <= 0 or max(width, height) ** 2 > MAX_SQUARE_PIXELS:
            raise ValueError("input dimensions exceed limit")
        with ImageOps.exif_transpose(source) as oriented:
            dimensions = oriented.size
    item = BatchInput(str(image_pin.path), FileSnapshot(image_pin.path, dimensions, image_pin.sha256,
                      info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns))
    wd_before = _stages(temp_parent, "sdfx-wd-v3-")
    batch_before = _stages(output_parent, ".sdfx-temporal-")
    provider = _TracedProvider(WdV3Provider(model_pin, tags_pin, temp_parent))
    registry = BackendRegistry({"wd_ext": provider})
    engine = _SyntheticGenerator()
    rows = run_registered_batch(engine, [item], {"params": {"prompt": prompt, "negative_prompt": ""}},
                                ("wd_ext",), BatchInterrogateOptions(), registry, output_parent, output_name, _Context())
    if len(rows) != 1 or rows[0]["status"] != "complete":
        raise RuntimeError(f"WD batch failed: {rows[0].get('error', 'unknown error') if rows else 'empty result'}")
    output = Path(rows[0]["result"]["images"][0]["path"])
    output_pin = _pin(output, MAX_IMAGE_BYTES)
    with Image.open(output) as result:
        png_prompt = result.info.get("sdfx_prompt")
        description = result.info.get("Description")
    if not provider.captions or not engine.prompts or png_prompt != engine.prompts[0] or description != "Synthetic test image; no Stable Diffusion inference":
        raise RuntimeError(f"published synthetic output metadata mismatch: {output}")
    unchanged = (_pin(image_pin.path, MAX_IMAGE_BYTES) == image_pin
                 and _pin(model_pin.path, MAX_MODEL_BYTES) == model_pin
                 and _pin(tags_pin.path, MAX_TAG_BYTES) == tags_pin)
    clean = (_stages(temp_parent, "sdfx-wd-v3-") == wd_before
             and _stages(output_parent, ".sdfx-temporal-") == batch_before)
    if not unchanged or not clean:
        raise RuntimeError(f"published synthetic output {output}; input or staging verification failed")
    return {"kind": "local-wd-v3-owned-batch-smoke",
            "generator": "synthetic test image; no Stable Diffusion inference",
            "caption": provider.captions[0], "generation_prompt": engine.prompts[0], "png_prompt": png_prompt,
            "input": {"path": str(image_pin.path), "size": image_pin.size, "sha256": image_pin.sha256,
                      "dimensions": list(dimensions)},
            "model": {"size": model_pin.size, "sha256": model_pin.sha256},
            "tags": {"size": tags_pin.size, "sha256": tags_pin.sha256},
            "output": {"path": str(output), "size": output_pin.size, "sha256": output_pin.sha256},
            "input_unchanged": unchanged, "staging_clean": clean}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("model", "tags", "image", "temp-parent", "output-parent", "output-name"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--prompt", default="batch smoke test")
    args = parser.parse_args()
    report = run_batch_smoke(args.model, args.tags, args.image, args.temp_parent,
                             args.output_parent, args.output_name, args.prompt)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
