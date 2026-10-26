"""Controlled Alternative diagnostics: record real latent statistics and compare normalization/endpoints.

Run with the GPU venv: alt_latent_probe.py <checkpoint> <source PNG> <output dir>.
Runtime monkeypatches affect this diagnostic process only; production files and model weights are not changed.
"""
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("TQDM_DISABLE", "1")

import torch
from PIL import Image
from sdfx.engine import Engine
from sdfx import extras
from sdfx.protocol import Emitter, JobContext

checkpoint, source, output = sys.argv[1:4]
out = Path(output).resolve()
out.mkdir(parents=True, exist_ok=True)
with Image.open(source) as im:
    width, height = im.size

engine = Engine()
ctx = JobContext(Emitter(io.BytesIO()), "probe")
original_invert = extras.invert_latents
original_run = engine._run
report = {}
state = {}


def stats(t):
    f = t.float()
    return {"mean": float(f.mean()), "std": float(f.std()), "min": float(f.min()), "max": float(f.max()),
            "channel_means": f.mean(dim=(0, 2, 3)).tolist(), "channel_stds": f.std(dim=(0, 2, 3)).tolist()}


def invert(*args, **kwargs):
    pipe, latents = args[1:3]
    state["encoded"] = stats(latents)
    state["forward_scheduler"] = dict(pipe.scheduler.config)
    saved = dict(pipe.scheduler.config)
    if "trailing" in state["variant"]:
        pipe.scheduler.register_to_config(timestep_spacing="trailing")
    try:
        raw = original_invert(*args, **kwargs)
    finally:
        pipe.scheduler.register_to_config(**saved)
    state["raw"] = raw
    state["inverse_noise"] = stats(raw)
    return raw


def run(*args, **kwargs):
    if kwargs.get("latents") is not None and state.get("raw") is not None:
        state["production_noise"] = stats(kwargs["latents"])
        if "raw" in state["variant"]:
            kwargs["latents"] = state["raw"]
        else:
            raw = state["raw"]
            kwargs["latents"] = raw / raw.float().std().clamp_min(1e-6).to(raw.dtype)
        state["applied_noise"] = stats(kwargs["latents"])
    return original_run(*args, **kwargs)


try:
    engine.load_model({"path": str(Path(checkpoint).resolve()), "dtype": "fp16", "vram_mode": "gpu"}, ctx)
    extras.invert_latents = invert
    engine._run = run
    variants = sys.argv[4].split(",") if len(sys.argv) > 4 else ("normalized", "raw", "trailing_normalized", "trailing_raw")
    for variant in variants:
        state.clear()
        state["variant"] = variant
        result = engine.img2img({"params": {
            "prompt": "portrait of a woman wearing (red scarf:0.8), in a park, detailed", "negative_prompt": "lowres",
            "sampler_name": "DDIM" if "ddim" in variant else "DPM++ 2M", "steps": 50 if "50" in variant else 16,
            "cfg_scale": 1 if "cfg1" in variant else 2 if "cfg2" in variant else 6,
            "seed": 11, "width": width, "height": height,
            "img2img": {"init_image": str(Path(source).resolve()), "denoising_strength": 1},
            "hooks": {"img2imgalt": {"decode_cfg_scale": 1}}}, "output_dir": str(out)}, ctx)
        path = result["images"][0]["path"]
        target = out / (variant + ".png")
        os.replace(path, target)
        report[variant] = {k: v for k, v in state.items() if k != "raw"}
        report[variant]["path"] = str(target)
        print(variant, report[variant]["inverse_noise"], flush=True)
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
finally:
    extras.invert_latents = original_invert
    engine._run = original_run
    engine.unload({}, ctx)
