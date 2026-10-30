"""Pixel upscalers: PIL Lanczos/Nearest/None and model upscalers (ESRGAN etc.) through spandrel, tiled."""
from __future__ import annotations

import importlib.util
import logging
import math
from typing import Optional

from .images import Image, require_pil

log = logging.getLogger("sdfx.upscalers")

PIL_UPSCALERS = {"None": None, "Lanczos": "LANCZOS", "Nearest": "NEAREST"}


def has_spandrel() -> bool:
    try:
        return importlib.util.find_spec("spandrel") is not None
    except (ImportError, ValueError):
        return False


_models: dict = {}


def _load(path: str, device, dtype):
    m = _models.get(path)
    if m is None:
        import spandrel  # type: ignore
        m = spandrel.ModelLoader().load_from_file(path)
        if not isinstance(m, spandrel.ImageModelDescriptor):
            raise ValueError(f"{path} is not a single-image upscaler model")
        m.model.eval()
        _models.clear()  # keep at most one model resident
        _models[path] = m
    use_half = dtype is not None and str(dtype).endswith("float16") and getattr(m, "supports_half", False)
    m.to(device)
    if use_half:
        m.model.half()
    else:
        m.model.float()
    return m


def _run_tiled(desc, img, device, tile: int = 512, overlap: int = 32, check_interrupt=None):
    import numpy as np
    import torch
    scale = int(desc.scale)
    x = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    dtype = next(desc.model.parameters()).dtype
    _, c, h, w = x.shape
    out = torch.zeros((1, c, h * scale, w * scale), dtype=torch.float32)
    weight = torch.zeros_like(out)
    step = max(1, tile - overlap)
    ys = list(range(0, max(1, h - overlap), step)) if h > tile else [0]
    xs = list(range(0, max(1, w - overlap), step)) if w > tile else [0]
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                if check_interrupt:
                    check_interrupt()
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                y0a, x0a = max(0, y1 - tile), max(0, x1 - tile)
                patch = x[:, :, y0a:y1, x0a:x1].to(device, dtype)
                res = desc(patch).float().cpu()
                out[:, :, y0a * scale:y1 * scale, x0a * scale:x1 * scale] += res
                weight[:, :, y0a * scale:y1 * scale, x0a * scale:x1 * scale] += 1
    out = (out / weight.clamp(min=1)).clamp(0, 1)
    arr = (out[0].permute(1, 2, 0).numpy() * 255.0 + 0.5).astype("uint8")
    return Image.fromarray(arr)


_ldsr = None
LDSR_REPO = "CompVis/ldm-super-resolution-4x-openimages"


def ldsr(img, device, steps: int = 100):
    """4x latent-diffusion super resolution (the LDSR built-in), via diffusers LDMSuperResolutionPipeline."""
    global _ldsr
    import torch
    from diffusers import LDMSuperResolutionPipeline  # type: ignore
    if _ldsr is None:
        _ldsr = LDMSuperResolutionPipeline.from_pretrained(LDSR_REPO)
        _ldsr.set_progress_bar_config(disable=True)
    _ldsr.to(device)
    try:
        w, h = img.width - img.width % 4 or 4, img.height - img.height % 4 or 4
        src = img.crop((0, 0, w, h)) if (w, h) != img.size else img
        with torch.inference_mode():
            return _ldsr(src, num_inference_steps=steps, eta=1.0).images[0]
    finally:
        _ldsr.to("cpu")


def upscale_to(img, width: int, height: int, name: str, model_path: Optional[str] = None, device=None, dtype=None,
               check_interrupt=None):
    """Upscale a PIL image to exactly width x height with upscaler `name` (A1111 semantics)."""
    require_pil()
    from . import waifu2x
    if name in waifu2x.NAMES:
        return waifu2x.upscale(img, width, height, waifu2x.parse_name(name), device=device or "cpu",
                              check_interrupt=check_interrupt or (lambda: None))
    if name in PIL_UPSCALERS:
        resample = Image.NEAREST if name == "Nearest" else Image.LANCZOS
        return img.resize((width, height), resample)
    if name == "LDSR":
        cur = img.convert("RGB")
        for _ in range(3):
            if cur.width >= width and cur.height >= height:
                break
            if check_interrupt:
                check_interrupt()
            cur = ldsr(cur, device or "cpu")
        return cur.resize((width, height), Image.LANCZOS) if cur.size != (width, height) else cur
    if not model_path:
        raise ValueError(f"no model path for upscaler '{name}'")
    desc = _load(model_path, device or "cpu", dtype)
    cur = img.convert("RGB")
    # run the model until the image is at least the target size (max 3 passes), then resize exactly
    for _ in range(3):
        if cur.width >= width and cur.height >= height:
            break
        cur = _run_tiled(desc, cur, device or "cpu", check_interrupt=check_interrupt)
    if cur.size != (width, height):
        cur = cur.resize((width, height), Image.LANCZOS)
    return cur


def upscale_by(img, scale: float, name: str, model_path: Optional[str] = None, device=None, dtype=None, check_interrupt=None):
    w, h = max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale)))
    if name == "None":
        return img.resize((w, h), Image.LANCZOS) if (w, h) != img.size else img
    return upscale_to(img, w, h, name, model_path, device, dtype, check_interrupt)


def unload_models():
    _models.clear()
    from . import waifu2x
    waifu2x.unload()


__all__ = ["upscale_to", "upscale_by", "has_spandrel", "PIL_UPSCALERS", "math"]
