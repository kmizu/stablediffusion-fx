"""ControlNet model loading and preprocessors.

Preprocessors: none / invert (PIL), canny (cv2), depth (transformers DPT, needs torch + HF weights),
and controlnet_aux detectors when that package is importable.
"""
from __future__ import annotations

import importlib.util
import logging
import os
from typing import Optional

from .images import Image, ImageOps, require_pil

log = logging.getLogger("sdfx.controlnet")

AUX_DETECTORS = {
    # name -> (controlnet_aux class, from_pretrained repo)
    "openpose": ("OpenposeDetector", "lllyasviel/Annotators"),
    "lineart": ("LineartDetector", "lllyasviel/Annotators"),
    "lineart_anime": ("LineartAnimeDetector", "lllyasviel/Annotators"),
    "softedge_hed": ("HEDdetector", "lllyasviel/Annotators"),
    "mlsd": ("MLSDdetector", "lllyasviel/Annotators"),
    "normal_bae": ("NormalBaeDetector", "lllyasviel/Annotators"),
}


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def has_controlnet_aux() -> bool:
    return _has("controlnet_aux")


def available_preprocessors() -> list[str]:
    out = ["none", "invert", "inpaint"] + list(PIL_PREPROCESSORS)
    if _has("cv2"):
        out.append("canny")
    if _has("transformers") and _has("torch"):
        out.append("depth")
    if _has("diffusers") and _has("torch"):
        out.extend(["depth_marigold", "normal_marigold"])
    if has_controlnet_aux():
        out.extend(AUX_DETECTORS.keys())
    return out


_aux_cache: dict = {}
_depth_pipe = None


PIL_PREPROCESSORS = ("recolor_luminance", "recolor_intensity", "tile_resample", "tile_colorfix", "blur_gaussian")


def pil_preprocess(name: str, img, args: dict):
    """forge_preprocessor_recolor / forge_preprocessor_tile equivalents (pure PIL/numpy)."""
    import numpy as np
    from PIL import ImageFilter
    rgb = img.convert("RGB")
    if name in ("recolor_luminance", "recolor_intensity"):
        a = np.asarray(rgb, dtype=np.float32) / 255.0
        if name == "recolor_luminance":
            y = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
        else:
            y = a.max(axis=2)
        y = np.clip(y, 0, 1) ** float(args.get("gamma_correction", 1.0))
        return Image.fromarray((y * 255).astype("uint8")).convert("RGB")
    if name == "tile_resample":
        down = float(args.get("down_sampling_rate", 1.0))
        if down <= 1.0:
            return rgb
        small = rgb.resize((max(1, int(rgb.width / down)), max(1, int(rgb.height / down))), Image.BICUBIC)
        return small.resize(rgb.size, Image.BICUBIC)
    if name == "tile_colorfix":
        r = int(args.get("variation", 8))
        return rgb.filter(ImageFilter.GaussianBlur(radius=max(1, r)))
    if name == "blur_gaussian":
        return rgb.filter(ImageFilter.GaussianBlur(radius=float(args.get("sigma", 9.0))))
    raise ValueError(name)


_marigold: dict = {}


def depth_to_image(pred):
    """Marigold affine-invariant depth (N,H,W[,1] or H,W, 0 = near) -> RGB grayscale with near = bright, like the
    ControlNet depth maps the models were trained on."""
    import numpy as np
    a = np.asarray(pred.cpu().numpy() if hasattr(pred, "cpu") else pred, dtype=np.float32)
    a = a.reshape(a.shape[-3], a.shape[-2]) if a.ndim == 4 and a.shape[-1] == 1 else a
    while a.ndim > 2:
        a = a[0]
    lo, hi = float(a.min()), float(a.max())
    a = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
    return Image.fromarray(((1.0 - a) * 255).round().astype("uint8")).convert("RGB")


def marigold(name: str, img, args: dict):
    """forge_preprocessor_marigold: diffusers Marigold depth / normals (LCM variants, few steps)."""
    import torch
    import diffusers  # type: ignore
    if name == "depth_marigold":
        cls, repo = getattr(diffusers, "MarigoldDepthPipeline", None), "prs-eth/marigold-depth-lcm-v1-0"
    else:
        cls, repo = getattr(diffusers, "MarigoldNormalsPipeline", None), "prs-eth/marigold-normals-lcm-v0-1"
    if cls is None:
        raise ValueError(f"{name} needs diffusers >= 0.28")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = _marigold.get(name)
    if pipe is None:
        pipe = cls.from_pretrained(repo, torch_dtype=torch.float16 if dev == "cuda" else torch.float32)
        pipe.set_progress_bar_config(disable=True)
        _marigold[name] = pipe
    pipe.to(dev)
    try:
        out = pipe(img.convert("RGB"), num_inference_steps=int(args.get("steps", 4)))
        if name == "depth_marigold":
            res = depth_to_image(out.prediction)
        else:
            res = pipe.image_processor.visualize_normals(out.prediction)[0].convert("RGB")
        return res.resize(img.size, Image.BILINEAR)
    finally:
        pipe.to("cpu")


def run_preprocessor(name: str, img, args: Optional[dict] = None):
    """Apply preprocessor `name` to a PIL RGB image; returns a PIL RGB image of the same size."""
    require_pil()
    args = args or {}
    if name in ("none", "", None):
        return img
    if name == "invert":
        return ImageOps.invert(img.convert("RGB"))
    if name in PIL_PREPROCESSORS:
        return pil_preprocess(name, img, args)
    if name == "inpaint":
        return img.convert("RGB")  # the masked area is set to -1 by the engine when a mask is available
    if name in ("depth_marigold", "normal_marigold"):
        return marigold(name, img, args)
    if name == "canny":
        import cv2  # type: ignore
        import numpy as np
        lo = int(args.get("low_threshold", 100))
        hi = int(args.get("high_threshold", 200))
        edges = cv2.Canny(np.asarray(img.convert("RGB")), lo, hi)
        return Image.fromarray(edges).convert("RGB")
    if name == "depth":
        global _depth_pipe
        from transformers import pipeline  # type: ignore
        if _depth_pipe is None:
            _depth_pipe = pipeline("depth-estimation", model=os.environ.get("SDFX_DEPTH_MODEL", "Intel/dpt-hybrid-midas"))
        d = _depth_pipe(img.convert("RGB"))["depth"]
        return d.convert("RGB").resize(img.size, Image.BILINEAR)
    if name in AUX_DETECTORS:
        import controlnet_aux  # type: ignore
        cls_name, repo = AUX_DETECTORS[name]
        det = _aux_cache.get(name)
        if det is None:
            det = getattr(controlnet_aux, cls_name).from_pretrained(repo)
            _aux_cache[name] = det
        res = int(args.get("resolution", min(img.width, img.height)))
        out = det(img.convert("RGB"), detect_resolution=res, image_resolution=res)
        return out.convert("RGB").resize(img.size, Image.BILINEAR)
    raise ValueError(f"unknown preprocessor '{name}'")


def load_controlnet(path: str, dtype):
    """ControlNetModel from a diffusers directory or a single .safetensors/.pth file."""
    from diffusers import ControlNetModel  # type: ignore
    if os.path.isdir(path):
        return ControlNetModel.from_pretrained(path, torch_dtype=dtype)
    return ControlNetModel.from_single_file(path, torch_dtype=dtype)
