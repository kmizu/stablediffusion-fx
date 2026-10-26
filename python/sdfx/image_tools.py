"""Image tools behind the `image_tool` engine request, re-implemented from extensions in the A1111 index.

  tool               extension(s)                                   what
  pixel_art          sd-pixel, sd-webui-pixelart, pixelization      downscale + k-means palette (+ dither, outline)
  color_enhance      sd-webui-color-enhance (GEGL "Color Enhance")  stretch chroma in LCh, keep lightness and hue
  cas_upscale        AMD FSR 1.0 integration                        Lanczos upscale + contrast adaptive sharpening
  person_mask        A Person Mask Generator                        YOLO person segmentation -> mask
  remove_background  rembg, ABG, PBRemTools, bg-mask, GeekyRemB     rembg models (u2net, isnet-anime, ...) or the
                                                                    person segmentation -> transparent PNG / mask
  depth_mask         depthmap2mask                                  depth estimation -> near / far mask

Every tool takes a PIL image and returns (image, extra_outputs) where extra_outputs are other PIL images
(e.g. the mask) keyed by name.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

TOOLS = ("pixel_art", "color_enhance", "cas_upscale", "person_mask", "remove_background", "depth_mask")
REMBG_MODELS = ("u2net", "u2netp", "u2net_human_seg", "isnet-general-use", "isnet-anime", "silueta")


# ------------------------------------------------------------------------------------------------ pixel art
def bayer(n: int) -> np.ndarray:
    m = np.array([[0, 2], [3, 1]], dtype=np.float32)
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / (m.size) - 0.5


def pixel_art(img, pixel_size: int = 8, colors: int = 16, dither: float = 0.0, outline: bool = False,
              upscale: bool = True, seed: int = 0):
    """Cell average downscale, k-means palette (OpenCV), optional ordered dither and dark outlines."""
    import cv2
    from PIL import Image
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[:2]
    ps = max(1, int(pixel_size))
    sw, sh = max(1, w // ps), max(1, h // ps)
    small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    if dither > 0:
        b = bayer(4)
        tile = np.tile(b, (sh // 4 + 1, sw // 4 + 1))[:sh, :sw]
        small = small + tile[..., None] * (255.0 / max(colors, 2)) * float(dither) * 2
    k = max(2, min(int(colors), 256))
    data = small.reshape(-1, 3).astype(np.float32)
    # OpenCV takes a signed C int; generation seeds can span the full uint32 range (or a Scala Long).
    cv2.setRNGSeed((int(seed) + 2**31) % 2**32 - 2**31)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    k = min(k, len(data))
    _, labels, centers = cv2.kmeans(data, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    q = centers[labels.flatten()].reshape(sh, sw, 3)
    if outline:
        gray = cv2.cvtColor(np.clip(q, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 40, 120) > 0
        q[edges] = q[edges] * 0.35
    out = Image.fromarray(np.clip(q, 0, 255).astype(np.uint8), "RGB")
    if upscale:
        out = out.resize((w, h), Image.NEAREST)
    return out, {"palette": int(k)}


# --------------------------------------------------------------------------------------------- colour enhance
def color_enhance(img, strength: float = 1.0, percentile: float = 99.5):
    """GEGL color-enhance: stretch chroma so its high percentile reaches the gamut-ish maximum, blend by strength."""
    import cv2
    from PIL import Image
    rgb = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)            # L 0..100, a/b about -128..127
    a, b = lab[..., 1], lab[..., 2]
    c = np.sqrt(a * a + b * b)
    hi = float(np.percentile(c, percentile))
    if hi < 1e-3:
        return img.convert("RGB"), {}
    target = 100.0
    k = 1.0 + (target / hi - 1.0) * float(strength)
    k = max(0.0, min(k, 4.0))
    lab[..., 1], lab[..., 2] = a * k, b * k
    out = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8), "RGB"), {"chroma_gain": round(k, 3)}


# ----------------------------------------------------------------------------------------------- CAS upscale
def cas(rgb: np.ndarray, sharpness: float) -> np.ndarray:
    """AMD FidelityFX contrast adaptive sharpening (3x3 cross neighbourhood), rgb float 0..1."""
    p = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode="edge")
    c = p[1:-1, 1:-1]
    n, s, w, e = p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]
    mn = np.minimum.reduce([c, n, s, w, e])
    mx = np.maximum.reduce([c, n, s, w, e])
    amp = np.sqrt(np.clip(np.minimum(mn, 1.0 - mx) / np.maximum(mx, 1e-5), 0, 1))
    peak = -1.0 / (8.0 - 3.0 * float(np.clip(sharpness, 0, 1)))
    wgt = amp * peak
    out = (c + (n + s + w + e) * wgt) / (1.0 + 4.0 * wgt)
    return np.clip(out, 0, 1)


def cas_upscale(img, scale: float = 2.0, sharpness: float = 0.5):
    from PIL import Image
    w, h = img.size
    up = img.convert("RGB").resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    rgb = np.asarray(up).astype(np.float32) / 255.0
    return Image.fromarray((cas(rgb, sharpness) * 255 + 0.5).astype(np.uint8), "RGB"), {}


# -------------------------------------------------------------------------------------------- masks / cut-out
def person_mask_array(img, confidence: float = 0.3, model: str = "person_yolov8n-seg.pt", device: str = "cuda",
                      dilate: int = 0) -> np.ndarray:
    from PIL import Image
    from . import adetailer as ad
    d = ad.detect(img, ad.resolve_model(model), confidence, device)
    m = np.zeros((img.height, img.width), dtype=np.uint8)
    for mk in d.masks:
        m = np.maximum(m, np.asarray(mk.convert("L")))
    if dilate:
        m = np.asarray(ad.offset_dilate(Image.fromarray(m, "L"), 0, 0, int(dilate)))
    return m


def person_mask(img, confidence: float = 0.3, dilate: int = 4, invert: bool = False, device: str = "cuda"):
    from PIL import Image
    m = person_mask_array(img, confidence, device=device, dilate=dilate)
    if invert:
        m = 255 - m
    mask = Image.fromarray(m, "L")
    return mask.convert("RGB"), {"mask": mask}


def _cutout(img, alpha: np.ndarray, background: str):
    from PIL import Image
    rgba = img.convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha.astype(np.uint8), "L"))
    if background in ("", "transparent", None):
        return rgba
    bg = Image.new("RGBA", rgba.size, {"white": (255, 255, 255, 255), "black": (0, 0, 0, 255),
                                         "green": (0, 177, 64, 255)}.get(background, (255, 255, 255, 255)))
    return Image.alpha_composite(bg, rgba).convert("RGB")


def remove_background(img, model: str = "u2net", alpha_matting: bool = False, background: str = "transparent",
                      device: str = "cuda"):
    from PIL import Image
    if model == "person_segmentation":
        alpha = person_mask_array(img, device=device)
    else:
        if model not in REMBG_MODELS:
            raise ValueError(f"unknown background removal model '{model}'")
        import rembg  # optional dependency
        sess = _rembg_sessions.get(model)
        if sess is None:
            sess = _rembg_sessions[model] = rembg.new_session(model)
        cut = rembg.remove(img.convert("RGB"), session=sess, alpha_matting=bool(alpha_matting))
        alpha = np.asarray(cut.convert("RGBA"))[..., 3]
    mask = Image.fromarray(alpha.astype(np.uint8), "L")
    return _cutout(img, alpha, background), {"mask": mask}


_rembg_sessions: dict = {}


def depth_mask(img, near: float = 0.5, far: float = 1.0, invert: bool = False, blur: int = 4):
    """depthmap2mask: DPT depth (near = bright), keep depths between the near / far thresholds (0..1 of the range)."""
    import cv2
    from PIL import Image
    from .controlnet import run_preprocessor
    d = np.asarray(run_preprocessor("depth", img.convert("RGB")).convert("L")).astype(np.float32)
    d = (d - d.min()) / max(float(d.max() - d.min()), 1e-6)       # 1 = nearest
    lo, hi = sorted((float(near), float(far)))
    m = ((d >= lo) & (d <= hi)).astype(np.uint8) * 255
    if invert:
        m = 255 - m
    if blur:
        m = cv2.GaussianBlur(m, (0, 0), float(blur))
    mask = Image.fromarray(m, "L")
    return mask.convert("RGB"), {"mask": mask, "depth": Image.fromarray((d * 255).astype(np.uint8), "L")}


def run(tool: str, img, params: dict, device: str = "cuda"):
    p = dict(params or {})
    if tool == "pixel_art":
        return pixel_art(img, int(p.get("pixel_size", 8)), int(p.get("colors", 16)), float(p.get("dither", 0.0)),
                         bool(p.get("outline", False)), bool(p.get("upscale", True)), int(p.get("seed", 0)))
    if tool == "color_enhance":
        return color_enhance(img, float(p.get("strength", 1.0)), float(p.get("percentile", 99.5)))
    if tool == "cas_upscale":
        return cas_upscale(img, float(p.get("scale", 2.0)), float(p.get("sharpness", 0.5)))
    if tool == "person_mask":
        return person_mask(img, float(p.get("confidence", 0.3)), int(p.get("dilate", 4)), bool(p.get("invert", False)),
                           device)
    if tool == "remove_background":
        return remove_background(img, str(p.get("model", "u2net")), bool(p.get("alpha_matting", False)),
                                 str(p.get("background", "transparent")), device)
    if tool == "depth_mask":
        return depth_mask(img, float(p.get("near", 0.5)), float(p.get("far", 1.0)), bool(p.get("invert", False)),
                          int(p.get("blur", 4)))
    raise ValueError(f"unknown image tool '{tool}' (one of {', '.join(TOOLS)})")


__all__ = ["TOOLS", "REMBG_MODELS", "run", "pixel_art", "color_enhance", "cas", "cas_upscale", "person_mask",
           "remove_background", "depth_mask", "bayer"]
