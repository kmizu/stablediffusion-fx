"""Hires fix helpers: latent upscaling modes (A1111 names) and image upscaling for the second pass."""
from __future__ import annotations

from .base import LATENT_UPSCALERS

LATENT_MODES = {
    "Latent": ("bilinear", False),
    "Latent (antialiased)": ("bilinear", True),
    "Latent (bicubic)": ("bicubic", False),
    "Latent (bicubic antialiased)": ("bicubic", True),
    "Latent (nearest)": ("nearest", False),
    "Latent (nearest-exact)": ("nearest-exact", False),
}


def is_latent(upscaler: str) -> bool:
    return upscaler in LATENT_UPSCALERS


def upscale_latents(latents, width: int, height: int, upscaler: str):
    """Resize a (B, C, h, w) latent tensor to the latent size of width x height."""
    import torch.nn.functional as F
    mode, aa = LATENT_MODES.get(upscaler, ("bilinear", False))
    size = (height // 8, width // 8)
    dt = latents.dtype
    x = latents.float()
    if mode in ("nearest", "nearest-exact"):
        y = F.interpolate(x, size=size, mode=mode)
    else:
        y = F.interpolate(x, size=size, mode=mode, antialias=aa, align_corners=False)
    return y.to(dt)
