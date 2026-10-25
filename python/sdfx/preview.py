"""Live previews: cheap linear latent->RGB approximation (A1111 'Approx cheap'), optional TAESD."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger("sdfx.preview")

# rows = latent channels, cols = RGB (A1111 modules/sd_vae_approx.cheap_approximation)
SD15_FACTORS = np.array([
    [0.298, 0.207, 0.208],
    [0.187, 0.286, 0.173],
    [-0.158, 0.189, 0.264],
    [-0.184, -0.271, -0.473],
], dtype=np.float32)

SDXL_FACTORS = np.array([
    [0.3448, 0.4168, 0.4395],
    [-0.1953, -0.0290, 0.0250],
    [0.1074, 0.0886, -0.0163],
    [-0.3730, -0.2499, -0.2088],
], dtype=np.float32)


def latent_to_rgb(latent: np.ndarray, sdxl: bool = False) -> np.ndarray:
    """(4, h, w) or (B, 4, h, w) latent -> uint8 (h, w, 3) RGB for the first image."""
    lat = np.asarray(latent, dtype=np.float32)
    if lat.ndim == 4:
        lat = lat[0]
    if lat.shape[0] != 4:
        lat = lat[:4]
    f = SDXL_FACTORS if sdxl else SD15_FACTORS
    rgb = np.einsum("lxy,lr->xyr", lat, f)
    rgb = np.clip((rgb + 1.0) / 2.0, 0.0, 1.0)
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def latent_preview_b64(latent, sdxl: bool = False, max_side: int = 256) -> Optional[str]:
    from .images import HAVE_PIL, rgb_array_to_image, to_jpeg_b64
    if not HAVE_PIL:
        return None
    arr = latent_to_rgb(latent, sdxl)
    return to_jpeg_b64(rgb_array_to_image(arr), max_side=max_side)


class TaesdPreviewer:
    """Decodes latents with TAESD (madebyollin/taesd or taesdxl); loaded lazily, falls back on failure."""

    def __init__(self):
        self._models = {}
        self._failed = set()

    def decode_b64(self, latents, sdxl: bool, device, dtype) -> Optional[str]:
        key = "sdxl" if sdxl else "sd15"
        if key in self._failed:
            return None
        try:
            import torch
            from diffusers import AutoencoderTiny  # type: ignore
            m = self._models.get(key)
            if m is None:
                repo = "madebyollin/taesdxl" if sdxl else "madebyollin/taesd"
                m = AutoencoderTiny.from_pretrained(repo, torch_dtype=dtype).to(device)
                self._models[key] = m
            with torch.no_grad():
                img = m.decode(latents[:1].to(device, dtype) / getattr(m.config, "scaling_factor", 1.0)).sample[0]
            arr = ((img.float().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            from .images import rgb_array_to_image, to_jpeg_b64
            return to_jpeg_b64(rgb_array_to_image(arr))
        except Exception as e:  # network / missing weights
            log.warning("TAESD preview unavailable (%s); using approx", e)
            self._failed.add(key)
            return None
