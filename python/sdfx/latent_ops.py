"""Per-step latent operations applied from diffusers' callback_on_step_end.

  {"latent_modifier": {"sharpness_multiplier","sharpness_method","tonemap_multiplier","tonemap_method",
                       "tonemap_percentile","contrast_multiplier","combat_method","combat_cfg_drift",
                       "extra_noise_type","extra_noise_multiplier","extra_noise_lowpass",
                       "divisive_norm_size","divisive_norm_multiplier",
                       "spectral_mod_mode","spectral_mod_percentile","spectral_mod_multiplier"}}
        sd_forge_latent_modifier (subset), applied like the original to the guided denoised prediction (x0)
        through guidance.GuidanceHook
  {"soft_inpainting": {"mask_blend_power","mask_blend_scale","inpaint_detail_preservation",
                       "composite_mask_influence","composite_difference_threshold","composite_difference_contrast"}}
        soft-inpainting: per-step blending of the original image's noised latents with a soft (blurred) mask,
        i.e. differential diffusion.
"""
from __future__ import annotations

import math


LATENT_MODIFIER_DEFAULTS = {
    "sharpness_multiplier": 0.0, "sharpness_method": "anisotropic",
    "tonemap_multiplier": 0.0, "tonemap_method": "reinhard", "tonemap_percentile": 100.0,
    "contrast_multiplier": 0.0,
    "combat_method": "subtract", "combat_cfg_drift": 0.0,
    "extra_noise_type": "gaussian", "extra_noise_multiplier": 0.0, "extra_noise_lowpass": 100,
    "divisive_norm_size": 127, "divisive_norm_multiplier": 0.0,
    "spectral_mod_mode": "hard_clamp", "spectral_mod_percentile": 5.0, "spectral_mod_multiplier": 0.0,
}

SOFT_INPAINT_DEFAULTS = {"mask_blend_power": 1.0, "mask_blend_scale": 0.5, "inpaint_detail_preservation": 4.0,
                         "composite_mask_influence": 0.0, "composite_difference_threshold": 0.5,
                         "composite_difference_contrast": 2.0}


def normalize_latent_modifier(cfg: dict) -> dict:
    out = dict(LATENT_MODIFIER_DEFAULTS)
    for k, d in LATENT_MODIFIER_DEFAULTS.items():
        if k in cfg:
            out[k] = type(d)(cfg[k]) if not isinstance(d, str) else str(cfg[k])
    return out


def normalize_soft_inpaint(cfg: dict) -> dict:
    out = dict(SOFT_INPAINT_DEFAULTS)
    for k in SOFT_INPAINT_DEFAULTS:
        if k in cfg:
            out[k] = float(cfg[k])
    return out


def _blur(x, k: int):
    import torch
    k = max(1, int(k) | 1)
    pad = k // 2
    w = torch.ones((x.shape[1], 1, k, k), device=x.device, dtype=x.dtype) / (k * k)
    return torch.nn.functional.conv2d(torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="replicate"), w,
                                      groups=x.shape[1])


def _gaussian_blur(x, sigma: float = 1.0):
    import torch
    r = max(1, int(math.ceil(sigma * 2)))
    xs = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    k1 = torch.exp(-(xs ** 2) / (2 * sigma * sigma))
    k1 = (k1 / k1.sum()).to(x.dtype)
    c = x.shape[1]
    xp = torch.nn.functional.pad(x, (r, r, r, r), mode="reflect" if min(x.shape[-2:]) > r else "replicate")
    y = torch.nn.functional.conv2d(xp, k1.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return torch.nn.functional.conv2d(y, k1.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def modify_latents(x, cfg: dict, progress: float, generator=None):
    """Returns modified latents (B, C, H, W). `progress` is 0..1 through the sampling schedule."""
    import torch
    dt = x.dtype
    y = x.float()
    # sharpness (unsharp mask in latent space); the effect fades out towards the end like the original
    m = cfg["sharpness_multiplier"]
    if m:
        blurred = _gaussian_blur(y, 1.0) if cfg["sharpness_method"] != "gaussian" else _gaussian_blur(y, 2.0)
        y = y + (y - blurred) * m * (1.0 - progress)
    # contrast around the per-channel mean
    m = cfg["contrast_multiplier"]
    if m:
        mean = y.mean(dim=(-2, -1), keepdim=True)
        y = mean + (y - mean) * (1.0 + m)
    # tonemapping of large magnitudes
    m = cfg["tonemap_multiplier"]
    if m:
        mag = y.norm(dim=1, keepdim=True)
        q = cfg["tonemap_percentile"] / 100.0
        top = torch.quantile(mag.flatten(1), min(max(q, 0.0), 1.0), dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
        if cfg["tonemap_method"] == "arctan":
            new = top * (2 / math.pi) * torch.atan(mag / top * (math.pi / 2))
        elif cfg["tonemap_method"] == "clamp":
            new = mag.clamp(max=top)
        else:  # reinhard
            norm = mag / top
            new = top * norm / (1 + norm)
        y = y + (y * (new / mag.clamp_min(1e-6)) - y) * m
    # combat CFG drift: pull the per-channel mean back to zero
    m = cfg["combat_cfg_drift"]
    if m:
        if cfg["combat_method"] == "median":
            c = y.flatten(2).median(dim=2).values[..., None, None]
        else:
            c = y.mean(dim=(-2, -1), keepdim=True)
        y = y - c * m
    # divisive normalisation (local contrast)
    m = cfg["divisive_norm_multiplier"]
    if m:
        size = max(3, min(int(cfg["divisive_norm_size"]), 2 * min(y.shape[-2:]) - 1))
        local = _blur(y.abs(), size).clamp_min(1e-4)
        y = y + (y / local * local.mean(dim=(-2, -1), keepdim=True) - y) * m
    # spectral modulation: clamp high-magnitude frequencies
    m = cfg["spectral_mod_multiplier"]
    if m:
        f = torch.fft.fftshift(torch.fft.fft2(y), dim=(-2, -1))
        mag = f.abs()
        thr = torch.quantile(mag.flatten(1), 1.0 - cfg["spectral_mod_percentile"] / 100.0, dim=1).view(-1, 1, 1, 1)
        if cfg["spectral_mod_mode"] == "soft_clamp":
            scale = torch.where(mag > thr, (thr / mag.clamp_min(1e-6)) ** 0.5, torch.ones_like(mag))
        else:
            scale = torch.where(mag > thr, thr / mag.clamp_min(1e-6), torch.ones_like(mag))
        f = f * (1 + (scale - 1) * m)
        y = torch.fft.ifft2(torch.fft.ifftshift(f, dim=(-2, -1))).real
    # extra noise (fades out)
    m = cfg["extra_noise_multiplier"]
    if m:
        n = torch.randn(y.shape, generator=generator, device=y.device if generator is None else generator.device,
                        dtype=torch.float32).to(y.device)
        if cfg["extra_noise_type"] == "uniform":
            n = (torch.rand_like(n) - 0.5) * 2 * math.sqrt(3)
        lp = int(cfg["extra_noise_lowpass"])
        if 0 < lp < 100:
            n = _gaussian_blur(n, max(0.5, (100 - lp) / 25))
        y = y + n * m * (1.0 - progress) * y.std()
    return y.to(dt)


def soft_mask_threshold(step: int, total: int, power: float) -> float:
    """Differential diffusion: pixels whose mask value is >= threshold are regenerated at this step."""
    p = (step + 1) / max(total, 1)
    return max(0.0, 1.0 - p) ** max(power, 1e-3)


def soft_inpaint_blend(latents, orig_noised, soft_mask, step: int, total: int, cfg: dict):
    """Keep generated latents where the soft mask exceeds the step threshold, original (noised) elsewhere."""
    import torch
    thr = soft_mask_threshold(step, total, cfg["mask_blend_power"])
    m = soft_mask.to(latents.dtype)
    # mask_blend_scale softens the hard threshold into a ramp; detail preservation sharpens it
    width = max(1e-3, cfg["mask_blend_scale"] / max(cfg["inpaint_detail_preservation"], 1e-3))
    keep = torch.clamp((m - thr) / width + 0.5, 0.0, 1.0)
    return latents * keep + orig_noised.to(latents.dtype) * (1 - keep)


def composite_mask(orig_rgb, gen_rgb, soft_mask, cfg: dict):
    """Final pixel composite: soft mask, boosted where generated and original differ (soft inpainting step 2).
    Arrays are numpy float32 HxWx3 in 0..1 and HxW mask."""
    import numpy as np
    diff = np.abs(gen_rgb - orig_rgb).mean(axis=2)
    thr = cfg["composite_difference_threshold"]
    contrast = cfg["composite_difference_contrast"]
    boost = np.clip((diff / max(thr, 1e-6)) ** contrast, 0.0, 1.0)
    infl = cfg["composite_mask_influence"]
    m = soft_mask * (1 - infl) + np.maximum(soft_mask, boost) * infl if infl else soft_mask
    m = np.clip(m, 0.0, 1.0)[..., None]
    return gen_rgb * m + orig_rgb * (1 - m)


__all__ = ["modify_latents", "soft_inpaint_blend", "soft_mask_threshold", "composite_mask",
           "normalize_latent_modifier", "normalize_soft_inpaint", "LATENT_MODIFIER_DEFAULTS", "SOFT_INPAINT_DEFAULTS"]
