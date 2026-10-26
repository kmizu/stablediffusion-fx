"""feynlee Latent Upscale behaviour (MIT): encode first, interpolate latents, denoise full steps.

No learned upscaler or external model. Schedule mathematics is supplied by sdfx.sigmas.
"""
from __future__ import annotations

import math
from .protocol import EngineError, INVALID_PARAMS, UNSUPPORTED

METHODS = ('bilinear', 'bilinear-antialiased', 'bicubic', 'bicubic-antialiased', 'area', 'nearest', 'nearest-exact')
SCHEDULES = {'simple': 'Simple', 'normal': 'Normal', 'karras': 'Karras', 'exponential': 'Exponential',
             'polyexponential': 'Polyexponential', 'automatic': None}
DEFAULTS = {'method': 'nearest-exact', 'scheduler': 'simple'}


def normalize(raw):
    c = dict(DEFAULTS, **raw)
    if not all(isinstance(c[k], str) for k in DEFAULTS):
        raise EngineError(INVALID_PARAMS, 'Latent Upscale method and scheduler must be strings')
    if c['method'] in ('linear', 'trilinear'):
        raise EngineError(UNSUPPORTED, 'Latent Upscale: linear/trilinear require 3-D/5-D tensors; image latents are 4-D. '
                          'Choose bilinear, bicubic, area or nearest instead.')
    if c['method'] not in METHODS or c['scheduler'] not in SCHEDULES:
        raise EngineError(INVALID_PARAMS, 'Latent Upscale: unknown interpolation method or scheduler')
    return {k: c[k] for k in DEFAULTS}


def schedule(c, current):
    return SCHEDULES[c['scheduler']] or current


def full_steps(steps, strength):
    strength = float(strength)
    if not math.isfinite(strength) or not 0 < strength <= 1:
        raise EngineError(INVALID_PARAMS, 'Latent Upscale: denoising strength must be greater than zero and at most one')
    steps = max(1, int(steps))
    # Check the schedule bound before division: positive subnormal strengths can overflow
    # steps / strength to infinity before math.ceil has a chance to enforce the count limit.
    if strength < steps / 10000:
        raise EngineError(INVALID_PARAMS, 'Latent Upscale: denoising strength is too small for this step count (schedule limit 10000)')
    count = max(steps, int(math.ceil(steps / strength - 1e-9)))
    if count > 10000:
        raise EngineError(INVALID_PARAMS, 'Latent Upscale: denoising strength is too small for this step count (schedule limit 10000)')
    # The pipeline chooses the schedule suffix via int(count * strength). Snap into that same
    # integer bin so float truncation cannot lose a step; selected timesteps/sigmas stay unchanged.
    # The scalar can shift by up to one schedule interval. GenerationParams keeps the user's
    # original strength for other hooks; this effective value is only sent to the pipeline slicer.
    effective = min(1., (steps + 1e-6) / count)
    return count, effective, steps


def resize(latents, width, height, method):
    import torch.nn.functional as F
    normalize({'method': method})
    if latents.ndim != 4 or width < 8 or height < 8 or width % 8 or height % 8:
        raise EngineError(INVALID_PARAMS, 'Latent Upscale needs 4-D image latents and target dimensions divisible by 8')
    mode = method.removesuffix('-antialiased')
    kw = {'antialias': method.endswith('-antialiased'), 'align_corners': False} if mode in ('bilinear', 'bicubic') else {}
    return F.interpolate(latents.float(), size=(height // 8, width // 8), mode=mode, **kw).to(latents.dtype)


def encode(pipe, source, width, height, method, generators, device, dtype, mask=None):
    """Seeded VAE posterior samples at source size; optional mask creates conditioning latents.

    VAE precision is restored even on failure. The same per-image generators continue into diffusion noise.
    ``mask`` is a source-sized PIL mask; white is repainted, as in native diffusers inpainting.
    """
    import torch
    import numpy as np
    vae = pipe.vae
    original_dtype = vae.dtype
    upcast = original_dtype == torch.float16 and getattr(vae.config, 'force_upcast', False)
    try:
        if upcast:
            vae.to(torch.float32)
        image = pipe.image_processor.preprocess(source).to(device=device, dtype=vae.dtype)
        if mask is not None:
            from PIL import Image
            m = mask.convert('L').resize((image.shape[-1], image.shape[-2]), Image.Resampling.NEAREST)
            mask_tensor = torch.from_numpy(np.asarray(m).copy()).to(device=device, dtype=image.dtype)[None, None] / 255
            image = image * (mask_tensor < .5)
        with torch.no_grad():
            posterior = vae.encode(image).latent_dist
            samples = torch.cat([posterior.sample(generator=g) for g in generators])
            mean = getattr(vae.config, 'latents_mean', None)
            std = getattr(vae.config, 'latents_std', None)
            if mean is not None and std is not None:
                means = torch.tensor(mean, device=samples.device, dtype=samples.dtype)[None, :, None, None]
                stds = torch.tensor(std, device=samples.device, dtype=samples.dtype)[None, :, None, None]
                samples = (samples - means) / stds
            samples = samples * vae.config.scaling_factor
            return resize(samples.to(device=device, dtype=dtype), width, height, method)
    finally:
        if upcast:
            vae.to(original_dtype)
