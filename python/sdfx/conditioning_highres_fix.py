"""Mask-conditioning strength modes, independent of pipeline implementation."""

import math
from contextlib import contextmanager

MODES = ('disabled', 'empty', 'lowest', 'low', 'medium', 'high', 'highest', 'full')


def condition_pixels(image, mask, weight):
    if image.shape[-2:] != mask.shape[-2:]:
        raise ValueError('conditioning image and mask dimensions differ')
    selected = mask >= .5
    # Boolean tensor times a Python float otherwise promotes FP16/BF16 image
    # conditioning to FP32, which FP16-safe VAEs do not automatically recast.
    selected = selected.to(dtype=image.dtype) if hasattr(selected, 'to') else selected.astype(image.dtype, copy=False)
    return image * (1. - selected * weight)


def source_denoising(pass_name, params, effective, requested=None):
    if requested is not None:
        return requested
    if pass_name == 'hires':
        return params.hires.denoising_strength
    if pass_name == 'base':
        return params.img2img.denoising_strength if params.img2img is not None else None
    return effective


def resolve_weight(mode, denoising):
    if mode not in MODES:
        raise ValueError('unknown conditioning highres mode')
    if mode == 'disabled' or denoising is None:
        return None
    if isinstance(denoising, bool) or not isinstance(denoising, (int, float)):
        raise ValueError('denoising strength must be a number between 0 and 1')
    d = float(denoising)
    if not math.isfinite(d) or not 0 <= d <= 1:
        raise ValueError('denoising strength must be a number between 0 and 1')
    return {'empty': 1., 'lowest': 1. - d / 4., 'low': 1. - d / 2.,
            'medium': .5, 'high': d / 2., 'highest': d / 4., 'full': 0.}[mode]


@contextmanager
def conditioning_scope(pipe, weight):
    """Override only pre-VAE masked-image conditioning for one pipeline call."""
    if weight is None or weight == 1.:
        yield
        return
    processor = pipe.image_processor
    original_preprocess = processor.preprocess
    original_prepare = pipe.prepare_mask_latents
    processor_own = processor.__dict__.get('preprocess')
    prepare_own = pipe.__dict__.get('prepare_mask_latents')
    had_processor = 'preprocess' in processor.__dict__
    had_prepare = 'prepare_mask_latents' in pipe.__dict__
    captured = []
    def preprocess(*args, **kwargs):
        result = original_preprocess(*args, **kwargs)
        captured.append(result)
        return result
    def prepare(mask, masked_image, *args, **kwargs):
        from .protocol import EngineError, UNSUPPORTED
        if len(captured) != 1 or getattr(captured[0], 'ndim', None) != 4 or captured[0].shape[1] != 3:
            raise EngineError(UNSUPPORTED, 'conditioning mask weight requires a pixel-space image')
        if masked_image is None or getattr(masked_image, 'ndim', None) != 4 or masked_image.shape[1] != 3:
            raise EngineError(UNSUPPORTED, 'conditioning mask weight cannot replace precomputed image latents')
        image = captured[0]
        adjusted = condition_pixels(image, mask, weight)
        return original_prepare(mask, adjusted, *args, **kwargs)
    try:
        processor.preprocess = preprocess
        pipe.prepare_mask_latents = prepare
        yield
    finally:
        if had_processor:
            processor.preprocess = processor_own
        elif 'preprocess' in processor.__dict__:
            del processor.preprocess
        if had_prepare:
            pipe.prepare_mask_latents = prepare_own
        elif 'prepare_mask_latents' in pipe.__dict__:
            del pipe.prepare_mask_latents
