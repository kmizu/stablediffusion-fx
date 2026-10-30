"""Source-faithful post-image refinement, not reuse of a first-pass latent.

Behavior reference: Amadeus-AI/img2img-hires-fix, MIT,
b8930527a8cc606dd79dd54ed23bc967314d1859. See THIRD_PARTY_NOTICES.md.
"""
from dataclasses import dataclass, replace
import math
from .protocol import EngineError, INVALID_PARAMS


@dataclass(frozen=True)
class Config:
    ratio: float = 2.
    width: int = 0
    height: int = 0
    steps: int = 0
    upscaler: str = 'R-ESRGAN 4x+'
    prompt: str = ''
    negative_prompt: str = ''
    denoise: float = .33
    sampler: str = 'Use same sampler'
    scheduler: str = 'Use same scheduler'
    cfg: float = 0.

    def target(self, width, height):
        w, h = ((self.width, self.height) if self.width and self.height else
                (int(width * self.ratio), int(height * self.ratio)))
        w, h = max(64, w // 8 * 8), max(64, h // 8 * 8)
        if w > 16384 or h > 16384:
            raise EngineError(INVALID_PARAMS, 'img2img Hires Fix output exceeds 16384 pixels per side')
        return w, h


def normalize(values):
    if not isinstance(values, dict):
        raise EngineError(INVALID_PARAMS, 'img2img Hires Fix configuration must be an object')
    defaults = Config()
    data = {name: values.get(name, getattr(defaults, name)) for name in defaults.__dataclass_fields__}
    for key, high in (('ratio', 4), ('denoise', 1), ('cfg', 30)):
        v = data[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= high:
            raise EngineError(INVALID_PARAMS, f'img2img Hires Fix invalid {key}')
    for key, high in (('width', 2048), ('height', 2048), ('steps', 25)):
        v = data[key]
        if type(v) is not int or not 0 <= v <= high or (key != 'steps' and v % 8):
            raise EngineError(INVALID_PARAMS, f'img2img Hires Fix invalid {key}')
    for key in ('prompt', 'negative_prompt', 'sampler', 'scheduler', 'upscaler'):
        if not isinstance(data[key], str):
            raise EngineError(INVALID_PARAMS, f'img2img Hires Fix invalid {key}')
        data[key] = data[key].strip()
    if not data['upscaler'] or data['upscaler'].startswith('Latent'):
        raise EngineError(INVALID_PARAMS, 'img2img Hires Fix requires a pixel upscaler')
    if not data['sampler'] or not data['scheduler']:
        raise EngineError(INVALID_PARAMS, 'img2img Hires Fix sampler and scheduler must be specified')
    if data['ratio'] == 0 and not (data['width'] and data['height']):
        raise EngineError(INVALID_PARAMS, 'img2img Hires Fix needs a positive ratio or both target dimensions')
    return Config(**data)


def effective(config, params):
    return replace(params, prompt=config.prompt or params.prompt,
        negative_prompt=config.negative_prompt or params.negative_prompt,
        steps=config.steps or params.steps, cfg_scale=config.cfg or params.cfg_scale,
        sampler_name=params.sampler_name if config.sampler == 'Use same sampler' else config.sampler,
        scheduler=params.scheduler if config.scheduler == 'Use same scheduler' else config.scheduler)


def refine(engine, params, req, ctx, config, image, seed, index, total, pag=None, controlnet=None,
           restore_params=None, lora_hashes=None):
    """Refine a completed image once, preserving the native first-pass pipeline."""
    from PIL import Image
    from . import upscalers
    from .params import img2img_steps
    from .hooks import hook_config
    ctx.check_interrupt()
    width, height = config.target(*image.size)
    source = upscalers.upscale_to(image, width, height, config.upscaler,
        req.upscaler_paths.get(config.upscaler), engine.device, engine.dtype, ctx.check_interrupt)
    if config.denoise == 0:
        return source
    sub = effective(config, params)
    sub = replace(sub, width=width, height=height, batch_size=1, n_iter=1, seed=seed, img2img=None,
        hires=replace(sub.hires, enabled=False),
        hooks={k: v for k, v in sub.hooks.items() if k not in
            ('img2img_hires_fix', 'latent_upscale', 'img2imgalt', 'soft_inpainting', 'inpaint_difference')})
    switch = req.img2img_hires_loras is not None and hook_config(sub.hooks, 'composable_lora', {}) is None
    if switch:
        sub = replace(sub, loras=req.img2img_hires_loras)
    steps, strength, actual = img2img_steps(sub.steps, config.denoise)
    try:
        if switch:
            hashes = engine._apply_loras(sub, req, ctx)
            if lora_hashes is not None:
                lora_hashes.update(hashes)
            if getattr(ctx, 'resource_hashes', None) is not None:
                ctx.resource_hashes.update({f'lora:{name}': value for name, value in hashes.items()})
        embeds = engine._adjust_embeds(sub, engine._encode_scheduled(sub.prompt, sub.negative_prompt,
            sub, 1, sub.steps, gradient_total=actual), ctx)
        task = 'inpaint' if engine.kind.endswith('inpaint') else 'img2img'
        images, _ = engine._run(task, sub, req, ctx, sampler=sub.sampler_name, schedule=sub.scheduler,
            cfg_scale=sub.cfg_scale, embeds=embeds, steps=steps, w=width, h=height, seeds=[seed],
            pass_='img2img_hires', image_idx=index, images_total=total, image=[source],
            mask=[Image.new('L', (width, height), 255)] if task == 'inpaint' else None,
            strength=strength, expected_steps=actual, cn=controlnet, pag=pag, hook_seed=seed,
            conditioning_denoise=config.denoise)
        return images[0]
    finally:
        if switch:
            engine._apply_loras(restore_params if restore_params is not None else params, req, ctx)
