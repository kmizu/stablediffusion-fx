"""Immutable Hires Fix Tweaks seed planning; no model or global RNG mutation.

Behavior reference: w-e-w/sd-webui-hires-fix-tweaks (AGPL-3.0),
28902f742a795afc31d86641ad85466afdd280e6. See THIRD_PARTY_NOTICES.md.
"""
from dataclasses import dataclass
from contextlib import contextmanager
from types import FunctionType, MethodType
import math
import secrets
from .protocol import EngineError, INVALID_PARAMS, UNSUPPORTED


def wire_seed(seed):
    """Keep effective seed facts exact in JSON clients backed by IEEE754 numbers."""
    return str(seed) if seed > 9007199254740991 else seed


@contextmanager
def prepare_noise(pipe, factory):
    """Override only this pipeline's latent-noise draw, including inpaint tuples.

    A private globals dictionary avoids touching diffusers module state or other
    pipeline instances. Restore exact instance ownership even after cancellation.
    """
    if factory is None:
        yield
        return
    method = getattr(pipe, 'prepare_latents', None)
    fn = getattr(method, '__func__', None)
    if (not isinstance(fn, FunctionType) or 'randn_tensor' not in fn.__code__.co_names
            or 'randn_tensor' not in fn.__globals__):
        raise EngineError(UNSUPPORTED, 'Hires custom noise requires a compatible prepare_latents')
    owned = 'prepare_latents' in vars(pipe)
    previous = vars(pipe).get('prepare_latents')
    namespace = dict(fn.__globals__, randn_tensor=factory)
    replacement = FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__)
    replacement.__kwdefaults__ = fn.__kwdefaults__
    pipe.prepare_latents = MethodType(replacement, pipe)
    try:
        yield
    finally:
        if owned:
            pipe.prepare_latents = previous
        else:
            delattr(pipe, 'prepare_latents')


def resized_noise(shape, seed, subseed, strength, width, height, draw, interpolate):
    """Draw seeded noise, vary its source size, then center crop/paste it.

    draw(seed, CHW) must be an independent deterministic draw. The untouched
    background uses the primary seed at the target size, matching seed resize.
    interpolate takes batched tensors, like the engine's existing slerp helper.
    """
    source = (shape[0], height // 8, width // 8) if width and height else tuple(shape)
    noise = draw(seed, source)
    if strength:
        noise = interpolate(strength, noise.unsqueeze(0), draw(subseed, source).unsqueeze(0))[0]
    if source == tuple(shape):
        return noise
    result = draw(seed, shape)
    dy, dx = (shape[1] - source[1]) // 2, (shape[2] - source[2]) // 2
    # Preserve source behavior for odd differences: symmetric crop drops the
    # extra edge rather than stretching noise or changing its random sequence.
    rows = source[1] if dy >= 0 else source[1] + 2 * dy
    cols = source[2] if dx >= 0 else source[2] + 2 * dx
    result[:, max(dy, 0):max(dy, 0) + rows, max(dx, 0):max(dx, 0) + cols] = (
        noise[:, max(-dy, 0):max(-dy, 0) + rows, max(-dx, 0):max(-dx, 0) + cols])
    return result


def _invalid(message):
    raise EngineError(INVALID_PARAMS, 'Hires Fix Tweaks: ' + message)


@dataclass(frozen=True)
class SeedPlan:
    seeds: tuple
    subseeds: tuple
    strength: float
    count: int
    resize_width: int
    resize_height: int

    def batch(self, start, size, repeat):
        if (any(type(v) is not int for v in (start, size, repeat))
                or start < 0 or size < 1 or start + size > len(self.seeds)
                or not 0 <= repeat < self.count):
            _invalid('invalid batch slice or repeat')
        increment = repeat if self.strength == 0 else 0
        return (tuple(x + increment for x in self.seeds[start:start + size]),
                tuple(x + repeat for x in self.subseeds[start:start + size]))


def seed_plan(config, base_seeds, base_subseeds, base_strength, random_seed=None):
    """Resolve random values once per job, before slicing into base batches.

    Zero inherits the corresponding full base list. A positive primary seed is
    held fixed when variation is active; subseeds always advance. Unlike the
    upstream typo, random subseeds depend on subseed=-1, not primary seed=-1.
    """
    if not isinstance(config, dict):
        _invalid('configuration must be an object')
    defaults = dict(batch_count=1, seed_enabled=False, seed=0, extras=False,
                    subseed=0, strength=0.0, resize_width=0, resize_height=0)
    c = {key: config.get(key, value) for key, value in defaults.items()}
    # JSON consumers using IEEE754 transport large integer seeds as decimal text.
    import re
    for key in ('seed', 'subseed'):
        if isinstance(c[key], str) and re.fullmatch(r'-?[0-9]{1,19}', c[key]):
            c[key] = int(c[key])
    for key in ('seed_enabled', 'extras'):
        if type(c[key]) is not bool:
            _invalid(key + ' must be boolean')
    for key, low, high in [('batch_count', 1, 16), ('seed', -1, 2**63 - 1),
                           ('subseed', -1, 2**63 - 1), ('resize_width', 0, 16384),
                           ('resize_height', 0, 16384)]:
        if type(c[key]) is not int or not low <= c[key] <= high:
            _invalid(key + ' is outside its integer range')
    for key in ('resize_width', 'resize_height'):
        if c[key] % 8:
            _invalid(key + ' must be a multiple of 8')
    for value in (c['strength'], base_strength):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            _invalid('variation strength must be finite in [0, 1]')
    seeds, subs = tuple(base_seeds), tuple(base_subseeds)
    if not seeds or len(seeds) != len(subs):
        _invalid('base seed lists must have matching nonzero lengths')
    if any(type(v) is not int or v < 0 for v in seeds + subs):
        _invalid('base seeds must already be resolved nonnegative integers')
    strength = base_strength
    width = height = 0
    if c['seed_enabled']:
        strength = c['strength'] if c['extras'] else 0.0
        random_seed = random_seed or (lambda: secrets.randbelow(4294967294))
        if c['seed'] != 0:
            seed = random_seed() if c['seed'] == -1 else c['seed']
            seeds = tuple(seed + (i if strength == 0 else 0) for i in range(len(seeds)))
        if c['extras']:
            if c['subseed'] != 0:
                sub = random_seed() if c['subseed'] == -1 else c['subseed']
                subs = tuple(sub + i for i in range(len(subs)))
            if c['resize_width'] and c['resize_height']:
                width, height = c['resize_width'], c['resize_height']
    increment = c['batch_count'] - 1
    if (any(s + (increment if strength == 0 else 0) > 2**63 - 1 for s in seeds)
            or any(s + increment > 2**63 - 1 for s in subs)):
        _invalid('batch or repeat seeds exceed the signed 64-bit range')
    return SeedPlan(seeds, subs, float(strength), c['batch_count'], width, height)
