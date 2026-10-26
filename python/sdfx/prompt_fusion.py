"""Standalone Prompt Fusion adapter; upstream parser/curve port lives in sdfx.fusion (MIT).

Sampling uses upstream's zero-based interpolation clock. Native scheduling uses one-based
inclusive endpoints; the conversion happens only at the leaf boundary.
"""
from __future__ import annotations

import math
import re


def active(text):
    """Ordinary prompts bypass the Fusion parser to preserve exact native tokenization."""
    return bool(re.search(
        r'(?<!\\)\$[A-Za-z_]|(?<!\\)\[[^\]]*:[^\]]*,[^\]]*\]|'
        r'(?<!\\)\[[^\]]*:\s*(?:linear|bezier|catmull|mean)\s*\]|'
        r'(?<!\\)\[[^\]]*\|[^\]]*:[^\]]+\]|(?<!\\)\([^()]*:[^()]*,[^()]*\)', text))


def builder(text, steps, is_hires=False):
    from .fusion.prompt_parser import parse_prompt
    from .fusion.interpolation_tensor import InterpolationTensorBuilder
    b = InterpolationTensorBuilder()
    parse_prompt(text).extend_tensor(b, (0, steps), steps, {}, is_hires, True)
    if len(b.get_prompt_database()) > 512:
        raise ValueError('Prompt Fusion expands to more than 512 conditioning prompts')
    return b


def leaf_texts(text, total):
    """Render upstream zero-based leaf editing through the native parser.

    Fusion emits step-zero attention windows, which the ordinary native schedule intentionally
    omits. Shift editing boundaries by one here, without changing ordinary A1111 scheduling.
    Alternation retains the native first-option-on-first-step convention.
    """
    from . import prompt_schedule as p
    try:
        tree = p._Parser(text).start()
    except p._Fail:
        return [text] * total
    def collect(n):
        if isinstance(n, p.Sched):
            v = float(n.when)
            n.step = int(v * total if 0 < v < 1 else v) + 1
    p._walk(tree, collect)
    return [p._text(tree, step) for step in range(1, total + 1)]


def sample(text, steps, encode, scale=0., epsilon=.0001, origins=None, is_hires=False):
    """Evaluate a full Fusion expression using a native text-conditioning callback.

    The callback returns a dictionary or [(native_end_step, dictionary), ...].
    It must encode one image (batch size 1), with positive keys even for the negative branch.
    """
    import torch
    from .fusion.interpolation_tensor import (
        DictCondWrapper, TensorCondWrapper, ScheduledPromptConditioning, InterpolationParams)
    if not math.isfinite(scale) or not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError('Prompt Fusion requires finite slerp scale and epsilon between 0 and 1')
    b = builder(text, steps, is_hires)

    def wrap(d):
        if 'pooled_prompt_embeds' in d:
            return DictCondWrapper({'crossattn': d['prompt_embeds'][0], 'vector': d['pooled_prompt_embeds'][0]})
        return TensorCondWrapper(d['prompt_embeds'][0])

    encode_empty = encode('')
    empty = wrap(encode_empty if isinstance(encode_empty, dict) else encode_empty[0][1])
    conds = []
    for leaf in b.get_prompt_database():
        encoded = encode(leaf)
        sch = [(steps, encoded)] if isinstance(encoded, dict) else encoded
        conds.append([ScheduledPromptConditioning(end - 1, wrap(d)) for end, d in sch])
    tensor = b.build(conds, empty)
    result = []
    for step in range(steps):
        origin = wrap(origins[step][1]) if origins is not None else empty
        cond = tensor.interpolate(InterpolationParams(step / steps, step, steps, scale, epsilon), origin, empty)
        raw = cond.original_cond
        d = ({'prompt_embeds': raw['crossattn'].unsqueeze(0), 'pooled_prompt_embeds': raw['vector'].unsqueeze(0)}
             if isinstance(raw, dict) else {'prompt_embeds': raw.unsqueeze(0)})
        if not all(torch.isfinite(v).all() for v in d.values()):
            raise ValueError('Prompt Fusion produced non-finite conditioning; check weights and curve points')
        result.append((step + 1, d))
    return result


def encode(engine, pos, neg, gp, bs, steps, hires_total=None, gradient_total=None):
    """Build native scheduled embeddings, composing with the existing scheduling/encoding path."""
    import torch
    from dataclasses import replace
    from .engine import ScheduledEmbeds
    from . import hooks
    from .protocol import EngineError, INVALID_PARAMS, UNSUPPORTED
    cfg = hooks.hook_config(gp.hooks, 'prompt_fusion', {})
    if hooks.hook_config(gp.hooks, 'photomaker', {}) is not None:
        raise EngineError(UNSUPPORTED, "Prompt Fusion cannot use PhotoMaker's text-only encoder")
    sub = replace(gp, hooks={k: v for k, v in gp.hooks.items() if k != 'prompt_fusion'})
    total = max(1, gradient_total or hires_total or steps)
    cache = {}

    def leaf(text, negative=False):
        key = (text, negative)
        if key not in cache:
            out = []
            rendered_cache = {}
            for end, rendered in enumerate(leaf_texts(text, total), 1):
                if rendered not in rendered_cache:
                    rendered_cache[rendered] = engine._encode_scheduled('' if negative else rendered,
                        rendered if negative else '', sub, 1, steps, hires_total, gradient_total)
                enc = rendered_cache[rendered]
                sch = getattr(enc, 'schedule', [(total, enc)])
                d = next((v for e, v in sch if end <= e), sch[-1][1])
                if d.get('_negpip'):
                    raise EngineError(UNSUPPORTED, 'Prompt Fusion interpolation with NegPiP negative terms is unsupported')
                if d.get('_neutral'):
                    raise EngineError(UNSUPPORTED, 'Use Neutral Prompt operators outside Prompt Fusion expressions')
                keys = [('prompt_embeds', 'negative_prompt_embeds' if negative else 'prompt_embeds'),
                        ('pooled_prompt_embeds', 'negative_pooled_prompt_embeds' if negative else 'pooled_prompt_embeds')]
                out.append((end, {target: d[source] for target, source in keys if source in d}))
            cache[key] = out
        return cache[key]

    try:
        scale, epsilon = float(cfg.get('slerp_scale', 0)), float(cfg.get('slerp_epsilon', .0001))
        ns = sample(neg, total, lambda t: leaf(t, True), scale, epsilon, is_hires=hires_total is not None)
        ps = sample(pos, total, leaf, scale, epsilon,
                    origins=ns if hooks.coerce_bool(cfg.get('negative_origin', True), True) else None,
                    is_hires=hires_total is not None)
    except (ValueError, KeyError, AssertionError, IndexError, ZeroDivisionError, RecursionError, OverflowError) as exc:
        raise EngineError(INVALID_PARAMS, f'Prompt Fusion: {exc or type(exc).__name__}') from exc
    longest = max(d['prompt_embeds'].shape[1] for _, d in ps + ns)
    empty = leaf('')[0][1]['prompt_embeds']

    def pad(d):
        t = d['prompt_embeds']
        missing = longest - t.shape[1]
        if missing:
            fill = torch.zeros_like(empty) if not bool(t.abs().any()) else empty
            d['prompt_embeds'] = torch.cat([t, fill.repeat(1, missing // fill.shape[1], 1)], dim=1)
        return {k: v.repeat(bs, *([1] * (v.ndim - 1))).to(engine.dtype) for k, v in d.items()}

    variants = [(e, {**pad(p), **{'negative_' + k: v for k, v in pad(n).items()}})
                for (e, p), (_, n) in zip(ps, ns)]
    out = ScheduledEmbeds(variants[0][1])
    out.schedule = variants
    return out
