"""NegPiP for CLIP-based SD pipelines; adapted from hako-mikan/sd-webui-negpip (AGPL-3.0).

Negative terms are encoded separately with absolute weights, appended as context, and have their value projections
negated. Keys stay positive. A private final channel transports signs through CFG/PAG and scheduled embedding
concatenation; it is removed before native attention processing. Torch imports are local for the dry-run engine.
"""
from __future__ import annotations

from . import prompt
from .protocol import EngineError, UNSUPPORTED

CONTEXT_KEYS = ("prompt_embeds", "negative_prompt_embeds")


def escape_text(text: str) -> str:
    text = text.replace("\\", "\\\\")
    for char in "()[]":
        text = text.replace(char, "\\" + char)
    return text


def extract(text: str) -> tuple[str, list[tuple[str, float]]]:
    parsed = prompt.parse_prompt_attention(text)
    targets = [(t, float(w)) for t, w in parsed if t and w < 0 and not (t == "BREAK" and w == -1)]
    if not targets:
        return text, []
    parts = []
    for text, weight in parsed:
        if text == "BREAK" and weight == -1:
            parts.append(" BREAK ")
        elif weight >= 0:
            literal = escape_text(text)
            parts.append(literal if weight == 1 else f"({literal}:{weight:g})")
    return "".join(parts), targets


def _positions(pipe, text: str, length: int) -> list[int]:
    tokenizers = [t for t in (getattr(pipe, "tokenizer", None), getattr(pipe, "tokenizer_2", None)) if t is not None]
    chunks = [(t, prompt._chunks_for(pipe, t, escape_text(text))) for t in tokenizers]
    positions = []
    for i in range(length // 77):
        ends = []
        for tokenizer, blocks in chunks:
            tokens = blocks[min(i, len(blocks) - 1)][0]
            ends.append(tokens.index(tokenizer.eos_token_id, 1))
        positions.extend(range(i * 77 + 1, i * 77 + max(ends) + 1))
    return positions


def _mark(tensor, sign=1):
    import torch
    flags = torch.full((*tensor.shape[:-1], 1), sign, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, flags], dim=-1)


def _pad(tensor, length):
    import torch
    if tensor.shape[1] == length:
        return tensor
    return torch.cat([tensor, tensor.new_zeros(tensor.shape[0], length - tensor.shape[1], tensor.shape[2])], dim=1)


def encode_photo_target(pipe, text, weight, *, clip_skip, dtype):
    """Use the native SDXL encoder and token slice for one signed target."""
    import torch
    with torch.no_grad():
        word, _ = prompt.encode_sdxl(pipe, f'({escape_text(text)}:{weight:g})', clip_skip)
        indices = _positions(pipe, text, word.shape[1])
        return word[0, indices].to(dtype=dtype)


def encode(engine, pos: str, neg: str, gp, bs: int) -> dict:
    import torch
    pos_clean, pos_targets = extract(pos)
    neg_clean, neg_targets = extract(neg)
    if not pos_targets and not neg_targets:
        return engine._encode_plain(pos, neg, gp, bs)
    out = dict(engine._encode_plain(pos_clean, neg_clean, gp, bs))
    for key, targets in zip(CONTEXT_KEYS, (pos_targets, neg_targets)):
        pieces = [_mark(out[key])]
        for text, weight in targets:
            word = engine._encode_plain(f"({escape_text(text)}:{abs(weight):g})", "", gp, 1)["prompt_embeds"]
            indices = _positions(engine.pipe, text, word.shape[1])
            pieces.append(_mark(word[:, indices].repeat(bs, 1, 1).to(out[key].device, out[key].dtype), -1))
        out[key] = torch.cat(pieces, dim=1)
    length = max(out[k].shape[1] for k in CONTEXT_KEYS)
    for key in CONTEXT_KEYS:
        out[key] = _pad(out[key], length)
    out["_negpip"] = True
    return out


def align_variants(variants: list) -> list:
    """Keep callback tensor shapes stable when a scheduled term appears/disappears or crosses zero."""
    if not any(d.get("_negpip") for _, d in variants):
        return variants
    length = max(d[k].shape[1] for _, d in variants for k in CONTEXT_KEYS)
    cache = {}
    for _, d in variants:
        if id(d) not in cache:
            result = dict(d)
            for key in CONTEXT_KEYS:
                tensor = d[key] if d.get("_negpip") else _mark(d[key])
                result[key] = _pad(tensor, length)
            result["_negpip"] = True
            cache[id(d)] = result
    return [(step, cache[id(d)]) for step, d in variants]


def adjust(embeds, apply):
    import torch
    if not embeds.get("_negpip"):
        return apply(embeds)
    plain = {k: v for k, v in embeds.items() if k != "_negpip"}
    for key in CONTEXT_KEYS:
        plain[key] = plain[key][..., :-1]
    out = dict(apply(plain))
    for key in CONTEXT_KEYS:
        out[key] = torch.cat([out[key], embeds[key][..., -1:]], dim=-1)
    out["_negpip"] = True
    return out


def _mask(existing, signs, heads):
    import torch
    import torch.nn.functional as F
    keep = signs != 0
    padding = torch.zeros_like(signs).masked_fill(~keep, -torch.inf)
    if existing is None:
        return padding[:, None, :]
    if existing.dtype == torch.bool:
        existing = torch.zeros_like(existing, dtype=signs.dtype).masked_fill(~existing, -torch.inf)
    if existing.shape[-1] < signs.shape[1]:
        existing = F.pad(existing, (0, signs.shape[1] - existing.shape[-1]), value=0)
    if existing.ndim == 4:
        b, h, q, n = existing.shape
        existing = existing.expand(signs.shape[0], heads, q, n).reshape(signs.shape[0] * heads, q, n)
    if existing.shape[0] == signs.shape[0] * heads and heads > 1:
        padding = padding.repeat_interleave(heads, dim=0)
    if existing.ndim == 3:
        padding = padding[:, None, :]
    return existing + padding.to(existing.device, existing.dtype)


def _forward(original, module):
    def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        import torch
        context = encoder_hidden_states[0] if isinstance(encoder_hidden_states, tuple) else encoder_hidden_states
        if not torch.is_tensor(context) or context.shape[-1] != module.to_k.in_features + 1:
            return original(hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, **kwargs)
        signs = context[..., -1]
        clean = context[..., :-1]
        encoder_hidden_states = (clean, *encoder_hidden_states[1:]) if isinstance(encoder_hidden_states, tuple) else clean
        attention_mask = _mask(attention_mask, signs, module.heads)
        handles = []

        def values(_m, _inputs, output):
            return output * signs.to(output.device, output.dtype)[..., None]

        def fused_values(_m, _inputs, output):
            key, value = output.chunk(2, dim=-1)
            return torch.cat([key, values(_m, _inputs, value)], dim=-1)

        try:
            handles.append(module.to_v.register_forward_hook(values))
            if getattr(module, "to_kv", None) is not None:
                handles.append(module.to_kv.register_forward_hook(fused_values))
            return original(hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
    return forward


class Rebindable:
    """Own a NegPiP forward wrapper across Accelerate hook replacement."""
    def __init__(self, pipe):
        self.pipe = pipe
        self._remove = None

    @property
    def active(self):
        return self._remove is not None

    def install(self):
        if self._remove is not None:
            raise RuntimeError('NegPiP already installed')
        self._remove = install(self.pipe)
        return self

    def remove(self):
        if self._remove is not None:
            remove, self._remove = self._remove, None
            remove()


def install(pipe):
    """Wrap native cross-attention without replacing processor modules or changing offload parameter names."""
    roots = [pipe.unet]
    if getattr(pipe, "controlnet", None) is not None:
        roots.append(pipe.controlnet)
    saved, seen = [], set()
    missing = object()

    def remove():
        while saved:
            module, previous = saved.pop()
            if previous is missing:
                del module.forward
            else:
                module.forward = previous

    try:
        for root in roots:
            projection = getattr(getattr(root, "config", None), "encoder_hid_dim_type", None)
            if projection not in (None, "ip_image_proj"):
                raise EngineError(UNSUPPORTED, f"NegPiP cannot use the pre-attention projection '{projection}'")
            for module in root.modules():
                if getattr(module, "is_cross_attention", False) and id(module) not in seen:
                    seen.add(id(module))
                    saved.append((module, module.__dict__.get("forward", missing)))
                    module.forward = _forward(module.forward, module)
        if not saved:
            raise EngineError(UNSUPPORTED, "NegPiP found no supported cross-attention layers")
    except BaseException:
        remove()
        raise
    return remove
