"""A1111 prompt syntax: attention weights, BREAK, 75-token chunking -> prompt embeddings.

The parser and chunker are pure python. The `encode_*` functions use torch + a diffusers pipeline.
"""
from __future__ import annotations

import re
from typing import Callable, Optional, Sequence

re_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:\s*(x?[+-]?[.\d]+)\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)
re_extra_networks = re.compile(r"<(?:lora|lyco|hypernet):[^>]+>")
re_prompt_editing = re.compile(r"\[[^\[\]]*:[^\[\]]*\]|\[[^\[\]]*\|[^\[\]]*\]")
re_and = re.compile(r"\bAND\b")


def parse_prompt_attention(text: str) -> list[list]:
    """Port of A1111/reForge modules/prompt_parser.parse_prompt_attention.

    Returns [[text, weight], ...]; BREAK markers are returned as ["BREAK", -1].
    """
    res: list[list] = []
    round_brackets: list[int] = []
    square_brackets: list[int] = []
    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        tok = m.group(0)
        weight = m.group(1)
        if tok.startswith("\\"):
            res.append([tok[1:], 1.0])
        elif tok == "(":
            round_brackets.append(len(res))
        elif tok == "[":
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            if weight.lower().startswith("x"):
                try:
                    val = float(weight[1:])
                except ValueError:
                    res.append([tok, 1.0])
                    continue
                start = round_brackets.pop()
                segment = res[start:]
                abs_val = abs(val)
                count = max(1, int(abs_val))
                tail = (abs_val % 1) + (abs_val >= 1)
                res[start:] = [[t, w] for _ in range(count) for t, w in segment]
                if tail > 1e-4 and segment:
                    for seg in res[-len(segment):]:
                        seg[1] *= tail
                if val < 0:
                    multiply_range(start, -1.0)
            else:
                try:
                    multiply_range(round_brackets.pop(), float(weight))
                except ValueError:
                    res.append([tok, 1.0])
        elif tok == ")" and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif tok == "]" and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, tok)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)
    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1
    return res


def strip_extra_networks(text: str) -> str:
    return re_extra_networks.sub("", text or "")


def unsupported_syntax_warnings(text: str) -> list[str]:
    w = []
    if re_and.search(text or ""):
        w.append("composable diffusion (AND) is not supported; treated as plain text")
    return w


def chunk_tokens(parsed: Sequence[Sequence], tokenize: Callable[[list[str]], list[list[int]]], id_start: int, id_end: int,
                 comma_token: Optional[int] = None, chunk_length: int = 75, comma_backtrack: int = 20) -> list[tuple[list[int], list[float]]]:
    """Port of A1111 tokenize_line: returns chunks of (77 token ids, 77 multipliers) incl. BOS/EOS.

    Padding uses id_end (A1111); callers replace post-EOS ids with a different pad id when needed.
    """
    texts = [t for t, _ in parsed]
    tokenized = tokenize(texts)
    chunks: list[tuple[list[int], list[float]]] = []
    cur_tokens: list[int] = []
    cur_mults: list[float] = []
    last_comma = -1

    def next_chunk():
        nonlocal cur_tokens, cur_mults, last_comma
        to_add = chunk_length - len(cur_tokens)
        if to_add > 0:
            cur_tokens += [id_end] * to_add
            cur_mults += [1.0] * to_add
        chunks.append(([id_start] + cur_tokens + [id_end], [1.0] + cur_mults + [1.0]))
        cur_tokens, cur_mults = [], []
        last_comma = -1

    for tokens, (text, weight) in zip(tokenized, parsed):
        if text == "BREAK" and weight == -1:
            next_chunk()
            continue
        for token in tokens:
            if comma_token is not None and token == comma_token:
                last_comma = len(cur_tokens)
            elif comma_backtrack != 0 and len(cur_tokens) == chunk_length and last_comma != -1 \
                    and len(cur_tokens) - last_comma <= comma_backtrack:
                bl = last_comma + 1
                reloc_t, reloc_m = cur_tokens[bl:], cur_mults[bl:]
                cur_tokens, cur_mults = cur_tokens[:bl], cur_mults[:bl]
                next_chunk()
                cur_tokens, cur_mults = reloc_t, reloc_m
            if len(cur_tokens) == chunk_length:
                next_chunk()
            cur_tokens.append(token)
            cur_mults.append(float(weight))

    if cur_tokens or not chunks:
        next_chunk()
    return chunks


def apply_pad(tokens: list[int], id_end: int, id_pad: int) -> list[int]:
    """A1111: everything after the first EOS becomes id_pad (OpenCLIP uses 0)."""
    if id_end == id_pad or id_end not in tokens[1:]:
        return list(tokens)
    idx = tokens.index(id_end, 1)
    return tokens[: idx + 1] + [id_pad] * (len(tokens) - idx - 1)


# --------------------------------------------------------------------------- torch encoding

class _TokInfo:
    def __init__(self, pipe, tokenizer):
        self.pipe = pipe
        self.tok = tokenizer
        self.id_start = tokenizer.bos_token_id
        self.id_end = tokenizer.eos_token_id
        pad = tokenizer.pad_token_id
        self.id_pad = pad if pad is not None else self.id_end
        ids = tokenizer(",", add_special_tokens=False).input_ids
        self.comma = ids[0] if len(ids) == 1 else None

    def tokenize(self, texts: list[str]) -> list[list[int]]:
        conv = getattr(self.pipe, "maybe_convert_prompt", None)
        out = []
        for t in texts:
            if conv is not None and t:
                try:
                    t = conv(t, self.tok)
                except Exception:
                    pass
            out.append(self.tok(t, add_special_tokens=False, truncation=False).input_ids if t else [])
        return out


def _chunks_for(pipe, tokenizer, text: str):
    info = _TokInfo(pipe, tokenizer)
    parsed = parse_prompt_attention(text)
    chunks = chunk_tokens(parsed, info.tokenize, info.id_start, info.id_end, info.comma)
    return [(apply_pad(t, info.id_end, info.id_pad), m) for t, m in chunks]


def _emphasize(z, mults):
    import torch
    m = torch.tensor(mults, dtype=z.dtype, device=z.device).reshape(1, -1, 1)
    original_mean = z.mean()
    z = z * m
    new_mean = z.mean()
    if float(new_mean) != 0.0:
        z = z * (original_mean / new_mean)
    return z


def _encode_chunk(te, tokens, layer_from_end: Optional[int], final_norm: bool, device):
    import torch
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    out = te(ids, output_hidden_states=True)
    if layer_from_end is None:
        z = out[0] if not hasattr(out, "last_hidden_state") else out.last_hidden_state
    else:
        z = out.hidden_states[-layer_from_end]
        if final_norm:
            z = te.text_model.final_layer_norm(z)
    pooled = getattr(out, "text_embeds", None)
    return z, pooled


def _te_device(pipe, te):
    dev = getattr(pipe, "_execution_device", None)
    if dev is None:
        dev = next(te.parameters()).device
    return dev


def sd_layer_choice(clip_skip: int, openclip: bool = False, num_layers: int = 12) -> Optional[int]:
    """Hidden-state index from the end (None = last_hidden_state) matching reForge modules_forge/forge_clip.py.

    SD1.x (CLIP_SD_15_L, minimal_clip_skip=1): clip_skip > 1 -> hidden_states[-clip_skip] + final norm.
    SD2.x (CLIP_SD_21_H, minimal_clip_skip=2): penultimate by default, hidden_states[-clip_skip] when > 2, + final norm.
    diffusers' single-file SD2 text encoder drops the last of the 24 layers (23 layers), so its last_hidden_state
    already is the original penultimate layer and every index shifts by one.
    """
    if not openclip:
        return clip_skip if clip_skip > 1 else None
    idx = max(int(clip_skip), 2)
    if num_layers == 23:
        idx -= 1
        return None if idx <= 1 else idx
    return idx


def sdxl_layer_choice(clip_skip: int) -> int:
    """reForge CLIP_SD_XL_L/G (minimal_clip_skip=2): penultimate for clip_skip 1..2, else hidden_states[-clip_skip];
    no final layer norm."""
    return max(int(clip_skip), 2)


def encode_sd15(pipe, text: str, clip_skip: int = 1, openclip: bool = False):
    """Returns prompt_embeds (1, 77*k, C) for SD1.x / SD2.x pipelines."""
    import torch
    te = pipe.text_encoder
    device = _te_device(pipe, te)
    chunks = _chunks_for(pipe, pipe.tokenizer, text)
    layer = sd_layer_choice(clip_skip, openclip, int(getattr(te.config, "num_hidden_layers", 12)))
    zs = []
    with torch.no_grad():
        for toks, mults in chunks:
            z, _ = _encode_chunk(te, toks, layer, layer is not None, device)
            zs.append(_emphasize(z, mults))
    return torch.cat(zs, dim=1)


def encode_sdxl(pipe, text: str, clip_skip: int = 1):
    """Returns (prompt_embeds (1, 77*k, 2048), pooled (1, 1280)) for SDXL pipelines."""
    import torch
    te1, te2 = pipe.text_encoder, pipe.text_encoder_2
    layer = sdxl_layer_choice(clip_skip)
    chunks1 = _chunks_for(pipe, pipe.tokenizer, text) if te1 is not None else None
    chunks2 = _chunks_for(pipe, pipe.tokenizer_2, text)
    n = len(chunks2) if chunks1 is None else max(len(chunks1), len(chunks2))
    pooled = None
    zs = []
    with torch.no_grad():
        for i in range(n):
            parts = []
            if chunks1 is not None:
                t1, m1 = chunks1[min(i, len(chunks1) - 1)]
                z1, _ = _encode_chunk(te1, t1, layer, False, _te_device(pipe, te1))
                parts.append(_emphasize(z1, m1))
            t2, m2 = chunks2[min(i, len(chunks2) - 1)]
            z2, p2 = _encode_chunk(te2, t2, layer, False, _te_device(pipe, te2))
            if pooled is None:
                pooled = p2
            z2 = _emphasize(z2, m2)
            parts = [p.to(z2.device, z2.dtype) for p in parts] + [z2]
            zs.append(torch.cat(parts, dim=-1))
    return torch.cat(zs, dim=1), pooled


def pad_to_match(a, b, empty):
    """A1111 pad_cond: append empty-prompt chunks to the shorter of a/b so both have equal length."""
    import torch
    la, lb, le = a.shape[1], b.shape[1], empty.shape[1]
    if la < lb:
        a = torch.cat([a] + [empty.to(a.device, a.dtype)] * ((lb - la) // le), dim=1)
    elif lb < la:
        b = torch.cat([b] + [empty.to(b.device, b.dtype)] * ((la - lb) // le), dim=1)
    return a, b
