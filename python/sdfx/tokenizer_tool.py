"""Inspect a pipeline's actual CLIP BPE vocabulary, without loading any model or tensor."""
from __future__ import annotations

import re

from .protocol import EngineError, INVALID_PARAMS, UNSUPPORTED


def inspect(pipe, params: dict, ctx=None) -> dict:
    text = params.get("text", "")
    mode = params.get("input_mode", "text")
    if not isinstance(text, str) or len(text) > 65536:
        raise EngineError(INVALID_PARAMS, "text must be a string of at most 65536 characters")
    if mode not in ("text", "ids"):
        raise EngineError(INVALID_PARAMS, "input_mode must be text or ids")
    ids = None
    if mode == "ids":
        if text.strip() and not re.fullmatch(r"\s*\d+\s*(?:,\s*\d+\s*)*", text):
            raise EngineError(INVALID_PARAMS, "IDs must be comma-separated nonnegative integers")
        try:
            ids = [int(x.strip()) for x in text.split(",")] if text.strip() else []
        except ValueError:
            raise EngineError(INVALID_PARAMS, "invalid integer ID")
    streams = []
    for name in ("tokenizer", "tokenizer_2"):
        tok = getattr(pipe, name, None)
        if tok is None:
            continue
        if ctx is not None:
            ctx.check_interrupt()
        streams.append(_stream(tok, name, text, ids))
    if not streams:
        raise EngineError(UNSUPPORTED, "loaded pipeline has no CLIP tokenizer")
    return {"input_mode": mode, "streams": streams}


def _stream(tok, name, text, input_ids):
    decoder = getattr(tok, "byte_decoder", None)
    if decoder is None:
        raise EngineError(UNSUPPORTED, f"{name} has no CLIP byte vocabulary")
    vocab = {int(v): k for k, v in tok.get_vocab().items()}
    added_ids = set(getattr(tok, "get_added_vocab", lambda: {})().values())
    special_ids = set(getattr(tok, "all_special_ids", (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id)))
    ids = list(input_ids) if input_ids is not None else list(
        tok(text, add_special_tokens=False, truncation=False).input_ids)
    if len(ids) > 8192:
        raise EngineError(INVALID_PARAMS, "inspection is limited to 8192 tokens per stream; shorten the input")
    invalid = next((i for i in ids if i not in vocab), None)
    if invalid is not None:
        raise EngineError(INVALID_PARAMS, f"ID {invalid} is not in {name}'s vocabulary")
    tokens, groups = [], []
    pending, pending_ids = bytearray(), []

    def flush(force=False):
        nonlocal pending, pending_ids
        if not pending_ids:
            return
        try:
            decoded = pending.decode("utf-8")
        except UnicodeDecodeError as e:
            if not force and e.reason == "unexpected end of data":
                return
            decoded = pending.decode("utf-8", errors="replace")
        groups.append({"ids": pending_ids, "text": decoded})
        pending, pending_ids = bytearray(), []

    for index, token_id in enumerate(ids):
        piece = vocab[token_id]
        # End-of-word markers become spaces for display; retain the exact BPE string too.
        special = token_id in special_ids
        literal = special or token_id in added_ids
        raw = piece.encode("utf-8") if literal else bytes(decoder[c] for c in piece.replace("</w>", ""))
        if not literal and piece.endswith("</w>"):
            raw += b" "
        tokens.append({"index": index + 1, "id": token_id, "piece": piece,
                       "text": raw.decode("utf-8", errors="replace"), "bytes_hex": raw.hex(" "), "special": special})
        if special:
            flush(True)
        pending.extend(raw)
        pending_ids.append(token_id)
        flush()
    flush(True)
    maximum = int(getattr(tok, "model_max_length", 77))
    return {"name": name, "source": str(getattr(tok, "name_or_path", "")), "count": len(ids), "ids": ids,
            "content_capacity": maximum - 2 if 2 < maximum < 100000 else 75,
            "bos_id": tok.bos_token_id, "eos_id": tok.eos_token_id, "pad_id": tok.pad_token_id,
            "decoded": tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) if ids else "",
            "tokens": tokens, "groups": groups}
