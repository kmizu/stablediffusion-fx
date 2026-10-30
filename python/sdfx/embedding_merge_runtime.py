"""Transaction orchestration for temporary merge vectors on the serial worker.

The model backend owns collision checks, snapshot fidelity and fail-closed
restoration; this layer never changes persistent TI ownership or file maps.
"""
from contextlib import contextmanager
from dataclasses import dataclass

from .embedding_merge import evaluate, parse_expression, MAX_ROWS
from .embedding_merge_prompts import substitute


@dataclass(frozen=True)
class MergedPrompts:
    raw_prompts: tuple[str, ...]
    prompts: tuple[str, ...]
    provenance: dict[str, str]


@contextmanager
def expression_scope(source, prompts, backend, check):
    raw = tuple(prompts)
    vectors = {}
    cache = {}
    provenance = {}

    def replace(span):
        check()
        key = span.expression.strip()
        if key not in cache:
            values = evaluate(parse_expression(key), source.resolve_text, source.resolve_token, source.stream_names)
            if values[0].shape[0] == 0:
                cache[key] = ''
            else:
                if sum(len(value[0]) for value in vectors.values()) + len(values[0]) > MAX_ROWS:
                    raise ValueError('total temporary vector row limit exceeded')
                number = len(vectors) + 1
                name = f'__nui_em_{number}__'
                while name in vectors or any(name in text for text in raw):
                    number += 1
                    name = f'__nui_em_{number}__'
                vectors[name] = values
                provenance[name] = span.original
                cache[key] = name
        return cache[key]

    transformed = tuple(substitute(prompt, replace) for prompt in raw)
    result = MergedPrompts(raw, transformed, provenance)
    check()
    if not vectors:
        yield result
        return
    snapshot = backend.capture()
    try:
        for name, values in vectors.items():
            check()
            backend.install(name, values)
        check()
        yield result
    finally:
        # A cancellation check here would prevent the restoration it requested.
        backend.restore(snapshot)
