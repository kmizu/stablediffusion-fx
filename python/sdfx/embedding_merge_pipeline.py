"""Temporary expression tokens for pipelines that encode text inside __call__."""
from copy import deepcopy
import sys
from types import SimpleNamespace

from .embedding_merge_backend import ModelBackend, streams
from .embedding_merge_model import VectorSource
from .embedding_merge_prompts import expressions
from .embedding_merge_runtime import expression_scope


def clone_tokenizers(components):
    """PhotoMaker adds a trigger token; keep it out of the base tokenizer."""
    return {name: deepcopy(value) if name in ('tokenizer', 'tokenizer_2') and value is not None else value
            for name, value in components.items()}


def run_pipeline(engine, pipe, prompts, operation, *, check, tensor_factory=None):
    raw = tuple(prompts)
    if not any(expressions(text) for text in raw):
        return operation(raw)
    if getattr(engine, '_embedding_encoding_transaction', None) is not None:
        raise ValueError('text-only pipeline cannot run inside an encoder-only transaction')
    original = engine.pipe
    base, target = streams(original), streams(pipe)
    if set(base) != set(target) or any(base[key][1] is not target[key][1] for key in base):
        raise ValueError('text-only pipeline must share the loaded model encoders')
    state = {}

    def prepare():
        check()
        source = VectorSource({name: (tokenizer, encoder.get_input_embeddings().weight)
                               for name, (tokenizer, encoder) in target.items()},
                              getattr(engine, 'embedding_token_owners', {}))
        proxy = SimpleNamespace(pipe=pipe, _free=engine._free)
        # A failed outer editor may discard the model. This also protects the
        # generator context's finalizer if an exception retains its traceback.
        backend = ModelBackend(proxy, tensor_factory=tensor_factory,
                               access=lambda call: call() if getattr(engine, 'pipe', None) is original else None)
        scope = expression_scope(source, raw, backend, check)
        merged = scope.__enter__()
        state['scope'] = scope
        return merged.prompts

    try:
        texts = engine._with_embedding_encoder_access(prepare)
        # Full pipeline execution requires its normal offload hooks. Both text
        # passes and the intervening token decode see the same temporary names.
        engine._activate(pipe)
        check()
        return operation(texts)
    finally:
        if 'scope' in state:
            failure = sys.exc_info()
            # Cancellation must not prevent restoration. The shared guard
            # restores normal placement after the original modules/maps return.
            if getattr(engine, 'pipe', None) is original:
                engine._with_embedding_encoder_access(lambda: state['scope'].__exit__(*failure))
            else:
                state['scope'].__exit__(*failure)
