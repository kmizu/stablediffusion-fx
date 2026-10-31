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


def validate_offload_multidiffusion(offload, photomaker, multidiffusion, prompts):
    if offload and photomaker and multidiffusion and any(expressions(text) for text in prompts):
        from .protocol import EngineError, UNSUPPORTED
        raise EngineError(UNSUPPORTED, 'MultiDiffusion with PhotoMaker and Embedding Merge is not supported with offload yet')


def validate_keyword_offload_multidiffusion(hooks, offload, photomaker, multidiffusion, prompts):
    from .hooks import hook_config
    if hook_config(hooks, 'model_keyword', {}) is not None:
        validate_offload_multidiffusion(offload, photomaker, multidiffusion, prompts)


def run_pipeline(engine, pipe, prompts, operation, *, check, tensor_factory=None, runtime_hook=None,
                 extra_runtime_hooks=(), pre_runtime_hooks=()):
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
    runtime_hooks = list(pre_runtime_hooks) + ([runtime_hook] if runtime_hook is not None else []) + list(extra_runtime_hooks)
    rebind_runtime = bool(runtime_hooks) and bool(getattr(engine, '_offload', False))
    rebound = []

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
        # Removing Accelerate hooks rewrites UNet.forward. Suspend Neutral
        # first, then bind it to the new offload wrapper just for invocation.
        if rebind_runtime:
            for hook in reversed(runtime_hooks):
                hook.remove()
        texts = engine._with_embedding_encoder_access(prepare)
        # Full pipeline execution requires its normal offload hooks. Both text
        # passes and the intervening token decode see the same temporary names.
        engine._activate(pipe)
        check()
        if rebind_runtime:
            for hook in runtime_hooks:
                hook.install()
                rebound.append(hook)
        # Transformers' legacy CLIP EOS=2 path pools at the largest token id.
        # Temporary expression tokens exceed the actual EOS id, so scope the
        # corrected SDXL pooling to PhotoMaker's own text encoding only.
        model = getattr(getattr(pipe, 'text_encoder_2', None), 'text_model', None)
        tokenizer = getattr(pipe, 'tokenizer_2', None)
        eos = getattr(tokenizer, 'eos_token_id', None)
        if getattr(pipe, 'id_encoder', None) is None or getattr(model, 'eos_token_id', None) != 2 or eos is None:
            return operation(texts)
        try:
            model.eos_token_id = eos
            return operation(texts)
        finally:
            model.eos_token_id = 2
    finally:
        if rebind_runtime:
            for hook in reversed(rebound):
                hook.remove()
        if 'scope' in state:
            failure = sys.exc_info()
            # Cancellation must not prevent restoration. The shared guard
            # restores normal placement after the original modules/maps return.
            if getattr(engine, 'pipe', None) is original:
                engine._with_embedding_encoder_access(lambda: state['scope'].__exit__(*failure))
            else:
                state['scope'].__exit__(*failure)
