"""Temporary CLIP rows using the shared #46 encoder-access ownership contract."""
import copy
import threading
from contextlib import contextmanager

from .embedding_merge_model import VectorSource
from .embedding_merge_prompts import expressions
from .embedding_merge_runtime import expression_scope


def restore_composable_provenance(result, provenance):
    """Keep raw expressions in routing metadata after transient rows are removed."""
    if not provenance or not isinstance(result, dict):
        return result
    seen = set()
    def raw(value):
        if isinstance(value, str):
            for name, expression in provenance.items():
                value = value.replace(name, expression)
            return value
        if isinstance(value, list):
            return [raw(item) for item in value]
        if isinstance(value, tuple):
            return tuple(raw(item) for item in value)
        return value
    def visit(branch):
        if not isinstance(branch, dict) or id(branch) in seen:
            return
        seen.add(id(branch))
        for key in ('branch', 'prompt', 'negative'):
            if '_clora_' + key in branch:
                branch['_photo_raw_' + key] = raw(branch['_clora_' + key])
        for _, variant in getattr(branch, 'schedule', None) or ():
            visit(variant)
        plan = branch.get('_neutral')
        if plan is not None:
            for nested in plan.branches:
                visit(nested)
    visit(result)
    return result


def run_encoded(engine, prompts, operation, *, check=lambda: None, tensor_factory=None):
    prompts = tuple(prompts)
    if not any(expressions(text) for text in prompts):
        return operation(prompts)

    def materialized():
        source = VectorSource({key: (tokenizer, encoder.get_input_embeddings().weight)
            for key, (tokenizer, encoder) in streams(engine.pipe).items()},
            getattr(engine, 'embedding_token_owners', {}))
        # Keep the existing offload access boundary around installation, encode
        # and rollback. Nested hook placement would invalidate snapshot weights.
        backend = ModelBackend(engine, tensor_factory=tensor_factory, access=lambda call: call())
        with expression_scope(source, prompts, backend, check) as merged:
            return restore_composable_provenance(operation(merged.prompts), merged.provenance)
    return run_encoding_transaction(engine, materialized)


def streams(pipe):
    result = {'clip_l': (getattr(pipe, 'tokenizer', None), getattr(pipe, 'text_encoder', None))}
    if getattr(pipe, 'tokenizer_2', None) is not None or getattr(pipe, 'text_encoder_2', None) is not None:
        result['clip_g'] = (getattr(pipe, 'tokenizer_2', None), getattr(pipe, 'text_encoder_2', None))
    if any(tokenizer is None or encoder is None for tokenizer, encoder in result.values()):
        raise ValueError('model requires complete tokenizer and text encoder streams')
    return result


class ModelBackend:
    def __init__(self, engine, tensor_factory=None, access=None):
        self.engine = engine
        self.tensor_factory = tensor_factory
        self.access = access or engine._with_embedding_encoder_access

    def capture(self):
        def operation():
            result = []
            for tokenizer, encoder in streams(self.engine.pipe).values():
                module = encoder.get_input_embeddings()
                weight = module.weight
                inference = False
                if hasattr(weight, 'detach'):
                    import torch
                    inference = torch.is_inference(weight)
                saved = weight.detach().clone() if hasattr(weight, 'detach') else weight.copy()
                result.append((tokenizer, copy.deepcopy(tokenizer.__dict__), encoder, module, saved,
                               getattr(module, 'num_embeddings', None), getattr(module, 'padding_idx', None),
                               getattr(encoder.config, 'vocab_size', None), getattr(encoder, 'vocab_size', None), inference))
            return result
        return self.access(operation)

    def install(self, name, values):
        def operation():
            active = streams(self.engine.pipe)
            if len(active) != len(values) or len({len(v) for v in values}) != 1:
                raise ValueError('temporary embedding stream rows must match')
            candidate_names = [name] + [f'{name}_{i}' for i in range(1, len(values[0]))]
            # Validate every stream before modifying the first encoder.
            for tokenizer, encoder in active.values():
                if set(candidate_names) & set(tokenizer.get_vocab()):
                    raise ValueError('temporary embedding token collision')
            factory = self.tensor_factory
            if factory is None:
                import torch
                factory = torch.from_numpy
            for (tokenizer, encoder), value in zip(active.values(), values):
                self.engine.pipe.load_textual_inversion(factory(value), token=name,
                    tokenizer=tokenizer, text_encoder=encoder)
        return self.access(operation)

    def restore(self, snapshot):
        def operation():
            for tokenizer, state, encoder, module, weight, size, padding, config_size, encoder_size, inference in snapshot:
                tokenizer.__dict__.clear()
                tokenizer.__dict__.update(state)
                if hasattr(module.weight, 'detach'):
                    # The loader may resize this same module in place. A
                    # snapshot taken in inference mode must not become the
                    # persistent Parameter's inference-only storage.
                    import torch
                    with torch.inference_mode(inference), torch.no_grad():
                        restored = weight.detach().clone()
                    module.weight.data = restored
                else:
                    module.weight = weight.copy()
                if size is not None:
                    module.num_embeddings = size
                if hasattr(module, 'padding_idx'):
                    module.padding_idx = padding
                encoder.set_input_embeddings(module)
                if config_size is not None:
                    encoder.config.vocab_size = config_size
                if encoder_size is not None:
                    encoder.vocab_size = encoder_size
        try:
            return self.access(operation)
        except BaseException as failure:
            try:
                self.engine._free()
            except BaseException:
                pass
            raise RuntimeError('temporary embedding restoration failed; reload checkpoint') from failure


@contextmanager
def encoding_device_scope(engine):
    """Encoder access removes offload hooks; explicitly align inputs and weights.

    Restore each encoder's materialized placement before the outer shared guard
    reinstalls accelerate hooks. No temporary execution-device hint survives.
    """
    pipe = engine.pipe
    missing = object()
    previous = getattr(pipe, '_embedding_merge_execution_device', missing)
    captured = [(encoder, getattr(encoder.get_input_embeddings().weight, 'device', None))
                for _, encoder in streams(pipe).values()]
    target = getattr(engine, 'device', None)
    try:
        if target is not None:
            for encoder, original in captured:
                if original is not None:
                    encoder.to(target)
            pipe._embedding_merge_execution_device = target
        yield
    finally:
        try:
            # An inner rollback may have invalidated the model. Do not revive
            # discarded encoders or mask its original NO_MODEL exception.
            if getattr(engine, 'pipe', None) is pipe:
                for encoder, original in captured:
                    if target is not None and original is not None:
                        encoder.to(original)
        except BaseException as failure:
            try:
                engine._free()
            except BaseException:
                pass
            raise RuntimeError('temporary encoder placement restoration failed; reload checkpoint') from failure
        finally:
            if previous is missing:
                if hasattr(pipe, '_embedding_merge_execution_device'):
                    del pipe._embedding_merge_execution_device
            else:
                pipe._embedding_merge_execution_device = previous



def run_encoding_transaction(engine, operation):
    """One materialization/device scope shared by merge and unsaved TI editing.

    The serial worker may compose multiple temporary-vector operations. Inner
    operations restore their own snapshots before the outer editor restores its
    rows; hooks are only detached/reattached at this outer transaction boundary.
    """
    owner = getattr(engine, '_embedding_encoding_transaction', None)
    if owner is not None:
        if owner != threading.get_ident():
            raise RuntimeError('embedding encoding transaction belongs to another worker')
        return operation()

    def materialized():
        engine._embedding_encoding_transaction = threading.get_ident()
        try:
            with encoding_device_scope(engine):
                return operation()
        finally:
            if hasattr(engine, '_embedding_encoding_transaction'):
                del engine._embedding_encoding_transaction
    return engine._with_embedding_encoder_access(materialized)
