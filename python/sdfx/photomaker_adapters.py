"""Keep PhotoMaker's shared PEFT adapter local to one pipeline invocation."""
from contextlib import contextmanager
from copy import deepcopy

from .protocol import EngineError, NO_MODEL, UNSUPPORTED


def load_owned_adapter(engine, pipe, weights, **options):
    """Called inside adapter_scope; only one fixed-name PM payload may be live.

    VAE changes can evict derived pipelines while retaining their shared UNet.
    Drop only our registered adapter and PM caches before rebuilding; distinct
    cached PM pipelines cannot safely refer to the same name with other weights.
    """
    base = engine.pipe
    identity = id(getattr(base, 'unet', None))
    registrations = getattr(base, 'get_list_adapters', lambda: {})()
    present = any('photomaker' in names for names in registrations.values())
    if present and getattr(engine, '_photomaker_adapter_owner', None) != identity:
        raise EngineError(UNSUPPORTED, 'PhotoMaker adapter name is already owned by another loader; reload the checkpoint')
    try:
        if present:
            engine._drop_hooks()
            base.delete_adapters('photomaker')
        for key in list(engine.derived):
            if isinstance(key, tuple) and key and key[0] == 'photomaker':
                del engine.derived[key]
        engine._photomaker_adapter_owner = identity
        pipe.load_photomaker_adapter(weights, **options)
    except BaseException as error:
        # The official loader may have injected only some PEFT layers. A
        # disabled partial registration is not a recoverable loaded model.
        try:
            engine._free()
        finally:
            engine._photomaker_adapter_owner = None
            raise EngineError(NO_MODEL, 'PhotoMaker adapter loading failed; checkpoint unloaded, reload before retrying') from error


def _layers(pipe, layer_type):
    seen = set()
    for name in ('unet', 'text_encoder', 'text_encoder_2'):
        component = getattr(pipe, name, None)
        if component is not None:
            for module in component.modules():
                if isinstance(module, layer_type) and id(module) not in seen:
                    seen.add(id(module))
                    yield module


class AdapterScope:
    def __init__(self, engine, layer_type, inference_scope):
        self.engine, self.pipe = engine, engine.pipe
        self.layer_type, self.inference_scope = layer_type, inference_scope
        self.missing = object()
        self.window_state = getattr(engine, '_lora_state', self.missing)
        self.window_key = getattr(engine, '_lora_key', self.missing)
        self.before = {id(layer): (list(layer.active_adapters), bool(layer.disable_adapters), deepcopy(layer.scaling))
                       for layer in _layers(self.pipe, layer_type)}

    def activate(self, pipe):
        with self.inference_scope():
            for layer in _layers(pipe, self.layer_type):
                active, disabled, scaling = self.before.get(id(layer), ([], True, {}))
                names = [] if disabled else list(active)
                if 'photomaker' in layer.scaling and 'photomaker' not in names:
                    names.append('photomaker')
                layer.set_adapter(names)
                layer.scaling.update(deepcopy(scaling))
                layer.enable_adapters(bool(names))
        if self.window_state is not self.missing:
            names, weights = self.window_state
            if 'photomaker' not in names:
                self.engine._lora_state = (list(names) + ['photomaker'], list(weights) + [1.0])
            self.engine._lora_key = None

    def restore(self):
        if self.engine.pipe is not self.pipe:
            return
        try:
            with self.inference_scope():
                for layer in _layers(self.pipe, self.layer_type):
                    active, disabled, scaling = self.before.get(id(layer), ([], True, {}))
                    layer.set_adapter(active)
                    layer.scaling.update(deepcopy(scaling))
                    layer.enable_adapters(not disabled)
            for name, value in (('_lora_state', self.window_state), ('_lora_key', self.window_key)):
                if value is not self.missing:
                    setattr(self.engine, name, value)
                elif hasattr(self.engine, name):
                    delattr(self.engine, name)
        except BaseException as error:
            try:
                self.engine._free()
            finally:
                raise EngineError(NO_MODEL, 'PhotoMaker adapter restoration failed; reload the checkpoint') from error


@contextmanager
def adapter_scope(engine, *, layer_type=None, inference_scope=None):
    if layer_type is None:
        from peft.tuners.tuners_utils import BaseTunerLayer
        layer_type = BaseTunerLayer
    if inference_scope is None:
        import torch
        inference_scope = torch.inference_mode
    scope = AdapterScope(engine, layer_type, inference_scope)
    # Adapter injection also toggles existing PEFT requires_grad flags. Keep
    # loading and this inference-only pipeline invocation in the same mode.
    with inference_scope():
        try:
            yield scope
        finally:
            scope.restore()


@contextmanager
def id_encoder_scope(engine, pipe):
    """Official v1's dynamically attached ID encoder is outside offload hooks."""
    encoder = getattr(pipe, 'id_encoder', None)
    if encoder is None:
        yield
        return
    devices = {parameter.device for parameter in encoder.parameters()}
    if len(devices) != 1:
        raise ValueError('PhotoMaker ID encoder requires one materialized device')
    original = next(iter(devices))
    if str(original) == 'meta':
        raise ValueError('PhotoMaker ID encoder is not materialized')
    base = engine.pipe
    try:
        encoder.to(engine.device)
        yield
    finally:
        if engine.pipe is base:
            try:
                encoder.to(original)
            except BaseException as error:
                try:
                    engine._free()
                finally:
                    raise EngineError(NO_MODEL, 'PhotoMaker ID encoder restoration failed; reload the checkpoint') from error
