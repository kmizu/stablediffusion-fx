from contextlib import contextmanager
from types import SimpleNamespace
import pytest

from sdfx.photomaker_adapters import adapter_scope


class Layer:
    def __init__(self, scaling=None, active=(), disabled=False):
        self.scaling = dict(scaling or {})
        self.active_adapters = list(active)
        self.disable_adapters = disabled
    def set_adapter(self, names): self.active_adapters = list(names)
    def enable_adapters(self, enabled): self.disable_adapters = not enabled


def fixture():
    layers = []
    pipe = SimpleNamespace(unet=SimpleNamespace(modules=lambda: iter(layers)), text_encoder=None, text_encoder_2=None)
    engine = SimpleNamespace(pipe=pipe)
    engine._free = lambda: setattr(engine, 'pipe', None)
    events = []
    @contextmanager
    def inference():
        events.append('enter')
        try: yield
        finally: events.append('exit')
    return engine, layers, events, inference


def test_new_photomaker_layer_is_enabled_only_inside_call_and_disabled_after_cancel():
    engine, layers, events, inference = fixture()
    with pytest.raises(RuntimeError, match='cancel'):
        with adapter_scope(engine, layer_type=Layer, inference_scope=inference) as scope:
            layer = Layer({'photomaker': 1}, ['photomaker'])
            layers.append(layer)
            scope.activate(engine.pipe)
            assert layer.active_adapters == ['photomaker'] and not layer.disable_adapters
            raise RuntimeError('cancel')
    assert layer.disable_adapters and layer.active_adapters == []
    assert events == ['enter', 'enter', 'exit', 'enter', 'exit', 'exit']


def test_adapter_injection_itself_runs_inside_inference_scope():
    engine, layers, events, inference = fixture()
    with adapter_scope(engine, layer_type=Layer, inference_scope=inference):
        assert events == ['enter'], 'PEFT injection may reactivate existing offloaded inference parameters'


def test_existing_manual_weights_and_disabled_cached_layers_restore_exactly():
    engine, layers, events, inference = fixture()
    manual = Layer({'manual': .375}, ['manual'])
    cached = Layer({'photomaker': 1}, [], True)
    layers.extend([manual, cached])
    with adapter_scope(engine, layer_type=Layer, inference_scope=inference) as scope:
        manual.scaling.update(photomaker=1, manual=1)
        manual.active_adapters = ['photomaker']
        scope.activate(engine.pipe)
        assert manual.active_adapters == ['manual', 'photomaker']
        assert manual.scaling['manual'] == .375
        assert not cached.disable_adapters
    assert manual.active_adapters == ['manual'] and manual.scaling['manual'] == .375
    assert not manual.disable_adapters
    assert cached.active_adapters == [] and cached.disable_adapters


def test_partial_adapter_load_failure_disables_new_layer_without_activation():
    engine, layers, events, inference = fixture()
    with pytest.raises(ValueError, match='load'):
        with adapter_scope(engine, layer_type=Layer, inference_scope=inference):
            layers.append(Layer({'photomaker': 1}, ['photomaker']))
            raise ValueError('load failed')
    assert layers[0].disable_adapters


def test_restore_failure_invalidates_checkpoint():
    from sdfx.protocol import EngineError, NO_MODEL
    engine, layers, events, inference = fixture()
    class Broken(Layer):
        def set_adapter(self, names): raise RuntimeError('restore failed')
    with pytest.raises(EngineError) as error:
        with adapter_scope(engine, layer_type=Layer, inference_scope=inference):
            layers.append(Broken({'photomaker': 1}))
    assert error.value.code == NO_MODEL and engine.pipe is None


def test_actual_engine_scope_starts_before_special_pipeline_partial_load(monkeypatch):
    import ast
    from pathlib import Path
    from sdfx import photomaker_adapters
    engine, layers, events, inference = fixture()
    tree = ast.parse((Path(__file__).parents[1] / 'sdfx/engine.py').read_text(encoding='utf-8'))
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in ('_run', '_run_unscoped')]
    cls = ast.ClassDef(name='Subject', bases=[], keywords=[], body=methods, decorator_list=[])
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    def partial(*args):
        layers.append(Layer({'photomaker': 1}, ['photomaker']))
        raise ValueError('partial load')
    ext = SimpleNamespace(pre_run=lambda *args, **kwargs: SimpleNamespace(warnings=[], task='txt2img'), special_pipe=partial)
    scope = {'__package__': 'sdfx', 'ext_mod': ext,
             'hooks_mod': SimpleNamespace(hook_config=lambda *args: {})}
    monkeypatch.setattr(photomaker_adapters, 'adapter_scope',
                        lambda subject: adapter_scope(subject, layer_type=Layer, inference_scope=inference))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])), '<actual-run>', 'exec'), scope)
    subject = scope['Subject']()
    subject.pipe = engine.pipe
    subject._free = lambda: setattr(subject, 'pipe', None)
    with pytest.raises(ValueError, match='partial load'):
        subject._run('txt2img', SimpleNamespace(hooks={'photomaker': {}}), None, None,
                     sampler=None, schedule=None, cfg_scale=1, embeds={}, steps=1, w=1, h=1,
                     seeds=[1], pass_='base', image_idx=0, images_total=1)
    assert layers[0].disable_adapters, 'partial loader mutation must not survive special_pipe failure'


def test_window_updates_retain_photomaker_and_restore_window_cache():
    import ast
    from pathlib import Path
    engine, layers, events, inference = fixture()
    layer = Layer({'manual': .5, 'photomaker': 1}, ['manual'])
    layers.append(layer)
    state = (['manual'], [.5])
    engine._lora_state, engine._lora_key = state, ('before',)
    engine._dora_step_weights = {}
    engine._lora_windows, engine.sdxl = [(0, 0, .5)], True
    engine.pipe.set_adapters = lambda names, adapter_weights: layer.set_adapter(names)
    engine._set_inference_adapters = lambda pipe, names, weights: pipe.set_adapters(names, adapter_weights=weights)
    tree = ast.parse((Path(__file__).parents[1] / 'sdfx/engine.py').read_text(encoding='utf-8'))
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == '_set_lora_step')
    namespace = {'lora_blocks': SimpleNamespace(active=lambda *args: False, zero_unet=lambda *args: 0)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<actual-window>', 'exec'), namespace)
    with adapter_scope(engine, layer_type=Layer, inference_scope=inference) as scope:
        scope.activate(engine.pipe)
        namespace['_set_lora_step'](engine, 6, 8)
        assert layer.active_adapters == ['manual', 'photomaker']
    assert engine._lora_state is state and engine._lora_key == ('before',)
    assert layer.active_adapters == ['manual']


def test_id_encoder_tracks_execution_device_and_restores_on_cancel():
    from sdfx.photomaker_adapters import id_encoder_scope
    engine, layers, events, inference = fixture()
    engine.device = 'cuda'
    parameter = SimpleNamespace(device='cpu')
    def move(device):
        events.append(device)
        parameter.device = device
    pipe = SimpleNamespace(id_encoder=SimpleNamespace(parameters=lambda: iter([parameter]), to=move))
    with pytest.raises(ValueError, match='cancel'):
        with id_encoder_scope(engine, pipe):
            assert parameter.device == 'cuda'
            raise ValueError('cancel')
    assert parameter.device == 'cpu' and events == ['cuda', 'cpu']


def test_id_encoder_restore_failure_invalidates_model():
    from sdfx.photomaker_adapters import id_encoder_scope
    from sdfx.protocol import EngineError, NO_MODEL
    engine, layers, events, inference = fixture()
    engine.device = 'cuda'
    parameter = SimpleNamespace(device='cpu')
    def move(device):
        if device == 'cpu': raise RuntimeError('restore')
        parameter.device = device
    pipe = SimpleNamespace(id_encoder=SimpleNamespace(parameters=lambda: iter([parameter]), to=move))
    with pytest.raises(EngineError) as error:
        with id_encoder_scope(engine, pipe): pass
    assert error.value.code == NO_MODEL and engine.pipe is None
def test_id_encoder_scope_allows_real_accelerate_sequential_offload():
    import torch
    from accelerate import cpu_offload
    from types import SimpleNamespace
    from sdfx.photomaker_adapters import id_encoder_scope
    encoder = torch.nn.Sequential(torch.nn.Linear(3, 2))
    source = torch.ones(1, 3)
    expected = encoder(source).detach()
    cpu_offload(encoder, execution_device=torch.device('cpu'))
    assert all(p.is_meta for p in encoder.parameters())
    pipe = SimpleNamespace(id_encoder=encoder)
    engine = SimpleNamespace(pipe=pipe, device=torch.device('cpu'))
    with id_encoder_scope(engine, pipe):
        torch.testing.assert_close(encoder(source), expected)
    assert all(p.is_meta for p in encoder.parameters())
    with pytest.raises(RuntimeError, match='cancelled'):
        with id_encoder_scope(engine, pipe):
            torch.testing.assert_close(encoder(source), expected)
            raise RuntimeError('cancelled')
    assert all(p.is_meta for p in encoder.parameters())


def test_first_photomaker_load_materializes_sequential_base_before_loader_reads_device():
    import torch
    from accelerate import cpu_offload
    from accelerate.hooks import remove_hook_from_module
    from sdfx.photomaker_adapters import load_owned_adapter
    unet = cpu_offload(torch.nn.Sequential(torch.nn.Linear(3, 2)), execution_device=torch.device('cpu'))
    base = SimpleNamespace(unet=unet, get_list_adapters=lambda: {})
    engine = SimpleNamespace(pipe=base, derived={}, _offload=True,
                             _drop_hooks=lambda: remove_hook_from_module(unet, recurse=True), _free=lambda: None)
    observed = []
    pipe = SimpleNamespace(load_photomaker_adapter=lambda *args, **kwargs:
                           observed.append(next(unet.parameters()).device.type))
    assert next(unet.parameters()).is_meta
    load_owned_adapter(engine, pipe, {})
    assert observed == ['cpu'], 'official loader must not move newly loaded ID weights to meta'


def test_id_encoder_scope_rejects_unbacked_meta_weights():
    import torch
    from types import SimpleNamespace
    from sdfx.photomaker_adapters import id_encoder_scope
    pipe = SimpleNamespace(id_encoder=torch.nn.Linear(3, 2, device='meta'))
    with pytest.raises(ValueError, match='not materialized'):
        with id_encoder_scope(SimpleNamespace(pipe=pipe, device=torch.device('cpu')), pipe):
            pass
