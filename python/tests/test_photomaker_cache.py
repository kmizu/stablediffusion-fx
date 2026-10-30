from types import SimpleNamespace
import pytest

def load_owned_adapter(*args, **kwargs):
    from sdfx.photomaker_adapters import load_owned_adapter as load
    return load(*args, **kwargs)


def fixture():
    registry = {'ordinary': 'unchanged'}
    events = []
    unet = object()
    base = SimpleNamespace(unet=unet, get_list_adapters=lambda: {'unet': list(registry)},
                           delete_adapters=lambda name: (events.append(('delete', name)), registry.pop(name)))
    engine = SimpleNamespace(pipe=base, derived={}, _drop_hooks=lambda: events.append('drop-hooks'))
    def free():
        engine.pipe = None
        engine.derived.clear()
        registry.clear()
    engine._free = free
    def pipeline(fail=False):
        def load(weights, **options):
            if 'photomaker' in registry: raise ValueError('duplicate adapter')
            registry['photomaker'] = weights
            if fail: raise ValueError('injected then failed')
        return SimpleNamespace(load_photomaker_adapter=load)
    return engine, registry, events, pipeline


def test_rebuild_after_vae_cache_clear_replaces_only_owned_adapter():
    engine, registry, events, pipeline = fixture()
    load_owned_adapter(engine, pipeline(), 'A')
    engine.derived[('photomaker', ('A',), None)] = 'old A'
    engine.derived.clear()  # Native VAE replacement retains the shared UNet.
    load_owned_adapter(engine, pipeline(), 'A')
    assert registry == {'ordinary': 'unchanged', 'photomaker': 'A'}
    assert ('delete', 'photomaker') in events


def test_switching_a_b_a_evicts_all_old_photomaker_caches():
    engine, registry, events, pipeline = fixture()
    engine.derived[('sag', (), None)] = 'keep'
    for name in ('A', 'B', 'A'):
        load_owned_adapter(engine, pipeline(), name)
        assert all(key[0] != 'photomaker' for key in engine.derived)
        engine.derived[('photomaker', (name,), None)] = name
        assert registry['photomaker'] == name
    assert engine.derived[('sag', (), None)] == 'keep'
    assert registry['ordinary'] == 'unchanged'


def test_partial_load_invalidates_checkpoint_and_reload_can_retry():
    from sdfx.protocol import EngineError, NO_MODEL
    engine, registry, events, pipeline = fixture()
    with pytest.raises(EngineError) as error:
        load_owned_adapter(engine, pipeline(fail=True), 'A')
    assert error.value.code == NO_MODEL and engine.pipe is None and not registry
    reloaded, new_registry, _, new_pipeline = fixture()
    load_owned_adapter(reloaded, new_pipeline(), 'A')
    assert new_registry['photomaker'] == 'A'


def test_unowned_reserved_adapter_is_never_deleted():
    from sdfx.protocol import EngineError, UNSUPPORTED
    engine, registry, events, pipeline = fixture()
    registry['photomaker'] = 'external'
    with pytest.raises(EngineError) as error:
        load_owned_adapter(engine, pipeline(), 'A')
    assert error.value.code == UNSUPPORTED
    assert registry['photomaker'] == 'external' and not events


def test_actual_special_pipeline_rebuilds_after_vae_cache_eviction(monkeypatch):
    import ast
    import os
    import sys
    from pathlib import Path
    engine, registry, events, pipeline = fixture()
    engine.pipe.components = {'unet': engine.pipe.unet}
    class Pipeline:
        def __init__(self, **components): self.__dict__.update(components)
        def load_photomaker_adapter(self, weights, **kwargs): pipeline().load_photomaker_adapter(weights, **kwargs)
        def set_progress_bar_config(self, **kwargs): pass
    monkeypatch.setitem(sys.modules, 'photomaker', SimpleNamespace(PhotoMakerStableDiffusionXLPipeline=Pipeline))
    monkeypatch.setitem(sys.modules, 'sdfx.engine', SimpleNamespace(safe_torch_load=lambda path, label: path))
    tree = ast.parse((Path(__file__).parents[1] / 'sdfx/extensions.py').read_text(encoding='utf-8'))
    method = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'special_pipe')
    scope = {'__package__': 'sdfx', 'RunPlan': object, 'os': os}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<actual-special>', 'exec'), scope)
    plan = SimpleNamespace(sag=None, photomaker={'model': 'A-v1.bin', 'trigger_word': 'img'})
    first = scope['special_pipe'](engine, plan)
    engine.derived.clear()
    second = scope['special_pipe'](engine, plan)
    assert second is not first and registry['photomaker'] == 'A-v1.bin'
