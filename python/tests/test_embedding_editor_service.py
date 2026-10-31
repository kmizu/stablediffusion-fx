from pathlib import Path
from types import SimpleNamespace
import importlib
import pytest


def test_service_opens_and_saves_through_engine_owned_guard(tmp_path):
    from sdfx.resource_hashes import FileSnapshot
    service = importlib.import_module('sdfx.embedding_editor_service')
    path = tmp_path / 'concept.safetensors'; path.write_bytes(b'original')
    engine = SimpleNamespace(model={'model_token': 'm'}, pipe=object(), loaded_embedding_files={'concept': str(path)},
        embedding_token_owners={'concept': {'clip_l': ['concept']}}, embedding_snapshots={'concept': FileSnapshot.capture(path)})
    events = []
    def read(engine, name, refs, guarded=True):
        events.append(('read', guarded))
        return {'vectors': {'clip_l': [[1., 2.]]}, 'ranges': {'clip_l': [[0., 2.], [0., 2.]]}, 'guidance': []}
    def publish(engine, name, values, expected, *, baseline, current, refresh):
        events.append(('write', values, baseline)); return {'path': expected['path'], 'reloaded': True}
    engine._with_embedding_encoder_access = lambda fn: (events.append('guard'), fn())[1]
    class Context:
        def check_interrupt(self): events.append('check')
        def publish(self, fn): events.append('publish'); return fn()
    opened = service.run_editor_service(engine, {'operation': 'open', 'name': 'concept'}, Context(),
                                       read=read, publish_safetensors=publish)
    opened['vectors']['clip_l'][0][1] = 1.5
    result = service.run_editor_service(engine, {'operation': 'save', 'name': 'concept', 'draft': opened, 'overwrite': True},
                                        Context(), read=read, publish_safetensors=publish, refresh=lambda _: None,
                                        validate=lambda e, paths: events.append(('validate', paths)))
    assert result['reloaded'] and 'publish' in events and 'guard' in events
    assert events.index(('validate', [str(path)])) < next(i for i, value in enumerate(events) if isinstance(value, tuple) and value[0] == 'write')
    assert ('write', {'clip_l': [[1., 1.5]]}, {'clip_l': [[1., 2.]]}) in events


def test_open_selected_catalog_embedding_loads_once_under_publication_guard(tmp_path):
    from sdfx.resource_hashes import FileSnapshot
    service = importlib.import_module('sdfx.embedding_editor_service')
    path = tmp_path / 'concept.safetensors'; path.write_bytes(b'fixture')
    engine = SimpleNamespace(model={'model_token': 'm'}, pipe=object(), loaded_embedding_files={},
        embedding_token_owners={}, embedding_snapshots={})
    events = []
    engine._with_embedding_encoder_access = lambda fn: (events.append('guard'), fn())[1]
    class Context:
        def check_interrupt(self): events.append('check')
        def publish(self, fn): events.append('publish'); return fn()
    def validate(active, paths): events.append(('validate', paths))
    def refresh(paths):
        events.append(('load', paths))
        engine.loaded_embedding_files['concept'] = str(path)
        engine.embedding_token_owners['concept'] = {'clip_l': ['concept']}
        engine.embedding_snapshots['concept'] = FileSnapshot.capture(path)
    read = lambda *a, **kw: {'vectors': {'clip_l': [[1.]]}, 'ranges': {}, 'guidance': []}
    params = {'operation': 'open', 'name': 'concept', 'path': str(path)}
    result = service.run_editor_service(engine, params, Context(), read=read, refresh=refresh, validate=validate)
    assert result['name'] == 'concept' and events.count('publish') == 1
    assert events.index('publish') < events.index('guard') < next(i for i, x in enumerate(events) if isinstance(x, tuple) and x[0] == 'load')
    events.clear()
    service.run_editor_service(engine, params, Context(), read=read, refresh=refresh, validate=validate)
    assert 'publish' not in events


def test_open_catalog_refuses_name_collision_and_interruption(tmp_path):
    service = importlib.import_module('sdfx.embedding_editor_service')
    path = tmp_path / 'concept.safetensors'; path.write_bytes(b'fixture')
    other = tmp_path / 'other' / 'concept.safetensors'; other.parent.mkdir(); other.write_bytes(b'other')
    engine = SimpleNamespace(model={'model_token': 'm'}, pipe=object(), loaded_embedding_files={'concept': str(other)})
    class Context:
        interrupted = False
        def check_interrupt(self):
            if self.interrupted: raise RuntimeError('interrupted')
        def publish(self, fn): pytest.fail('must not mutate')
    with pytest.raises(ValueError, match='already loaded'):
        service.run_editor_service(engine, {'operation': 'open', 'name': 'concept', 'path': str(path)}, Context(),
                                   validate=lambda *_: None)
    engine.loaded_embedding_files = {}
    ctx = Context(); ctx.interrupted = True
    with pytest.raises(RuntimeError, match='interrupted'):
        service.run_editor_service(engine, {'operation': 'open', 'name': 'concept', 'path': str(path)}, ctx,
                                   validate=lambda *_: None)


def test_open_catalog_reports_unloaded_checkpoint_if_activation_rollback_failed(tmp_path):
    from sdfx.protocol import EngineError, NO_MODEL
    service = importlib.import_module('sdfx.embedding_editor_service')
    path = tmp_path / 'concept.safetensors'; path.write_bytes(b'fixture')
    engine = SimpleNamespace(model={'model_token': 'm'}, pipe=object(), loaded_embedding_files={})
    engine._with_embedding_encoder_access = lambda fn: fn()
    class Context:
        def check_interrupt(self): pass
        def publish(self, fn): return fn()
    def failed_refresh(paths):
        engine.pipe = None
        raise RuntimeError('rollback failed')
    with pytest.raises(EngineError) as failure:
        service.run_editor_service(engine, {'operation': 'open', 'name': 'concept', 'path': str(path)}, Context(),
                                   refresh=failed_refresh, validate=lambda *_: None)
    assert failure.value.code == NO_MODEL
