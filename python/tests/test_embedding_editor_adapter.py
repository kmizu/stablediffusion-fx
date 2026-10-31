from types import SimpleNamespace
import importlib
import os
import pytest


def test_model_adapter_reads_loaded_embedding_and_fingerprints_source(tmp_path):
    mod = importlib.import_module('sdfx.embedding_editor_adapter')
    path = tmp_path / 'concept.safetensors'; path.write_bytes(b'original bytes')
    from sdfx.resource_hashes import FileSnapshot
    engine = SimpleNamespace(model={'model_token': 'model-a'}, loaded_embedding_files={'concept': str(path)},
                             embedding_token_owners={'concept': {'clip_l': ['concept']}},
                             embedding_snapshots={'concept': FileSnapshot.capture(path)})
    read = lambda engine, name, refs: {'vectors': {'clip_l': [[1., 2.]]}, 'ranges': {}, 'guidance': {}}
    adapter = mod.EngineEditorAdapter(engine, read_vectors=read)
    snapshot = adapter.read('concept', [])
    assert snapshot['path'] == str(path) and snapshot['model_token'] == 'model-a'
    assert snapshot['vectors'] == {'clip_l': [[1., 2.]]}
    assert snapshot['file_identity'] == adapter.current('concept')['file_identity']
    # ujson uses binary64 numbers: nanosecond timestamps cannot be sent as JSON numbers.
    assert all(isinstance(value, str) for value in snapshot['file_identity'])


def test_model_adapter_detects_changed_file_even_when_size_and_mtime_match(tmp_path):
    mod = importlib.import_module('sdfx.embedding_editor_adapter')
    path = tmp_path / 'concept.safetensors'; path.write_bytes(b'abcdefgh')
    from sdfx.resource_hashes import FileSnapshot
    engine = SimpleNamespace(model={'model_token': 'model-a'}, loaded_embedding_files={'concept': str(path)},
                             embedding_token_owners={'concept': {'clip_l': ['concept']}},
                             embedding_snapshots={'concept': FileSnapshot.capture(path)})
    read = lambda engine, name, refs: {'vectors': {'clip_l': [[1., 2.]]}, 'ranges': {}, 'guidance': {}}
    adapter = mod.EngineEditorAdapter(engine, read_vectors=read)
    before = adapter.current('concept'); timestamp = path.stat().st_mtime_ns
    path.write_bytes(b'ABCDEFGH'); os.utime(path, ns=(timestamp, timestamp))
    try:
        after = adapter.current('concept')
    except ValueError as error:
        assert 'changed' in str(error)
    else:
        assert before['file_identity'] != after['file_identity']


def test_model_adapter_rejects_untracked_or_ambiguous_embedding(tmp_path):
    mod = importlib.import_module('sdfx.embedding_editor_adapter')
    path = tmp_path / 'concept.safetensors'; path.write_bytes(b'x')
    engine = SimpleNamespace(model={'model_token': 'model-a'}, loaded_embedding_files={'concept': str(path), 'CONCEPT': str(path)},
                             embedding_token_owners={})
    adapter = mod.EngineEditorAdapter(engine, read_vectors=lambda *_: {})
    with pytest.raises(ValueError, match='ambiguous'):
        adapter.current('concept')
    engine.loaded_embedding_files = {'concept': str(path)}
    with pytest.raises(ValueError, match='ownership'):
        adapter.current('concept')


def test_model_adapter_rechecks_stamp_before_publish_and_preview(tmp_path):
    from contextlib import contextmanager
    from sdfx.resource_hashes import FileSnapshot
    mod = importlib.import_module('sdfx.embedding_editor_adapter')
    path = tmp_path / 'concept.safetensors'; path.write_bytes(b'abcdefgh')
    engine = SimpleNamespace(model={'model_token': 'model-a'}, loaded_embedding_files={'concept': str(path)},
        embedding_token_owners={'concept': {'clip_l': ['concept']}}, embedding_snapshots={'concept': FileSnapshot.capture(path)})
    events = []
    def publish(engine, name, values, expected):
        events.append('publish'); return {'path': str(path)}
    @contextmanager
    def preview(engine, name, values, expected):
        events.append('install')
        try: yield
        finally: events.append('restore')
    adapter = mod.EngineEditorAdapter(engine, read_vectors=lambda *_: {}, publish_vectors=publish,
                                      preview_vectors=preview, generate=lambda *_: {})
    expected = adapter.current('concept')
    values = {'clip_l': [[1., 2.]]}
    with adapter.preview_scope('concept', values, expected=expected):
        events.append('generate')
    assert events == ['install', 'generate', 'restore']
    path.write_bytes(b'ABCDEFGH')
    with pytest.raises(ValueError, match='changed'):
        adapter.publish('concept', values, expected=expected)
    assert 'publish' not in events
