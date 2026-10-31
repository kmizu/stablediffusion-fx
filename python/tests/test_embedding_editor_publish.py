from types import SimpleNamespace
from pathlib import Path
import importlib
import pytest


def module(): return importlib.import_module('sdfx.embedding_editor_publish')


def test_publish_rechecks_before_write_and_refreshes_full_draft():
    engine = SimpleNamespace()
    expected = {'model_token': 'm', 'path': 'source', 'file_identity': ('dev', 'sha')}
    events = []
    values = {'clip_l': [[1., 2.]], 'clip_g': [[3., 4., 5.]]}
    def current(name): events.append('identity'); return expected.copy()
    def write(path, vectors): events.append(('write', path, vectors))
    def refresh(paths): events.append(('refresh', paths))
    result = module().publish_editor(engine, 'concept', values, expected, current=current, write=write, refresh=refresh)
    assert result['reloaded'] and events == ['identity', ('write', 'source', values), ('refresh', ['source'])]


def test_publish_fails_closed_if_refresh_fails_after_file_was_written():
    engine = SimpleNamespace(); expected = {'model_token': 'm', 'path': 'source', 'file_identity': ('dev', 'sha')}
    written = []
    def fail(_): raise RuntimeError('refresh error')
    with pytest.raises(RuntimeError, match='saved.*reload'):
        module().publish_editor(engine, 'concept', {'clip_l': [[1.]]}, expected,
            current=lambda _: expected, write=lambda path, values: written.append(path), refresh=fail)
    assert written == ['source'] and 'refresh error' in engine._embedding_reload_error


def test_publish_rejects_stale_file_before_any_write():
    engine = SimpleNamespace(); expected = {'model_token': 'm', 'path': 'source', 'file_identity': ('a',)}
    actual = {'model_token': 'm', 'path': 'source', 'file_identity': ('b',)}; written = []
    with pytest.raises(ValueError, match='file changed'):
        module().publish_editor(engine, 'concept', {'clip_l': [[1.]]}, expected,
            current=lambda _: actual, write=lambda path, values: written.append(path), refresh=lambda _: None)
    assert not written


def test_model_save_retains_verified_backup_and_full_metadata(tmp_path):
    import hashlib
    import json
    import struct
    source = tmp_path / 'concept.safetensors'
    header = {'__metadata__': {'step': '42'}, 'clip_l': {'dtype': 'F32', 'shape': [1, 2], 'data_offsets': [0, 8]},
              'extra': {'dtype': 'U8', 'shape': [2], 'data_offsets': [8, 10]}}
    encoded = json.dumps(header, separators=(',', ':')).encode('utf-8')
    original = struct.pack('<Q', len(encoded)) + encoded + struct.pack('<2f', 1., 2.) + b'zz'
    source.write_bytes(original); digest = hashlib.sha256(original).hexdigest()
    expected = {'model_token': 'm', 'path': str(source), 'file_identity': ('id', digest)}
    engine = SimpleNamespace(); calls = []
    result = module().publish_safetensors_editor(engine, 'concept', {'clip_l': [[1., 7.]]}, expected,
        baseline={'clip_l': [[1., 2.]]}, current=lambda _: expected.copy(), refresh=lambda paths: calls.append(paths))
    assert result['reloaded'] and calls == [[str(source)]]
    assert Path(result['backup']).read_bytes() == original
    updated = source.read_bytes(); start = 8 + len(encoded)
    assert updated[:start] == original[:start] and updated[start+8:] == b'zz'
    assert struct.unpack('<2f', updated[start:start+8]) == (1., 7.)


def test_preexisting_stage_path_is_never_deleted(tmp_path, monkeypatch):
    import hashlib, json, struct, uuid
    source = tmp_path / 'concept.safetensors'
    header = {'clip_l': {'dtype': 'F32', 'shape': [1, 1], 'data_offsets': [0, 4]}}
    encoded = json.dumps(header, separators=(',', ':')).encode('utf-8')
    original = struct.pack('<Q', len(encoded)) + encoded + struct.pack('<f', 1.)
    source.write_bytes(original)
    marker = '0123456789abcdef0123456789abcdef'
    stage = tmp_path / ('.concept.safetensors.editing-' + marker); stage.write_bytes(b'keep')
    monkeypatch.setattr(uuid, 'uuid4', lambda: SimpleNamespace(hex=marker))
    expected = {'model_token': 'm', 'path': str(source), 'file_identity': ('id', hashlib.sha256(original).hexdigest())}
    with pytest.raises(FileExistsError):
        module().publish_safetensors_editor(SimpleNamespace(), 'concept', {'clip_l': [[2.]]}, expected,
            baseline={'clip_l': [[1.]]}, current=lambda _: expected, refresh=lambda _: None)
    assert stage.read_bytes() == b'keep' and source.read_bytes() == original


def test_interruption_after_replace_keeps_backup_and_blocks_generation(tmp_path, monkeypatch):
    import hashlib, json, os, struct
    source = tmp_path / 'concept.safetensors'
    header = {'clip_l': {'dtype': 'F32', 'shape': [1, 1], 'data_offsets': [0, 4]}}
    encoded = json.dumps(header, separators=(',', ':')).encode('utf-8')
    original = struct.pack('<Q', len(encoded)) + encoded + struct.pack('<f', 1.)
    source.write_bytes(original)
    expected = {'model_token': 'm', 'path': str(source), 'file_identity': ('id', hashlib.sha256(original).hexdigest())}
    engine = SimpleNamespace(); original_replace = os.replace
    def interrupted(src, dst):
        original_replace(src, dst)
        raise KeyboardInterrupt('after replace')
    monkeypatch.setattr(os, 'replace', interrupted)
    with pytest.raises(RuntimeError, match='reload'):
        module().publish_safetensors_editor(engine, 'concept', {'clip_l': [[2.]]}, expected,
            baseline={'clip_l': [[1.]]}, current=lambda _: expected, refresh=lambda _: None)
    backups = list(tmp_path.glob('concept.safetensors.before-edit-*.backup'))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    assert engine._embedding_reload_error


def test_pt_model_save_preserves_metadata_and_backup(tmp_path):
    import hashlib, json
    source = tmp_path / 'concept.pt'
    original = json.dumps({'string_to_param': {'*': [[1., 2.]]}, 'step': 42, 'extra': 'keep'}).encode()
    source.write_bytes(original)
    expected = {'model_token': 'm', 'path': str(source), 'file_identity': ('id', hashlib.sha256(original).hexdigest())}
    def load(path): return json.loads(path.read_text(encoding='utf-8'))
    def dump(data, stream): stream.write(json.dumps(data).encode())
    def tensor(old, rows, baseline): return [row[:] for row in rows]
    result = module().publish_pt_editor(SimpleNamespace(), 'concept', {'clip_l': [[1., 7.]]}, expected,
        baseline={'clip_l': [[1., 2.]]}, current=lambda _: expected, refresh=lambda _: None,
        load=load, dump=dump, tensor=tensor)
    assert json.loads(source.read_bytes())['step'] == 42
    assert json.loads(source.read_bytes())['extra'] == 'keep'
    assert Path(result['backup']).read_bytes() == original


def test_model_save_keeps_backup_if_live_refresh_fails(tmp_path):
    import hashlib, json, struct
    source = tmp_path / 'concept.safetensors'
    header = {'clip_l': {'dtype': 'F32', 'shape': [1, 1], 'data_offsets': [0, 4]}}
    encoded = json.dumps(header, separators=(',', ':')).encode('utf-8')
    original = struct.pack('<Q', len(encoded)) + encoded + struct.pack('<f', 1.)
    source.write_bytes(original)
    expected = {'model_token': 'm', 'path': str(source), 'file_identity': ('id', hashlib.sha256(original).hexdigest())}
    engine = SimpleNamespace()
    def fail(_): raise RuntimeError('refresh failed')
    with pytest.raises(RuntimeError, match='saved.*reload'):
        module().publish_safetensors_editor(engine, 'concept', {'clip_l': [[2.]]}, expected,
            baseline={'clip_l': [[1.]]}, current=lambda _: expected, refresh=fail)
    backups = list(tmp_path.glob('concept.safetensors.before-edit-*.backup'))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    assert 'refresh failed' in engine._embedding_reload_error
    assert source.read_bytes() != original


def test_external_replace_between_final_check_and_publication_survives(tmp_path):
    import hashlib, json, struct
    source = tmp_path / 'concept.safetensors'
    header = {'clip_l': {'dtype': 'F32', 'shape': [1, 1], 'data_offsets': [0, 4]}}
    encoded = json.dumps(header, separators=(',', ':')).encode('utf-8')
    original = struct.pack('<Q', len(encoded)) + encoded + struct.pack('<f', 1.)
    external = struct.pack('<Q', len(encoded)) + encoded + struct.pack('<f', 9.)
    source.write_bytes(original)
    expected = {'model_token': 'm', 'path': str(source), 'file_identity': ('id', hashlib.sha256(original).hexdigest())}
    calls = 0
    def current(_):
        nonlocal calls
        calls += 1
        if calls == 3:
            source.write_bytes(external)
        return expected
    with pytest.raises(ValueError, match='changed'):
        module().publish_safetensors_editor(SimpleNamespace(), 'concept', {'clip_l': [[2.]]}, expected,
            baseline={'clip_l': [[1.]]}, current=current, refresh=lambda _: None)
    assert source.read_bytes() == external
    assert not list(tmp_path.glob('.concept.safetensors.editing-*'))


def test_external_create_while_source_is_vacated_is_never_overwritten(tmp_path, monkeypatch):
    import hashlib, json, os, struct
    source = tmp_path / 'concept.safetensors'
    header = {'clip_l': {'dtype': 'F32', 'shape': [1, 1], 'data_offsets': [0, 4]}}
    encoded = json.dumps(header, separators=(',', ':')).encode('utf-8')
    original = struct.pack('<Q', len(encoded)) + encoded + struct.pack('<f', 1.)
    external = struct.pack('<Q', len(encoded)) + encoded + struct.pack('<f', 9.)
    source.write_bytes(original)
    expected = {'model_token': 'm', 'path': str(source), 'file_identity': ('id', hashlib.sha256(original).hexdigest())}
    original_link = os.link
    def racing_link(src, dst):
        if Path(dst) == source and Path(src).name.startswith('.concept.safetensors.editing-'):
            source.write_bytes(external)
        return original_link(src, dst)
    monkeypatch.setattr(os, 'link', racing_link)
    with pytest.raises(FileExistsError):
        module().publish_safetensors_editor(SimpleNamespace(), 'concept', {'clip_l': [[2.]]}, expected,
            baseline={'clip_l': [[1.]]}, current=lambda _: expected, refresh=lambda _: None)
    assert source.read_bytes() == external
    backups = list(tmp_path.glob('concept.safetensors.before-edit-*.backup'))
    assert len(backups) == 1 and backups[0].read_bytes() == original


def test_interruption_after_hardlink_marks_published_model_for_reload(tmp_path, monkeypatch):
    import hashlib, json, os, struct
    source = tmp_path / 'concept.safetensors'
    header = {'clip_l': {'dtype': 'F32', 'shape': [1, 1], 'data_offsets': [0, 4]}}
    encoded = json.dumps(header, separators=(',', ':')).encode('utf-8')
    original = struct.pack('<Q', len(encoded)) + encoded + struct.pack('<f', 1.)
    source.write_bytes(original)
    expected = {'model_token': 'm', 'path': str(source), 'file_identity': ('id', hashlib.sha256(original).hexdigest())}
    engine = SimpleNamespace(); original_link = os.link
    def interrupted(src, dst):
        original_link(src, dst)
        raise KeyboardInterrupt('after hardlink')
    monkeypatch.setattr(os, 'link', interrupted)
    with pytest.raises(KeyboardInterrupt, match='after hardlink'):
        module().publish_safetensors_editor(engine, 'concept', {'clip_l': [[2.]]}, expected,
            baseline={'clip_l': [[1.]]}, current=lambda _: expected, refresh=lambda _: None)
    assert source.read_bytes() != original
    backups = list(tmp_path.glob('concept.safetensors.before-edit-*.backup'))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    assert engine._embedding_reload_error
