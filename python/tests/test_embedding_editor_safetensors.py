import hashlib
import importlib
import json
import struct
import pytest


def module(): return importlib.import_module('sdfx.embedding_editor_safetensors')


def fixture(path):
    header = {'__metadata__': {'step': '42', 'sd_checkpoint_name': 'model'},
              'clip_l': {'dtype': 'F32', 'shape': [2, 2], 'data_offsets': [0, 16]},
              'unrelated': {'dtype': 'U8', 'shape': [4], 'data_offsets': [16, 20]}}
    encoded = json.dumps(header, separators=(',', ':')).encode('utf-8')
    payload = struct.pack('<Q', len(encoded)) + encoded + struct.pack('<4f', 1., 2., 3., 4.) + b'abcd'
    path.write_bytes(payload)
    return payload, 8 + len(encoded)


def test_build_edit_preserves_header_other_vector_and_unrelated_tensor(tmp_path):
    src = tmp_path / 'in.safetensors'; dest = tmp_path / 'new.safetensors'
    original, data = fixture(src)
    digest = hashlib.sha256(original).hexdigest()
    module().build_safetensors_copy(src, dest, {'clip_l': [[1., 9.], [3., 4.]]}, digest)
    result = dest.read_bytes()
    assert src.read_bytes() == original
    assert result[:data + 4] == original[:data + 4]
    assert struct.unpack('<f', result[data + 4:data + 8])[0] == 9.
    assert result[data + 8:] == original[data + 8:]


def test_build_edit_rejects_stale_source_or_wrong_shape_without_output(tmp_path):
    src = tmp_path / 'in.safetensors'; dest = tmp_path / 'new.safetensors'
    original, _ = fixture(src)
    with pytest.raises(ValueError, match='changed'):
        module().build_safetensors_copy(src, dest, {'clip_l': [[1., 2.], [3., 4.]]}, '0'*64)
    assert not dest.exists()
    with pytest.raises(ValueError, match='shape'):
        module().build_safetensors_copy(src, dest, {'clip_l': [[1., 2.]]}, hashlib.sha256(original).hexdigest())
    assert not dest.exists() and src.read_bytes() == original


def test_build_edit_never_overwrites_existing_destination(tmp_path):
    src = tmp_path / 'in.safetensors'; dest = tmp_path / 'new.safetensors'
    original, _ = fixture(src); dest.write_bytes(b'keep')
    with pytest.raises(FileExistsError):
        module().build_safetensors_copy(src, dest, {'clip_l': [[1., 2.], [3., 4.]]}, hashlib.sha256(original).hexdigest())
    assert dest.read_bytes() == b'keep'


def test_bfloat16_overflow_is_rejected_before_copy(tmp_path):
    src = tmp_path / 'in.safetensors'; dest = tmp_path / 'new.safetensors'
    header = {'clip_l': {'dtype': 'BF16', 'shape': [1, 1], 'data_offsets': [0, 2]}}
    encoded = json.dumps(header, separators=(',', ':')).encode('utf-8')
    payload = struct.pack('<Q', len(encoded)) + encoded + b'\x80\x3f'
    src.write_bytes(payload)
    with pytest.raises(ValueError, match='dtype range'):
        module().build_safetensors_copy(src, dest, {'clip_l': [[3.4e38]]}, hashlib.sha256(payload).hexdigest())
    assert not dest.exists() and src.read_bytes() == payload


def test_unchanged_draft_coordinate_preserves_source_precision(tmp_path):
    src = tmp_path / 'in.safetensors'; dest = tmp_path / 'new.safetensors'
    header = {'clip_l': {'dtype': 'F64', 'shape': [1, 2], 'data_offsets': [0, 16]}}
    encoded = json.dumps(header, separators=(',', ':')).encode('utf-8')
    payload = struct.pack('<Q', len(encoded)) + encoded + struct.pack('<2d', 1.0000000001, 2.)
    src.write_bytes(payload)
    module().build_safetensors_copy(src, dest, {'clip_l': [[1., 9.]]}, hashlib.sha256(payload).hexdigest(),
                                   baseline={'clip_l': [[1., 2.]]})
    output = dest.read_bytes(); start = 8 + len(encoded)
    assert output[start:start+8] == payload[start:start+8]
    assert struct.unpack('<d', output[start+8:start+16])[0] == 9.


def test_safetensors_copy_edits_two_encoder_streams_and_preserves_extra(tmp_path):
    src = tmp_path / 'dual.safetensors'; dest = tmp_path / 'copy.safetensors'
    header = {'__metadata__': {'step': '8'},
              'clip_l': {'dtype': 'F32', 'shape': [1, 2], 'data_offsets': [0, 8]},
              'clip_g': {'dtype': 'F32', 'shape': [1, 3], 'data_offsets': [8, 20]},
              'extra': {'dtype': 'U8', 'shape': [2], 'data_offsets': [20, 22]}}
    encoded = json.dumps(header, separators=(',', ':')).encode('utf-8')
    payload = struct.pack('<Q', len(encoded)) + encoded + struct.pack('<5f', 1., 2., 3., 4., 5.) + b'zz'
    src.write_bytes(payload)
    values = {'clip_l': [[1., 9.]], 'clip_g': [[3., 4., 7.]]}
    baseline = {'clip_l': [[1., 2.]], 'clip_g': [[3., 4., 5.]]}
    module().build_safetensors_copy(src, dest, values, hashlib.sha256(payload).hexdigest(), baseline=baseline)
    result = dest.read_bytes(); start = 8 + len(encoded)
    assert result[:start] == payload[:start]
    assert struct.unpack('<5f', result[start:start+20]) == (1., 9., 3., 4., 7.)
    assert result[start+20:] == b'zz' and src.read_bytes() == payload
