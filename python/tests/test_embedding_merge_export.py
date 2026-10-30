import json
import struct
import numpy as np
import pytest

from sdfx.embedding_merge_export import export_embeddings


def read(path):
    raw = path.read_bytes()
    size = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8 + size])
    values = {}
    for name, descriptor in header.items():
        if name == '__metadata__':
            continue
        start, end = descriptor['data_offsets']
        values[name] = np.frombuffer(raw[8 + size + start:8 + size + end], dtype='<f4').reshape(descriptor['shape'])
    return values


def test_sd1_safetensors_and_zero_g_sdxl_conversion(tmp_path):
    rows = np.arange(1536, dtype=np.float32).reshape(2, 768)
    paths = export_embeddings(tmp_path, 'mixed', {'clip_l': rows})
    assert [p.relative_to(tmp_path).as_posix() for p in paths] == ['mixed.safetensors', 'sdxl/mixed.safetensors']
    np.testing.assert_array_equal(read(paths[0])['emb_params'], rows)
    converted = read(paths[1])
    np.testing.assert_array_equal(converted['clip_l'], rows)
    assert converted['clip_g'].shape == (2, 1280)
    assert not converted['clip_g'].any()


def test_sdxl_safetensors_and_sd1_l_conversion(tmp_path):
    left, right = np.ones((1, 768), dtype=np.float32), np.full((1, 1280), 2, dtype=np.float32)
    paths = export_embeddings(tmp_path, '双', {'clip_l': left, 'clip_g': right})
    np.testing.assert_array_equal(read(paths[0])['clip_g'], right)
    np.testing.assert_array_equal(read(paths[1])['emb_params'], left)


def test_zero_l_skips_conversion_and_sd2_does_not_invent_sdxl(tmp_path):
    assert len(export_embeddings(tmp_path, 'zero', {'clip_l': np.zeros((1, 768))})) == 1
    assert len(export_embeddings(tmp_path, 'sd2', {'clip_l': np.ones((1, 1024))})) == 1


def test_collision_and_cancel_do_not_create_any_destination(tmp_path):
    (tmp_path / 'sdxl').mkdir()
    existing = tmp_path / 'sdxl/test.safetensors'
    existing.write_bytes(b'keep')
    with pytest.raises(FileExistsError):
        export_embeddings(tmp_path, 'test', {'clip_l': np.ones((1, 768))})
    assert not (tmp_path / 'test.safetensors').exists()
    assert existing.read_bytes() == b'keep'
    with pytest.raises(RuntimeError, match='cancel'):
        export_embeddings(tmp_path, 'other', {'clip_l': np.ones((1, 768))},
                          check=lambda: (_ for _ in ()).throw(RuntimeError('cancel')))
    assert not (tmp_path / 'other.safetensors').exists()


@pytest.mark.parametrize('name', ['../escape', 'CON', 'a/b', 'x.', ''])
def test_invalid_names_fail_without_side_effect(tmp_path, name):
    with pytest.raises(ValueError):
        export_embeddings(tmp_path, name, {'clip_l': np.ones((1, 768))})
    assert list(tmp_path.iterdir()) == []


def test_late_conversion_collision_rolls_back_primary_without_touching_racing_file(tmp_path, monkeypatch):
    import os
    link = os.link
    target = tmp_path / 'sdxl/race.safetensors'
    def racing_link(source, destination):
        if destination == target:
            target.write_bytes(b'other-owner')
        return link(source, destination)
    monkeypatch.setattr(os, 'link', racing_link)
    with pytest.raises(FileExistsError):
        export_embeddings(tmp_path, 'race', {'clip_l': np.ones((1, 768))})
    assert not (tmp_path / 'race.safetensors').exists()
    assert target.read_bytes() == b'other-owner'
    assert len(list(tmp_path.rglob('*'))) == 2  # conversion dir and racer, no private temps
