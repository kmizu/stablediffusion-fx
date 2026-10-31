import hashlib
import importlib
import json
import pytest


def module(): return importlib.import_module('sdfx.embedding_editor_pt')


def load(path): return json.loads(path.read_text(encoding='utf-8'))
def dump(value, stream): stream.write(json.dumps(value, sort_keys=True).encode('utf-8'))
def tensor(old, rows):
    if len(old) != len(rows) or any(len(a) != len(b) for a, b in zip(old, rows)):
        raise ValueError('embedding shape mismatch')
    return [row[:] for row in rows]


def test_pt_copy_preserves_training_metadata_and_unrelated_payload(tmp_path):
    source = tmp_path / 'source.pt'; target = tmp_path / 'copy.pt'
    initial = {'string_to_param': {'*': [[1., 2.], [3., 4.]], 'aux': [[9.]]}, 'step': 42,
               'sd_checkpoint_name': 'test', 'other': {'kept': True}}
    source.write_text(json.dumps(initial), encoding='utf-8')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    module().build_pt_copy(source, target, {'clip_l': [[1., 5.], [3., 4.]]}, digest,
                           load=load, dump=dump, tensor=tensor)
    result = load(target)
    assert result['string_to_param']['*'] == [[1., 5.], [3., 4.]]
    assert result['string_to_param']['aux'] == [[9.]] and result['step'] == 42
    assert result['sd_checkpoint_name'] == 'test' and result['other'] == {'kept': True}
    assert load(source) == initial


def test_pt_copy_rejects_stale_or_incompatible_without_clobber(tmp_path):
    source = tmp_path / 'source.pt'; target = tmp_path / 'copy.pt'
    source.write_text(json.dumps({'string_to_param': {'*': [[1., 2.]]}, 'step': 42}), encoding='utf-8')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='changed'):
        module().build_pt_copy(source, target, {'clip_l': [[1., 2.]]}, '0'*64, load=load, dump=dump, tensor=tensor)
    assert not target.exists()
    with pytest.raises(ValueError, match='shape'):
        module().build_pt_copy(source, target, {'clip_l': [[1.]]}, digest, load=load, dump=dump, tensor=tensor)
    assert not target.exists()
    target.write_bytes(b'keep')
    with pytest.raises(FileExistsError):
        module().build_pt_copy(source, target, {'clip_l': [[1., 2.]]}, digest, load=load, dump=dump, tensor=tensor)
    assert target.read_bytes() == b'keep'


def test_pt_copy_preserves_unchanged_source_precision_with_baseline(tmp_path):
    source = tmp_path / 'source.pt'; target = tmp_path / 'copy.pt'
    initial = {'string_to_param': {'*': [[1.0000000001, 2.]]}, 'step': 42}
    source.write_text(json.dumps(initial), encoding='utf-8')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    def preserve(old, rows, baseline):
        return [[old[i][j] if value == baseline[i][j] else value for j, value in enumerate(row)]
                for i, row in enumerate(rows)]
    module().build_pt_copy(source, target, {'clip_l': [[1., 9.]]}, digest,
        baseline={'clip_l': [[1., 2.]]}, load=load, dump=dump, tensor=preserve)
    assert load(target)['string_to_param']['*'] == [[1.0000000001, 9.]]


def test_pt_copy_preserves_dual_encoder_and_extra_fields(tmp_path):
    source = tmp_path / 'dual.pt'; target = tmp_path / 'copy.pt'
    initial = {'clip_l': [[1., 2.], [3., 4.]], 'clip_g': [[5., 6., 7.], [8., 9., 10.]],
               'step': 22, 'sd_checkpoint_name': 'original', 'unrelated': 'keep'}
    source.write_text(json.dumps(initial), encoding='utf-8')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    values = {'clip_l': [[1., 7.], [3., 4.]], 'clip_g': [[5., 6., 7.], [8., 9., 10.]]}
    def preserve(old, rows, baseline):
        return [[old[i][j] if value == baseline[i][j] else value for j, value in enumerate(row)]
                for i, row in enumerate(rows)]
    module().build_pt_copy(source, target, values, digest, baseline={'clip_l': initial['clip_l'], 'clip_g': initial['clip_g']},
                           load=load, dump=dump, tensor=preserve)
    result = load(target)
    assert result['clip_l'] == values['clip_l'] and result['clip_g'] == initial['clip_g']
    assert result['step'] == 22 and result['sd_checkpoint_name'] == 'original' and result['unrelated'] == 'keep'


@pytest.mark.parametrize('filename,initial,expected', [
    ('single.bin', {'custom_embedding': [[1., 2.]]}, {'custom_embedding': [[1., 7.]]}),
    ('raw.pt', [[1., 2.]], [[1., 7.]]),
])
def test_pt_copy_supports_reader_accepted_single_key_and_tensor_root(tmp_path, filename, initial, expected):
    source = tmp_path / filename; target = tmp_path / ('edited-' + filename)
    source.write_text(json.dumps(initial), encoding='utf-8')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    module().build_pt_copy(source, target, {'clip_l': [[1., 7.]]}, digest,
                           load=load, dump=dump, tensor=tensor)
    assert load(target) == expected
    assert load(source) == initial
