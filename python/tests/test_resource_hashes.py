import hashlib
import pytest
from sdfx.resource_hashes import file_resource_hash


def test_conflicting_filenames_are_omitted_instead_of_misattributed(tmp_path):
    from sdfx.resource_hashes import ResourceHashes, record_file
    first, second = tmp_path / 'one' / 'same.pt', tmp_path / 'two' / 'same.pt'
    first.parent.mkdir(); second.parent.mkdir()
    first.write_bytes(b'one'); second.write_bytes(b'two')
    resources, warnings = ResourceHashes(), []
    for path in (first, second, first):
        record_file(resources, 'adetailer', path, warnings.append)
    assert not resources
    assert warnings


def test_only_current_prompt_embedding_tokens_are_recorded(tmp_path):
    from sdfx.resource_hashes import embedding_hashes
    used = tmp_path / 'used.safetensors'
    used.write_bytes(b'used')
    files = {'token': str(used), 'other': str(tmp_path / 'missing')}
    assert embedding_hashes(files, ['a token, portrait']) == {'embed:token': file_resource_hash(used)}
    assert embedding_hashes(files, ['token-extra tokenizer']) == {}


def test_resource_hash_is_actual_file_sha256(tmp_path):
    path = tmp_path / 'model.safetensors'
    path.write_bytes(b'actual weight bytes')
    assert file_resource_hash(path) == hashlib.sha256(path.read_bytes()).hexdigest()[:10]


def test_directory_and_missing_resource_are_not_guessed(tmp_path):
    with pytest.raises(ValueError, match='file'):
        file_resource_hash(tmp_path)
    with pytest.raises(ValueError, match='file'):
        file_resource_hash(tmp_path / 'missing')


@pytest.mark.parametrize('fail', [False, True])
def test_embedding_provenance_records_only_successful_load(tmp_path, fail):
    torch = pytest.importorskip('torch')
    from safetensors.torch import save_file
    from types import SimpleNamespace
    from sdfx.engine import Engine
    path = tmp_path / 'my_token.safetensors'
    save_file({'emb_params': torch.zeros(1, 8)}, str(path))
    class Tokenizer:
        def __init__(self): self.added = {}
        def get_added_vocab(self): return dict(self.added)
    class Pipe:
        def __init__(self): self.tokenizer = Tokenizer()
        def load_textual_inversion(self, *args, **kwargs):
            if fail:
                raise ValueError('incompatible embedding')
            self.tokenizer.added[kwargs['token']] = 1
    engine = Engine.__new__(Engine)
    engine.pipe, engine.sdxl = Pipe(), False
    engine.loaded_embeddings = set()
    engine.loaded_embedding_files = {}
    engine.embedding_token_owners = {}
    engine._load_embeddings(SimpleNamespace(embedding_paths=[str(path)]), ['my_token'], SimpleNamespace(warn=lambda _: None))
    assert engine.loaded_embedding_files == ({} if fail else {'my_token': str(path)})
    assert engine.embedding_token_owners == ({} if fail else {'my_token': {'clip_l': ['my_token']}})


def test_snapshot_rejects_replaced_or_modified_loaded_file(tmp_path):
    from sdfx.resource_hashes import FileSnapshot
    source = tmp_path / 'weights.pt'
    source.write_bytes(b'loaded bytes')
    snapshot = FileSnapshot.capture(source)
    assert snapshot.digest() == file_resource_hash(source)
    source.write_bytes(b'new weights on disk')
    with pytest.raises(ValueError, match='changed since loading'):
        snapshot.digest()


def test_changed_cached_resource_is_omitted_with_warning(tmp_path):
    from sdfx.resource_hashes import FileSnapshot, ResourceHashes, record_file
    path = tmp_path / 'cached.pt'
    path.write_bytes(b'loaded')
    snapshot = FileSnapshot.capture(path)
    path.write_bytes(b'changed')
    resources, warnings = ResourceHashes(), []
    record_file(resources, 'controlnet', path, warnings.append, snapshot)
    assert resources == {}
    assert len(warnings) == 1 and 'changed since loading' in warnings[0]
