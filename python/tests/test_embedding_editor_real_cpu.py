"""Actual small Diffusers/CLIP tests; do not require a GPU."""
import json
from types import SimpleNamespace
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('diffusers')


@pytest.fixture
def engine_fixture(tmp_path):
    from diffusers.loaders import TextualInversionLoaderMixin
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
    from safetensors.torch import save_file
    from sdfx.resource_hashes import FileSnapshot
    vocab = tmp_path / 'vocab.json'; merges = tmp_path / 'merges.txt'
    vocab.write_text(json.dumps({'base</w>': 0, '<|startoftext|>': 1, '<|endoftext|>': 2}), encoding='utf-8')
    merges.write_text('#version: 0.2\n', encoding='utf-8')
    class Pipe(TextualInversionLoaderMixin):
        hf_device_map = None
        components = {}
    pipe = Pipe(); pipe.tokenizer = CLIPTokenizer(str(vocab), str(merges))
    pipe.text_encoder = CLIPTextModel(CLIPTextConfig(vocab_size=3, hidden_size=4, intermediate_size=8,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8))
    pipe.tokenizer_2 = None; pipe.text_encoder_2 = None
    source = tmp_path / 'concept.safetensors'
    original = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    save_file({'clip_l': original}, str(source), metadata={'step': '42', 'sd_checkpoint_name': 'original'})
    pipe.load_textual_inversion(original, token='concept')
    engine = SimpleNamespace(pipe=pipe, model={'model_token': 'm'}, sdxl=False, loaded_embeddings={'concept'},
        loaded_embedding_files={'concept': str(source)}, embedding_token_owners={'concept': {'clip_l': ['concept', 'concept_1']}},
        embedding_snapshots={'concept': FileSnapshot.capture(source)}, derived={})
    engine._with_embedding_encoder_access = lambda fn: fn()
    return engine, source, original


def test_actual_diffusers_editor_open_save_metadata_and_live_refresh(engine_fixture):
    from safetensors import safe_open
    from safetensors.torch import load_file
    from sdfx.embedding_editor_service import run_editor_service
    engine, source, original = engine_fixture
    class Context:
        def check_interrupt(self): pass
        def publish(self, fn): return fn()
    draft = run_editor_service(engine, {'operation': 'open', 'name': 'concept'}, Context())
    assert draft['vectors']['clip_l'] == original.tolist()
    draft['vectors']['clip_l'][0][0] = 0.75
    result = run_editor_service(engine, {'operation': 'save', 'name': 'concept', 'draft': draft, 'overwrite': True}, Context())
    assert result['reloaded']
    with safe_open(str(source), framework='pt', device='cpu') as handle:
        assert handle.metadata() == {'step': '42', 'sd_checkpoint_name': 'original'}
    output = load_file(str(source))['clip_l']
    assert output[0, 0].item() == pytest.approx(0.75)
    assert torch.equal(output[1], original[1])
    ids = [engine.pipe.tokenizer.convert_tokens_to_ids(name) for name in ('concept', 'concept_1')]
    assert torch.equal(engine.pipe.text_encoder.get_input_embeddings().weight[ids].float(), output.float())
    assert load_file(result['backup'])['clip_l'].equal(original)


def test_actual_diffusers_open_catalog_activates_unloaded_embedding_without_disturbing_other(engine_fixture, tmp_path):
    from safetensors.torch import save_file
    from sdfx.embedding_editor_service import run_editor_service
    engine, _, original = engine_fixture
    source = tmp_path / 'newconcept.safetensors'
    new = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
    save_file({'clip_l': new}, str(source))
    tok = engine.pipe.tokenizer; encoder = engine.pipe.text_encoder
    original_ids = [tok.convert_tokens_to_ids(name) for name in ('concept', 'concept_1')]
    class Context:
        def check_interrupt(self): pass
        def publish(self, fn): return fn()
    result = run_editor_service(engine, {'operation': 'open', 'name': 'newconcept', 'path': str(source)}, Context())
    assert result['vectors']['clip_l'] == new.tolist()
    assert engine.embedding_token_owners['newconcept']['clip_l'] == ['newconcept']
    assert torch.equal(encoder.get_input_embeddings().weight[original_ids].float(), original)


def test_actual_independent_suffix_collision_does_not_replace_source(engine_fixture, tmp_path):
    from copy import deepcopy
    from safetensors.torch import save_file
    from sdfx.embedding_editor_service import run_editor_service
    engine, _, _ = engine_fixture
    class Context:
        def check_interrupt(self): pass
        def publish(self, fn): return fn()
    root = tmp_path / 'separate.safetensors'
    suffix = tmp_path / 'separate_1.safetensors'
    save_file({'clip_l': torch.tensor([[0.1, 0.2, 0.3, 0.4]])}, str(root))
    save_file({'clip_l': torch.tensor([[0.5, 0.6, 0.7, 0.8]])}, str(suffix))
    ctx = Context()
    draft = run_editor_service(engine, {'operation': 'open', 'name': 'separate', 'path': str(root)}, ctx)
    run_editor_service(engine, {'operation': 'open', 'name': 'separate_1', 'path': str(suffix)}, ctx)
    before_root, before_suffix = root.read_bytes(), suffix.read_bytes()
    before_weights = engine.pipe.text_encoder.get_input_embeddings().weight.detach().clone()
    before_owners = deepcopy(engine.embedding_token_owners)
    draft['vectors']['clip_l'][0][0] = 0.9
    with pytest.raises((ValueError, RuntimeError), match='collision'):
        run_editor_service(engine, {'operation': 'save', 'name': 'separate', 'draft': draft, 'overwrite': True}, ctx)
    assert root.read_bytes() == before_root
    assert suffix.read_bytes() == before_suffix
    assert torch.equal(engine.pipe.text_encoder.get_input_embeddings().weight, before_weights)
    assert engine.embedding_token_owners == before_owners
    assert not list(tmp_path.glob('separate.safetensors.before-edit-*.backup'))


def test_actual_diffusers_preview_rows_restore_after_error(engine_fixture):
    from sdfx.embedding_editor_preview import ModelEditorScope, temporary_edit
    engine, _, original = engine_fixture
    encoder = engine.pipe.text_encoder; tok = engine.pipe.tokenizer
    ids = [tok.convert_tokens_to_ids(name) for name in ('concept', 'concept_1')]
    before = encoder.get_input_embeddings().weight[ids].detach().clone()
    changed = {'clip_l': [[2., 2., 2., 2.], [3., 3., 3., 3.]]}
    with pytest.raises(RuntimeError, match='injected'):
        with temporary_edit(ModelEditorScope(engine, 'concept'), changed):
            assert not torch.equal(encoder.get_input_embeddings().weight[ids], before)
            raise RuntimeError('injected')
    assert torch.equal(encoder.get_input_embeddings().weight[ids], before)


def test_actual_diffusers_pipeline_preview_restores_after_parameter_replacement(engine_fixture):
    from sdfx.embedding_editor_preview import run_pipeline_editor
    engine, _, _ = engine_fixture
    encoder = engine.pipe.text_encoder; tok = engine.pipe.tokenizer
    ids = [tok.convert_tokens_to_ids(name) for name in ('concept', 'concept_1')]
    before = encoder.get_input_embeddings().weight.detach().clone()
    changed = {'clip_l': [[2., 2., 2., 2.], [3., 3., 3., 3.]]}
    def pipeline():
        weight = encoder.get_input_embeddings().weight.detach().clone()
        assert torch.equal(weight[ids], torch.tensor(changed['clip_l']))
        encoder.set_input_embeddings(torch.nn.Embedding.from_pretrained(weight, freeze=False))
        raise RuntimeError('injected pipeline failure')
    with pytest.raises(RuntimeError, match='injected pipeline failure'):
        run_pipeline_editor(engine, 'concept', changed, pipeline)
    assert torch.equal(encoder.get_input_embeddings().weight, before)


def test_actual_diffusers_editor_uses_shared_merge_transaction(engine_fixture):
    from sdfx.embedding_editor_generation import preview_encoding
    from sdfx.embedding_merge_backend import run_encoded
    engine, _, _ = engine_fixture
    engine.device = torch.device('cpu')
    encoder = engine.pipe.text_encoder; tok = engine.pipe.tokenizer
    ids = [tok.convert_tokens_to_ids(name) for name in ('concept', 'concept_1')]
    before = encoder.get_input_embeddings().weight.detach().clone()
    changed = {'clip_l': [[2., 2., 2., 2.], [3., 3., 3., 3.]]}
    engine._editor_preview = ('concept', changed)
    def encode(texts):
        assert torch.equal(encoder.get_input_embeddings().weight[ids], torch.tensor(changed['clip_l']))
        return texts
    assert preview_encoding(engine, lambda: run_encoded(engine, ('concept', ''), encode)) == ('concept', '')
    assert torch.equal(encoder.get_input_embeddings().weight, before)
    assert not hasattr(engine, '_embedding_encoding_transaction')


def test_actual_torch_pt_copy_preserves_metadata_and_unchanged_precision(tmp_path):
    import hashlib
    from sdfx.embedding_editor_pt import build_pt_copy
    source = tmp_path / 'source.pt'; target = tmp_path / 'copy.pt'
    original = torch.tensor([[1.0000000001, 2.]], dtype=torch.float64)
    torch.save({'string_to_param': {'*': original}, 'step': 42, 'sd_checkpoint_name': 'original',
                'other': torch.tensor([99])}, source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    build_pt_copy(source, target, {'clip_l': [[1., 9.]]}, digest, baseline={'clip_l': [[1., 2.]]})
    result = torch.load(target, map_location='cpu', weights_only=True)
    assert result['step'] == 42 and result['sd_checkpoint_name'] == 'original'
    assert result['other'].equal(torch.tensor([99]))
    assert result['string_to_param']['*'][0, 0].item() == original[0, 0].item()
    assert result['string_to_param']['*'][0, 1].item() == 9.


@pytest.mark.parametrize('single_key', [False, True])
def test_actual_torch_pt_copy_supports_tensor_root_and_single_key(tmp_path, single_key):
    import hashlib
    from sdfx.embedding_editor_pt import build_pt_copy
    source = tmp_path / ('source.bin' if single_key else 'source.pt')
    target = tmp_path / 'copy.pt'
    original = torch.tensor([[1.0000000001, 2.]], dtype=torch.float64)
    torch.save({'custom': original} if single_key else original, source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    build_pt_copy(source, target, {'clip_l': [[1., 9.]]}, digest, baseline={'clip_l': [[1., 2.]]})
    result = torch.load(target, map_location='cpu', weights_only=True)
    edited = result['custom'] if single_key else result
    assert edited[0, 0].item() == original[0, 0].item()
    assert edited[0, 1].item() == 9.
