from types import SimpleNamespace
import numpy as np
import pytest

from sdfx.embedding_merge_service import run
from sdfx.protocol import EngineError


class Tokenizer:
    byte_decoder = {chr(i): i for i in range(256)}
    bos_token_id, eos_token_id, pad_token_id = 0, 1, 1
    def tokenize(self, text): return [word + '</w>' for word in text.split()]
    def convert_tokens_to_ids(self, tokens): return [2 for _ in tokens]
    def get_vocab(self): return {'<s>': 0, '</s>': 1, 'cat</w>': 2}


class Context:
    def check_interrupt(self): pass
    def publish(self, operation): return operation()


def engine():
    weights = np.ones((3, 768), dtype=np.float32)
    return SimpleNamespace(dry_run=False, model={'name': 'fixture'}, embedding_token_owners={},
        pipe=SimpleNamespace(tokenizer=Tokenizer(), text_encoder=SimpleNamespace(
            get_input_embeddings=lambda: SimpleNamespace(weight=weights))),
        _with_embedding_encoder_access=lambda call: call())


def test_inspect_expression_returns_stream_rows_and_original_text():
    result = run(engine(), {'text': "'cat'*2", 'grouping': 'parts'}, Context())
    assert result['text'] == "'cat'*2"
    assert result['streams'][0]['name'] == 'clip_l'
    assert result['streams'][0]['rows'][0]['statistics']['min'] == 2
    assert result['paths'] == []


def test_plain_prompt_all_groupings_use_loaded_raw_vectors():
    for grouping in ('none', 'comma', 'parts', 'words', 'tokens', 'vectors'):
        result = run(engine(), {'text': 'cat cat', 'grouping': grouping}, Context())
        assert result['streams'][0]['count'] == 2


def test_save_is_explicit_and_preserves_primary_and_converted_files(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'sdfx.embedding_refresh', SimpleNamespace(
        validate_saved_names=lambda *args: None, refresh_saved_embeddings=lambda *args: None))
    params = {'text': "'cat'", 'operation': 'inspect', 'name': 'test', 'output_dir': str(tmp_path)}
    assert run(engine(), params, Context())['paths'] == []
    assert list(tmp_path.iterdir()) == []
    result = run(engine(), {**params, 'operation': 'save'}, Context())
    assert len(result['paths']) == 2
    assert result['active'] is True


def test_no_model_dry_run_and_bad_operation_are_explicit():
    for model in (SimpleNamespace(dry_run=True), SimpleNamespace(dry_run=False, model=None)):
        with pytest.raises(EngineError):
            run(model, {'text': 'cat'}, Context())
    with pytest.raises(EngineError):
        run(engine(), {'text': 'cat', 'operation': 'delete'}, Context())


def test_worker_registration_exposes_queued_route_and_honest_dryrun_error():
    from sdfx.base import BaseEngine
    from sdfx.server import QUEUED_TYPES
    assert 'embedding_merge' in QUEUED_TYPES
    fake = SimpleNamespace(dry_run=True)
    with pytest.raises(EngineError, match='real loaded model'):
        BaseEngine.embedding_merge(fake, {'text': 'cat'}, Context())


def test_save_validates_and_refreshes_only_primary_inside_publication(tmp_path, monkeypatch):
    import sys
    events = []
    monkeypatch.setitem(sys.modules, 'sdfx.embedding_refresh', SimpleNamespace(
        validate_saved_names=lambda engine, paths: events.append(('validate', paths)),
        refresh_saved_embeddings=lambda engine, paths: events.append(('refresh', paths))))
    class OrderedContext(Context):
        def publish(self, operation):
            events.append(('publish', []))
            value = operation()
            events.append(('published', []))
            return value
    result = run(engine(), {'text': "'cat'", 'operation': 'save', 'name': 'fresh',
                            'output_dir': str(tmp_path)}, OrderedContext())
    assert result['active'] is True
    assert [event[0] for event in events] == ['validate', 'publish', 'refresh', 'published']
    assert len(events[2][1]) == 1
    assert str(events[2][1][0]) == result['paths'][0]


def test_save_refresh_failure_keeps_outputs_and_blocks_generation(tmp_path, monkeypatch):
    import sys
    def fail(*args): raise ValueError('fixture refresh failure')
    monkeypatch.setitem(sys.modules, 'sdfx.embedding_refresh', SimpleNamespace(
        validate_saved_names=lambda *args: None, refresh_saved_embeddings=fail))
    model = engine()
    result = run(model, {'text': "'cat'", 'operation': 'save', 'name': 'fresh',
                         'output_dir': str(tmp_path)}, Context())
    assert result['active'] is False
    assert len(result['paths']) == 2
    assert 'reload' in result['warning'].lower()
    assert model._embedding_reload_error


def test_pasted_png_infotext_restores_without_encoding_or_export(tmp_path):
    model = engine()
    model.pipe.tokenizer.tokenize = lambda text: pytest.fail('restoration must not encode infotext')
    result = run(model, {'text': "<'EM_1'>\nSteps: 20, EmbeddingMerge: <'EM_1'>=<'cat'>",
                         'operation': 'inspect'}, Context())
    assert result['text'] == "<'cat'>\nSteps: 20"
    assert result['streams'] == [] and result['paths'] == []


def test_file_catalog_selects_compatible_conversion_before_shared_loader(tmp_path, monkeypatch):
    import sys
    from sdfx.embedding_merge_export import export_embeddings
    paths = export_embeddings(tmp_path, 'cat', {'clip_l': np.full((2, 768), 3)})
    seen = []
    def loader(selected, streams):
        seen.extend(selected)
        return (lambda name: {'clip_l': np.full((2, 768), 3)}), ['cat'], lambda name: {}
    monkeypatch.setitem(sys.modules, 'sdfx.embedding_inspector', SimpleNamespace(_file_embeddings=loader))
    result = run(engine(), {'text': "'cat'", 'embedding_paths': [str(p) for p in paths]}, Context())
    assert seen == [str(paths[0])]
    assert result['streams'][0]['count'] == 2
