"""Exercise actual engine method bodies without importing torch/diffusers."""
import ast
from pathlib import Path
from types import SimpleNamespace

from sdfx import embedding_merge_backend


def test_plain_encoding_does_not_access_merge_cancellation_context():
    tree = ast.parse((Path(__file__).parents[1] / 'sdfx/engine.py').read_text(encoding='utf-8'))
    engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Engine')
    method = next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == '_merge_encoding')
    scope = {'__name__': 'sdfx.engine', '__package__': 'sdfx'}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<actual-engine>', 'exec'), scope)
    subject = SimpleNamespace(_embedding_merge_ctx=SimpleNamespace())
    assert scope['_merge_encoding'](subject, '{cat::0-1}', '', lambda texts: texts) == ('{cat::0-1}', '')


def test_expression_resolution_precedes_attention_schedule_parsing(monkeypatch):
    tree = ast.parse((Path(__file__).parents[1] / 'sdfx/engine.py').read_text(encoding='utf-8'))
    engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Engine')
    methods = [n for n in engine.body if isinstance(n, ast.FunctionDef) and
               (n.name.startswith('_encode') or n.name == '_merge_encoding')]
    cls = ast.ClassDef(name='Subject', bases=[], keywords=[], body=methods, decorator_list=[])
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    calls = []

    def resolve(subject, texts, operation, **kwargs):
        calls.append(tuple(texts))
        return operation(tuple(text.replace("<'a':2>", 'TEMP') for text in texts))

    monkeypatch.setattr(embedding_merge_backend, 'run_encoded', resolve)

    def schedule(text, *args):
        assert "<'" not in text, 'merge expressions must be resolved before colon/attention scheduling'
        return [(1, text)]

    scope = {'__name__': 'sdfx.engine', '__package__': 'sdfx',
             'hooks_mod': SimpleNamespace(hook_config=lambda *args: None),
             'prompt': SimpleNamespace(strip_extra_networks=lambda s: s),
             'prompt_schedule': SimpleNamespace(schedule=schedule)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])), '<actual-engine>', 'exec'), scope)
    subject = scope['Subject']()
    subject._encode_plain = lambda pos, neg, gp, bs: {'positive': pos, 'negative': neg}
    gp = SimpleNamespace(hooks={})
    result = subject._encode_scheduled("(<'a':2>:0.5)", 'negative', gp, 1, 10)
    assert result == {'positive': '(TEMP:0.5)', 'negative': 'negative'}
    assert calls[0] == ("(<'a':2>:0.5)", 'negative')


def test_invalid_merge_reports_parameter_error_without_masking_interrupt(monkeypatch):
    import pytest
    from sdfx.protocol import EngineError, INVALID_PARAMS, INTERRUPTED
    tree = ast.parse((Path(__file__).parents[1] / 'sdfx/engine.py').read_text(encoding='utf-8'))
    engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Engine')
    method = next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == '_merge_encoding')
    scope = {'__name__': 'sdfx.engine', '__package__': 'sdfx', 'EngineError': EngineError, 'INVALID_PARAMS': INVALID_PARAMS}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<actual-engine>', 'exec'), scope)
    def invalid(*args, **kwargs): raise ValueError('divide by zero')
    monkeypatch.setattr(embedding_merge_backend, 'run_encoded', invalid)
    with pytest.raises(EngineError) as error:
        scope['_merge_encoding'](SimpleNamespace(), "<'cat'/0>", '', lambda x: x)
    assert error.value.code == INVALID_PARAMS
    def interrupted(*args, **kwargs): raise EngineError(INTERRUPTED, 'cancel')
    monkeypatch.setattr(embedding_merge_backend, 'run_encoded', interrupted)
    with pytest.raises(EngineError) as error:
        scope['_merge_encoding'](SimpleNamespace(), "<'cat'>", '', lambda x: x)
    assert error.value.code == INTERRUPTED


def test_native_loader_resolves_referenced_conversion_siblings_before_ownership_load(monkeypatch):
    import os, re
    from sdfx import embedding_merge_files
    tree = ast.parse((Path(__file__).parents[1] / 'sdfx/engine.py').read_text(encoding='utf-8'))
    engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Engine')
    method = next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == '_load_embeddings')
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    scope = {'__name__': 'sdfx.engine', '__package__': 'sdfx', 'os': os, 're': re}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])), '<actual-loader>', 'exec'), scope)
    seen = []
    monkeypatch.setattr(embedding_merge_files, 'requested_paths', lambda paths, texts, widths: seen.append((paths, texts, widths)) or [])
    encoder = SimpleNamespace(get_input_embeddings=lambda: SimpleNamespace(weight=SimpleNamespace(shape=(10, 768))))
    subject = SimpleNamespace(pipe=SimpleNamespace(text_encoder=encoder), sdxl=False, loaded_embeddings=set())
    request = SimpleNamespace(embedding_paths=['a/style.safetensors', 'a/sdxl/style.safetensors'])
    scope['_load_embeddings'](subject, request, ['style'], SimpleNamespace())
    assert seen == [(request.embedding_paths, ['style'], {'clip_l': 768})]
