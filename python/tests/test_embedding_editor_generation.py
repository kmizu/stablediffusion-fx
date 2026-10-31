from contextlib import contextmanager

import pytest

from sdfx.embedding_editor_generation import preview_encoding, validate_preview


def test_preview_validates_owned_draft_and_checks_interruption():
    events = []
    class Adapter:
        def current(self, name): return {'model_token': 'm', 'path': 'p', 'file_identity': (1, 'hash')}
        def read(self, name, refs):
            return {**self.current(name), 'vectors': {'clip_l': [[1., 2.]]}}
    class Context:
        def check_interrupt(self): events.append('check')
    draft = {'name': 'cat', 'model_token': 'm', 'path': 'p', 'file_identity': [1, 'hash'],
             'vectors': {'clip_l': [[3., 4.]]}}
    assert validate_preview(Adapter(), draft, Context()) == ('cat', {'clip_l': [[3., 4.]]})
    assert events == ['check', 'check']
    draft['file_identity'] = [2, 'hash']
    with pytest.raises(ValueError, match='changed'):
        validate_preview(Adapter(), draft, Context())


def test_preview_encoding_wraps_once_and_cleans_after_nested_error():
    class Engine: pass
    engine = Engine(); engine._editor_preview = ('cat', {'clip_l': [[3.]]})
    calls = []
    def runner(engine, name, values, operation, *, transaction):
        calls.append(('enter', name, values))
        try: return operation()
        finally: calls.append(('exit', name))
    def outer():
        return preview_encoding(engine, lambda: (_ for _ in ()).throw(ValueError('failure')), runner=runner, transaction=lambda *_: None)
    with pytest.raises(ValueError, match='failure'):
        preview_encoding(engine, outer, runner=runner, transaction=lambda *_: None)
    assert calls == [('enter', 'cat', {'clip_l': [[3.]]}), ('exit', 'cat')]
    assert not getattr(engine, '_editor_preview_active', False)


def test_preview_pipeline_wraps_actual_internal_encoder_call_and_cleans_state():
    from sdfx.embedding_editor_generation import preview_pipeline
    class Engine: pass
    engine = Engine(); engine._editor_preview = ('cat', {'clip_l': [[3.]]})
    calls = []
    def runner(engine, name, values, operation):
        calls.append(('install', name, values))
        try: return operation()
        finally: calls.append(('restore', name))
    def call():
        calls.append('pipeline')
        raise RuntimeError('injected')
    with pytest.raises(RuntimeError, match='injected'):
        preview_pipeline(engine, call, runner=runner)
    assert calls == [('install', 'cat', {'clip_l': [[3.]]}), 'pipeline', ('restore', 'cat')]
    assert not engine._editor_preview_active


def test_editor_and_real_merge_pipeline_compose_scopes_in_order(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from sdfx import embedding_merge_pipeline as merge
    from sdfx.embedding_editor_preview import run_pipeline_editor
    events = []
    class Engine:
        pipe = object()
        embedding_token_owners = {}
        def _with_embedding_encoder_access(self, operation): return operation()
        def _activate(self, pipe): events.append('activate')
        def _free(self): events.append('free')
    class Scope:
        def snapshot(self): return 'old'
        def install(self, values): events.append('editor-install')
        def capture_installed(self): return 'new'
        def restore_rebound(self, snapshot, installed): events.append('editor-restore')
    encoder = SimpleNamespace(get_input_embeddings=lambda: SimpleNamespace(weight='matrix'))
    monkeypatch.setattr(merge, 'streams', lambda _: {'clip_l': ('tokenizer', encoder)})
    monkeypatch.setattr(merge, 'expressions', lambda text: text == 'merge')
    monkeypatch.setattr(merge, 'VectorSource', lambda *a, **kw: object())
    monkeypatch.setattr(merge, 'ModelBackend', lambda *a, **kw: object())
    @contextmanager
    def merged(*args):
        events.append('merge-install')
        try: yield SimpleNamespace(prompts=('temporary', 'negative'))
        finally: events.append('merge-restore')
    monkeypatch.setattr(merge, 'expression_scope', merged)
    engine = Engine()
    def invoke(texts):
        assert texts == ('temporary', 'negative')
        events.append('pipeline')
    def pipeline(): return merge.run_pipeline(engine, object(), ('merge', 'negative'), invoke, check=lambda: None)
    run_pipeline_editor(engine, 'cat', {}, pipeline, scope_factory=lambda *_: Scope())
    assert events == ['editor-install', 'merge-install', 'activate', 'pipeline', 'merge-restore', 'editor-restore']


def test_photomaker_reactivates_derived_pipeline_after_editor_guard():
    from sdfx.embedding_editor_generation import invoke_pipeline
    events = []
    class Engine:
        _editor_preview = ('cat', {'clip_l': [[3.]]})
        def _activate(self, pipe): events.append(('activate', pipe))
    def editor(engine, name, values, operation):
        events.append('editor-install-reattaches-base-hooks')
        return operation()
    def merge(engine, pipe, prompts, operation, *, check):
        events.append('merge-bypass')
        return operation(prompts)
    def call(prompts): events.append(('pipeline', prompts)); return 'ok'
    assert invoke_pipeline(Engine(), 'derived', ('cat', ''), call, check=lambda: None,
                           editor_runner=editor, merge_runner=merge) == 'ok'
    assert events == ['editor-install-reattaches-base-hooks', ('activate', 'derived'),
                      'merge-bypass', ('pipeline', ('cat', ''))]


@pytest.mark.parametrize("draft", [None, ('cat', {'clip_l': [[3.]]})])
def test_photomaker_forwards_keyword_runtime_scopes_with_and_without_editor(draft):
    from types import SimpleNamespace
    from sdfx.embedding_editor_generation import invoke_pipeline
    engine = SimpleNamespace(_editor_preview=draft, _activate=lambda pipe: None)
    hooks = dict(runtime_hook=object(), extra_runtime_hooks=(object(),), pre_runtime_hooks=(object(),))
    def merge(engine, pipe, prompts, operation, *, check, **kwargs):
        assert kwargs == hooks
        assert prompts == ('base', 'negative', 'branch1', 'branch2')
        return operation(prompts)
    def editor(engine, name, values, operation): return operation()
    assert invoke_pipeline(engine, 'derived', ('base', 'negative', 'branch1', 'branch2'),
        lambda prompts: 'ok', check=lambda: None, editor_runner=editor, merge_runner=merge, **hooks) == 'ok'


@pytest.mark.parametrize('fail', [False, True])
def test_editor_offload_guards_suspend_runtime_hooks_even_without_merge(fail):
    from types import SimpleNamespace
    from sdfx.embedding_editor_generation import invoke_pipeline
    active = ['pre', 'neutral', 'negative']
    events = []
    class Hook:
        def __init__(self, name): self.name = name
        def remove(self):
            if self.name in active:
                assert active[-1] == self.name
                active.pop(); events.append(('remove', self.name))
        def install(self):
            assert self.name not in active
            active.append(self.name); events.append(('install', self.name))
    pre, neutral, negative = [Hook(n) for n in active.copy()]
    engine = SimpleNamespace(_editor_preview=('cat', {}), _offload=True,
                             _activate=lambda pipe: events.append('activate'))
    def editor(engine, name, values, operation):
        assert not active, 'editor installation guard must not overwrite active hooks'
        events.append('install-draft')
        try: return operation()
        finally:
            assert not active, 'editor restoration guard must not overwrite active hooks'
            events.append('restore-draft')
    def merge(engine, pipe, prompts, operation, **kwargs): return operation(prompts)
    def operation(prompts):
        assert active == ['pre', 'neutral', 'negative']
        if fail: raise ValueError('pipeline failed')
        return 'ok'
    def run(): return invoke_pipeline(engine, 'derived', ('cat', ''), operation, check=lambda: None,
        editor_runner=editor, merge_runner=merge, runtime_hook=neutral,
        pre_runtime_hooks=(pre,), extra_runtime_hooks=(negative,))
    if fail:
        with pytest.raises(ValueError, match='pipeline failed'): run()
    else: assert run() == 'ok'
    assert not active
    assert not engine._editor_preview_active
    assert events[-1] == 'restore-draft'
