import importlib
import pytest


def module(): return importlib.import_module('sdfx.embedding_editor_preview')


class Model:
    def __init__(self):
        self.rows = {'clip_l': [1., 2.], 'clip_g': [3., 4.]}
        self.invalidated = False; self.fail_install = False; self.fail_restore = False
    def snapshot(self): return {key: row[:] for key, row in self.rows.items()}
    def install(self, values):
        self.rows['clip_l'] = values['clip_l'][:]
        if self.fail_install: raise RuntimeError('install failed')
        self.rows['clip_g'] = values['clip_g'][:]
    def restore(self, snapshot):
        if self.fail_restore: raise RuntimeError('restore failed')
        self.rows = {key: row[:] for key, row in snapshot.items()}
    def invalidate(self): self.invalidated = True


def test_preview_scope_restores_both_encoders_on_success_and_interruption():
    model = Model(); original = model.snapshot(); values = {'clip_l': [8., 9.], 'clip_g': [10., 11.]}
    with module().temporary_edit(model, values):
        assert model.rows == values
    assert model.rows == original
    with pytest.raises(KeyboardInterrupt):
        with module().temporary_edit(model, values):
            raise KeyboardInterrupt()
    assert model.rows == original and not model.invalidated


def test_partial_install_restores_original_and_failed_restore_invalidates():
    model = Model(); original = model.snapshot(); model.fail_install = True
    with pytest.raises(RuntimeError, match='install failed'):
        with module().temporary_edit(model, {'clip_l': [8., 9.], 'clip_g': [10., 11.]}): pass
    assert model.rows == original and not model.invalidated
    model.fail_install = False; model.fail_restore = True
    from sdfx.protocol import EngineError, NO_MODEL
    with pytest.raises(EngineError, match='reload the checkpoint') as error:
        with module().temporary_edit(model, {'clip_l': [8., 9.], 'clip_g': [10., 11.]}): pass
    assert model.invalidated
    assert error.value.code == NO_MODEL


def test_real_scope_adapter_uses_owned_rows_and_restores_both_streams(monkeypatch):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace
    preview = module()
    class Rows:
        def __init__(self, rows): self.rows = [r[:] for r in rows]; self.dtype = 'f32'; self.device = 'cpu'; self.shape = (len(rows), len(rows[0])); self.is_meta = False
        def __getitem__(self, ids): return Rows([self.rows[i] for i in ids])
        def __setitem__(self, ids, value):
            for i, row in zip(ids, value.rows): self.rows[i] = row[:]
        def detach(self): return self
        def clone(self): return Rows(self.rows)
    class Tokenizer:
        def get_added_vocab(self): return {'concept': 1, 'concept_1': 3, 'unrelated': 2}
    left = Rows([[0., 0.], [1., 2.], [8., 8.], [3., 4.]])
    right = Rows([[0., 0., 0.], [5., 6., 7.], [9., 9., 9.], [8., 9., 10.]])
    def encoder(weight): return SimpleNamespace(get_input_embeddings=lambda: SimpleNamespace(weight=weight))
    streams = {'clip_l': (Tokenizer(), encoder(left)), 'clip_g': (Tokenizer(), encoder(right))}
    monkeypatch.setitem(sys.modules, 'sdfx.embedding_inspector', SimpleNamespace(_streams=lambda pipe: streams))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(no_grad=nullcontext,
        as_tensor=lambda rows, dtype=None, device=None: Rows(rows),
        isfinite=lambda value: SimpleNamespace(all=lambda: True)))
    engine = SimpleNamespace(pipe=object(), embedding_token_owners={'concept': {'clip_l': ['concept_1', 'concept'],
                                                                                'clip_g': ['concept', 'concept_1']}})
    adapter = preview.ModelEditorScope(engine, 'concept')
    changed = {'clip_l': [[11., 12.], [13., 14.]], 'clip_g': [[15., 16., 17.], [18., 19., 20.]]}
    before = (left.rows[:], right.rows[:])
    with preview.temporary_edit(adapter, changed):
        assert left.rows[1] == [11., 12.] and left.rows[3] == [13., 14.]
        assert right.rows[2] == [9., 9., 9.]
    assert left.rows == before[0] and right.rows == before[1]


def test_model_scope_rejects_unrepresentable_dtype_before_mutation(monkeypatch):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace
    preview = module()
    class Rows:
        def __init__(self, rows): self.rows = [r[:] for r in rows]; self.dtype = 'f16'; self.device = 'cpu'; self.shape = (len(rows), len(rows[0])); self.is_meta = False
        def __getitem__(self, ids): return Rows([self.rows[i] for i in ids])
        def __setitem__(self, ids, value):
            for i, row in zip(ids, value.rows): self.rows[i] = row[:]
        def detach(self): return self
        def clone(self): return Rows(self.rows)
    class Tokenizer:
        def get_added_vocab(self): return {'concept': 1}
    weights = Rows([[0.], [1.]])
    encoder = SimpleNamespace(get_input_embeddings=lambda: SimpleNamespace(weight=weights))
    monkeypatch.setitem(sys.modules, 'sdfx.embedding_inspector', SimpleNamespace(_streams=lambda pipe: {'clip_l': (Tokenizer(), encoder)}))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(no_grad=nullcontext,
        as_tensor=lambda rows, dtype=None, device=None: Rows([[float('inf')]]),
        isfinite=lambda value: SimpleNamespace(all=lambda: False)))
    engine = SimpleNamespace(pipe=object(), embedding_token_owners={'concept': {'clip_l': ['concept']}})
    adapter = preview.ModelEditorScope(engine, 'concept')
    with pytest.raises(ValueError, match='finite'):
        with preview.temporary_edit(adapter, {'clip_l': [[1e100]]}): pass
    assert weights.rows == [[0.], [1.]]


def test_editor_encoding_composes_with_shared_transaction_without_nested_guards():
    preview = module(); model = Model(); original = model.snapshot(); events = []
    def transaction(engine, operation):
        assert engine == 'engine'
        events.append('guard-enter')
        try: return operation()
        finally: events.append('guard-exit')
    def encode():
        assert model.rows['clip_l'] == [7., 8.]
        events.append('encode')
        return 'conditioning'
    result = preview.run_editor_encoding('engine', 'concept', {'clip_l': [7., 8.], 'clip_g': [9., 10.]},
        encode, transaction=transaction, scope_factory=lambda engine, name: model)
    assert result == 'conditioning' and events == ['guard-enter', 'encode', 'guard-exit']
    assert model.rows == original


def test_editor_encoding_restores_after_nested_merge_encoding_failure():
    preview = module(); model = Model(); original = model.snapshot(); events = []
    def transaction(engine, operation):
        events.append('enter')
        try: return operation()
        finally: events.append('exit')
    def encode():
        assert model.rows == {'clip_l': [7., 8.], 'clip_g': [9., 10.]}
        events.append('nested-merge')
        raise ValueError('encoding failed')
    with pytest.raises(ValueError, match='encoding failed'):
        preview.run_editor_encoding('engine', 'concept', {'clip_l': [7., 8.], 'clip_g': [9., 10.]},
            encode, transaction=transaction, scope_factory=lambda engine, name: model)
    assert model.rows == original
    assert events == ['enter', 'nested-merge', 'exit']


def test_pipeline_editor_restores_after_pipeline_error_and_guard_reentry():
    preview = module(); events = []
    class Engine:
        def _with_embedding_encoder_access(self, operation):
            events.append('guard-enter')
            try: return operation()
            finally: events.append('guard-exit')
    class Scope:
        def snapshot(self): events.append('snapshot'); return 'original'
        def install(self, values): events.append('install')
        def capture_installed(self): events.append('installed'); return 'new'
        def restore_rebound(self, snapshot, installed):
            assert (snapshot, installed) == ('original', 'new')
            events.append('restore')
        def invalidate(self): events.append('invalidate')
    def pipeline(): events.append('pipeline'); raise ValueError('injected')
    with pytest.raises(ValueError, match='injected'):
        preview.run_pipeline_editor(Engine(), 'concept', {}, pipeline, scope_factory=lambda *_: Scope())
    assert events == ['guard-enter', 'snapshot', 'install', 'installed', 'guard-exit',
                      'pipeline', 'guard-enter', 'restore', 'guard-exit']


def test_pipeline_editor_fails_closed_when_restore_cannot_enter_guard():
    from sdfx.protocol import EngineError, NO_MODEL
    preview = module(); events = []
    class Engine:
        count = 0
        def _with_embedding_encoder_access(self, operation):
            self.count += 1
            if self.count == 2: raise RuntimeError('materialization failed')
            return operation()
    class Scope:
        def snapshot(self): return 'old'
        def install(self, values): pass
        def capture_installed(self): return 'new'
        def invalidate(self): events.append('invalidate')
    with pytest.raises(EngineError) as failure:
        preview.run_pipeline_editor(Engine(), 'concept', {}, lambda: None, scope_factory=lambda *_: Scope())
    assert failure.value.code == NO_MODEL and events == ['invalidate']


def test_pipeline_editor_rolls_back_partial_install_without_calling_pipeline():
    preview = module(); events = []
    class Engine:
        def _with_embedding_encoder_access(self, operation): return operation()
    class Scope:
        def snapshot(self): return 'old'
        def install(self, values): events.append('partial-install'); raise ValueError('invalid')
        def restore(self, snapshot): assert snapshot == 'old'; events.append('restore')
        def invalidate(self): events.append('invalidate')
    with pytest.raises(ValueError, match='invalid'):
        preview.run_pipeline_editor(Engine(), 'concept', {}, lambda: events.append('pipeline'),
                                    scope_factory=lambda *_: Scope())
    assert events == ['partial-install', 'restore']


def test_pipeline_editor_wraps_merge_install_and_cleanup_in_order():
    preview = module(); events = []
    class Engine:
        def _with_embedding_encoder_access(self, operation):
            events.append('guard-enter')
            try: return operation()
            finally: events.append('guard-exit')
    class Scope:
        def snapshot(self): events.append('editor-snapshot'); return 'old'
        def install(self, values): events.append('editor-install')
        def capture_installed(self): return 'new'
        def restore_rebound(self, snapshot, installed): events.append('editor-restore')
    engine = Engine()
    def merge():
        engine._with_embedding_encoder_access(lambda: events.append('merge-install'))
        try: events.append('pipeline')
        finally: engine._with_embedding_encoder_access(lambda: events.append('merge-restore'))
    preview.run_pipeline_editor(engine, 'concept', {}, merge, scope_factory=lambda *_: Scope())
    assert [x for x in events if not x.startswith('guard-')] == [
        'editor-snapshot', 'editor-install', 'merge-install', 'pipeline', 'merge-restore', 'editor-restore']


def test_pipeline_editor_preserves_original_no_model_after_inner_invalidation():
    from sdfx.protocol import EngineError, NO_MODEL
    preview = module(); events = []
    class Engine:
        pipe = object()
        def _with_embedding_encoder_access(self, operation):
            if self.pipe is None: raise AssertionError('must not reattach discarded model')
            return operation()
    class Scope:
        def snapshot(self): return 'old'
        def install(self, values): pass
        def capture_installed(self): return 'new'
        def restore_rebound(self, *args): events.append('restore')
        def invalidate(self): events.append('invalidate')
    engine = Engine()
    def inner():
        engine.pipe = None
        raise EngineError(NO_MODEL, 'inner scope invalidated the checkpoint')
    with pytest.raises(EngineError, match='inner scope invalidated') as failure:
        preview.run_pipeline_editor(engine, 'concept', {}, inner, scope_factory=lambda *_: Scope())
    assert failure.value.code == NO_MODEL and events == []
    del failure
    import gc
    gc.collect()
    assert events == []


def test_scheduled_edit_preserves_inner_no_model_without_touching_discarded_pipe():
    from sdfx.protocol import EngineError, NO_MODEL
    from types import SimpleNamespace
    preview = module(); events = []
    engine = SimpleNamespace(pipe=object())
    class Scope:
        def __init__(self): self.engine = engine
        def snapshot(self): return 'old'
        def install(self, values): pass
        def restore(self, snapshot): events.append('restore')
        def invalidate(self): events.append('invalidate')
    with pytest.raises(EngineError, match='inner invalidated') as failure:
        with preview.temporary_edit(Scope(), {}):
            engine.pipe = None
            raise EngineError(NO_MODEL, 'inner invalidated')
    assert failure.value.code == NO_MODEL and events == []


def test_pipeline_scope_rebinds_changed_parameter_only_when_owned_rows_match(monkeypatch):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace
    preview = module()
    class Rows:
        def __init__(self, rows):
            self.rows = [r[:] for r in rows]; self.dtype = 'f32'; self.device = 'cpu'
            self.shape = (len(rows), len(rows[0])); self.is_meta = False
        def __getitem__(self, ids): return Rows([self.rows[i] for i in ids])
        def __setitem__(self, ids, value):
            for i, row in zip(ids, value.rows): self.rows[i] = row[:]
        def detach(self): return self
        def clone(self): return Rows(self.rows)
        def to(self, **kwargs): return self
    class Tokenizer:
        def get_added_vocab(self): return {'concept': 1, 'unrelated': 2}
    class Encoder:
        def __init__(self): self.weight = Rows([[0.], [1.], [9.]])
        def get_input_embeddings(self): return SimpleNamespace(weight=self.weight)
    enc = Encoder(); tok = Tokenizer()
    monkeypatch.setitem(sys.modules, 'sdfx.embedding_inspector', SimpleNamespace(_streams=lambda _: {'clip_l': (tok, enc)}))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(no_grad=nullcontext,
        as_tensor=lambda rows, dtype=None, device=None: Rows(rows),
        isfinite=lambda value: SimpleNamespace(all=lambda: True),
        equal=lambda a, b: a.rows == b.rows))
    engine = SimpleNamespace(pipe=object(), embedding_token_owners={'concept': {'clip_l': ['concept']}})
    scope = preview.ModelEditorScope(engine, 'concept')
    old = scope.snapshot(); scope.install({'clip_l': [[4.]]}); installed = scope.capture_installed()
    enc.weight = Rows(enc.weight.rows)
    scope.restore_rebound(old, installed)
    assert enc.weight.rows == [[0.], [1.], [9.]]
    scope = preview.ModelEditorScope(engine, 'concept')
    old = scope.snapshot(); scope.install({'clip_l': [[4.]]}); installed = scope.capture_installed()
    enc.weight = Rows([[0.], [5.], [9.]])
    with pytest.raises(RuntimeError, match='changed'):
        scope.restore_rebound(old, installed)
    assert enc.weight.rows[1] == [5.]


def test_pipeline_scope_rejects_swapped_owned_names_with_same_ids(monkeypatch):
    import sys
    from types import SimpleNamespace
    preview = module()
    class Rows:
        dtype = 'f32'; device = 'cpu'; is_meta = False
        shape = (3, 1)
        def __getitem__(self, ids): return self
        def detach(self): return self
        def clone(self): return self
        def to(self, **kwargs): return self
    ids = {'concept': 1, 'concept_1': 2}
    tokenizer = SimpleNamespace(get_added_vocab=lambda: ids)
    encoder = SimpleNamespace(get_input_embeddings=lambda: SimpleNamespace(weight=Rows()))
    monkeypatch.setitem(sys.modules, 'sdfx.embedding_inspector', SimpleNamespace(_streams=lambda _: {'clip_l': (tokenizer, encoder)}))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(equal=lambda *_: True))
    engine = SimpleNamespace(pipe=object(), embedding_token_owners={'concept': {'clip_l': ['concept', 'concept_1']}})
    scope = preview.ModelEditorScope(engine, 'concept')
    old = scope.snapshot(); installed = scope.capture_installed()
    ids.update(concept=2, concept_1=1)
    with pytest.raises(RuntimeError, match='changed'):
        scope.restore_rebound(old, installed)
