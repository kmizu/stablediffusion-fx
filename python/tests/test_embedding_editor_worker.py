import importlib
from contextlib import contextmanager
import pytest


def worker():
    return importlib.import_module('sdfx.embedding_editor_worker')


class Adapter:
    def __init__(self):
        self.live = {'clip_l': [[1., 2.], [3., 4.]], 'clip_g': [[5., 6., 7.], [8., 9., 10.]]}
        self.model = 'model-a'
        self.identity = ('file-a', 1, 100)
        self.path = 'model/embedding.safetensors'
        self.saved = None
        self.temporary = None
        self.raise_generate = False
    def read(self, name, guidance):
        return {'vectors': self.live, 'model_token': self.model, 'path': self.path,
                'file_identity': self.identity, 'ranges': {}, 'guidance': {}}
    def current(self, name):
        return {'model_token': self.model, 'path': self.path, 'file_identity': self.identity}
    def publish(self, name, values, *, expected):
        self.saved = values
        return {'path': self.path}
    @contextmanager
    def preview_scope(self, name, values, *, expected):
        assert self.temporary is None
        self.temporary = values
        try:
            yield
        finally:
            self.temporary = None
    def generate(self, params):
        assert self.temporary is not None
        if self.raise_generate:
            raise RuntimeError('generation failure')
        return {'image': 'preview'}


def opened(adapter):
    return worker().open_editor(adapter, 'concept', 'apple:#ff0000')


def test_open_and_save_preserve_both_full_encoder_drafts():
    adapter = Adapter(); draft = opened(adapter)
    assert draft['vectors'] == adapter.live
    draft['vectors']['clip_l'][0][1] = -4.
    result = worker().save_editor(adapter, 'concept', draft, overwrite=True)
    assert result['path'] == adapter.path
    assert adapter.saved['clip_l'] == [[1., -4.], [3., 4.]]
    assert adapter.saved['clip_g'] == adapter.live['clip_g']
    assert adapter.live['clip_l'][0][1] == 2.


def test_guidance_refresh_preserves_edits_and_rejects_stale_source():
    adapter = Adapter(); draft = opened(adapter)
    draft['vectors']['clip_l'][0][0] = 7.
    events = []
    class Context:
        def check_interrupt(self): events.append('check')
    result = worker().dispatch_editor(adapter, {'operation': 'guidance', 'name': 'concept',
        'draft': draft, 'guidance': 'apple:#ff0000'}, Context())
    assert 'guidance' in result and 'vectors' not in result
    assert draft['vectors']['clip_l'][0][0] == 7. and adapter.live['clip_l'][0][0] == 1.
    assert len(events) >= 2
    adapter.identity = ('changed',)
    with pytest.raises(ValueError, match='changed'):
        worker().dispatch_editor(adapter, {'operation': 'guidance', 'name': 'concept',
            'draft': draft, 'guidance': 'apple:#ff0000'}, Context())


@pytest.mark.parametrize('change', ['model', 'file'])
def test_stale_draft_cannot_overwrite_user_file(change):
    adapter = Adapter(); draft = opened(adapter)
    if change == 'model': adapter.model = 'model-b'
    else: adapter.identity = ('file-b', 1, 100)
    with pytest.raises(ValueError, match='changed'):
        worker().save_editor(adapter, 'concept', draft, overwrite=True)
    assert adapter.saved is None


def test_preview_cleanup_on_success_and_failure_without_saving():
    adapter = Adapter(); draft = opened(adapter)
    draft['vectors']['clip_g'][1][1] = 20.
    assert worker().preview_editor(adapter, 'concept', draft, {}) == {'image': 'preview'}
    assert adapter.temporary is None and adapter.saved is None
    adapter.raise_generate = True
    with pytest.raises(RuntimeError, match='generation failure'):
        worker().preview_editor(adapter, 'concept', draft, {})
    assert adapter.temporary is None and adapter.saved is None


def test_source_change_during_revalidation_cannot_publish():
    class RacingAdapter(Adapter):
        race = False
        def read(self, name, guidance):
            result = super().read(name, guidance)
            if self.race:
                self.identity = ('file-b', 1, 100)
                result['file_identity'] = self.identity
            return result
    adapter = RacingAdapter(); draft = opened(adapter); adapter.race = True
    with pytest.raises(ValueError, match='changed'):
        worker().save_editor(adapter, 'concept', draft, overwrite=True)
    assert adapter.saved is None


def test_existing_file_requires_explicit_overwrite_consent():
    adapter = Adapter(); draft = opened(adapter)
    with pytest.raises(ValueError, match='overwrite'):
        worker().save_editor(adapter, 'concept', draft)
    assert adapter.saved is None


def test_publish_receives_captured_stamp_for_final_atomic_revalidation():
    class GuardedAdapter(Adapter):
        def publish(self, name, values, *, expected):
            assert expected['model_token'] == self.model
            assert expected['path'] == self.path
            assert tuple(expected['file_identity']) == self.identity
            return super().publish(name, values, expected=expected)
    adapter = GuardedAdapter(); draft = opened(adapter)
    worker().save_editor(adapter, 'concept', draft, overwrite=True)
    assert adapter.saved is not None


def test_preview_scope_receives_identity_for_revalidation():
    class GuardedAdapter(Adapter):
        @contextmanager
        def preview_scope(self, name, values, *, expected):
            assert expected['model_token'] == self.model
            assert tuple(expected['file_identity']) == self.identity
            with super().preview_scope(name, values, expected=expected):
                yield
    adapter = GuardedAdapter(); draft = opened(adapter)
    assert worker().preview_editor(adapter, 'concept', draft, {}) == {'image': 'preview'}


def test_dispatch_serializes_save_publication_and_checks_preview_interruption():
    adapter = Adapter(); draft = opened(adapter)
    events = []
    class Context:
        def publish(self, fn): events.append('publication'); return fn()
        def check_interrupt(self): events.append('interrupt-check')
    api = worker()
    saved = api.dispatch_editor(adapter, {'operation': 'save', 'name': 'concept', 'draft': draft, 'overwrite': True}, Context())
    assert saved['path'] == adapter.path and events == ['interrupt-check', 'interrupt-check', 'publication']
    events.clear()
    assert api.dispatch_editor(adapter, {'operation': 'preview', 'name': 'concept', 'draft': draft,
                                         'generation': {}}, Context()) == {'image': 'preview'}
    assert events == ['interrupt-check']


def test_dispatch_rejects_unknown_operations_and_missing_owned_context():
    adapter = Adapter(); draft = opened(adapter)
    with pytest.raises(ValueError, match='operation'):
        worker().dispatch_editor(adapter, {'operation': 'missing'}, None)
    with pytest.raises(ValueError, match='context'):
        worker().dispatch_editor(adapter, {'operation': 'save', 'name': 'concept', 'draft': draft}, None)


def test_cancel_during_save_validation_prevents_publication():
    events = []
    class Context:
        interrupted = False
        def check_interrupt(self):
            if self.interrupted: raise RuntimeError('interrupted')
        def publish(self, fn): events.append('publish'); return fn()
    class SlowAdapter(Adapter):
        ctx = None
        def read(self, name, guidance):
            result = super().read(name, guidance)
            if self.ctx is not None: self.ctx.interrupted = True
            return result
    adapter = SlowAdapter(); draft = opened(adapter); ctx = Context(); adapter.ctx = ctx
    with pytest.raises(RuntimeError, match='interrupted'):
        worker().dispatch_editor(adapter, {'operation': 'save', 'name': 'concept', 'draft': draft, 'overwrite': True}, ctx)
    assert events == [] and adapter.saved is None
