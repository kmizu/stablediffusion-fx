import numpy as np
import pytest

from sdfx.embedding_merge_runtime import expression_scope


class Source:
    stream_names = ('clip_l',)

    def resolve_text(self, text):
        return (np.array([[1, 2]], dtype=np.float32),)

    def resolve_token(self, token):
        return (np.array([[token, token]], dtype=np.float32),)


class Backend:
    def __init__(self):
        self.state = {'persistent': 'untouched'}
        self.calls = []

    def capture(self):
        self.calls.append('capture')
        return self.state.copy()

    def install(self, name, vectors):
        self.calls.append('install')
        self.state[name] = vectors

    def restore(self, state):
        self.calls.append('restore')
        self.state = state


def test_repeated_positive_negative_hires_expressions_share_scoped_rows():
    backend = Backend()
    prompts = ["a <'cat'*2>", "(<'cat'*2>:0.5)", "{'cat':0}"]
    with expression_scope(Source(), prompts, backend, lambda: None) as result:
        assert backend.calls == ['capture', 'install']
        assert result.prompts[0].split()[-1] in result.prompts[1]
        assert result.prompts[2] == ''
        assert result.raw_prompts == tuple(prompts)
    assert backend.state == {'persistent': 'untouched'}
    assert backend.calls[-1] == 'restore'


def test_scope_without_expressions_does_not_touch_model():
    backend = Backend()
    with expression_scope(Source(), ['ordinary <lora:x:1>'], backend, lambda: None) as result:
        assert result.prompts == ('ordinary <lora:x:1>',)
    assert backend.calls == []


def test_partial_install_failure_restores_snapshot():
    backend = Backend()
    original = backend.install

    def fail(name, vectors):
        original(name, vectors)
        raise RuntimeError('second encoder failed')

    backend.install = fail
    with pytest.raises(RuntimeError, match='second encoder'):
        with expression_scope(Source(), ["<'a'>"], backend, lambda: None):
            pytest.fail('must not yield')
    assert backend.state == {'persistent': 'untouched'}
    assert backend.calls[-1] == 'restore'


def test_interrupted_generation_restores_and_propagates_original_failure():
    backend = Backend()
    with pytest.raises(RuntimeError, match='interrupted'):
        with expression_scope(Source(), ["<'a'>"], backend, lambda: None):
            raise RuntimeError('interrupted')
    assert backend.state == {'persistent': 'untouched'}


def test_parse_failure_happens_before_snapshot_or_mutation():
    backend = Backend()
    with pytest.raises(ValueError):
        with expression_scope(Source(), ["<'a'>", "<'b'/0>"], backend, lambda: None):
            pass
    assert backend.calls == []


def test_combined_unique_expression_rows_are_bounded_before_model_snapshot():
    backend = Backend()
    with pytest.raises(ValueError, match='total temporary'):
        with expression_scope(Source(), ["<'cat':3000> <'dog':3000>"], backend, lambda: None):
            pytest.fail('must reject excessive total allocation')
    assert backend.calls == []


def test_temporary_name_cannot_reinterpret_literal_user_text():
    backend = Backend()
    with expression_scope(Source(), ["literal __nui_em_1__ <'cat'>"], backend, lambda: None) as result:
        assert result.prompts[0] == 'literal __nui_em_1__ __nui_em_2__'
        assert '__nui_em_1__' not in backend.state
