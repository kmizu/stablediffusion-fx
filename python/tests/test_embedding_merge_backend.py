from types import SimpleNamespace
import numpy as np
import pytest

from sdfx.embedding_merge_backend import ModelBackend, run_encoded


def test_temporary_merge_names_are_restored_only_in_composable_metadata_and_schedules():
    from sdfx.embedding_merge_backend import restore_composable_provenance
    class Scheduled(dict): pass
    first = {'_clora_branch': '__nui_em_1__ <lora:x:1>', '_clora_prompt': '__nui_em_1__',
             '_clora_negative': '__nui_em_2__', 'payload': '__nui_em_1__'}
    second = dict(first, _clora_branch=['__nui_em_1__', 'plain'])
    branch = Scheduled(first); branch.schedule = [(1, first), (2, second)]
    plan = SimpleNamespace(branches=[branch])
    result = {'_neutral': plan, '_clora_prompt': '__nui_em_1__', 'payload': '__nui_em_1__'}
    restore_composable_provenance(result, {'__nui_em_1__': "<'cat'>", '__nui_em_2__': "<'bad'>"})
    assert first['_photo_raw_branch'] == "<'cat'> <lora:x:1>"
    assert second['_photo_raw_branch'] == ["<'cat'>", 'plain']
    assert branch['_photo_raw_negative'] == "<'bad'>"
    assert first['_clora_branch'] == '__nui_em_1__ <lora:x:1>'
    assert result['payload'] == first['payload'] == '__nui_em_1__'


def test_run_encoded_restores_raw_branch_metadata_after_temporary_rows_are_removed():
    engine = Engine()
    def encode(texts):
        branch = {'_clora_branch': texts[0], '_clora_prompt': texts[0], '_clora_negative': texts[1]}
        return {'_neutral': SimpleNamespace(branches=[branch])}
    result = run_encoded(engine, ("<'a'> person img", "<'a'*0.5>"), encode, tensor_factory=lambda a: a)
    branch = result['_neutral'].branches[0]
    assert branch['_photo_raw_branch'] == "<'a'> person img"
    assert branch['_photo_raw_negative'] == "<'a'*0.5>"
    assert '__nui_em_' in branch['_clora_branch']
    assert engine.pipe.tokenizer.get_vocab() == {'a': 0}


class Tokenizer:
    byte_decoder = {chr(i): i for i in range(256)}
    def __init__(self):
        self.vocab = {'a': 0}

    def get_vocab(self):
        return self.vocab.copy()

    def tokenize(self, text):
        return [text]

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab[token] for token in tokens]


class Encoder:
    def __init__(self):
        self.module = SimpleNamespace(weight=np.array([[1., 2.]], dtype=np.float32), num_embeddings=1, padding_idx=None)
        self.config = SimpleNamespace(vocab_size=1)

    def get_input_embeddings(self):
        return self.module

    def set_input_embeddings(self, module):
        self.module = module


class Engine:
    def __init__(self):
        self.calls = []
        self.pipe = SimpleNamespace(tokenizer=Tokenizer(), text_encoder=Encoder(),
                                    tokenizer_2=Tokenizer(), text_encoder_2=Encoder(),
                                    load_textual_inversion=self.load)
        self.embedding_token_owners = {'persistent': {'clip_l': ('persistent',)}}
        self.freed = False

    def _with_embedding_encoder_access(self, call):
        self.calls.append('access')
        return call()

    def load(self, values, token, tokenizer, text_encoder):
        tokenizer.vocab[token] = 1
        text_encoder.module = SimpleNamespace(weight=np.concatenate((text_encoder.module.weight, values)),
                                             num_embeddings=2, padding_idx=None)
        text_encoder.config.vocab_size = 2

    def _free(self):
        self.freed = True


def test_both_encoders_restore_original_modules_weights_maps_and_config():
    engine = Engine()
    backend = ModelBackend(engine, tensor_factory=lambda a: a)
    originals = [engine.pipe.text_encoder.module, engine.pipe.text_encoder_2.module]
    snapshot = backend.capture()
    backend.install('temp', (np.array([[3., 4.]]), np.array([[5., 6.]])))
    backend.restore(snapshot)
    for encoder, module, tokenizer in zip((engine.pipe.text_encoder, engine.pipe.text_encoder_2), originals,
                                          (engine.pipe.tokenizer, engine.pipe.tokenizer_2)):
        assert encoder.module is module
        np.testing.assert_array_equal(module.weight, [[1, 2]])
        assert tokenizer.vocab == {'a': 0}
        assert encoder.config.vocab_size == 1
    assert engine.embedding_token_owners == {'persistent': {'clip_l': ('persistent',)}}
    assert engine.calls == ['access', 'access', 'access']


def test_collision_on_either_stream_prevents_any_installation():
    engine = Engine()
    engine.pipe.tokenizer_2.vocab['temp_1'] = 1
    backend = ModelBackend(engine, tensor_factory=lambda a: a)
    with pytest.raises(ValueError, match='collision'):
        backend.install('temp', (np.ones((2, 2)), np.ones((2, 2))))
    assert engine.pipe.tokenizer.vocab == {'a': 0}


def test_failed_rollback_invalidates_entire_model():
    engine = Engine()
    backend = ModelBackend(engine, tensor_factory=lambda a: a)
    snapshot = backend.capture()
    engine.pipe.text_encoder.set_input_embeddings = lambda _: (_ for _ in ()).throw(RuntimeError('broken'))
    with pytest.raises(RuntimeError, match='reload'):
        backend.restore(snapshot)
    assert engine.freed


def test_encode_scope_materializes_once_and_restores_before_return():
    engine = Engine()
    engine.embedding_token_owners = {}
    def encode(prompts):
        assert 'nui_em' in prompts[0]
        np.testing.assert_array_equal(engine.pipe.text_encoder.module.weight[-1], [2, 4])
        return 'encoded'
    assert run_encoded(engine, ["<'a'*2>", ''], encode, tensor_factory=lambda a: a) == 'encoded'
    assert engine.calls == ['access']
    np.testing.assert_array_equal(engine.pipe.text_encoder.module.weight, [[1, 2]])


def test_encode_scope_plain_prompt_never_touches_backend():
    engine = SimpleNamespace()
    assert run_encoded(engine, ['ordinary'], lambda texts: texts) == ('ordinary',)


def test_encoding_device_scope_restores_both_cpu_encoders_and_prompt_override():
    from sdfx.embedding_merge_backend import encoding_device_scope
    from sdfx.prompt import _te_device
    class DeviceEncoder:
        def __init__(self): self.weight = SimpleNamespace(device='cpu')
        def get_input_embeddings(self): return SimpleNamespace(weight=self.weight)
        def to(self, device): self.weight.device = device
    engine = Engine()
    engine.device = 'cuda:0'
    engine.pipe.text_encoder = DeviceEncoder()
    engine.pipe.text_encoder_2 = DeviceEncoder()
    engine.pipe._execution_device = 'cpu'
    with encoding_device_scope(engine):
        assert engine.pipe.text_encoder.weight.device == 'cuda:0'
        assert engine.pipe.text_encoder_2.weight.device == 'cuda:0'
        assert _te_device(engine.pipe, engine.pipe.text_encoder) == 'cuda:0'
    assert engine.pipe.text_encoder.weight.device == 'cpu'
    assert engine.pipe.text_encoder_2.weight.device == 'cpu'
    assert _te_device(engine.pipe, engine.pipe.text_encoder) == 'cpu'
    assert not hasattr(engine.pipe, '_embedding_merge_execution_device')


def test_encoding_device_scope_rolls_back_first_encoder_on_second_move_failure():
    from sdfx.embedding_merge_backend import encoding_device_scope
    class DeviceEncoder:
        def __init__(self, fail=False): self.weight, self.fail = SimpleNamespace(device='cpu'), fail
        def get_input_embeddings(self): return SimpleNamespace(weight=self.weight)
        def to(self, device):
            if self.fail and device == 'cuda:0': raise RuntimeError('move failed')
            self.weight.device = device
    engine = Engine()
    engine.device = 'cuda:0'
    engine.pipe.text_encoder, engine.pipe.text_encoder_2 = DeviceEncoder(), DeviceEncoder(True)
    with pytest.raises(RuntimeError, match='move failed'):
        with encoding_device_scope(engine): pytest.fail('not reached')
    assert engine.pipe.text_encoder.weight.device == 'cpu'
    assert not hasattr(engine.pipe, '_embedding_merge_execution_device')


def test_outer_editor_transaction_composes_with_merge_once_and_restores_after_failure(monkeypatch):
    from contextlib import contextmanager
    from sdfx import embedding_merge_backend as backend_module
    from sdfx.embedding_merge_backend import run_encoding_transaction
    engine = Engine()
    original = engine.pipe.text_encoder.module.weight.copy()
    events = []
    @contextmanager
    def placement(subject):
        events.append('enter')
        try: yield
        finally: events.append('exit')
    monkeypatch.setattr(backend_module, 'encoding_device_scope', placement)
    def edit_and_encode():
        engine.pipe.text_encoder.module.weight[:] = 4
        try:
            def encoded(texts):
                assert engine.pipe.text_encoder.module.weight[-1].tolist() == [8, 8]
                raise RuntimeError('nested failure')
            return run_encoded(engine, ["<'a'*2>"], encoded, tensor_factory=lambda x: x)
        finally:
            engine.pipe.text_encoder.module.weight[:] = original
    with pytest.raises(RuntimeError, match='nested failure'):
        run_encoding_transaction(engine, edit_and_encode)
    assert engine.calls == ['access']
    assert events == ['enter', 'exit']
    assert not hasattr(engine, '_embedding_encoding_transaction')
    assert engine.pipe.tokenizer.vocab == {'a': 0}
    np.testing.assert_array_equal(engine.pipe.text_encoder.module.weight, original)


def test_transaction_marker_is_removed_after_placement_failure(monkeypatch):
    from contextlib import contextmanager
    from sdfx import embedding_merge_backend as backend_module
    from sdfx.embedding_merge_backend import run_encoding_transaction
    engine = Engine()
    @contextmanager
    def broken(subject):
        try: yield
        finally: raise RuntimeError('rollback failed')
    monkeypatch.setattr(backend_module, 'encoding_device_scope', broken)
    with pytest.raises(RuntimeError, match='rollback failed'):
        run_encoding_transaction(engine, lambda: None)
    assert not hasattr(engine, '_embedding_encoding_transaction')


def test_invalidated_model_does_not_restore_old_placement_or_mask_no_model():
    from sdfx.embedding_merge_backend import run_encoding_transaction
    from sdfx.protocol import EngineError, NO_MODEL
    engine = Engine()
    class DeviceEncoder:
        def __init__(self): self.weight = SimpleNamespace(device='cpu')
        def get_input_embeddings(self): return SimpleNamespace(weight=self.weight)
        def to(self, device):
            if engine.pipe is None: raise RuntimeError('must not touch invalidated encoder')
            self.weight.device = device
    engine.device = 'cuda:0'
    engine.pipe.text_encoder, engine.pipe.text_encoder_2 = DeviceEncoder(), DeviceEncoder()
    def invalidate():
        engine.pipe = None
        raise EngineError(NO_MODEL, 'editor rollback failed; reload')
    with pytest.raises(EngineError) as error:
        run_encoding_transaction(engine, invalidate)
    assert error.value.code == NO_MODEL
    assert not hasattr(engine, '_embedding_encoding_transaction')


def test_long_plain_generation_never_enters_merge_transaction():
    from sdfx.embedding_merge_backend import run_encoded
    text = 'ordinary words ' * 6000
    assert run_encoded(object(), (text, ''), lambda prompts: prompts) == (text, '')
