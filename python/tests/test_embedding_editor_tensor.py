import importlib
import sys
from types import SimpleNamespace


def test_tensor_reader_uses_recorded_owners_and_each_encoder_vocabulary(monkeypatch):
    worker = importlib.import_module('sdfx.embedding_editor_tensor')
    class Tensor:
        def __init__(self, rows): self.rows = rows; self.shape = (len(rows), len(rows[0]))
        def __getitem__(self, index): return Tensor(self.rows[index])
        def detach(self): return self
        def to(self, **kwargs): return self
        def amin(self, dim): return Vector([min(row[i] for row in self.rows) for i in range(self.shape[1])])
        def amax(self, dim): return Vector([max(row[i] for row in self.rows) for i in range(self.shape[1])])
        def tolist(self): return self.rows
    class Vector:
        def __init__(self, values): self.values = values
        def tolist(self): return self.values
    class Tokenizer:
        def __init__(self, size): self.vocab_size = size
        def encode(self, text, add_special_tokens=False): return [0]
    def stream(rows):
        weights = Tensor(rows)
        return Tokenizer(2), SimpleNamespace(get_input_embeddings=lambda: SimpleNamespace(weight=weights))
    streams = {'clip_l': stream([[1., -2.], [3., 4.], [9., 9.]]),
               'clip_g': stream([[0., 1., 2.], [2., 3., 4.], [8., 8., 8.]])}
    calls = []
    def resolve(tok, weights, name, owners=None):
        calls.append((name, owners))
        return [2], Tensor([weights.rows[2]]) if not name.startswith('#') else Tensor([weights.rows[0]])
    monkeypatch.setitem(sys.modules, 'sdfx.embedding_inspector', SimpleNamespace(_streams=lambda pipe: streams, _resolve=resolve,
        _file_embeddings=lambda paths, streams: (lambda name: {'clip_l': Tensor([[9., 9.]]), 'clip_g': Tensor([[8., 8., 8.]])}, [], None)))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(float32='float32', equal=lambda a, b: a.tolist() == b.tolist()))
    engine = SimpleNamespace(pipe=object(), loaded_embedding_files={'concept': 'source'},
        embedding_token_owners={'concept': {'clip_l': ['concept'], 'clip_g': ['concept']}})
    engine._with_embedding_encoder_access = lambda fn: fn()
    result = worker.read_editor_vectors(engine, 'concept', [('apple', '#FF0000')])
    assert result['vectors'] == {'clip_l': [[9., 9.]], 'clip_g': [[8., 8., 8.]]}
    assert result['ranges']['clip_l'] == [[1., 3.], [-2., 4.]]
    assert result['ranges']['clip_g'] == [[0., 2.], [1., 3.], [2., 4.]]
    assert result['guidance'][0]['color'] == '#FF0000'
    assert result['guidance'][0]['markers']['clip_g'] == [0., 0., 0.]
    assert any(name == 'concept' and owners == {'concept': ['concept']} for name, owners in calls)


def test_reader_rejects_loaded_vector_if_file_was_changed(monkeypatch):
    import pytest
    worker = importlib.import_module('sdfx.embedding_editor_tensor')
    class Tensor:
        shape = (1, 1)
        def __init__(self, value): self.value = value
        def __getitem__(self, index): return self
        def detach(self): return self
        def to(self, **kwargs): return self
        def amin(self, dim): return self
        def amax(self, dim): return self
        def tolist(self): return [[self.value]] if self.value in (5., 9.) else [self.value]
    class Tokenizer: vocab_size = 1
    encoder = SimpleNamespace(get_input_embeddings=lambda: SimpleNamespace(weight=Tensor(1.)))
    streams = {'clip_l': (Tokenizer(), encoder)}
    monkeypatch.setitem(sys.modules, 'sdfx.embedding_inspector', SimpleNamespace(_streams=lambda pipe: streams,
        _resolve=lambda *args: ([0], Tensor(5.)),
        _file_embeddings=lambda paths, streams: (lambda name: {'clip_l': Tensor(9.)}, [], None)))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(float32='float32', equal=lambda a, b: a.tolist() == b.tolist()))
    engine = SimpleNamespace(pipe=object(), loaded_embedding_files={'concept': 'source'},
        embedding_token_owners={'concept': {'clip_l': ['concept']}})
    engine._with_embedding_encoder_access = lambda fn: fn()
    with pytest.raises(ValueError, match='differs from loaded'):
        worker.read_editor_vectors(engine, 'concept', [])
