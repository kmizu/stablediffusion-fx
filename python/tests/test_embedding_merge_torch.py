"""Tiny actual transformers/diffusers TI restoration; CPU only, no downloads."""
import json
from types import SimpleNamespace
import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')
from diffusers.loaders import TextualInversionLoaderMixin
from sdfx.embedding_merge_backend import run_encoded
from sdfx.prompt import encode_sd15, encode_sdxl


def tiny_engine(tmp_path, sdxl):
    vocab, merges = tmp_path / 'vocab.json', tmp_path / 'merges.txt'
    vocab.write_text(json.dumps({'<|startoftext|>': 0, 'a</w>': 1, '<|endoftext|>': 2}))
    merges.write_text('#version: 0.2\n')
    def stream(projected=False):
        tokenizer = transformers.CLIPTokenizer(str(vocab), str(merges))
        config = transformers.CLIPTextConfig(vocab_size=3, hidden_size=8, intermediate_size=16,
            num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=77,
            eos_token_id=2, attention_dropout=0.0, projection_dim=8)
        cls = transformers.CLIPTextModelWithProjection if projected else transformers.CLIPTextModel
        return tokenizer, cls(config).eval()
    class Pipe(TextualInversionLoaderMixin):
        hf_device_map = None
        components = {}
        _execution_device = torch.device('cpu')
    torch.manual_seed(173)
    pipe = Pipe()
    pipe.tokenizer, pipe.text_encoder = stream()
    if sdxl:
        pipe.tokenizer_2, pipe.text_encoder_2 = stream(True)
    return SimpleNamespace(pipe=pipe, device=torch.device('cpu'), embedding_token_owners={},
        _with_embedding_encoder_access=lambda operation: operation(), _free=lambda: None)


@pytest.mark.parametrize('sdxl', [False, True])
def test_actual_clip_equal_vectors_equal_context_and_restore_every_row(tmp_path, sdxl):
    engine = tiny_engine(tmp_path, sdxl)
    pipe = engine.pipe
    streams = [(pipe.tokenizer, pipe.text_encoder)]
    if sdxl: streams.append((pipe.tokenizer_2, pipe.text_encoder_2))
    before = [(tokenizer.get_vocab().copy(), encoder.get_input_embeddings(),
               encoder.get_input_embeddings().weight.detach().clone()) for tokenizer, encoder in streams]
    def encode(text):
        with torch.no_grad():
            return encode_sdxl(pipe, text) if sdxl else (encode_sd15(pipe, text),)
    ordinary = encode('a')
    actual = run_encoded(engine, ["<'a'>"], lambda texts: encode(texts[0]))
    for left, right in zip(ordinary, actual):
        assert torch.equal(left, right)
    for (tokenizer, encoder), (vocab, module, weight) in zip(streams, before):
        assert tokenizer.get_vocab() == vocab
        assert encoder.get_input_embeddings() is module
        assert torch.equal(module.weight, weight)
    assert not engine.embedding_token_owners
    assert not hasattr(pipe, '_embedding_merge_execution_device')
    with pytest.raises(RuntimeError, match='encode failure'):
        run_encoded(engine, ["<'a'*2>"], lambda texts: (_ for _ in ()).throw(RuntimeError('encode failure')))
    for (tokenizer, encoder), (vocab, module, weight) in zip(streams, before):
        assert tokenizer.get_vocab() == vocab and encoder.get_input_embeddings() is module
        assert torch.equal(module.weight, weight)


@pytest.mark.parametrize('sdxl', [False, True])
def test_actual_editor_row_change_then_merge_reuses_one_transaction(tmp_path, sdxl):
    from sdfx.embedding_merge_backend import run_encoding_transaction
    engine = tiny_engine(tmp_path, sdxl)
    pipe = engine.pipe
    encoders = [pipe.text_encoder] + ([pipe.text_encoder_2] if sdxl else [])
    original = [(encoder.get_input_embeddings().weight, encoder.get_input_embeddings().weight.detach().clone())
                for encoder in encoders]
    def encode(text):
        with torch.no_grad():
            return encode_sdxl(pipe, text) if sdxl else (encode_sd15(pipe, text),)
    ordinary = encode('a')
    calls = []
    engine._with_embedding_encoder_access = lambda operation: calls.append('guard') or operation()
    def edit():
        try:
            with torch.no_grad():
                for parameter, _ in original:
                    parameter[1].mul_(2)
            return run_encoded(engine, ["<'a'*0.5>"], lambda texts: encode(texts[0]))
        finally:
            with torch.no_grad():
                for parameter, before in original:
                    parameter.copy_(before)
    actual = run_encoding_transaction(engine, edit)
    assert calls == ['guard']
    for left, right in zip(ordinary, actual): assert torch.equal(left, right)
    for encoder, (parameter, before) in zip(encoders, original):
        assert encoder.get_input_embeddings().weight is parameter
        assert torch.equal(parameter, before)
    assert not hasattr(engine, '_embedding_encoding_transaction')
