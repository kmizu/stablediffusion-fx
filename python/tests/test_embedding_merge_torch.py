"""Tiny actual transformers/diffusers TI restoration; CPU only, no downloads."""
import json
from types import SimpleNamespace
import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')
from diffusers.loaders import TextualInversionLoaderMixin
from sdfx.embedding_merge_backend import run_encoded
from sdfx.prompt import encode_sd15, encode_sdxl


@pytest.mark.parametrize('original_inference', [False, True])
def test_inference_scoped_merge_preserves_original_embedding_storage_kind(tmp_path, original_inference):
    from sdfx.embedding_merge_pipeline import run_pipeline
    engine = tiny_engine(tmp_path, True)
    engine._activate = lambda pipe: None
    modules = [engine.pipe.text_encoder.get_input_embeddings(), engine.pipe.text_encoder_2.get_input_embeddings()]
    if original_inference:
        with torch.inference_mode():
            for module in modules:
                module.weight.data = module.weight.detach().clone()
    original = [(m.weight, torch.is_inference(m.weight), m.weight.detach().clone()) for m in modules]
    with torch.inference_mode():
        run_pipeline(engine, engine.pipe, ["<'a'>"], lambda texts: None, check=lambda: None)
    for module, (parameter, inference, value) in zip(modules, original):
        assert module.weight is parameter
        assert torch.is_inference(module.weight) == inference
        assert torch.equal(module.weight, value)


def test_photo_signed_target_uses_the_same_encoder_and_positions_as_native_negpip(tmp_path):
    from sdfx import negpip
    engine = tiny_engine(tmp_path, True)
    engine.dtype = torch.float32
    engine._encode_plain = lambda p, n, gp, bs: {
        'prompt_embeds': encode_sdxl(engine.pipe, p)[0].repeat(bs, 1, 1),
        'negative_prompt_embeds': encode_sdxl(engine.pipe, n)[0].repeat(bs, 1, 1)}
    native = negpip.encode(engine, 'a (a:-2)', '', SimpleNamespace(), 1)
    signed = negpip.encode_photo_target(engine.pipe, 'a', 2., clip_skip=1, dtype=torch.float32)
    assert torch.equal(signed, native['prompt_embeds'][0, 77:, :-1])
    assert torch.all(native['prompt_embeds'][0, 77:, -1] == -1)


def test_repeated_photo_merge_with_real_sequential_offload_hooks(tmp_path):
    from accelerate import cpu_offload
    from accelerate.hooks import remove_hook_from_module
    from sdfx.embedding_merge_pipeline import run_pipeline
    engine = tiny_engine(tmp_path, True)
    encoders = [engine.pipe.text_encoder, engine.pipe.text_encoder_2]
    def install():
        with torch.inference_mode():
            for encoder in encoders: cpu_offload(encoder, execution_device=torch.device('cpu'))
    def remove():
        with torch.inference_mode():
            for encoder in encoders: remove_hook_from_module(encoder, recurse=True)
    def access(operation):
        remove()
        try: return operation()
        finally: install()
    engine._offload = True
    engine._with_embedding_encoder_access = access
    engine._activate = lambda pipe: None
    install()
    def encode(texts):
        for encoder in encoders:
            output = encoder(input_ids=torch.tensor([[0, 1, 2]]), output_hidden_states=True)
            assert torch.isfinite(output.hidden_states[-1]).all()
        return 'ok'
    for _ in range(3):
        with torch.inference_mode():
            assert run_pipeline(engine, engine.pipe, ["<'a'>"], encode, check=lambda: None) == 'ok'


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


def test_composable_metadata_recreates_photo_merge_tokens_after_first_scope(tmp_path):
    from sdfx.embedding_merge_pipeline import run_pipeline
    engine = tiny_engine(tmp_path, True)
    engine._activate = lambda pipe: None
    pipe = engine.pipe
    original = [(tokenizer, tokenizer.get_vocab().copy(), encoder.get_input_embeddings()) for tokenizer, encoder in
                ((pipe.tokenizer, pipe.text_encoder), (pipe.tokenizer_2, pipe.text_encoder_2))]
    def branch_encode(texts):
        branch = {'_clora_branch': texts[0], '_clora_prompt': texts[0], '_clora_negative': texts[1]}
        return {'_neutral': SimpleNamespace(branches=[branch])}
    result = run_encoded(engine, ("<'a'> person img", "<'a'*0.5>"), branch_encode)
    branch = result['_neutral'].branches[0]
    assert branch['_photo_raw_branch'] == "<'a'> person img"
    assert branch['_photo_raw_negative'] == "<'a'*0.5>"
    for tokenizer, vocab, encoder_module in original:
        assert tokenizer.get_vocab() == vocab
    seen = []
    def photo(texts):
        assert '__nui_em_' in texts[0] and '__nui_em_' in texts[1]
        assert any(name.startswith('__nui_em_') for name in pipe.tokenizer.get_vocab())
        seen.append(tuple(texts))
        return 'generated'
    assert run_pipeline(engine, pipe, (branch['_photo_raw_branch'], branch['_photo_raw_negative']), photo, check=lambda: None) == 'generated'
    assert seen
    for tokenizer, vocab, encoder_module in original:
        assert tokenizer.get_vocab() == vocab
