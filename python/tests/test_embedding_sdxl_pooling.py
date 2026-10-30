import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from sdfx.prompt import encode_sdxl


def test_sdxl_added_token_uses_same_pool_as_equal_original_without_changing_normal_prompts():
    cfg = transformers.CLIPTextConfig(vocab_size=12, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=77,
        eos_token_id=2, attention_dropout=0.0, projection_dim=8)
    torch.manual_seed(17)
    encoder = transformers.CLIPTextModelWithProjection(cfg).eval()
    class Tok:
        bos_token_id = 0
        eos_token_id = 3
        pad_token_id = 3
        def __call__(self, text, **kwargs):
            ids = {"cat": [1], "new": [10], ",": [4], "": []}.get(text, [])
            return type("Tokens", (), {"input_ids": ids})()
    class Pipe:
        tokenizer = Tok()
        tokenizer_2 = Tok()
        text_encoder = encoder
        text_encoder_2 = encoder
        _execution_device = "cpu"
    with torch.no_grad():
        encoder.text_model.embeddings.token_embedding.weight[10].copy_(
            encoder.text_model.embeddings.token_embedding.weight[1])
        ids = torch.tensor([[0, 1] + [3] * 75])
        upstream_plain_pool = encoder(ids, output_hidden_states=True).text_embeds
        z1, pooled1 = encode_sdxl(Pipe(), "cat")
        z2, pooled2 = encode_sdxl(Pipe(), "new")
    assert torch.equal(z1, z2)
    assert torch.equal(pooled1, upstream_plain_pool)
    assert torch.equal(pooled1, pooled2)


def test_corrected_encoder_eos_configuration_preserves_native_pool():
    cfg = transformers.CLIPTextConfig(vocab_size=12, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=77,
        eos_token_id=3, attention_dropout=0.0, projection_dim=8)
    torch.manual_seed(42)
    encoder = transformers.CLIPTextModelWithProjection(cfg).eval()
    class Tok:
        bos_token_id, eos_token_id, pad_token_id = 0, 3, 3
        def __call__(self, text, **kwargs):
            return type("Tokens", (), {"input_ids": {"new": [10], ",": [4], "": []}.get(text, [])})()
    class Pipe:
        tokenizer = Tok(); tokenizer_2 = Tok()
        text_encoder = encoder; text_encoder_2 = encoder
        _execution_device = "cpu"
    with torch.no_grad():
        expected = encoder(torch.tensor([[0, 10] + [3] * 75]), output_hidden_states=True).text_embeds
        _, actual = encode_sdxl(Pipe(), "new")
    assert torch.equal(actual, expected)
