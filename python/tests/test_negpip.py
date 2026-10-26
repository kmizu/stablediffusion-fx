import pytest

from sdfx.negpip import extract, escape_text
from sdfx.prompt import parse_prompt_attention


def test_extract_negative_segments_and_preserve_ordinary_prompts_exactly():
    for text in ("a woman, (red scarf:1.2)", r"a \(literal\) [word]", "one BREAK two"):
        clean, targets = extract(text)
        assert clean == text and targets == []
    clean, targets = extract("a woman, (black:-2), (red:1.3) scarf")
    assert targets == [("black", -2.0)]
    assert parse_prompt_attention(clean) == [["a woman, , ", 1.0], ["red", 1.3], [" scarf", 1.0]]


def test_nested_weights_escaped_brackets_and_break_marker():
    clean, targets = extract(r"((black:2):-1) BREAK \(literal\)")
    assert targets == [("black", -2.0)]
    parsed = parse_prompt_attention(clean)
    assert ["BREAK", -1] in parsed
    assert any("(literal)" in text for text, _ in parsed)
    assert extract("word BREAK word")[1] == []
    assert parse_prompt_attention(escape_text(r"a (b) [c] \ d")) == [[r"a (b) [c] \ d", 1.0]]


def test_attention_keeps_keys_negates_projected_values_and_masks_padding():
    torch = pytest.importorskip("torch")
    from diffusers.models.attention_processor import Attention
    from types import SimpleNamespace
    from sdfx.negpip import install
    torch.manual_seed(3)
    attention = Attention(query_dim=4, cross_attention_dim=4, heads=2, dim_head=2, bias=True)
    root = torch.nn.ModuleList([attention])
    pipe = SimpleNamespace(unet=root)
    x, context = torch.randn(2, 3, 4), torch.randn(2, 5, 4)
    signs = torch.tensor([[1, 1, -1, 0, 0], [1, -1, -1, 1, 0.]])
    names = list(root.state_dict())
    baseline = attention(x, encoder_hidden_states=context)
    q = attention.to_q(x).view(2, 3, 2, 2).transpose(1, 2)
    k = attention.to_k(context).view(2, 5, 2, 2).transpose(1, 2)
    v = (attention.to_v(context) * signs[..., None]).view(2, 5, 2, 2).transpose(1, 2)
    mask = torch.zeros(2, 1, 1, 5).masked_fill(signs[:, None, None, :] == 0, -torch.inf)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    expected = expected.transpose(1, 2).reshape(2, 3, 4)
    expected = attention.to_out[1](attention.to_out[0](expected))
    remove = install(pipe)
    marked = torch.cat([context, signs[..., None]], dim=-1)
    torch.testing.assert_close(attention(x, encoder_hidden_states=marked), expected)
    torch.testing.assert_close(attention(x, encoder_hidden_states=context), baseline, rtol=0, atol=0)
    assert list(root.state_dict()) == names
    assert not attention.to_v._forward_hooks
    remove()
    remove()
    assert "forward" not in attention.__dict__
    torch.testing.assert_close(attention(x, encoder_hidden_states=context), baseline, rtol=0, atol=0)


def test_projection_hooks_removed_when_processor_raises():
    torch = pytest.importorskip("torch")
    from diffusers.models.attention_processor import Attention
    from types import SimpleNamespace
    from sdfx.negpip import install
    attention = Attention(query_dim=4, cross_attention_dim=4, heads=1, dim_head=4)
    class Broken:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("broken processor")
    attention.set_processor(Broken())
    remove = install(SimpleNamespace(unet=torch.nn.ModuleList([attention])))
    with pytest.raises(RuntimeError, match="broken processor"):
        attention(torch.randn(1, 2, 4), encoder_hidden_states=torch.ones(1, 3, 5))
    assert not attention.to_v._forward_hooks
    remove()


def test_schedule_alignment_and_embedding_modifiers_preserve_signs():
    torch = pytest.importorskip("torch")
    from sdfx.negpip import align_variants, adjust
    plain = {k: torch.ones(1, 77, 4) for k in ("prompt_embeds", "negative_prompt_embeds")}
    ordinary = [(3, plain)]
    assert align_variants(ordinary) is ordinary
    signed = {k: torch.cat([torch.ones(1, 80, 4), -torch.ones(1, 80, 1)], -1) for k in plain}
    signed["_negpip"] = True
    schedule = align_variants([(2, plain), (4, signed), (6, plain)])
    assert schedule[0][1] is schedule[2][1]
    first = schedule[0][1]
    assert first["prompt_embeds"].shape == (1, 80, 5)
    assert first["prompt_embeds"][0, 76, -1] == 1
    assert first["prompt_embeds"][0, 77, -1] == 0
    def zero(d):
        assert "_negpip" not in d
        assert d["prompt_embeds"].shape[-1] == 4
        return {k: torch.zeros_like(v) for k, v in d.items()}
    out = adjust(first, zero)
    assert not out["prompt_embeds"][..., :-1].any()
    torch.testing.assert_close(out["prompt_embeds"][..., -1], first["prompt_embeds"][..., -1])


def test_encoder_separates_branches_crops_tokens_and_keeps_clean_pooled():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from sdfx.engine import Engine
    from sdfx.params import GenerationParams
    class Tokenizer:
        bos_token_id, eos_token_id, pad_token_id = 100, 101, 101
        def __call__(self, text, **kw):
            return SimpleNamespace(input_ids=list(range(1, len(text.split()) + 1)))
    engine = Engine.__new__(Engine)
    engine.pipe = SimpleNamespace(tokenizer=Tokenizer())
    calls = []
    def plain(pos, neg, gp, bs):
        calls.append((pos, neg, bs))
        return {"prompt_embeds": torch.arange(77.).view(1, 77, 1).expand(bs, 77, 4),
                "negative_prompt_embeds": torch.zeros(bs, 77, 4),
                "pooled_prompt_embeds": torch.full((bs, 8), len(pos)),
                "negative_pooled_prompt_embeds": torch.full((bs, 8), len(neg))}
    engine._encode_plain = plain
    gp = GenerationParams(hooks={"negpip": {}})
    out = engine._encode("a (black hair:-2)", "bad (red:-1)", gp, 2)
    assert calls == [("a ", "bad ", 2), ("(black hair:2)", "", 1), ("(red:1)", "", 1)]
    assert out["prompt_embeds"].shape == (2, 80, 5)
    assert out["prompt_embeds"][0, 77:, 0].tolist() == [1, 2, 3]  # two words plus EOS
    assert out["prompt_embeds"][0, 77:, -1].tolist() == [-1, -1, -1]
    assert out["negative_prompt_embeds"][0, 77:, -1].tolist() == [-1, -1, 0]
    assert out["pooled_prompt_embeds"][0, 0] == 2
    schedule = engine._encode_scheduled("{black::1--1}", "", GenerationParams(hooks={"negpip": {}, "weight_gradient": {}}), 2, 3)
    assert [d["prompt_embeds"].shape for _, d in schedule.schedule] == [(2, 79, 5)] * 3
    assert schedule.schedule[0][1]["prompt_embeds"][0, 77:, -1].tolist() == [0, 0]
    assert schedule.schedule[-1][1]["prompt_embeds"][0, 77:, -1].tolist() == [-1, -1]


@pytest.mark.parametrize("fused", [False, True])
def test_existing_masks_and_fused_processors(fused):
    torch = pytest.importorskip("torch")
    from diffusers.models.attention_processor import Attention, FusedAttnProcessor2_0
    from types import SimpleNamespace
    from sdfx.negpip import install
    torch.manual_seed(4)
    a = Attention(query_dim=4, cross_attention_dim=4, heads=2, dim_head=2, bias=True)
    if fused:
        a.fuse_projections()
        a.set_processor(FusedAttnProcessor2_0())
    x, c = torch.randn(1, 2, 4), torch.randn(1, 4, 4)
    signs = torch.tensor([[1., -1, 0, -1]])
    mask = torch.tensor([[[True, False, True, True]]])
    q = a.to_q(x).view(1, 2, 2, 2).transpose(1, 2)
    k = a.to_k(c).view(1, 4, 2, 2).transpose(1, 2)
    v = (a.to_v(c) * signs[..., None]).view(1, 4, 2, 2).transpose(1, 2)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=torch.tensor([[[[True, False, False, True]]]]))
    expected = a.to_out[1](a.to_out[0](expected.transpose(1, 2).reshape(1, 2, 4)))
    remove = install(SimpleNamespace(unet=torch.nn.ModuleList([a])))
    try:
        actual = a(x, encoder_hidden_states=torch.cat([c, signs[..., None]], -1), attention_mask=mask)
        torch.testing.assert_close(actual, expected)
    finally:
        remove()


def test_torch_free_feature_advertisement():
    from sdfx.hooks import KNOWN_HOOKS
    from sdfx.dryrun import FakeEngine
    assert "negpip" in KNOWN_HOOKS
    assert "negpip" in FakeEngine().features()
