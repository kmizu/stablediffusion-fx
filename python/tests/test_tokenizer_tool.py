from types import SimpleNamespace

import pytest

from sdfx.base import BaseEngine
from sdfx.protocol import EngineError


class TinyClip:
    """Small real byte vocabulary; no network/model dependencies in adapter tests."""
    name_or_path = "test-byte-vocabulary"
    model_max_length = 77
    bos_token_id = 10
    eos_token_id = 11
    pad_token_id = 11
    byte_decoder = {"a": 97, "b": 98, "Ã": 195, "©": 169}
    vocab = {"a</w>": 1, "b</w>": 2, "Ã": 3, "©</w>": 4, "<bos>": 10, "<eos>": 11}

    def get_vocab(self):
        return self.vocab

    def __call__(self, text, **kw):
        assert kw == {"add_special_tokens": False, "truncation": False}
        return SimpleNamespace(input_ids={"": [], "a b": [1, 2], "é": [3, 4]}.get(text, [1] * len(text)))

    def decode(self, ids, **kw):
        raw = "".join({v: k for k, v in self.vocab.items()}[i] for i in ids)
        return raw.replace("</w>", " ")


def engine():
    e = BaseEngine()
    e.model = {"name": "test", "kind": "sdxl"}
    e.pipe = SimpleNamespace(tokenizer=TinyClip(), tokenizer_2=TinyClip())
    return e


def test_text_returns_both_actual_streams_and_no_special_tokens():
    r = engine().tokenize({"text": "a b"}, None)
    assert [s["name"] for s in r["streams"]] == ["tokenizer", "tokenizer_2"]
    assert r["streams"][0]["ids"] == [1, 2]
    assert r["streams"][0]["count"] == 2
    assert r["streams"][0]["content_capacity"] == 75
    assert r["streams"][0]["tokens"][0]["piece"] == "a</w>"


def test_unicode_group_keeps_all_ids_and_byte_fragments():
    s = engine().tokenize({"text": "é"}, None)["streams"][0]
    assert s["groups"] == [{"ids": [3, 4], "text": "é "}]
    assert [t["bytes_hex"] for t in s["tokens"]] == ["c3", "a9 20"]


def test_id_input_preserves_ids_specials_and_empty():
    e = engine()
    assert e.tokenize({"text": "1, 2,10", "input_mode": "ids"}, None)["streams"][0]["ids"] == [1, 2, 10]
    assert e.tokenize({"text": "", "input_mode": "ids"}, None)["streams"][0]["count"] == 0
    assert e.tokenize({"text": ""}, None)["streams"][0]["ids"] == []


@pytest.mark.parametrize("params", [{"text": "1,,2", "input_mode": "ids"}, {"text": "999", "input_mode": "ids"},
                                      {"text": "-1", "input_mode": "ids"}, {"text": 2}, {"input_mode": "bad"},
                                      {"text": "x" * 65537}])
def test_rejects_invalid_input(params):
    with pytest.raises(EngineError) as err:
        engine().tokenize(params, None)
    assert err.value.code == "invalid_params"


def test_sd15_has_one_stream_and_long_input_is_not_truncated():
    e = engine()
    e.pipe.tokenizer_2 = None
    r = e.tokenize({"text": "a" * 100}, None)
    assert len(r["streams"]) == 1
    assert r["streams"][0]["count"] == 100


def test_no_model_and_dry_run_are_explicit():
    with pytest.raises(EngineError) as err:
        BaseEngine().tokenize({"text": "cat"}, None)
    assert err.value.code == "no_model"
    e = engine()
    e.dry_run = True
    with pytest.raises(EngineError) as err:
        e.tokenize({"text": "cat"}, None)
    assert err.value.code == "unsupported"


def test_added_unicode_vocabulary_is_literal_not_a_byte_encoded_piece():
    e = engine()
    tok = TinyClip()
    tok.vocab = {**tok.vocab, "猫": 12}
    tok.get_added_vocab = lambda: {"猫": 12}
    e.pipe.tokenizer = tok
    e.pipe.tokenizer_2 = None
    s = e.tokenize({"text": "12", "input_mode": "ids"}, None)["streams"][0]
    assert s["groups"] == [{"ids": [12], "text": "猫"}]


def test_added_special_ids_are_classified_and_retained_in_text_input():
    e = engine()
    class SpecialClip(TinyClip):
        def __call__(self, text, **kw):
            assert text == "<added-special>"
            assert kw == {"add_special_tokens": False, "truncation": False}
            return SimpleNamespace(input_ids=[12])
    tok = SpecialClip()
    tok.vocab = {**tok.vocab, "<added-special>": 12}
    tok.all_special_ids = [10, 11, 12]
    tok.get_added_vocab = lambda: {"<added-special>": 12}
    e.pipe.tokenizer = tok
    e.pipe.tokenizer_2 = None
    s = e.tokenize({"text": "12", "input_mode": "ids"}, None)["streams"][0]
    assert s["tokens"][0]["special"] is True
    assert s["tokens"][0]["piece"] == "<added-special>"
    assert s["groups"] == [{"ids": [12], "text": "<added-special>"}]
    literal = e.tokenize({"text": "<added-special>"}, None)["streams"][0]
    assert literal["ids"] == [12]
    assert literal["tokens"][0]["special"] is True


def test_excessive_token_count_is_rejected_instead_of_truncated():
    with pytest.raises(EngineError) as err:
        engine().tokenize({"text": "a" * 8193}, None)
    assert err.value.code == "invalid_params"


def test_cached_real_clip_unicode_and_dual_streams():
    """Opt in for CPU-only real vocab validation; never download model files."""
    import os
    if os.environ.get("SDFX_TEST_CACHED_TOKENIZERS") != "1":
        pytest.skip("set SDFX_TEST_CACHED_TOKENIZERS=1 with cached CLIP tokenizers")
    from transformers import CLIPTokenizer
    from sdfx.tokenizer_tool import inspect
    a = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
    b = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2", local_files_only=True)
    p = SimpleNamespace(tokenizer=a, tokenizer_2=b)
    text = "a cat, 日本語 café 🐈"
    r = inspect(p, {"text": text})
    assert len(r["streams"]) == 2
    assert r["streams"][0]["ids"] == [320, 2368, 267, 39121, 44353, 34002, 508, 15304, 2074, 486]
    assert [s["pad_id"] for s in r["streams"]] == [49407, 0]
    for s, tok in zip(r["streams"], (a, b)):
        assert s["ids"] == tok(text, add_special_tokens=False, truncation=False).input_ids
        assert "".join(g["text"] for g in s["groups"]).strip() == s["decoded"].strip()
        assert [t["id"] for t in s["tokens"]] == s["ids"]
        decoded = inspect(p, {"text": ",".join(map(str, s["ids"])), "input_mode": "ids"})
        assert decoded["streams"][0]["decoded"] == s["decoded"]


def test_tokenize_dryrun_json_lines_route_is_explicitly_unsupported():
    from test_server_dryrun import Engine
    e = Engine()
    try:
        hello = e.next()
        assert "tokenizer" not in hello["features"]
        r, _ = e.call("tokenize", {"text": "a cat"})
        assert r["type"] == "error"
        assert r["code"] == "unsupported"
        assert "dry-run has no tokenizer" in r["message"]
        pong, _ = e.call("ping")
        assert pong["type"] == "result"
    finally:
        e.close()
