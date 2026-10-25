import pytest

from sdfx.prompt import (apply_pad, chunk_tokens, parse_prompt_attention, sd_layer_choice, sdxl_layer_choice,
                         strip_extra_networks, unsupported_syntax_warnings)


@pytest.mark.parametrize("cs,expected", [(1, 2), (2, 2), (3, 3), (4, 4)])
def test_sdxl_clip_skip_matches_reforge(cs, expected):
    # modules_forge/forge_clip.py CLIP_SD_XL_L/G: minimal_clip_skip = 2 -> penultimate for 1..2
    assert sdxl_layer_choice(cs) == expected


def test_sd15_and_sd2_clip_skip():
    assert sd_layer_choice(1) is None and sd_layer_choice(2) == 2 and sd_layer_choice(3) == 3
    # SD2 (CLIP_SD_21_H, minimal 2): penultimate by default; diffusers SD2 encoder has 23 of 24 layers
    assert sd_layer_choice(1, True, 23) is None and sd_layer_choice(2, True, 23) is None
    assert sd_layer_choice(3, True, 23) == 2
    assert sd_layer_choice(1, True, 24) == 2 and sd_layer_choice(3, True, 24) == 3


def approx_pairs(got, expected):
    assert len(got) == len(expected), got
    for (t, w), (et, ew) in zip(got, expected):
        assert t == et
        assert w == pytest.approx(ew)


@pytest.mark.parametrize("text,expected", [
    ("normal text", [["normal text", 1.0]]),
    ("an (important) word", [["an ", 1.0], ["important", 1.1], [" word", 1.0]]),
    ("(unbalanced", [["unbalanced", 1.1]]),
    ("\\(literal\\]", [["(literal]", 1.0]]),
    ("(unnecessary)(parens)", [["unnecessaryparens", 1.1]]),
    ("a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).",
     [["a ", 1.0], ["house", 1.5730000000000004], [" ", 1.1], ["on", 1.0], [" a ", 1.1], ["hill", 0.55],
      [", sun, ", 1.1], ["sky", 1.4641000000000006], [".", 1.1]]),
    ("[word]", [["word", 1 / 1.1]]),
    ("(word:0.8) x", [["word", 0.8], [" x", 1.0]]),
    ("", [["", 1.0]]),
])
def test_a1111_doctests(text, expected):
    approx_pairs(parse_prompt_attention(text), expected)


def test_break():
    got = parse_prompt_attention("a cat BREAK a dog")
    assert got == [["a cat", 1.0], ["BREAK", -1], ["a dog", 1.0]]


def test_repeat_syntax():
    approx_pairs(parse_prompt_attention("(hello world, a poem, :x2.5)"),
                 [["hello world, a poem, ", 1.0], ["hello world, a poem, ", 1.5]])
    approx_pairs(parse_prompt_attention("(hello world, a poem, :x1.4)"), [["hello world, a poem, ", 1.4]])


def _tok(texts):
    # whitespace "tokenizer": token id = len(word) + 10, ',' = 1
    out = []
    for t in texts:
        ids = []
        for w in t.replace(",", " , ").split():
            ids.append(1 if w == "," else 10 + len(w))
        out.append(ids)
    return out


def test_chunking_basic():
    chunks = chunk_tokens(parse_prompt_attention("a (bb:1.5) c"), _tok, 49406, 49407, comma_token=1)
    assert len(chunks) == 1
    toks, mults = chunks[0]
    assert len(toks) == 77 and len(mults) == 77
    assert toks[0] == 49406 and toks[1:4] == [11, 12, 11] and toks[4] == 49407 and toks[-1] == 49407
    assert mults[1:4] == [1.0, 1.5, 1.0]


def test_chunking_break_and_overflow():
    chunks = chunk_tokens(parse_prompt_attention("a BREAK b"), _tok, 0, 9, comma_token=1)
    assert len(chunks) == 2
    long = " ".join(["w"] * 100)
    chunks = chunk_tokens(parse_prompt_attention(long), _tok, 0, 9, comma_token=1)
    assert len(chunks) == 2
    assert all(len(t) == 77 for t, _ in chunks)
    assert chunks[1][0][1:26] == [11] * 25


def test_comma_backtrack():
    words = ["w"] * 70 + [","] + ["w"] * 10
    chunks = chunk_tokens([[" ".join(words), 1.0]], _tok, 0, 9, comma_token=1)
    assert len(chunks) == 2
    first = chunks[0][0]
    # the comma (position 71 incl. BOS) ends the first chunk; the following words moved to the next chunk
    assert first[71] == 1 and first[72] == 9
    assert chunks[1][0][1:5] == [11, 11, 11, 11]


def test_apply_pad():
    assert apply_pad([0, 5, 9, 9, 9], 9, 7) == [0, 5, 9, 7, 7]
    assert apply_pad([0, 5, 9, 9], 9, 9) == [0, 5, 9, 9]


def test_extra_networks_and_warnings():
    assert strip_extra_networks("a <lora:foo:0.5> b") == "a  b"
    assert unsupported_syntax_warnings("[cat:dog:0.5]")
    assert unsupported_syntax_warnings("x AND y")
    assert unsupported_syntax_warnings("(cat:1.2) [dog]") == []
