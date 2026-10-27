"""Per-image prompt composition with synthetic interrogator results."""
import pytest
from sdfx.batch_interrogate import BatchInterrogateOptions, compose


def test_ordered_results_prepend_and_independent_inputs():
    options = BatchInterrogateOptions()
    first = compose("base", "bad", ["cat, blue", "dog, cat"], options)
    second = compose("base", "bad", ["bird"], options)
    assert first.positive == "cat, blue, dog, base"
    assert first.negative == "bad"
    assert second.positive == "bird, base"
    assert second.negative == "bad"


def test_reverse_append_and_empty_do_not_add_syntax():
    opts = BatchInterrogateOptions(reverse=True, position="append")
    result = compose("positive", "bad", ["blue, cat"], opts)
    assert result.positive == "positive"
    assert result.negative == "bad, blue, cat"
    assert compose("positive", "bad", [" , "], opts).negative == "bad"
    assert compose("", "", [""], BatchInterrogateOptions()).positive == ""


def test_dedup_and_exaggeration():
    assert compose("", "", ["cat, dog, cat", "dog, bird"], BatchInterrogateOptions()).positive == "cat, dog, bird"
    assert compose("", "", ["cat, dog, cat"], BatchInterrogateOptions(exaggeration=True)).positive == "cat, dog, cat"


def test_replacement_filter_and_weight_order():
    opts = BatchInterrogateOptions(replacements=(("cat", "feline"), ("remove", "")),
        filter_positive=True, filter_negative=True, custom_filter="(blue:1.2)", use_custom_filter=True,
        weight=0.5)
    result = compose("(feline:1.2), base", "(bad:0.5)", ["cat, bad, blue, remove, red"], opts)
    assert result.positive == "(red:0.5), (feline:1.2), base"
    assert result.negative == "(bad:0.5)"


def test_punctuation_preserves_emoticons_and_underscore_exceptions():
    opts = BatchInterrogateOptions(remove_punctuation=True)
    result = compose("", "", ["hello!, ^_^, smile :), cat-dog"], opts)
    assert result.positive == "hello, _, smile :), catdog"


def test_bad_options_and_oversized_text_rejected():
    for options in (BatchInterrogateOptions(position="middle"), BatchInterrogateOptions(weight=float("nan")),
                    BatchInterrogateOptions(weight=1.5), BatchInterrogateOptions(replacements=(("", "x"),))):
        with pytest.raises(ValueError): compose("a", "", ["b"], options)
    with pytest.raises(ValueError): compose("x" * 1_000_001, "", [], BatchInterrogateOptions())



def test_placeholder_selection_is_bounded_for_long_matching_prefix():
    from sdfx.batch_interrogate import _placeholder_prefix
    text = "SDFXEMOTICON" + "X" * 20000 + ":)"
    marker = _placeholder_prefix(text)
    assert len(marker) <= 80 and marker not in text
    result = compose("", "", [text], BatchInterrogateOptions(remove_punctuation=True))
    assert result.positive.endswith(":)")


def test_aggregate_replacement_budget_and_work_budget_rejected():
    options = BatchInterrogateOptions(replacements=tuple((f"key{i}", "v" * 5000) for i in range(30)))
    with pytest.raises(ValueError, match="(?i)(limit|budget)"):
        compose("base", "", ["small"], options)
    many = BatchInterrogateOptions(replacements=tuple((f"k{i}", "v") for i in range(100)))
    with pytest.raises(ValueError, match="(?i)(limit|budget)"):
        compose("", "", ["a" * 20000], many)


def test_underscore_emoji_survives_punctuation_filter():
    result = compose("", "", [">_<, cat!"], BatchInterrogateOptions(remove_punctuation=True))
    assert result.positive == ">_<, cat"


def test_control_characters_rejected_and_replacements_are_literal():
    with pytest.raises(ValueError, match="(?i)control"):
        compose("base", "", ["bad\x00text"], BatchInterrogateOptions())
    result = compose("", "", ["cat"], BatchInterrogateOptions(replacements=(("cat", r"\1\path"),)))
    assert result.positive == r"\1\path"
