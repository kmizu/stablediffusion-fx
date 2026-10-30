import pytest

from sdfx.embedding_merge_prompts import expressions, substitute, restore_provenance


def test_expressions_ignore_network_tags_and_delimiters_inside_literals():
    prompt = "<lora:style:.3> (<'a>b'+'it''s'>:1.2), {'x}y'*2}"
    spans = expressions(prompt)
    assert [s.expression for s in spans] == ["'a>b'+'it''s'", "'x}y'*2"]
    assert substitute(prompt, lambda s: 'TOKEN') == '<lora:style:.3> (TOKEN:1.2), TOKEN'


def test_no_expression_preserves_exact_text_and_repetition_order():
    assert substitute('words  < lora >', lambda _: 'bad') == 'words  < lora >'
    seen = []
    result = substitute("<'a'> + <'a'>", lambda s: seen.append(s.original) or str(len(seen)))
    assert result == '1 + 2'
    assert seen == ["<'a'>", "<'a'>"]


def test_unclosed_expression_and_malformed_operators_fail():
    for text in ["a <'b'", "{'a' +}", "<'a'/'b'>"]:
        with pytest.raises(ValueError):
            expressions(text)


def test_source_png_provenance_roundtrip_with_internal_delimiters_and_quotes():
    metadata = "<'EM_1'>=<'a>b'+'it''s'>, {'EM_2'}={'negative'*0.5}"
    assert restore_provenance("(<'EM_1'>:1.2), {'EM_2'}", metadata) == \
        "(<'a>b'+'it''s'>:1.2), {'negative'*0.5}"


def test_provenance_is_single_pass_and_rejects_duplicate_keys():
    assert restore_provenance("<'EM_1'>", "<'EM_1'>=<'EM_2'>, <'EM_2'>=<'other'>") == "<'EM_2'>"
    with pytest.raises(ValueError, match='duplicate'):
        restore_provenance("<'EM_1'>", "<'EM_1'>=<'a'>, <'EM_1'>=<'b'>")


@pytest.mark.parametrize('metadata', ["<'name'>=<'a'>", "<'EM_1'>=bad", "<'EM_1'>=<'a'> trailing"])
def test_malformed_metadata_is_not_silently_applied(metadata):
    with pytest.raises(ValueError):
        restore_provenance("<'EM_1'>", metadata)


def test_pasted_infotext_restores_both_prompts_and_removes_only_provenance():
    from sdfx.embedding_merge_prompts import restore_infotext
    import json
    provenance = "<'EM_1'>=<'cat' * 2>, {'EM_2'}={'dog':1}"
    text = "portrait <'EM_1'>\nNegative prompt: {'EM_2'}\nSteps: 20, EmbeddingMerge: " + json.dumps(provenance) + ", Seed: 17"
    restored = restore_infotext(text)
    assert restored == "portrait <'cat' * 2>\nNegative prompt: {'dog':1}\nSteps: 20, Seed: 17"


def test_unquoted_single_provenance_handles_comma_in_expression():
    from sdfx.embedding_merge_prompts import restore_infotext
    text = "<'EM_1'>\nSteps: 20, EmbeddingMerge: <'EM_1'>=<'cat', 'dog'>, Seed: 9"
    assert restore_infotext(text) == "<'cat', 'dog'>\nSteps: 20, Seed: 9"


def test_plain_text_without_provenance_is_unchanged():
    from sdfx.embedding_merge_prompts import restore_infotext
    assert restore_infotext("<'cat'>") == "<'cat'>"
