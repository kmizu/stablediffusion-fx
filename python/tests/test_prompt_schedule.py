"""The doctests of A1111 modules/prompt_parser.get_learned_conditioning_prompt_schedules, against our port."""
import pytest

from sdfx.prompt_schedule import has_schedule, schedule, text_at

CASES = [
    ("test", [[10, "test"]]),
    ("a [b:3]", [[3, "a "], [10, "a b"]]),
    ("a [b: 3]", [[3, "a "], [10, "a b"]]),
    ("a [[[b]]:2]", [[2, "a "], [10, "a [[b]]"]]),
    ("[(a:2):3]", [[3, ""], [10, "(a:2)"]]),
    ("a [b : c : 1] d", [[1, "a b  d"], [10, "a  c  d"]]),
    ("a[b:[c:d:2]:1]e", [[1, "abe"], [2, "ace"], [10, "ade"]]),
    ("a [unbalanced", [[10, "a [unbalanced"]]),
    ("a [b:.5] c", [[5, "a  c"], [10, "a b c"]]),
    # A1111 marks this doctest "not handling this right now"; real lark fails to parse it -> unchanged
    ("a [{b|d{:.5] c", [[10, "a [{b|d{:.5] c"]]),
    ("((a][:b:c [d:3]", [[3, "((a][:b:c "], [10, "((a][:b:c d"]]),
    ("[a|(b:1.1)]", [[1, "a"], [2, "(b:1.1)"], [3, "a"], [4, "(b:1.1)"], [5, "a"], [6, "(b:1.1)"], [7, "a"], [8, "(b:1.1)"],
                     [9, "a"], [10, "(b:1.1)"]]),
    ("[fe|]male", [[1, "female"], [2, "male"], [3, "female"], [4, "male"], [5, "female"], [6, "male"], [7, "female"],
                   [8, "male"], [9, "female"], [10, "male"]]),
    ("[fe|||]male", [[1, "female"], [2, "male"], [3, "male"], [4, "male"], [5, "female"], [6, "male"], [7, "male"],
                     [8, "male"], [9, "female"], [10, "male"]]),
]


@pytest.mark.parametrize("prompt,expected", CASES)
def test_a1111_doctests(prompt, expected):
    assert schedule(prompt, 10) == expected


def test_hires_offsets():
    assert schedule("a [b:.5] c", 10, 10) == [[10, "a b c"]]
    assert schedule("a [b:1.5] c", 10, 10) == [[5, "a  c"], [10, "a b c"]]


def test_helpers():
    s = schedule("a [b:c:4] d", 10)
    assert text_at(s, 1) == "a b d" and text_at(s, 4) == "a b d" and text_at(s, 5) == "a c d"
    assert has_schedule("a [b:c:0.5]") and has_schedule("[x|y]")
    assert not has_schedule("a (b:1.2), [c]") and not has_schedule("plain")
