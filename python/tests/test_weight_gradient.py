import pytest

from sdfx.weight_gradient import curve, render, schedule, has_gradient


def test_readme_linear_example_and_floored_fractional_steps():
    p = "{dog : 0.1 - 0.3 : 1 - 0}"
    assert [render(p, s, 22) for s in range(1, 8)] == [
        "(dog:1)", "(dog:1)", "(dog:0.75)", "(dog:0.5)", "(dog:0.25)", "(dog:0)", "(dog:0)"]
    assert render("{dog:10%-30%:1-0}", 4, 22) == "(dog:0.5)"


def test_readme_exponential_return_example():
    p = "{cat : 1 - 15 : 1 - 0 - 1 : e}"
    assert [render(p, s, 22) for s in (1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 22)] == [
        "(cat:1)", "(cat:0.9965)", "(cat:0.9744)", "(cat:0.8143)", "(cat:0.1857)", "(cat:0.0256)",
        "(cat:0.0035)", "(cat:0)", "(cat:0.0035)", "(cat:0.9965)", "(cat:1)", "(cat:1)"]


def test_readme_circular_example():
    p = "{redhead girl, drinking coffee :: 0 - 1 : c}"
    assert [render(p, s, 22) for s in (1, 2, 3, 10, 11, 12, 21, 22)] == [
        "(redhead girl, drinking coffee:0)", "(redhead girl, drinking coffee:0.0023)",
        "(redhead girl, drinking coffee:0.0092)", "(redhead girl, drinking coffee:0.2425)",
        "(redhead girl, drinking coffee:0.3475)", "(redhead girl, drinking coffee:0.6525)",
        "(redhead girl, drinking coffee:0.9977)", "(redhead girl, drinking coffee:1)"]


@pytest.mark.parametrize("mode", ["a", "e", "ei", "eo", "c", "ci", "co"])
def test_curves_endpoints_monotonicity_and_symmetry(mode):
    values = [curve(mode, i / 100) for i in range(101)]
    assert values[0] == 0 and values[-1] == 1
    assert values == sorted(values)
    if mode in ("a", "e", "c"):
        assert values == pytest.approx([1 - x for x in reversed(values)])
    assert curve("ei", 0.5) == pytest.approx(0.03125)
    assert curve("eo", 0.5) == pytest.approx(0.96875)
    assert curve("ci", 0.5) == pytest.approx(1 - 0.75**0.5)


def test_signed_weights_multiple_tags_and_default_interval():
    p = "photo of {cat:: -1 - 1}, {dog:2-4:2-0-1:a}"
    assert render(p, 1, 5) == "photo of (cat:-1), (dog:2)"
    assert render(p, 3, 5) == "photo of (cat:0), (dog:0)"
    assert render(p, 5, 5) == "photo of (cat:1), (dog:1)"
    assert render("{cat::0-1}", 1, 1) == "(cat:1)"


@pytest.mark.parametrize("p", [r"\{cat::0-1}", "{cat::bad}", "{cat::0-1:unknown}", "{cat:8-2:0-1}",
                               "{cat::nan-1}", "{unclosed", "{red|blue}"])
def test_escaped_and_malformed_tags_remain_literal(p):
    assert render(p, 3, 10) == p
    assert not has_gradient(p)


def test_nested_emphasis_and_editing_composes_with_existing_schedule():
    p = "{[cat:dog:0.5], (red:1.2)::1-2} [day|night]"
    assert has_gradient(p)
    assert schedule(p, 4) == [[1, "(cat, (red:1.2):1) day"], [2, "(cat, (red:1.2):1.3333) night"],
                              [3, "(dog, (red:1.2):1.6667) day"], [4, "(dog, (red:1.2):2) night"]]


def test_hires_restarts_gradient_clock_but_preserves_a1111_offsets():
    assert schedule("{cat::0-1} [day:night:1.5]", 10, 4) == [
        [1, "(cat:0) day"], [2, "(cat:0.3333) day"], [3, "(cat:0.6667) night"], [4, "(cat:1) night"]]
    assert schedule("{cat::1-1}", 10) == [[10, "(cat:1)"]]


def test_img2img_clock_uses_actual_steps_without_retiming_a1111_editing():
    assert schedule("{cat::0-1} [day:night:0.5]", 10, gradient_steps=4) == [
        [1, "(cat:0) day"], [2, "(cat:0.3333) day"], [3, "(cat:0.6667) day"], [4, "(cat:1) day"]]


def test_feature_is_advertised_by_torch_free_engine():
    from sdfx.hooks import KNOWN_HOOKS
    from sdfx.dryrun import FakeEngine
    assert "weight_gradient" in KNOWN_HOOKS
    assert "weight_gradient" in FakeEngine().features()
