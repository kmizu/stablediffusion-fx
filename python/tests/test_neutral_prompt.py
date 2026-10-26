import pytest

from sdfx.neutral_prompt import parse, leaves, compose, perpendicular, salient_blend, topk, rescale


def test_parser_keywords_weights_and_nested_groups():
    root = parse("castle :2 AND_PERP [monster :-1 AND_SALT tree :3] :0.5 AND_TOPK lights :2")
    assert [(n.kind, n.weight, n.text) for n in root.children] == [
        ("AND", 2, "castle "), ("AND_PERP", 0.5, None), ("AND_TOPK", 2, " lights ")]
    assert [(n.kind, n.weight, n.text.strip()) for n in root.children[1].children] == [
        ("AND", -1, "monster"), ("AND_SALT", 3, "tree")]
    assert [n.text.strip() for n in leaves(root)] == ["castle", "monster", "tree", "lights"]


@pytest.mark.parametrize("text", ["a [b c : d e :1.5]", "[(long shot:1.2):0.1] detail..", "word :not_float", "(word:1.2)"])
def test_parser_preserves_ordinary_attention_and_editing(text):
    assert leaves(parse(text))[0].text == text
    assert len(leaves(parse(text))) == 1


def test_empty_literal_boundaries_and_escaped_keywords():
    assert leaves(parse(""))[0].text == ""
    assert leaves(parse("CANDY ANDROID"))[0].text == "CANDY ANDROID"
    assert leaves(parse(r"word \AND literal"))[0].text == r"word \AND literal"


@pytest.mark.parametrize("text", ["word :nan", "word :inf", "word :-inf"])
def test_nonfinite_weights_fail(text):
    with pytest.raises(ValueError, match="finite"):
        parse(text)


def test_nesting_limit_is_explicit():
    with pytest.raises(ValueError, match="nesting"):
        parse("x AND_PERP [" * 80 + "y" + "]" * 80)


def test_projection_batch_isolation_and_zero_normal():
    torch = pytest.importorskip("torch")
    normal = torch.tensor([[1., 0], [0, 0]])
    vector = torch.tensor([[2., 3], [4, 5]])
    torch.testing.assert_close(perpendicular(normal, vector), torch.tensor([[0., 3], [4, 5]]))


def test_projection_is_invariant_to_extreme_nonzero_normal_scale():
    torch = pytest.importorskip("torch")
    normal = torch.tensor([[1., 0], [1, 0]]) * torch.tensor([[1e-20], [1e20]])
    vector = torch.tensor([[2., 3], [2, 3]])
    torch.testing.assert_close(perpendicular(normal, vector), torch.tensor([[0., 3], [0, 3]]))


def test_saliency_compares_normalized_maps_and_weights_after_selection():
    torch = pytest.importorskip("torch")
    normal = torch.tensor([[4., 1, 1, 1]])
    other = torch.tensor([[0., 2, 0, 0]])
    torch.testing.assert_close(salient_blend(normal, [(other, 2)]), torch.tensor([[0., 2, -2, -2]]))
    assert not salient_blend(normal, []).any()


def test_topk_matches_upstream_threshold_including_ties_and_small_tensors():
    torch = pytest.importorskip("torch")
    vector = torch.arange(-50., 50).reshape(1, 2, 5, 10)
    out = topk(vector)
    assert torch.equal(out, vector * (vector.abs() >= 47))  # kthvalue(95), includes equal values
    assert torch.equal(topk(torch.tensor([[2.]])), torch.tensor([[2.]]))
    assert torch.equal(topk(torch.ones(2, 10)), torch.ones(2, 10))


def test_tree_composition_scales_normal_and_auxiliary_groups_once():
    torch = pytest.importorskip("torch")
    unconditional = torch.tensor([[10., 20]])
    predictions = [unconditional + torch.tensor([[1., 0]]), unconditional + torch.tensor([[2., 3]])]
    out = compose(parse("one :2 AND_PERP two :-1"), predictions, unconditional, 3)
    torch.testing.assert_close(out, torch.tensor([[16., 11.]]))
    nested = compose(parse("[one :2 AND_PERP two :-1] :0.5"), predictions, unconditional, 3)
    torch.testing.assert_close(nested, torch.tensor([[13., 15.5]]))
    torch.testing.assert_close(compose(parse("one :2 AND two :-1"), predictions, unconditional, 3), torch.tensor([[10., 11.]]))


def test_rescale_matches_mean_and_std_per_image_without_nan():
    torch = pytest.importorskip("torch")
    guided = torch.tensor([[2., 4, 6], [100, 110, 120]])
    cond = torch.tensor([[3., 4, 5], [-2, 0, 2]])
    assert rescale(guided, cond, 0) is guided
    torch.testing.assert_close(rescale(guided, cond, 1), cond)
    half = rescale(guided, cond, 0.5)
    torch.testing.assert_close(half.mean(1), (guided.mean(1) + cond.mean(1)) / 2)
    torch.testing.assert_close(half.std(1), (guided.std(1) + cond.std(1)) / 2)
    assert torch.isfinite(rescale(torch.zeros(2, 1), torch.zeros(2, 1), 1)).all()


def test_composition_rescale_reference_includes_auxiliary_contribution():
    torch = pytest.importorskip("torch")
    u = torch.tensor([[1., 2, 4]])
    delta = torch.tensor([[0., 1, 2]])
    root = parse("one AND_TOPK two :2")
    predictions = [u + delta, u + torch.tensor([[2., -3, 1]])]
    full = compose(root, predictions, u, 1)
    guided = compose(root, predictions, u, 4)
    torch.testing.assert_close(compose(root, predictions, u, 4, phi=0.7), rescale(guided, full, 0.7))


def test_rescale_does_not_flatten_extremely_small_nonzero_variance():
    torch = pytest.importorskip("torch")
    guided = torch.tensor([[1e-40, 2e-40, 3e-40]])
    cond = torch.tensor([[1., 2, 3]])
    torch.testing.assert_close(rescale(guided, cond, 1), cond, atol=2e-5, rtol=2e-5)
