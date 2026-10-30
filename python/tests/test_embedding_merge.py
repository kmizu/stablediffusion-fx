"""Hand-calculated expression fixtures derived from pinned upstream operations."""
import numpy as np
import pytest

from sdfx.embedding_merge import parse_expression, evaluate


def run(text, streams=('clip_l',)):
    values = {'a': [[1, 2], [3, 4]], 'b': [[10, 20]], '': [], "it's": [[7, 8]]}

    def literal(value):
        base = np.asarray(values[value], dtype=np.float32).reshape(-1, 2)
        return tuple(base.copy() * (i + 1) for i in range(len(streams)))

    def token(value):
        number = {'pad': 90, 'end': 91, 'start': 92}.get(value, value)
        return tuple(np.array([[number, number]], dtype=np.float32) for _ in streams)

    return evaluate(parse_expression(text), literal, token, streams)


def test_sum_zero_pads_shorter_term_without_changing_row_order():
    np.testing.assert_array_equal(run("'a'+'b'")[0], [[11, 22], [3, 4]])


def test_result_modifier_differs_from_literal_modifier():
    np.testing.assert_array_equal(run("'a'+'b'*2")[0], [[21, 42], [3, 4]])
    np.testing.assert_array_equal(run("'a'+'b'=*2")[0], [[22, 44], [6, 8]])
    np.testing.assert_array_equal(run("'a'+'b'=*2-'b'")[0], [[12, 24], [6, 8]])


def test_resize_shift_and_result_crop():
    np.testing.assert_array_equal(run("'a':3:+1")[0], [[0, 0], [1, 2], [3, 4]])
    np.testing.assert_array_equal(run("'a'+'a':-1=:1")[0], [[4, 6]])
    assert run("'a':0")[0].shape == (0, 2)


def test_concatenated_literal_ids_and_special_aliases():
    np.testing.assert_array_equal(run("'b','a',0,00,000,0000")[0],
        [[10, 20], [1, 2], [3, 4], [0, 0], [90, 90], [91, 91], [92, 92]])


def test_doubled_apostrophe_and_whitespace():
    np.testing.assert_array_equal(run("  'it''s' / 2  ")[0], [[3.5, 4]])


def test_sdxl_stream_modifiers_are_independent():
    left, right = run("'a'*2L/2G", ('clip_l', 'clip_g'))
    np.testing.assert_array_equal(left, [[2, 4], [6, 8]])
    np.testing.assert_array_equal(right, [[1, 2], [3, 4]])
    with pytest.raises(ValueError, match='SDXL'):
        run("'a'*2G")


@pytest.mark.parametrize('text', ["'a", "'a'+", "'a' 'b'", "'a'=,3", "'a',-1",
    "'a',00000", "'a'*", "'a',", "'a'**2", "'a'*nan", "'a'*1e300"])
def test_malformed_expressions_reject_before_resolution(text):
    with pytest.raises(ValueError):
        parse_expression(text)


def test_nonfinite_zero_division_and_allocation_limits():
    with pytest.raises(ValueError, match='zero'):
        run("'a'/0")
    with pytest.raises(ValueError, match='limit'):
        run("'a':1000000000")


def test_mismatched_stream_rows_rejected_without_truncation():
    with pytest.raises(ValueError, match='rows'):
        evaluate(parse_expression("'a'"), lambda _: (np.zeros((2, 2)), np.zeros((1, 3))),
                 lambda _: (), ('clip_l', 'clip_g'))


def test_source_arrays_are_not_mutated():
    source = np.array([[1, 2]], dtype=np.float32)
    evaluate(parse_expression("'a'*3+'a'"), lambda _: (source,), lambda _: (), ('clip_l',))
    np.testing.assert_array_equal(source, [[1, 2]])
