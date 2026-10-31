import importlib
import math
import pytest


def module():
    return importlib.import_module('sdfx.embedding_editor_state')


def test_edits_preserve_other_vectors_and_encoder_without_mutating_input():
    source = {'clip_l': [[1., 2.], [3., 4.]], 'clip_g': [[5., 6., 7.], [8., 9., 10.]]}
    draft = module().EditorDraft(source)
    draft.set_coordinate('clip_l', 0, 1, -2.5)
    assert draft.values() == {'clip_l': [[1., -2.5], [3., 4.]], 'clip_g': [[5., 6., 7.], [8., 9., 10.]]}
    assert source['clip_l'][0][1] == 2.
    draft.set_coordinate('clip_g', 1, 2, 42.)
    assert draft.values()['clip_l'][0][1] == -2.5
    assert draft.values()['clip_g'][1] == [8., 9., 42.]


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_coordinate_is_rejected_without_changing_draft(value):
    draft = module().EditorDraft({'clip_l': [[1., 2.]]})
    with pytest.raises(ValueError, match='finite'):
        draft.set_coordinate('clip_l', 0, 1, value)
    assert draft.values() == {'clip_l': [[1., 2.]]}


def test_invalid_paired_vector_count_and_indices_are_rejected():
    with pytest.raises(ValueError, match='vector counts'):
        module().EditorDraft({'clip_l': [[1.]], 'clip_g': [[2.], [3.]]})
    draft = module().EditorDraft({'clip_l': [[1., 2.]]})
    for stream, row, coordinate in [('clip_g', 0, 0), ('clip_l', -1, 0), ('clip_l', 0, 2)]:
        with pytest.raises(ValueError):
            draft.set_coordinate(stream, row, coordinate, 3.)


def test_invalid_coordinate_does_not_echo_private_input_in_error():
    secret = 'private draft contents'
    with pytest.raises(ValueError) as failure:
        module().EditorDraft({'clip_l': [[secret]]})
    assert secret not in str(failure.value)


def test_guidance_colors_and_flat_distribution_are_explicit():
    m = module()
    assert m.parse_guidance(' apple : #ff0000, dog:#00aAFF ') == [('apple', '#FF0000'), ('dog', '#00AAFF')]
    assert m.marker_position(2., 0., 4.) == pytest.approx(.5)
    assert m.marker_position(5., 0., 4.) == 1.
    assert m.marker_position(2., 2., 2.) is None
    with pytest.raises(ValueError, match='color'):
        m.parse_guidance('cat:red')


def test_replace_and_reset_vector_preserve_other_unsaved_drafts():
    draft = module().EditorDraft({'clip_l': [[1., 2.], [3., 4.]], 'clip_g': [[5., 6., 7.], [8., 9., 10.]]})
    draft.replace_vector('clip_l', 0, [20., 30.])
    draft.replace_vector('clip_g', 1, [40., 50., 60.])
    assert draft.is_dirty()
    draft.reset_vector('clip_l', 0)
    assert draft.values()['clip_l'][0] == [1., 2.]
    assert draft.values()['clip_g'][1] == [40., 50., 60.]
    draft.reset_all()
    assert not draft.is_dirty()
    assert draft.values()['clip_g'][1] == [8., 9., 10.]


def test_invalid_vector_replacement_is_atomic():
    draft = module().EditorDraft({'clip_l': [[1., 2.]]})
    for invalid in ([3.], [3., float('nan')], [3., 4., 5.]):
        with pytest.raises(ValueError):
            draft.replace_vector('clip_l', 0, invalid)
        assert draft.values() == {'clip_l': [[1., 2.]]}


def test_guidance_markers_keep_each_encoder_independent():
    markers = module().guidance_markers(
        {'clip_l': [(-2., 2.), (1., 1.)], 'clip_g': [(0., 10.), (-4., 4.), (0., 1.)]},
        {'clip_l': [0., 1.], 'clip_g': [5., -2., 2.]})
    assert markers == {'clip_l': [0.5, None], 'clip_g': [0.5, 0.25, 1.]}


def test_guidance_rejects_missing_encoder_and_dimension_mismatch():
    fn = module().guidance_markers
    with pytest.raises(ValueError, match='encoder'):
        fn({'clip_l': [(0., 1.)], 'clip_g': [(0., 1.)]}, {'clip_l': [0.5]})
    with pytest.raises(ValueError, match='dimension'):
        fn({'clip_l': [(0., 1.), (0., 1.)]}, {'clip_l': [0.5]})


def test_draft_source_stamp_rejects_changed_model_or_file():
    m = module()
    stamp = m.SourceStamp(model_token='model-a', path='C:/emb/a.safetensors', file_identity=('inode-a', 10, 100))
    stamp.require_current(model_token='model-a', path='C:/emb/a.safetensors', file_identity=('inode-a', 10, 100))
    with pytest.raises(ValueError, match='model changed'):
        stamp.require_current(model_token='model-b', path='C:/emb/a.safetensors', file_identity=('inode-a', 10, 100))
    with pytest.raises(ValueError, match='file changed'):
        stamp.require_current(model_token='model-a', path='C:/emb/a.safetensors', file_identity=('inode-b', 10, 100))


def test_slider_bounds_contain_current_and_allow_flat_distribution():
    helper = module().slider_bounds
    assert helper(-1., 2., 0.5) == (-1., 2.)
    assert helper(-1., 2., 4.) == (-1., 4.)
    assert helper(1., 1., 1.) == (1., 1.)
    assert helper(1., 1., -3.) == (-3., 1.)
    with pytest.raises(ValueError):
        helper(2., -1., 0.)


def test_coordinate_paging_is_complete_and_never_wraps():
    helper = module().coordinate_page
    assert helper(7, 0, 3) == [0, 1, 2]
    assert helper(7, 1, 3) == [3, 4, 5]
    assert helper(7, 2, 3) == [6]
    with pytest.raises(ValueError):
        helper(7, 3, 3)
    with pytest.raises(ValueError):
        helper(7, -1, 3)


def test_source_stamp_rejects_missing_model_or_unverified_file_identity():
    m = module()
    with pytest.raises(ValueError, match='model'):
        m.SourceStamp(model_token='', path='C:/emb/a', file_identity=('ok',)).require_current(
            model_token='', path='C:/emb/a', file_identity=('ok',))
    with pytest.raises(ValueError, match='file'):
        m.SourceStamp(model_token='model-a', path='C:/emb/a', file_identity=None).require_current(
            model_token='model-a', path='C:/emb/a', file_identity=None)
