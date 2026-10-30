import numpy as np
import pytest

from sdfx.embedding_merge_inspection import Part, Piece, rows, statistics


def fixture():
    return [Part(np.array([[1, 0], [0, 2], [3, 0]], dtype=np.float32), 'blue sky ', (1, 2, 3),
                 (Piece('blue ', (1,)), Piece('sky ', (2, 3))), 0),
            Part(np.array([[0, 4], [5, 0]], dtype=np.float32), 'style', None, (), 0),
            Part(np.array([[0, 6]], dtype=np.float32), 'end ', (4,), (Piece('end ', (4,)),), 1)]


def test_six_grouping_modes_keep_all_vector_rows_and_embedding_identity():
    expected = {'none': [6], 'comma': [5, 1], 'parts': [3, 2, 1],
                'words': [1, 2, 2, 1], 'tokens': [1, 2, 2, 1], 'vectors': [1] * 6}
    for mode, counts in expected.items():
        result = rows(fixture(), mode)
        assert [r['count'] for r in result] == counts
        assert result[0]['start'] == 1
        assert result[-1]['end'] == 6
    assert rows(fixture(), 'parts')[1]['tokens'] == ['*']
    assert rows(fixture(), 'vectors')[3]['tokens'] == ['*_1']


def test_unicode_multivector_piece_stays_intact_in_token_mode():
    parts = [Part(np.ones((3, 2)), '雪 a ', (5, 6, 7),
                  (Piece('雪', (5, 6)), Piece(' a ', (7,))), 0)]
    assert [r['count'] for r in rows(parts, 'tokens')] == [2, 1]
    assert [r['count'] for r in rows(parts, 'words')] == [3]
    assert rows(parts, 'tokens')[0]['text'] == '雪'


def test_grouped_norm_is_source_matrix_spectral_not_flattened_euclidean():
    result = statistics(np.array([[3, 0], [0, 4]], dtype=np.float32))
    assert result['norm'] == 4
    assert result['sum'] == 7
    assert result['abs_sum'] == 7
    assert result['std'] == pytest.approx(np.std([3, 0, 0, 4], ddof=1))
    assert statistics(np.array([3, 4], dtype=np.float32))['norm'] == 5


def test_invalid_piece_lengths_and_nonfinite_vectors_reject():
    with pytest.raises(ValueError, match='piece'):
        rows([Part(np.ones((2, 2)), 'a', (1, 2), (Piece('a', (1,)),), 0)], 'tokens')
    with pytest.raises(ValueError, match='finite'):
        statistics(np.array([np.nan, 1]))
    assert statistics(np.array([1.]))['std'] is None
