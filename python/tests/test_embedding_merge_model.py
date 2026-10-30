import numpy as np
import pytest

from sdfx.embedding_merge_model import VectorSource


class Tokenizer:
    byte_decoder = {chr(i): i for i in range(256)}
    bos_token_id, eos_token_id, pad_token_id = 0, 1, 1
    vocab = {'<s>': 0, '</s>': 1, 'cat</w>': 2, 'style': 3, 'style_1': 4, 'foreign_1': 5,
             '\xe9\x9b': 6, '\xaa</w>': 7}

    def tokenize(self, text):
        result = []
        for word in text.strip().split():
            result.extend(['\xe9\x9b', '\xaa</w>'] if word == '雪' else [word if word in self.vocab else word + '</w>'])
        return result

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab[t] for t in tokens]

    def get_vocab(self):
        return self.vocab


def source():
    weights = np.arange(16, dtype=np.float32).reshape(8, 2)
    return VectorSource({'clip_l': (Tokenizer(), weights)}, {'style': {'clip_l': ('style', 'style_1')}})


def test_raw_token_rows_and_exact_owned_ti_expansion():
    result = source().resolve_text('cat style foreign_1')[0]
    np.testing.assert_array_equal(result, [[4, 5], [6, 7], [8, 9], [10, 11]])
    parts = source().parts('cat style foreign_1')['clip_l']
    assert [p.text for p in parts] == ['cat ', 'style', 'foreign_1']
    assert parts[1].tokens is None
    assert parts[2].tokens == (5,)


def test_utf8_token_pieces_join_without_replacement_characters():
    part = source().parts('雪 cat')['clip_l'][0]
    assert [p.text for p in part.pieces] == ['雪 ', 'cat ']
    assert [p.tokens for p in part.pieces] == [(6, 7), (2,)]


def test_special_ids_and_empty_text():
    np.testing.assert_array_equal(source().resolve_token('start')[0], [[0, 1]])
    assert source().resolve_text('')[0].shape == (0, 2)
    with pytest.raises(ValueError, match='token'):
        source().resolve_token(99)


def test_inconsistent_owner_cannot_silently_fall_back_to_suffix():
    weights = np.zeros((8, 2))
    bad = VectorSource({'clip_l': (Tokenizer(), weights)}, {'style': {'clip_l': ('style', 'missing')}})
    with pytest.raises(ValueError, match='ownership'):
        bad.resolve_text('style')


def test_comma_grouping_drops_separator_and_keeps_groups_separate():
    parts = source().parts('cat,style', split_commas=True)['clip_l']
    assert [p.comma_group for p in parts] == [0, 1]
    assert [p.text for p in parts] == ['cat', 'style']


def test_case_sensitive_loaded_owner_is_preserved_before_tokenization():
    class MixedTokenizer(Tokenizer):
        vocab = {**Tokenizer.vocab, 'MixedTI': 3, 'MixedTI_1': 4}
    vector_source = VectorSource({'clip_l': (MixedTokenizer(), np.arange(16).reshape(8, 2))},
                                {'MixedTI': {'clip_l': ('MixedTI', 'MixedTI_1')}})
    assert vector_source.resolve_text('MixedTI')[0].shape == (2, 2)


def test_unloaded_file_embedding_matches_token_sequence_without_tokenizer_mutation():
    original = dict(Tokenizer.vocab)
    loaded = []
    def load(name):
        loaded.append(name)
        return {'clip_l': np.full((3, 2), 9, dtype=np.float32)}
    vector_source = VectorSource({'clip_l': (Tokenizer(), np.arange(16).reshape(8, 2))}, {},
                                file_names=['cat cat'], load_file=load)
    parts = vector_source.parts('cat cat cat')['clip_l']
    assert parts[0].text == 'cat cat' and parts[0].tokens is None
    np.testing.assert_array_equal(parts[0].vectors, np.full((3, 2), 9))
    assert parts[1].tokens == (2,)
    assert loaded == ['cat cat']
    assert Tokenizer.vocab == original


def test_loaded_ownership_takes_precedence_over_stale_file_vectors():
    vector_source = VectorSource({'clip_l': (Tokenizer(), np.arange(16).reshape(8, 2))},
        {'style': {'clip_l': ('style', 'style_1')}}, file_names=['style'],
        load_file=lambda name: pytest.fail('active TI must win'))
    np.testing.assert_array_equal(vector_source.resolve_text('style')[0], [[6, 7], [8, 9]])


def test_unloaded_file_stream_alignment_is_validated():
    vector_source = VectorSource({'clip_l': (Tokenizer(), np.zeros((8, 2)))}, {},
        file_names=['cat'], load_file=lambda name: {'clip_l': np.zeros((2, 3))})
    with pytest.raises(ValueError, match='dimension'):
        vector_source.resolve_text('cat')


def test_incomplete_paired_owner_is_rejected_instead_of_plain_token_fallback():
    weights = np.zeros((8, 2))
    paired = VectorSource({'clip_l': (Tokenizer(), weights), 'clip_g': (Tokenizer(), weights)},
                          {'style': {'clip_l': ('style',)}})
    with pytest.raises(ValueError, match='ownership'):
        paired.resolve_text('style')


def test_parts_keep_source_internal_spaces_while_token_grouping_uses_decoded_pieces():
    from sdfx.embedding_merge_inspection import rows
    parts = source().parts('cat   cat style   cat')['clip_l']
    assert [part.text for part in parts] == ['cat   cat ', 'style', 'cat']
    assert [row['text'] for row in rows(parts, 'tokens')] == ['cat ', 'cat ', 'style', 'cat ']
    assert rows(parts, 'none')[0]['text'] == 'cat cat style cat '
