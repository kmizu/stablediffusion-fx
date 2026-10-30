"""Model-independent grouping and statistics for native Embedding Merge tables."""
from dataclasses import dataclass

import numpy as np

from .embedding_merge import MAX_ROWS


@dataclass(frozen=True)
class Piece:
    text: str
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class Part:
    vectors: np.ndarray
    text: str
    tokens: tuple[int, ...] | None
    pieces: tuple[Piece, ...]
    comma_group: int = 0


def statistics(values):
    values = np.asarray(values, dtype=np.float32)
    if not values.size or values.ndim not in (1, 2) or not np.isfinite(values).all():
        raise ValueError('statistics require nonempty finite vectors')
    # Upstream uses ord=2: Euclidean for rank1, spectral for grouped rank2.
    return {'min': float(values.min()), 'max': float(values.max()), 'sum': float(values.sum()),
            'abs_sum': float(np.abs(values).sum()), 'norm': float(np.linalg.norm(values, ord=2)),
            'std': float(values.std(ddof=1)) if values.size > 1 else None}


def rows(parts, grouping):
    if grouping not in ('none', 'comma', 'parts', 'words', 'tokens', 'vectors'):
        raise ValueError('unknown row grouping')
    if sum(len(part.vectors) for part in parts) > MAX_ROWS:
        raise ValueError('inspection row limit exceeded')
    groups = []
    for part in parts:
        array = np.asarray(part.vectors, dtype=np.float32)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError('inspection requires finite rank2 vectors')
        if not len(array):
            continue
        if part.tokens is not None:
            flattened = tuple(token for piece in part.pieces for token in piece.tokens)
            if flattened != part.tokens or len(part.tokens) != len(array) or any(not p.tokens for p in part.pieces):
                raise ValueError('token piece rows do not match vectors')
        if grouping in ('none', 'comma'):
            key = 0 if grouping == 'none' else part.comma_group
            tokens = list(part.tokens) if part.tokens is not None else [f'*_{len(array)}']
            text = ((''.join(piece.text for piece in part.pieces) if part.tokens is not None else part.text + ' ')
                    if grouping == 'none' else part.text)
            if groups and groups[-1][3] == key:
                old_array, old_text, old_tokens, _ = groups[-1]
                groups[-1] = (np.concatenate((old_array, array)), old_text + text, old_tokens + tokens, key)
            else:
                groups.append((array, text, tokens, key))
        elif grouping == 'parts' or (part.tokens is None and grouping != 'vectors'):
            groups.append((array, part.text, list(part.tokens) if part.tokens is not None else ['*'], None))
        elif grouping == 'vectors':
            offset = 0
            for index, vector in enumerate(array):
                if part.tokens is None:
                    text, token = part.text, f'*_{index + 1}'
                else:
                    offset = 0
                    text = part.text
                    for piece in part.pieces:
                        offset += len(piece.tokens)
                        if index < offset:
                            text = piece.text
                            break
                    token = part.tokens[index]
                groups.append((vector, text, [token], None))
        else:
            pieces = []
            for piece in part.pieces:
                if grouping == 'words' and pieces and not pieces[-1].text.endswith(' '):
                    previous = pieces[-1]
                    pieces[-1] = Piece(previous.text + piece.text, previous.tokens + piece.tokens)
                else:
                    pieces.append(piece)
            offset = 0
            for piece in pieces:
                end = offset + len(piece.tokens)
                groups.append((array[offset:end], piece.text, list(piece.tokens), None))
                offset = end
    result = []
    position = 1
    for array, text, tokens, _ in groups:
        count = 1 if array.ndim == 1 else len(array)
        result.append({'start': position, 'end': position + count - 1, 'count': count,
                       'text': text, 'tokens': tokens, 'statistics': statistics(array)})
        position += count
    return result
