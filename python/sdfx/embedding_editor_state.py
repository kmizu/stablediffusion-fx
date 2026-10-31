"""Pure validation for an embedding editor's unsaved coordinate draft."""

from dataclasses import dataclass
import math
import re


def _coordinate(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError('coordinate must be a finite number')
    try:
        number = float(value)
    except (ValueError, OverflowError):
        raise ValueError('coordinate must be a finite number') from None
    if not math.isfinite(number):
        raise ValueError('coordinate must be a finite number')
    return number


class EditorDraft:
    """Own separate mutable copies for each encoder and each vector."""

    def __init__(self, values):
        if not isinstance(values, dict) or set(values) not in ({'clip_l'}, {'clip_l', 'clip_g'}):
            raise ValueError('invalid encoder set')
        copied = {}
        for key, rows in values.items():
            if not isinstance(rows, (list, tuple)) or not rows:
                raise ValueError('encoder must contain vectors')
            width = None
            result = []
            for row in rows:
                if not isinstance(row, (list, tuple)) or not row or (width is not None and len(row) != width):
                    raise ValueError('invalid vector dimensions')
                width = len(row)
                vector = [_coordinate(value) for value in row]
                result.append(vector)
            copied[key] = result
        if len(copied) == 2 and len(copied['clip_l']) != len(copied['clip_g']):
            raise ValueError('encoder vector counts differ')
        self._values = copied
        self._original = self.values()

    def set_coordinate(self, encoder, vector, coordinate, value):
        rows = self._values.get(encoder)
        if rows is None or not isinstance(vector, int) or isinstance(vector, bool) or not 0 <= vector < len(rows):
            raise ValueError('invalid encoder or vector index')
        if not isinstance(coordinate, int) or isinstance(coordinate, bool) or not 0 <= coordinate < len(rows[vector]):
            raise ValueError('invalid coordinate index')
        number = _coordinate(value)
        rows[vector][coordinate] = number

    def replace_vector(self, encoder, vector, values):
        rows = self._values.get(encoder)
        if rows is None or not isinstance(vector, int) or isinstance(vector, bool) or not 0 <= vector < len(rows):
            raise ValueError('invalid encoder or vector index')
        if not isinstance(values, (list, tuple)) or len(values) != len(rows[vector]):
            raise ValueError('invalid vector dimensions')
        numbers = [_coordinate(value) for value in values]
        rows[vector] = numbers

    def reset_vector(self, encoder, vector):
        rows = self._original.get(encoder)
        if rows is None or not isinstance(vector, int) or isinstance(vector, bool) or not 0 <= vector < len(rows):
            raise ValueError('invalid encoder or vector index')
        self._values[encoder][vector] = rows[vector].copy()

    def reset_all(self):
        self._values = {key: [row.copy() for row in rows] for key, rows in self._original.items()}

    def is_dirty(self):
        return self._values != self._original

    def values(self):
        return {key: [row.copy() for row in rows] for key, rows in self._values.items()}


def parse_guidance(text):
    """Parse comma-separated reference text and six-digit color pairs."""
    if not isinstance(text, str):
        raise ValueError('guidance must be text')
    if not text.strip():
        return []
    result = []
    for item in text.split(','):
        if ':' not in item:
            raise ValueError('guidance needs a word and color')
        word, color = (part.strip() for part in item.rsplit(':', 1))
        if not word or not re.fullmatch(r'#[0-9a-fA-F]{6}', color):
            raise ValueError('guidance needs a word and #RRGGBB color')
        result.append((word, color.upper()))
    return result


def marker_position(value, minimum, maximum):
    value, minimum, maximum = float(value), float(minimum), float(maximum)
    if not all(math.isfinite(x) for x in (value, minimum, maximum)) or maximum < minimum:
        raise ValueError('invalid guidance range')
    if maximum == minimum:
        return None
    return max(0., min(1., (value - minimum) / (maximum - minimum)))


def guidance_markers(ranges, reference):
    """Return a normalized marker for each coordinate in each encoder."""
    if not isinstance(ranges, dict) or not isinstance(reference, dict) or set(ranges) != set(reference):
        raise ValueError('guidance encoder sets differ')
    result = {}
    for encoder, limits in ranges.items():
        values = reference[encoder]
        if len(limits) != len(values):
            raise ValueError('guidance vector dimension differs')
        result[encoder] = [marker_position(value, lower, upper)
                           for (lower, upper), value in zip(limits, values)]
    return result


@dataclass(frozen=True)
class SourceStamp:
    """Identity at read time, checked again before publishing a draft."""
    model_token: str
    path: str
    file_identity: tuple

    def require_current(self, *, model_token, path, file_identity):
        if not self.model_token or self.model_token != model_token:
            raise ValueError('model changed; reopen the embedding')
        if not self.path or not self.file_identity or self.path != path or self.file_identity != file_identity:
            raise ValueError('embedding file changed; reopen the embedding')


def slider_bounds(minimum, maximum, current):
    minimum, maximum, current = float(minimum), float(maximum), float(current)
    if not all(math.isfinite(x) for x in (minimum, maximum, current)) or maximum < minimum:
        raise ValueError('invalid slider bounds')
    return min(minimum, current), max(maximum, current)


def coordinate_page(width, page, page_size):
    if any(not isinstance(x, int) or isinstance(x, bool) for x in (width, page, page_size)):
        raise ValueError('invalid coordinate page')
    if width <= 0 or page < 0 or page_size <= 0 or page * page_size >= width:
        raise ValueError('invalid coordinate page')
    start = page * page_size
    return list(range(start, min(width, start + page_size)))
