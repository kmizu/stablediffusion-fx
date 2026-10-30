"""Bounded, non-evaluating Embedding Merge grammar and float32 vector algebra."""
from dataclasses import dataclass
import re

import numpy as np

MAX_EXPRESSION = 65536
MAX_ROWS = 4096
MAX_ACTIONS = 4096
_NUMBER = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)')


@dataclass(frozen=True)
class Action:
    kind: str
    value: object = None
    whole: bool = False
    stream: str | None = None


def parse_expression(text):
    if not isinstance(text, str) or len(text) > MAX_EXPRESSION:
        raise ValueError('expression input limit exceeded')
    text = text.replace('\0', ' ')
    actions = []
    position = 0

    def spaces():
        nonlocal position
        while position < len(text) and text[position].isspace():
            position += 1

    def quoted():
        nonlocal position
        spaces()
        if position == len(text) or text[position] != "'":
            raise ValueError(f'expected quoted text at {position}')
        position += 1
        result = []
        while position < len(text):
            char = text[position]
            position += 1
            if char == "'":
                if position < len(text) and text[position] == "'":
                    result.append("'")
                    position += 1
                else:
                    return ''.join(result)
            else:
                result.append(char)
        raise ValueError('last quote not closed')

    actions.append(Action('text', quoted()))
    while True:
        spaces()
        if position == len(text):
            return tuple(actions)
        if len(actions) >= MAX_ACTIONS:
            raise ValueError('expression action limit exceeded')
        whole = text[position] == '='
        if whole:
            position += 1
            spaces()
        if position == len(text):
            raise ValueError('unfinished result operator')
        operator = text[position]
        position += 1
        spaces()
        if operator in '+-' and not whole:
            actions.append(Action('add' if operator == '+' else 'sub'))
            actions.append(Action('text', quoted()))
        elif operator in '*/':
            match = _NUMBER.match(text, position)
            if not match:
                raise ValueError('expected numeric multiplier')
            position = match.end()
            value = float(match.group())
            spaces()
            stream = None
            if position < len(text) and text[position] in 'LG':
                stream = 'clip_l' if text[position] == 'L' else 'clip_g'
                position += 1
            actions.append(Action('mul' if operator == '*' else 'div', value, whole, stream))
        elif operator == ':':
            shift = 0
            if position < len(text) and text[position] in '+-':
                shift = 1 if text[position] == '+' else -1
                position += 1
                spaces()
            match = re.match(r'-?[0-9]+' if shift else r'[0-9]+', text[position:])
            if not match:
                raise ValueError('expected integer vector count or rotation')
            position += len(match.group())
            value = int(match.group())
            actions.append(Action('roll' if shift else 'resize', value * shift if shift else value, whole))
        elif operator == ',' and not whole:
            if position < len(text) and text[position] == "'":
                actions.append(Action('append', quoted()))
            else:
                match = re.match(r'[0-9]+', text[position:])
                if not match:
                    raise ValueError('expected quoted text or token ID after comma')
                raw = match.group()
                position += len(raw)
                if raw.startswith('0'):
                    if raw not in ('0', '00', '000', '0000'):
                        raise ValueError('invalid zero-prefixed token ID')
                    value = {'0': 0, '00': 'pad', '000': 'end', '0000': 'start'}[raw]
                else:
                    value = int(raw)
                actions.append(Action('token', value))
        else:
            raise ValueError(f'invalid operator at {position - 1}')


def evaluate(actions, resolve_text, resolve_token, stream_names):
    """Resolvers return one [rows,width] array per CLIP stream; never mutate them."""
    names = tuple(stream_names)
    if not names or len(names) > 2 or len(set(names)) != len(names):
        raise ValueError('invalid CLIP streams')

    def checked(values):
        if len(values) != len(names):
            raise ValueError('mismatched CLIP streams')
        arrays = tuple(np.array(value, dtype=np.float32, copy=True) for value in values)
        if any(a.ndim != 2 or a.shape[1] < 1 for a in arrays):
            raise ValueError('expected two-dimensional vectors')
        if len({len(a) for a in arrays}) != 1:
            raise ValueError('CLIP stream rows must match')
        if any(len(a) > MAX_ROWS for a in arrays):
            raise ValueError('vector row limit exceeded')
        if any(not np.isfinite(a).all() for a in arrays):
            raise ValueError('nonfinite vector value')
        return arrays

    def resized(array, count):
        if count < 0 or count > MAX_ROWS:
            raise ValueError('vector row limit exceeded')
        result = np.zeros((count, array.shape[1]), dtype=np.float32)
        keep = min(count, len(array))
        result[:keep] = array[:keep]
        return result

    left = None
    right = None
    sign = 1

    def combine():
        nonlocal left, right
        if left is not None:
            if right is None or any(a.shape[1] != b.shape[1] for a, b in zip(left, right)):
                raise ValueError('cannot merge different embedding dimensions')
            count = max(len(left[0]), len(right[0]))
            with np.errstate(over='ignore', invalid='ignore'):
                right = checked(tuple(resized(a, count) + sign * resized(b, count)
                                      for a, b in zip(left, right)))
            left = None

    for action in actions:
        if action.whole or action.kind in ('add', 'sub'):
            combine()
        if action.kind == 'text':
            right = checked(resolve_text(action.value))
        elif action.kind in ('add', 'sub'):
            left, right = right, None
            sign = 1 if action.kind == 'add' else -1
        elif action.kind in ('append', 'token'):
            values = checked(resolve_text(action.value) if action.kind == 'append' else resolve_token(action.value))
            if any(a.shape[1] != b.shape[1] for a, b in zip(right, values)):
                raise ValueError('cannot concatenate different embedding dimensions')
            if len(right[0]) + len(values[0]) > MAX_ROWS:
                raise ValueError('vector row limit exceeded')
            right = tuple(np.concatenate((a, b)) for a, b in zip(right, values))
        elif action.kind == 'resize':
            right = tuple(resized(a, action.value) for a in right)
        elif action.kind == 'roll':
            right = tuple(np.roll(a, action.value, axis=0) for a in right)
        elif action.kind in ('mul', 'div'):
            if action.stream and len(names) != 2:
                raise ValueError('L/G suffix requires SDXL streams')
            if action.kind == 'div' and action.value == 0:
                raise ValueError('division by zero')
            if not np.isfinite(action.value):
                raise ValueError('nonfinite scalar')
            with np.errstate(over='ignore', invalid='ignore'):
                right = checked(tuple((a * action.value if action.kind == 'mul' else a / action.value)
                    if action.stream is None or action.stream == name else a
                    for name, a in zip(names, right)))
        else:
            raise ValueError('unknown expression action')
    combine()
    if right is None:
        raise ValueError('empty expression')
    return checked(right)
