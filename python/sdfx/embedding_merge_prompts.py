"""Lossless expression spans and source-compatible temporary-name provenance."""
from dataclasses import dataclass
import json
import re

from .embedding_merge import MAX_EXPRESSION, parse_expression


@dataclass(frozen=True)
class Expression:
    start: int
    end: int
    original: str
    expression: str


def _span(text, start):
    close = '>' if text[start] == '<' else '}'
    quote = False
    position = start + 1
    while position < len(text):
        char = text[position]
        if char == "'":
            if quote and position + 1 < len(text) and text[position + 1] == "'":
                position += 2
                continue
            quote = not quote
        elif char == close and not quote:
            original = text[start:position + 1]
            expression = text[start + 1:position]
            parse_expression(expression)
            return Expression(start, position + 1, original, expression)
        position += 1
    raise ValueError(f'unclosed merge expression at {start}')


def expressions(prompt):
    if not isinstance(prompt, str):
        raise ValueError('prompt input limit exceeded')
    # Ordinary generation keeps its existing prompt policy. The bound applies
    # only when this extension actually parses an embedded expression.
    if "<'" not in prompt and "{'" not in prompt:
        return ()
    if len(prompt) > MAX_EXPRESSION:
        raise ValueError('prompt input limit exceeded')
    result = []
    position = 0
    while position < len(prompt):
        match = re.search(r"[<{]'", prompt[position:])
        if match is None:
            break
        span = _span(prompt, position + match.start())
        result.append(span)
        position = span.end
    return tuple(result)


def substitute(prompt, replacement):
    parts = []
    position = 0
    for span in expressions(prompt):
        parts.extend((prompt[position:span.start], replacement(span)))
        position = span.end
    parts.append(prompt[position:])
    return ''.join(parts)


def restore_provenance(prompt, metadata):
    if not isinstance(metadata, str) or len(metadata) > MAX_EXPRESSION:
        raise ValueError('provenance input limit exceeded')
    if metadata.startswith('"'):
        metadata = json.loads(metadata)
        if not isinstance(metadata, str):
            raise ValueError('provenance must be text')
    mapping = {}
    position = 0
    while position < len(metadata):
        while position < len(metadata) and metadata[position].isspace():
            position += 1
        match = re.match(r"(?:<'EM[_/-][0-9]+'>|\{'EM[_/-][0-9]+'\})=", metadata[position:])
        if match is None:
            raise ValueError('invalid ephemeral provenance key')
        key = match.group()[:-1]
        if key in mapping:
            raise ValueError('duplicate provenance key')
        position += len(match.group())
        if metadata[position:position + 2] not in ("<'", "{'"):
            raise ValueError('expected merge expression in provenance')
        span = _span(metadata, position)
        mapping[key] = span.original
        position = span.end
        while position < len(metadata) and metadata[position].isspace():
            position += 1
        if position < len(metadata):
            if metadata[position] != ',' or position + 1 == len(metadata):
                raise ValueError('invalid provenance separator')
            position += 1
    # Match original spans once: an expression introduced by restoration is never
    # interpreted as another ephemeral name in this same operation.
    return substitute(prompt, lambda span: mapping.get(span.original, span.original))


def restore_infotext(text):
    """Restore A1111's EmbeddingMerge field without consuming adjacent fields."""
    if not isinstance(text, str) or len(text) > MAX_EXPRESSION:
        raise ValueError('infotext input limit exceeded')
    match = re.search(r'(?:^|,\s*|\n)EmbeddingMerge:\s*', text)
    if match is None:
        return text
    start = match.end()
    if text[start:start + 1] == '"':
        metadata, length = json.JSONDecoder().raw_decode(text[start:])
        if not isinstance(metadata, str):
            raise ValueError('provenance must be text')
        end = start + length
    else:
        end = start
        while True:
            key = re.match(r"(?:<'EM[_/-][0-9]+'>|\{'EM[_/-][0-9]+'\})=", text[end:])
            if key is None:
                raise ValueError('invalid ephemeral provenance key')
            end += len(key.group())
            if text[end:end + 2] not in ("<'", "{'"):
                raise ValueError('expected merge expression in provenance')
            end = _span(text, end).end
            following = re.match(r",\s*(?=[<{]'EM[_/-][0-9]+['}>=])", text[end:])
            if following is None:
                break
            end += len(following.group())
        metadata = text[start:end]
    before, after = text[:match.start()], text[end:]
    if not before and after.startswith(','):
        after = after[1:].lstrip()
    return restore_provenance(before + after, metadata)
