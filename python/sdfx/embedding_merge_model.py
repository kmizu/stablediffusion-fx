"""Read-only raw CLIP/TI vector access; caller materializes offloaded weights first."""
import numpy as np
import re

from .embedding_merge import MAX_EXPRESSION, MAX_ROWS
from .embedding_merge_inspection import Part, Piece


class VectorSource:
    def __init__(self, streams, owners, *, file_names=(), load_file=None):
        self.streams = dict(streams)
        self.owners = owners
        self.stream_names = tuple(streams)
        self.load_file = load_file
        self.file_cache = {}
        self.file_sequences = {}
        active = {name.casefold() for name in owners}
        for stream, (tokenizer, _) in self.streams.items():
            sequences = []
            for name in file_names:
                if name.casefold() in active:
                    continue
                ids = tuple(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(name)))
                if ids:
                    sequences.append((ids, name))
            self.file_sequences[stream] = sorted(sequences, key=lambda item: -len(item[0]))

    def _file(self, name, stream):
        if name not in self.file_cache:
            values = self.load_file(name)
            if values is None or set(values) != set(self.streams):
                raise ValueError('embedding encoder set differs from loaded model')
            normalized = {}
            for key, value in values.items():
                if hasattr(value, 'detach'):
                    value = value.detach().float().cpu().numpy()
                value = np.array(value, dtype=np.float32, copy=True)
                if (value.ndim != 2 or value.shape[1] != self.streams[key][1].shape[1]
                        or not 0 < len(value) <= MAX_ROWS or not np.isfinite(value).all()):
                    raise ValueError('embedding dimension or value mismatch')
                normalized[key] = value
            if len({len(value) for value in normalized.values()}) != 1:
                raise ValueError('embedding stream row counts differ')
            self.file_cache[name] = normalized
        return self.file_cache[name][stream].copy()

    @staticmethod
    def _take(weights, ids):
        if len(ids) > MAX_ROWS:
            raise ValueError('vector row limit exceeded')
        if any(not isinstance(i, int) or i < 0 or i >= weights.shape[0] for i in ids):
            raise ValueError('unknown token ID')
        # Works with NumPy in pure tests and materialized torch weights at runtime.
        array = weights[list(ids)]
        if hasattr(array, 'detach'):
            array = array.detach().float().cpu().numpy()
        return np.array(array, dtype=np.float32, copy=True)

    @staticmethod
    def _pieces(tokenizer, tokens, ids):
        result, raw, pending_ids = [], bytearray(), []
        decoder = getattr(tokenizer, 'byte_decoder', None)
        if decoder is None:
            raise ValueError('CLIP tokenizer byte decoder unavailable')
        for token, token_id in zip(tokens, ids):
            # End-of-word is a CLIP marker, not part of its byte alphabet.
            terminal = token.endswith('</w>')
            value = token[:-4] if terminal else token
            try:
                raw.extend(decoder[char] for char in value)
            except KeyError:
                if raw:
                    raise ValueError('incomplete UTF8 token before special token')
                result.append(Piece(token, (token_id,)))
                continue
            if terminal:
                raw.extend(b' ')
            pending_ids.append(token_id)
            try:
                decoded = raw.decode('utf-8', errors='strict')
            except UnicodeDecodeError as error:
                if error.reason == 'unexpected end of data':
                    continue
                raise ValueError('invalid UTF8 token sequence') from error
            result.append(Piece(decoded, tuple(pending_ids)))
            raw.clear()
            pending_ids.clear()
        if raw:
            raise ValueError('incomplete UTF8 token sequence')
        return tuple(result)

    def parts(self, text, split_commas=False):
        if not isinstance(text, str) or len(text) > MAX_EXPRESSION:
            raise ValueError('text input limit exceeded')
        grouped = text.split(',') if split_commas else [text]
        result = {}
        for name, (tokenizer, weights) in self.streams.items():
            parts = []
            vocab = tokenizer.get_vocab()
            for group_index, group in enumerate(grouped):
                # CLIP normalizes ordinary words itself. Preserve added-token
                # case so an existing native TI owner remains identifiable.
                raw_group = group.strip()
                raw_cursor = 0
                tokens = tokenizer.tokenize(raw_group)
                ids = tokenizer.convert_tokens_to_ids(tokens)
                if len(tokens) != len(ids):
                    raise ValueError('tokenizer row mismatch')
                plain_tokens, plain_ids = [], []

                def locate(display_name, index):
                    for match in re.finditer(re.escape(display_name), raw_group, re.IGNORECASE):
                        if raw_cursor is not None and match.start() < raw_cursor:
                            continue
                        prefix = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_group[:match.start()]))
                        if tuple(prefix) == tuple(ids[:index]):
                            return match.start(), match.end()
                    return None

                def flush(end=None, final=False):
                    if plain_ids:
                        pieces = self._pieces(tokenizer, plain_tokens, plain_ids)
                        original = (raw_group[raw_cursor:end] if raw_cursor is not None and (end is not None or final)
                                    else ''.join(p.text for p in pieces))
                        parts.append(Part(self._take(weights, plain_ids), original,
                                          tuple(plain_ids), pieces, group_index))
                        plain_tokens.clear()
                        plain_ids.clear()

                position = 0
                while position < len(ids):
                    token, token_id = tokens[position], ids[position]
                    matches = [(sequence, file_name) for sequence, file_name in self.file_sequences[name]
                               if tuple(ids[position:position + len(sequence)]) == sequence]
                    if matches:
                        length = len(matches[0][0])
                        if sum(len(sequence) == length for sequence, _ in matches) > 1:
                            raise ValueError('ambiguous embedding token sequence')
                        sequence, file_name = matches[0]
                        location = locate(file_name, position)
                        flush(location[0] if location else None)
                        raw_cursor = location[1] if location else None
                        if raw_cursor is not None:
                            while raw_cursor < len(raw_group) and raw_group[raw_cursor].isspace():
                                raw_cursor += 1
                        parts.append(Part(self._file(file_name, name), file_name, None, (), group_index))
                        position += len(sequence)
                        continue
                    position += 1
                    ownership = self.owners.get(token)
                    if ownership is not None and set(ownership) != set(self.streams):
                        raise ValueError(f'incomplete embedding ownership: {token}')
                    owner = (ownership or {}).get(name)
                    if owner is None:
                        plain_tokens.append(token)
                        plain_ids.append(token_id)
                    else:
                        location = locate(token, position - 1)
                        flush(location[0] if location else None)
                        raw_cursor = location[1] if location else None
                        if raw_cursor is not None:
                            while raw_cursor < len(raw_group) and raw_group[raw_cursor].isspace():
                                raw_cursor += 1
                        if not owner or token not in owner or any(item not in vocab for item in owner):
                            raise ValueError(f'inconsistent embedding ownership: {token}')
                        owned_ids = [vocab[item] for item in owner]
                        if len(set(owned_ids)) != len(owned_ids):
                            raise ValueError(f'duplicate embedding ownership: {token}')
                        parts.append(Part(self._take(weights, owned_ids), token, None, (), group_index))
                flush(final=True)
            if sum(len(part.vectors) for part in parts) > MAX_ROWS:
                raise ValueError('vector row limit exceeded')
            result[name] = parts
        return result

    def resolve_text(self, text):
        parts = self.parts(text)
        result = []
        for name, (_, weights) in self.streams.items():
            arrays = [part.vectors for part in parts[name]]
            result.append(np.concatenate(arrays) if arrays else np.zeros((0, weights.shape[1]), dtype=np.float32))
        return tuple(result)

    def resolve_token(self, value):
        result = []
        for tokenizer, weights in self.streams.values():
            special = {'start': tokenizer.bos_token_id, 'end': tokenizer.eos_token_id, 'pad': tokenizer.pad_token_id}
            token_id = special.get(value, value)
            if token_id not in tokenizer.get_vocab().values():
                raise ValueError('unknown token ID')
            result.append(self._take(weights, [token_id]))
        return tuple(result)
