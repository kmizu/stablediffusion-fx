"""Create a byte-preserving safetensors edit copy without touching the source.

The caller owns publication. Existing target files are never overwritten.
"""

import hashlib
import json
import math
import os
import struct
from pathlib import Path

_MAX_HEADER = 16 * 1024 * 1024
_MAX_FILE = 512 * 1024 * 1024
_TYPES = {'F16': ('e', 2), 'F32': ('f', 4), 'F64': ('d', 8), 'BF16': (None, 2)}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate safetensors header key')
        result[key] = value
    return result


def _encode(value, dtype):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('embedding value must be finite')
    try:
        if dtype == 'BF16':
            bits = struct.unpack('<I', struct.pack('<f', number))[0]
            bits += 0x7fff + ((bits >> 16) & 1)
            rounded = (bits >> 16) & 0xffff
            if rounded & 0x7f80 == 0x7f80:
                raise ValueError('embedding value outside dtype range')
            return struct.pack('<H', rounded)
        return struct.pack('<' + _TYPES[dtype][0], number)
    except (OverflowError, struct.error) as failure:
        raise ValueError('embedding value outside dtype range') from failure


def build_safetensors_copy(source, destination, vectors, expected_sha256, *, baseline=None):
    """Create an exclusive copy with selected tensor bytes replaced in-place.

    `vectors` maps exact tensor names to complete rank-2 numeric vectors. All
    other bytes, including the JSON header, metadata and other tensors, survive.
    """
    source, destination = Path(source), Path(destination)
    if not isinstance(vectors, dict) or not vectors:
        raise ValueError('no tensors to edit')
    if baseline is not None and (not isinstance(baseline, dict) or set(baseline) != set(vectors)):
        raise ValueError('baseline encoder set differs')
    with source.open('rb') as inp:
        before = os.fstat(inp.fileno())
        if before.st_size < 8 or before.st_size > _MAX_FILE:
            raise ValueError('unsupported safetensors file size')
        prefix = inp.read(8)
        length = struct.unpack('<Q', prefix)[0]
        if length < 2 or length > _MAX_HEADER or length > before.st_size - 8:
            raise ValueError('invalid safetensors header size')
        try:
            header = json.loads(inp.read(length), object_pairs_hook=_unique)
        except (ValueError, UnicodeDecodeError) as failure:
            raise ValueError('invalid safetensors header') from failure
        if not isinstance(header, dict):
            raise ValueError('invalid safetensors header')
        data_start = 8 + length
        spans = []
        for key, item in header.items():
            if key == '__metadata__':
                continue
            if not isinstance(item, dict) or not isinstance(item.get('data_offsets'), list) or len(item['data_offsets']) != 2:
                raise ValueError('invalid tensor offsets')
            start, end = item['data_offsets']
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in (start, end)) or start < 0 or end < start or data_start + end > before.st_size:
                raise ValueError('invalid tensor offsets')
            spans.append((start, end))
        spans.sort()
        if any(right[0] < left[1] for left, right in zip(spans, spans[1:])):
            raise ValueError('overlapping tensor data')
        changes = []
        for key, rows in vectors.items():
            item = header.get(key)
            if not isinstance(item, dict) or item.get('dtype') not in _TYPES:
                raise ValueError(f'unsupported tensor or dtype: {key}')
            shape, dtype = item.get('shape'), item['dtype']
            if not isinstance(shape, list) or len(shape) not in (1, 2) or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in shape):
                raise ValueError('invalid embedding shape')
            nrows, width = (1, shape[0]) if len(shape) == 1 else shape
            if not isinstance(rows, (list, tuple)) or len(rows) != nrows or any(not isinstance(row, (list, tuple)) or len(row) != width for row in rows):
                raise ValueError('embedding shape does not match source')
            start, end = item['data_offsets']
            if end - start != nrows * width * _TYPES[dtype][1]:
                raise ValueError('embedding shape and offsets disagree')
            if baseline is None:
                encoded = b''.join(_encode(value, dtype) for row in rows for value in row)
                changes.append((data_start + start, encoded))
            else:
                previous = baseline[key]
                if not isinstance(previous, (list, tuple)) or len(previous) != nrows or any(not isinstance(row, (list, tuple)) or len(row) != width for row in previous):
                    raise ValueError('baseline embedding shape differs')
                for row_index, (row, old_row) in enumerate(zip(rows, previous)):
                    for column, (value, old_value) in enumerate(zip(row, old_row)):
                        if not math.isfinite(float(value)) or not math.isfinite(float(old_value)):
                            raise ValueError('embedding value must be finite')
                        if float(value) != float(old_value):
                            offset = data_start + start + (row_index * width + column) * _TYPES[dtype][1]
                            changes.append((offset, _encode(value, dtype)))
        inp.seek(0)
        created = False
        try:
            with destination.open('xb') as out:
                created = True
                digest = hashlib.sha256()
                for block in iter(lambda: inp.read(1024 * 1024), b''):
                    digest.update(block)
                    out.write(block)
                after = os.fstat(inp.fileno())
                identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                if identity(before) != identity(after) or digest.hexdigest() != expected_sha256:
                    raise ValueError('source file changed')
                for offset, payload in changes:
                    out.seek(offset)
                    out.write(payload)
        except BaseException:
            if created:
                destination.unlink(missing_ok=True)
            raise
    return destination
