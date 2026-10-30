"""Bounded safetensors metadata editing without deserializing tensor data."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile

MAX_HEADER = 100 * 1024 * 1024
MAX_METADATA = 4 * 1024 * 1024
CHUNK = 1024 * 1024
_BITS = {'BOOL': 8, 'U8': 8, 'I8': 8, 'I16': 16, 'U16': 16, 'F16': 16, 'BF16': 16,
         'I32': 32, 'U32': 32, 'F32': 32, 'F64': 64, 'I64': 64, 'U64': 64,
         'F8_E4M3': 8, 'F8_E5M2': 8, 'F8_E8M0': 8, 'F8_E4M3FNUZ': 8, 'F8_E5M2FNUZ': 8,
         'C64': 64, 'F4': 4, 'F6_E2M3': 6, 'F6_E3M2': 6}


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError('nonfinite JSON value: ' + value)


def _json(text):
    try:
        return json.loads(text, object_pairs_hook=_pairs, parse_constant=_invalid_constant)
    except (RecursionError, UnicodeError) as exc:
        raise ValueError('invalid or excessively nested JSON') from exc


def _identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _unchanged(stream, path, identity):
    if _identity(os.fstat(stream.fileno())) != identity or _identity(path.stat()) != identity:
        raise ValueError('Source changed; reload metadata before saving.')


def _token(identity, digest):
    return json.dumps([identity, digest], separators=(',', ':'))


def _expected_token(token):
    if not isinstance(token, str) or not token or len(token) > 1024:
        raise ValueError('Load metadata first; a source token is required for saving.')
    value = _json(token)
    if (not isinstance(value, list) or len(value) != 2 or not isinstance(value[0], list)
            or len(value[0]) != 5 or any(type(x) is not int for x in value[0])
            or not isinstance(value[1], str) or not re.fullmatch('[0-9a-f]{64}', value[1])):
        raise ValueError('Invalid source token; reload metadata before saving.')
    return tuple(value[0]), value[1]


def _source(path):
    path = Path(path)
    if path.suffix.lower() != '.safetensors' or not path.is_file():
        raise ValueError('Choose an existing .safetensors file')
    return path.resolve()


def _header(stream):
    raw = stream.read(8)
    if len(raw) != 8:
        raise ValueError('truncated safetensors length')
    size, = struct.unpack('<Q', raw)
    if not 0 < size <= MAX_HEADER:
        raise ValueError('invalid or oversized safetensors header')
    raw = stream.read(size)
    if len(raw) != size:
        raise ValueError('truncated safetensors header')
    header = _json(raw)
    if not isinstance(header, dict):
        raise ValueError('safetensors header must be an object')
    metadata = header.get('__metadata__', {})
    if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
        raise ValueError('safetensors metadata must be a string map')
    intervals = []
    for name, desc in header.items():
        if name == '__metadata__':
            continue
        if not isinstance(desc, dict):
            raise ValueError('invalid tensor descriptor')
        shape, offsets, dtype = desc.get('shape'), desc.get('data_offsets'), desc.get('dtype')
        if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int or n < 0 for n in offsets)
                or not isinstance(dtype, str) or dtype not in _BITS):
            raise ValueError('invalid tensor shape, offsets or dtype')
        start, end = offsets
        bits = math.prod(shape) * _BITS[dtype]
        if bits % 8 or end - start != bits // 8:
            raise ValueError('tensor size does not match descriptor')
        intervals.append((start, end))
    end = 0
    for start, stop in sorted(intervals):
        if start != end:
            raise ValueError('tensor data has overlaps or holes')
        end = stop
    if os.fstat(stream.fileno()).st_size != 8 + size + end:
        raise ValueError('truncated or trailing tensor data')
    return header, end, raw


def read_metadata(path, *, check=lambda: None, progress=lambda done, total: None):
    path = _source(path)
    check()
    with path.open('rb') as stream:
        identity = _identity(os.fstat(stream.fileno()))
        header, total, raw = _header(stream)
        digest = hashlib.sha256(struct.pack('<Q', len(raw)))
        digest.update(raw)
        done = 0
        _unchanged(stream, path, identity)
        while done < total:
            check()
            _unchanged(stream, path, identity)
            chunk = stream.read(min(CHUNK, total - done))
            if not chunk:
                raise ValueError('Source changed; reload metadata before saving.')
            digest.update(chunk)
            done += len(chunk)
            progress(done, total)
        check()
        _unchanged(stream, path, identity)
    metadata = header.get('__metadata__', {})
    editable = dict(metadata)
    for key, value in metadata.items():
        if value.startswith('{'):
            try:
                editable[key] = _json(value)
            except ValueError:
                pass
    return {'path': str(path), 'metadata': metadata,
            'editable': json.dumps(editable, ensure_ascii=False, indent=2), 'tensor_bytes': total,
            'source_token': _token(identity, digest.hexdigest())}


def _metadata(text):
    if not isinstance(text, str) or len(text.encode('utf-8')) > MAX_METADATA:
        raise ValueError('metadata exceeds 4 MiB')
    value = _json(text)
    if not isinstance(value, dict):
        raise ValueError('metadata must be a JSON object')
    try:
        return {key: item if isinstance(item, str) else json.dumps(item, ensure_ascii=False,
                separators=(',', ':'), allow_nan=False) for key, item in value.items()}
    except (RecursionError, ValueError) as exc:
        raise ValueError('invalid metadata value') from exc


def _target(path, name):
    if not isinstance(name, str):
        raise ValueError('output name must be text')
    name = name or path.stem + '_md'
    if (len(name) > 200 or name in ('.', '..') or name[-1:] in (' ', '.')
            or any(ord(c) < 32 or c in '<>:"/\\|?*' for c in name)
            or re.match(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)', name, re.I)):
        raise ValueError('output name must be a valid simple filename stem')
    target = path.with_name(name + '.safetensors')
    if os.path.lexists(target):
        raise FileExistsError('Output already exists: ' + str(target))
    return target


def write_metadata(path, metadata_json, new_name='', *, source_token='', check=lambda: None,
                   progress=lambda done, total: None, publish=lambda operation: operation()):
    expected_identity, expected_hash = _expected_token(source_token)
    metadata = _metadata(metadata_json)
    path = _source(path)
    target = _target(path, new_name)
    temp = None
    check()
    try:
        with path.open('rb') as source:
            initial = _identity(os.fstat(source.fileno()))
            if initial != expected_identity:
                raise ValueError('Source changed; reload metadata before saving.')
            header, total, original_raw = _header(source)
            source_digest = hashlib.sha256(struct.pack('<Q', len(original_raw)))
            source_digest.update(original_raw)
            header['__metadata__'] = metadata
            raw = json.dumps(header, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
            raw += b' ' * (-len(raw) % 8)
            if len(raw) > MAX_HEADER:
                raise ValueError('updated header exceeds 100 MiB')
            fd, temp = tempfile.mkstemp(prefix='.metadata-', suffix='.part', dir=target.parent)
            digest = hashlib.sha256()
            with os.fdopen(fd, 'wb') as output:
                output.write(struct.pack('<Q', len(raw)))
                output.write(raw)
                done = 0
                while done < total:
                    check()
                    _unchanged(source, path, initial)
                    chunk = source.read(min(CHUNK, total - done))
                    if not chunk:
                        raise ValueError('source changed during metadata copy')
                    output.write(chunk)
                    digest.update(chunk)
                    source_digest.update(chunk)
                    done += len(chunk)
                    progress(done, total)
                check()
                _unchanged(source, path, initial)
                if source_digest.hexdigest() != expected_hash:
                    raise ValueError('Source changed; reload metadata before saving.')
                output.flush()
                os.fsync(output.fileno())
            check()
            def commit():
                _unchanged(source, path, initial)
                os.link(temp, target)
            publish(commit)
        return {'path': str(target), 'tensor_bytes': total, 'tensor_sha256': digest.hexdigest()}
    finally:
        if temp is not None:
            os.unlink(temp)


def run(params, ctx):
    from .protocol import EngineError, INVALID_PARAMS
    ctx.check_interrupt()
    try:
        operation = params.get('operation', 'read')
        if operation == 'read':
            result = read_metadata(params.get('path', ''), check=ctx.check_interrupt,
                progress=lambda done, total: ctx.progress((done + CHUNK - 1) // CHUNK,
                    (total + CHUNK - 1) // CHUNK, 'metadata_utils_read'))
            ctx.check_interrupt()
            return result
        if operation == 'write':
            return write_metadata(params.get('path', ''), params.get('metadata_json', ''),
                                  params.get('new_name', ''), source_token=params.get('source_token', ''), check=ctx.check_interrupt,
                                  progress=lambda done, total: ctx.progress((done + CHUNK - 1) // CHUNK,
                                      (total + CHUNK - 1) // CHUNK, 'metadata_utils'),
                                  publish=ctx.publish)
        raise ValueError('operation must be read or write')
    except (ValueError, TypeError, OSError) as exc:
        raise EngineError(INVALID_PARAMS, f'Metadata Utils: {exc}') from exc
