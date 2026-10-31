"""Prepare a metadata-preserving PyTorch embedding copy using a safe loader."""

import copy
import hashlib
import math
from pathlib import Path

from .resource_hashes import FileSnapshot


def _digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _default_load(path):
    from .engine import safe_torch_load
    return safe_torch_load(str(path), 'embedding')


def _default_dump(data, stream):
    import torch
    torch.save(data, stream)


def _default_tensor(original, rows, baseline=None):
    import torch
    if not isinstance(original, torch.Tensor) or not original.is_floating_point():
        raise ValueError('embedding tensor is invalid')
    expected = original.shape if original.ndim == 2 else (1, original.shape[0]) if original.ndim == 1 else ()
    if not expected or not isinstance(rows, (list, tuple)) or len(rows) != expected[0]:
        raise ValueError('embedding shape mismatch')
    if any(not isinstance(row, (list, tuple)) or len(row) != expected[1] for row in rows):
        raise ValueError('embedding shape mismatch')
    if baseline is None:
        result = torch.as_tensor(rows, dtype=original.dtype, device='cpu').reshape(original.shape)
    else:
        if not isinstance(baseline, (list, tuple)) or len(baseline) != expected[0] or any(
                not isinstance(row, (list, tuple)) or len(row) != expected[1] for row in baseline):
            raise ValueError('baseline embedding shape mismatch')
        result = original.detach().to(device='cpu').clone().reshape(expected)
        for i, (row, old_row) in enumerate(zip(rows, baseline)):
            for j, (value, old_value) in enumerate(zip(row, old_row)):
                if not math.isfinite(float(value)) or not math.isfinite(float(old_value)):
                    raise ValueError('embedding values must be finite')
                if float(value) != float(old_value):
                    result[i, j] = float(value)
        result = result.reshape(original.shape)
    if not torch.isfinite(result).all():
        raise ValueError('embedding values must be finite')
    return result


def build_pt_copy(source, destination, vectors, expected_sha256, *, baseline=None, load=None, dump=None, tensor=None):
    """Write a new copy; preserve every non-edited field of the safely loaded data."""
    source, destination = Path(source), Path(destination)
    load, dump, tensor = load or _default_load, dump or _default_dump, tensor or _default_tensor
    if not isinstance(vectors, dict) or set(vectors) not in ({'clip_l'}, {'clip_l', 'clip_g'}):
        raise ValueError('invalid encoder set')
    if baseline is not None and (not isinstance(baseline, dict) or set(baseline) != set(vectors)):
        raise ValueError('baseline encoder set differs')
    def replace(old, key):
        return tensor(old, vectors[key]) if baseline is None else tensor(old, vectors[key], baseline[key])
    snapshot = FileSnapshot.capture(source)
    if _digest(source) != expected_sha256:
        raise ValueError('source file changed')
    data = load(source)
    if FileSnapshot.identity_of(source) != snapshot.identity or _digest(source) != expected_sha256:
        raise ValueError('source file changed while loading')
    result = copy.deepcopy(data)
    if not isinstance(result, dict):
        if set(vectors) != {'clip_l'}:
            raise ValueError('unsupported PyTorch embedding format')
        result = replace(result, 'clip_l')
    elif 'string_to_param' in result and set(vectors) == {'clip_l'}:
        params = result['string_to_param']
        if not isinstance(params, dict) or '*' not in params:
            raise ValueError('unsupported PyTorch embedding format')
        params['*'] = replace(params['*'], 'clip_l')
    elif 'emb_params' in result and set(vectors) == {'clip_l'}:
        result['emb_params'] = replace(result['emb_params'], 'clip_l')
    elif all(key in result for key in vectors):
        for key in vectors:
            result[key] = replace(result[key], key)
    elif set(vectors) == {'clip_l'} and len(result) == 1:
        key = next(iter(result))
        result[key] = replace(result[key], 'clip_l')
    else:
        raise ValueError('unsupported PyTorch embedding encoder set')
    created = False
    try:
        with destination.open('xb') as stream:
            created = True
            dump(result, stream)
        if FileSnapshot.identity_of(source) != snapshot.identity or _digest(source) != expected_sha256:
            raise ValueError('source file changed during copy')
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise
    return destination
