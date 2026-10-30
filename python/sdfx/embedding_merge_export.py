"""No-overwrite embedding export with the pinned source's architecture conversions."""
import json
import os
from pathlib import Path
import re
import struct
import tempfile

import numpy as np

from .embedding_merge import MAX_ROWS


def export_plan(root, name, encoders, safetensors=True):
    if (not isinstance(name, str) or not name or len(name) > 200 or name[-1] in ' .'
            or any(not (c.isalnum() or c in '._- ') for c in name)
            or name in ('.', '..') or re.match(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)', name, re.I)):
        raise ValueError('invalid simple embedding filename')
    if set(encoders) not in ({'clip_l'}, {'clip_l', 'clip_g'}):
        raise ValueError('embedding requires CLIP-L and optional CLIP-G')
    values = {key: np.asarray(value, dtype='<f4') for key, value in encoders.items()}
    if any(value.ndim != 2 or not 0 < len(value) <= MAX_ROWS or value.shape[1] not in (768, 1024, 1280)
           or not np.isfinite(value).all() for value in values.values()):
        raise ValueError('invalid finite embedding shape')
    if len({len(value) for value in values.values()}) != 1:
        raise ValueError('encoder rows must match')
    if 'clip_g' in values and (values['clip_l'].shape[1] != 768 or values['clip_g'].shape[1] != 1280):
        raise ValueError('SDXL requires 768/1280 encoder widths')
    root = Path(root).resolve()
    primary = values if 'clip_g' in values else {'emb_params': values['clip_l']}
    result = [(root / (name + ('.safetensors' if safetensors else '.pt')), primary)]
    left = values['clip_l']
    if safetensors and left.shape[1] == 768 and left.any():
        if 'clip_g' in values:
            result.append((root / 'sd1' / (name + '.safetensors'), {'emb_params': left}))
        else:
            result.append((root / 'sdxl' / (name + '.safetensors'),
                           {'clip_l': left, 'clip_g': np.zeros((len(left), 1280), dtype='<f4')}))
    for path, _ in result:
        if not path.parent.resolve().is_relative_to(root):
            raise ValueError('embedding output folder escapes managed directory')
    return result


def _write_safetensors(stream, values):
    offset, header, arrays = 0, {}, []
    for key, value in sorted(values.items()):
        array = np.ascontiguousarray(value, dtype='<f4')
        header[key] = {'dtype': 'F32', 'shape': list(array.shape), 'data_offsets': [offset, offset + array.nbytes]}
        arrays.append(array)
        offset += array.nbytes
    raw = json.dumps(header, separators=(',', ':')).encode('utf-8')
    raw += b' ' * (-len(raw) % 8)
    stream.write(struct.pack('<Q', len(raw)))
    stream.write(raw)
    for array in arrays:
        stream.write(array.tobytes())


def export_embeddings(root, name, encoders, safetensors=True, *, check=lambda: None,
                      publish=lambda operation: operation()):
    plan = export_plan(root, name, encoders, safetensors)
    check()
    for path, _ in plan:
        if os.path.lexists(path):
            raise FileExistsError(f'embedding already exists: {path}')
    temporaries, created = [], []
    try:
        for path, values in plan:
            check()
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix='.embedding-merge-', suffix='.part', dir=path.parent)
            temporaries.append((Path(temporary), path))
            with os.fdopen(fd, 'wb') as stream:
                if path.suffix == '.safetensors':
                    _write_safetensors(stream, values)
                else:
                    import torch
                    tensors = {key: torch.from_numpy(value.copy()) for key, value in values.items()}
                    if 'emb_params' in tensors:
                        tensors = {'string_to_token': {'*': 265}, 'string_to_param': {'*': tensors['emb_params']},
                                   'name': name, 'step': 0, 'sd_checkpoint': None, 'sd_checkpoint_name': None}
                    torch.save(tensors, stream)
                stream.flush()
                os.fsync(stream.fileno())
        check()

        def commit():
            for temporary, target in temporaries:
                os.link(temporary, target)
                created.append((target, temporary.stat()))
        publish(commit)
        return [path for path, _ in plan]
    except BaseException:
        # Remove only our own newly linked outputs if a later link raced/fails.
        for target, identity in reversed(created):
            try:
                current = target.stat()
                if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                    target.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for temporary, _ in temporaries:
            temporary.unlink(missing_ok=True)
