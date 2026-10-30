"""Resolve source-format SD1/SDXL conversion siblings without hiding ambiguity."""
from pathlib import Path
import re

from .models import read_safetensors_header


def _shapes(path):
    if Path(path).suffix.lower() == '.safetensors':
        header = read_safetensors_header(path)
        return {key: value.get('shape') for key, value in header.items() if key != '__metadata__'}
    from .engine import safe_torch_load
    values = safe_torch_load(str(path), 'embedding')
    if isinstance(values, dict) and 'string_to_param' in values:
        values = {'emb_params': values['string_to_param']['*']}
    elif not isinstance(values, dict):
        values = {'emb_params': values}
    return {key: list(value.shape) for key, value in values.items() if hasattr(value, 'shape')}


def select_path(name, paths, widths, *, probe=_shapes):
    candidates = [Path(path) for path in paths if Path(path).stem.casefold() == name.casefold()]
    compatible = []
    for path in candidates:
        shapes = probe(path)
        if set(widths) == {'clip_l', 'clip_g'}:
            selected = {key: shapes.get(key) for key in widths}
        elif len(shapes) == 1 and 'clip_g' not in shapes:
            selected = {'clip_l': next(iter(shapes.values()))}
        else:
            continue
        if all(isinstance(shape, (tuple, list)) and len(shape) in (1, 2)
               and shape[-1] == widths[key] for key, shape in selected.items()):
            compatible.append(path)
    if len(compatible) > 1:
        raise ValueError(f'ambiguous compatible embedding name: {name}')
    if not compatible:
        raise ValueError(f'no model-compatible embedding file: {name}')
    return str(compatible[0])


def file_provider(paths, active_streams):
    if not isinstance(paths, list) or len(paths) > 100000 or any(not isinstance(path, str) for path in paths):
        raise ValueError('invalid embedding paths')
    from .embedding_inspector import _file_embeddings
    widths = {key: encoder.get_input_embeddings().weight.shape[1]
              for key, (_, encoder) in active_streams.items()}
    names = sorted({Path(path).stem.lower() for path in paths})
    def load(name):
        chosen = select_path(name, paths, widths)
        resolve, _, _ = _file_embeddings([chosen], active_streams)
        return resolve(name)
    return load, names


def requested_paths(paths, texts, widths):
    """Only disambiguate referenced duplicate names; preserve ordinary loader policy."""
    joined = ' '.join(texts)
    groups = {}
    for path in paths:
        groups.setdefault(Path(path).stem, []).append(path)
    result = []
    for name, candidates in groups.items():
        if len(candidates) > 1 and re.search(r'(?<![\w-])' + re.escape(name) + r'(?![\w-])', joined):
            result.append(select_path(name, candidates, widths))
        else:
            result.extend(candidates)
    return result
