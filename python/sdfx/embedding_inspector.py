"""Tensor operations for embedding inspector; no dependency on web UI internals."""

import math
import random

import torch


def mix_vectors(inputs, concat=False, combine=False, *, remove_zeros=True):
    """Mix weighted rank-2 tensors, padding rows as needed, or concatenate them.

    Operates on one encoder at a time. The caller must validate paired SDXL
    encoders before publishing an embedding.
    """
    result = None
    width = None
    for tensor, weight in inputs:
        if not math.isfinite(float(weight)):
            raise ValueError("weight must be finite")
        if float(weight) == 0:
            continue
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or not tensor.shape[0] or not tensor.shape[1]:
            raise ValueError("embedding must be a nonempty rank-2 tensor")
        if width is None:
            width = tensor.shape[1]
        elif tensor.shape[1] != width:
            raise ValueError("embedding dimension mismatch")
        vec = tensor.detach().to(device="cpu", dtype=torch.float32) * float(weight)
        if result is None:
            result = vec
        elif concat:
            result = torch.cat((result, vec), dim=0)
        else:
            count = max(len(result), len(vec))
            out = torch.zeros((count, width), dtype=torch.float32)
            out[:len(result)] += result
            out[:len(vec)] += vec
            result = out
    if result is None:
        raise ValueError("no embeddings to mix")
    if combine and len(result) > 1:
        result = result.sum(dim=0, keepdim=True)
    if remove_zeros:
        result = result[torch.count_nonzero(result, dim=1) > 0]
    if not len(result):
        raise ValueError("all mixed vectors are zero")
    return result


def mix_encoders(inputs, concat=False, combine=False, *, remove_zeros=True):
    """Mix complete SD1 or SDXL encoder sets without dropping one side."""
    active = [(values, weight) for values, weight in inputs if float(weight) != 0]
    if not active:
        raise ValueError("no embeddings to mix")
    keys = set(active[0][0])
    if keys not in ({"clip_l"}, {"clip_l", "clip_g"}):
        raise ValueError("unsupported encoder set")
    if any(set(values) != keys for values, _ in active):
        raise ValueError("embedding encoder set mismatch")
    if len(keys) == 2 and any(values["clip_l"].shape[0] != values["clip_g"].shape[0]
                              for values, _ in active):
        raise ValueError("embedding encoder vector counts differ")
    result = {key: mix_vectors([(values[key], weight) for values, weight in active], concat, combine, remove_zeros=False)
              for key in sorted(keys)}
    if not remove_zeros:
        return result
    masks = {key: torch.count_nonzero(tensor, dim=1) > 0 for key, tensor in result.items()}
    # A position belongs to both SDXL encoders. Retain it if either side is
    # nonzero; filtering separately would pair different token positions.
    keep = masks["clip_l"] | masks["clip_g"] if len(keys) == 2 else masks["clip_l"]
    for key in result:
        result[key] = result[key][keep]
        if not len(result[key]):
            raise ValueError("all mixed vectors are zero")
    return result


def evaluate_vectors(tensor, expression, *, allow_custom_code=False):
    """Run upstream-style Python expressions; caller must enforce user consent.

    This is arbitrary Python, not a sandbox. Never invoke it on an untrusted
    expression without the same authorization as the custom-code facility.
    """
    if not expression.strip():
        return tensor.clone()
    if not allow_custom_code:
        raise PermissionError("embedding eval requires Allow custom code")
    vec = tensor.detach().to(device="cpu", dtype=torch.float32).clone()
    if vec.ndim != 2 or not vec.numel():
        raise ValueError("embedding must be a nonempty rank-2 tensor")
    expr = expression.strip()
    itemwise = expr.startswith("=")
    code = compile(expr[1:] if itemwise else expr, "<embedding-eval>", "eval")
    maxn, maxi = vec.shape
    tot_vec = tensor.detach().to(device="cpu", dtype=torch.float32).clone()
    for n in range(maxn):
        vec_mag, vec_min, vec_max = torch.linalg.norm(vec[n]), vec[n].min(), vec[n].max()
        scope = {"torch": torch, "math": math, "random": random, "n": n, "maxn": maxn, "maxi": maxi,
                 "tot_vec": tot_vec, "vec": vec, "vec_mag": vec_mag, "vec_min": vec_min, "vec_max": vec_max}
        if itemwise:
            for i in range(maxi):
                scope.update(i=i, v=vec[n, i])
                vec[n, i] = eval(code, scope)
        else:
            scope["v"] = vec[n]
            vec[n] = eval(code, scope)
    return vec


def save_embedding(path, encoders, *, step=None, overwrite=False):
    """Publish safetensors atomically, without replacing a file unless requested."""
    import os
    import tempfile
    from pathlib import Path
    from safetensors.torch import save_file

    path = Path(path)
    if path.suffix.lower() not in (".safetensors", ".pt"):
        raise ValueError("embedding output must be safetensors or .pt")
    if set(encoders) not in ({"clip_l"}, {"clip_l", "clip_g"}):
        raise ValueError("unsupported encoder set")
    tensors = {k: v.detach().to(device="cpu", dtype=torch.float32).contiguous()
               for k, v in encoders.items()}
    if any(v.ndim != 2 or not v.numel() or not torch.isfinite(v).all() for v in tensors.values()):
        raise ValueError("invalid embedding tensor")
    if len(tensors) == 2 and tensors["clip_l"].shape[0] != tensors["clip_g"].shape[0]:
        raise ValueError("embedding encoder vector counts differ")
    metadata = {} if step is None else {"step": str(int(step))}
    descriptor, name = tempfile.mkstemp(prefix=".embedding-", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        if path.suffix.lower() == ".pt":
            if set(tensors) == {"clip_l"}:
                state = {"string_to_param": {"*": tensors["clip_l"]}, "name": path.stem,
                         "step": None if step is None else int(step)}
            else:
                state = dict(tensors)
                if step is not None:
                    state["step"] = int(step)
            torch.save(state, name)
        else:
            save_file(tensors, name, metadata=metadata)
        if overwrite:
            os.replace(name, path)
        else:
            os.link(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return path


def nearest_tokens(query, matrix, *, limit=20):
    """Return vocabulary row IDs and cosine similarity for a single vector."""
    if query.ndim != 1 or matrix.ndim != 2 or query.shape[0] != matrix.shape[1]:
        raise ValueError("embedding dimension mismatch")
    if not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise ValueError("invalid result limit")
    query = query.detach().to(device="cpu", dtype=torch.float32)
    matrix = matrix.detach().to(device="cpu", dtype=torch.float32)
    norm = torch.linalg.vector_norm(query)
    if not torch.isfinite(query).all() or not torch.isfinite(matrix).all():
        raise ValueError("embedding contains non-finite values")
    if norm == 0:
        raise ValueError("zero query vector")
    scores = torch.nn.functional.cosine_similarity(matrix, query.unsqueeze(0), dim=1, eps=1e-6)
    order = torch.argsort(scores, descending=True, stable=True)[:limit].tolist()
    return [(int(i), float(scores[i])) for i in order]


def _streams(pipe):
    streams = {"clip_l": (getattr(pipe, "tokenizer", None), getattr(pipe, "text_encoder", None))}
    if getattr(pipe, "tokenizer_2", None) is not None or getattr(pipe, "text_encoder_2", None) is not None:
        streams["clip_g"] = (getattr(pipe, "tokenizer_2", None), getattr(pipe, "text_encoder_2", None))
    if any(tokenizer is None or encoder is None for tokenizer, encoder in streams.values()):
        raise ValueError("loaded model does not have the required tokenizer and encoder")
    return streams


def _resolve(tokenizer, weights, name, owners=None):
    name = str(name).strip()
    if not name:
        raise ValueError("token is empty")
    added = tokenizer.get_added_vocab()
    if name.startswith("#"):
        ids = [int(name[1:])]
    elif name in added:
        owned = (owners or {}).get(name)
        if owned is None:
            if f"{name}_1" in added:
                raise ValueError(f"loaded embedding ownership is unknown: {name}")
            owned = [name]
        if name not in owned or any(token not in added for token in owned):
            raise ValueError(f"loaded embedding ownership is inconsistent: {name}")
        ids = sorted(added[token] for token in owned)
    else:
        ids = tokenizer.encode(name.lower(), add_special_tokens=False)[:1]
    if not ids or any(i < 0 or i >= len(weights) for i in ids):
        raise ValueError("token ID is outside the loaded encoder vocabulary")
    return ids, weights[ids].detach().to(device="cpu", dtype=torch.float32).clone()


class EmbeddingDimensionMismatch(ValueError):
    """A file is readable but incompatible with the loaded encoder width."""


def _file_embeddings(paths, streams):
    from pathlib import Path
    from safetensors.torch import load_file
    if not isinstance(paths, list) or len(paths) > 100000:
        raise ValueError("invalid embedding paths")
    found = {}
    for raw in paths:
        path = Path(raw)
        if path.suffix.lower() not in (".safetensors", ".pt", ".bin"):
            continue
        name = path.stem.lower()
        found.setdefault(name, []).append(path)
    def load(name):
        candidates = found.get(name.strip().lower())
        if candidates is None:
            return None
        if len(candidates) != 1:
            raise ValueError(f"duplicate embedding name: {name}")
        path = candidates[0]
        if path.suffix.lower() == ".safetensors":
            data = load_file(str(path))
        else:
            from .engine import safe_torch_load
            data = safe_torch_load(str(path), "embedding")
        if isinstance(data, dict) and "clip_l" in data and "clip_g" in data:
            values = {"clip_l": data["clip_l"], "clip_g": data["clip_g"]}
        elif isinstance(data, dict) and "string_to_param" in data:
            values = {"clip_l": data["string_to_param"]["*"]}
        elif isinstance(data, dict) and len(data) == 1:
            values = {"clip_l": next(iter(data.values()))}
        elif isinstance(data, torch.Tensor):
            values = {"clip_l": data}
        else:
            raise ValueError(f"unsupported embedding format: {path.name}")
        if set(values) != set(streams):
            raise ValueError(f"embedding encoder set differs from loaded model: {path.name}")
        normalized = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.ndim == 1 else v
                      for k, v in values.items()}
        for key, value in normalized.items():
            expected = streams[key][1].get_input_embeddings().weight.shape[1]
            if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != expected:
                raise EmbeddingDimensionMismatch(f"embedding dimension mismatch: {path.name} ({key})")
        return normalized
    def metadata(name):
        candidates = found.get(name.strip().lower())
        if not candidates or len(candidates) != 1:
            return {}
        path = candidates[0]
        result = {"path": str(path)}
        if path.suffix.lower() == ".safetensors":
            from safetensors import safe_open
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                result.update(handle.metadata() or {})
            result["path"] = str(path)
        else:
            from .engine import safe_torch_load
            data = safe_torch_load(str(path), "embedding")
            if isinstance(data, dict):
                for key in ("step", "sd_checkpoint", "sd_checkpoint_name", "name"):
                    if isinstance(data.get(key), (str, int, float)):
                        result[key] = str(data[key])
        return result
    return load, sorted(found), metadata


PRESET_EXPRESSIONS = [("Boost", "=v*8"), ("Digitize", "=math.ceil(v*8)/8"),
                      ("Binary", "=(1*(v>=0)-1*(v<0))/50"), ("Randomize", "=v*random.random()"),
                      ("Sine", "=v*math.sin(i/maxi*math.pi)"), ("Comb", "=v*((i%2)==0)"),
                      ("Crop_high", "=v*(i<maxi//2)"), ("Crop_low", "=v*(i>=maxi//2)")]


def planned_output_paths(params):
    from pathlib import Path
    raw = str(params.get("output", ""))
    if not raw:
        raise ValueError("choose an output path")
    base = Path(raw)
    if base.suffix.lower() not in (".safetensors", ".pt"):
        raise ValueError("embedding output must be safetensors or .pt")
    names = [""] + ([name for name, _ in PRESET_EXPRESSIONS] if params.get("batch_presets") else [])
    return [base if not suffix else base.with_name(base.stem + "_" + suffix + base.suffix) for suffix in names]


def run_inspector(pipe, params):
    """Operate on the loaded model; eval is authorized only by worker environment."""
    from .extras import code_allowed
    streams = _streams(pipe)
    load_file_embedding, file_names, file_metadata = _file_embeddings(params.get("embedding_paths", []), streams)
    operation = params.get("operation")
    def owners_for(key):
        return {name: values[key] for name, values in params.get("_loaded_owners", {}).items() if key in values}
    def active_name(name):
        folded = str(name).strip().casefold()
        matches = [token for token in params.get("_loaded_owners", {}) if token.casefold() == folded]
        if len(matches) > 1:
            raise ValueError(f"ambiguous loaded embedding name: {name}")
        if matches and set(params["_loaded_owners"][matches[0]]) != set(streams):
            raise ValueError(f"loaded embedding ownership is incomplete: {matches[0]}")
        return matches[0] if matches else None
    if operation == "tokenize":
        text = str(params.get("text", ""))
        if len(text) > 65536:
            raise ValueError("text is too long")
        token_ids = {key: tok.encode(text.lower(), add_special_tokens=False) for key, (tok, _) in streams.items()}
        warnings = [f"{key}: only the first 128 tokens are displayed and sent to the mixer"
                    for key, ids in token_ids.items() if len(ids) > 128]
        return {"encoders": {key: [{"id": int(i), "token": tok.convert_ids_to_tokens(int(i))}
                                   for i in token_ids[key][:128]] for key, (tok, _) in streams.items()},
                "warnings": warnings}
    if operation == "list":
        loaded = []
        for name, info in sorted(params.get("_list_loaded", {}).items(), key=lambda item: item[0].casefold()):
            entry = {"name": name, "path": info["path"], "encoders": {}}
            for key, (_, encoder) in streams.items():
                owned = info.get("owners", {}).get(key)
                entry["encoders"][key] = None if owned is None else [len(owned), encoder.get_input_embeddings().weight.shape[1]]
            if info.get("metadata_valid"):
                try:
                    _, _, metadata = _file_embeddings([info["path"]], streams)
                    entry["file_metadata"] = metadata(name)
                except Exception:
                    entry["file_metadata_status"] = "unavailable"
            else:
                entry["file_metadata_status"] = "file changed or could not be verified"
            loaded.append(entry)
        loaded_names = {entry["name"].casefold() for entry in loaded}
        return {"loaded": loaded, "available": [name for name in file_names if name.casefold() not in loaded_names]}
    if operation == "inspect":
        name = str(params.get("text", "")).strip()
        output = {}
        active = active_name(name)
        file_values = None if active is not None else load_file_embedding(name)
        for key, (tok, encoder) in streams.items():
            weights = encoder.get_input_embeddings().weight
            if file_values is None:
                ids, values = _resolve(tok, weights, active or name, owners_for(key))
                token_name = tok.convert_ids_to_tokens(ids[0])
            else:
                ids, values, token_name = [], file_values[key], name
                if values.ndim != 2 or values.shape[1] != weights.shape[1]:
                    raise ValueError("embedding dimension mismatch")
            per_vector = []
            for vector in values:
                limit = int(params.get("limit", 30))
                if torch.count_nonzero(vector) == 0:
                    if not 1 <= limit <= 1000:
                        raise ValueError("invalid result limit")
                    closest = [(i, 0.0) for i in range(min(limit, len(weights)))]
                else:
                    closest = nearest_tokens(vector, weights, limit=limit)
                per_vector.append({"magnitude": float(torch.linalg.vector_norm(vector)),
                                   "min": float(vector.min()), "max": float(vector.max()),
                                   "nearest": [{"id": i, "token": tok.convert_ids_to_tokens(i), "score": score}
                                               for i, score in closest]})
            output[key] = {"ids": ids, "token": token_name, "shape": list(values.shape),
                           "vectors": values.tolist(), "magnitude": torch.linalg.vector_norm(values, dim=1).tolist(),
                           "min": float(values.min()), "max": float(values.max()),
                           "nearest": per_vector[0]["nearest"], "per_vector": per_vector}
        if active is not None:
            info = params.get("_list_loaded", {}).get(active, {})
            metadata = {"source": "loaded model"}
            if info.get("metadata_valid"):
                try:
                    _, _, loaded_metadata = _file_embeddings([info["path"]], streams)
                    metadata.update(loaded_metadata(active))
                except Exception:
                    metadata["file_metadata_status"] = "unavailable"
            else:
                metadata["file_metadata_status"] = "file changed or could not be verified"
        else:
            metadata = file_metadata(name)
        return {"encoders": output, "metadata": metadata}
    if operation == "export":
        import json
        import os
        import tempfile
        from pathlib import Path
        name = str(params.get("text", "")).strip()
        target = Path(str(params.get("output", "")))
        if target.suffix.lower() != ".txt":
            raise ValueError("vector export requires a .txt path")
        active = active_name(name)
        file_values = None if active is not None else load_file_embedding(name)
        values = file_values if file_values is not None else {
            key: _resolve(tok, encoder.get_input_embeddings().weight, active or name, owners_for(key))[1]
            for key, (tok, encoder) in streams.items()}
        payload = json.dumps({key: value.tolist() for key, value in values.items()}, indent=2).encode("utf-8")
        fd, temp = tempfile.mkstemp(prefix=".embedding-vector-", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
            if params.get("overwrite"):
                os.replace(temp, target)
            else:
                os.link(temp, target)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        return {"path": str(target)}
    if operation != "save":
        raise ValueError("unknown embedding inspector operation")
    rows = params.get("rows")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 128:
        raise ValueError("choose 1 to 128 mixer rows")
    expr = str(params.get("eval", ""))
    if (expr or params.get("batch_presets")) and not code_allowed():
        raise PermissionError("embedding eval requires Allow custom code and engine restart")
    prepared = []
    row_warnings = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid mixer row")
        weight = float(row.get("weight", 1))
        if not math.isfinite(weight):
            raise ValueError("weight must be finite")
        selected_ids = row.get("ids")
        if weight == 0 or (selected_ids is None and not str(row.get("name", "")).strip()):
            continue
        if selected_ids is not None:
            if not isinstance(selected_ids, dict) or set(selected_ids) != set(streams):
                raise ValueError("token IDs must be supplied for every encoder")
            values = {key: _resolve(tok, enc.get_input_embeddings().weight, f"#{selected_ids[key]}", owners_for(key))[1]
                      for key, (tok, enc) in streams.items()}
        else:
            if len(streams) == 2 and str(row.get("name", "")).strip().startswith("#"):
                raise ValueError("supply token IDs for each encoder on SDXL")
            active = active_name(row.get("name", ""))
            try:
                values = None if active is not None else load_file_embedding(str(row.get("name", "")))
            except EmbeddingDimensionMismatch as exc:
                row_warnings.append(str(exc) + "; skipped")
                continue
            if values is None:
                values = {key: _resolve(tok, enc.get_input_embeddings().weight, active or row.get("name", ""), owners_for(key))[1]
                          for key, (tok, enc) in streams.items()}
        prepared.append((values, weight))
    if not prepared and row_warnings:
        raise ValueError("No compatible embeddings: " + "; ".join(row_warnings))
    original = mix_encoders(prepared, concat=bool(params.get("concat")), remove_zeros=False)
    output = str(params.get("output", ""))
    if not output:
        raise ValueError("choose an output path")
    targets = planned_output_paths(params)
    import os
    from pathlib import Path
    def canonical(path):
        return os.path.normcase(str(Path(path).resolve()))
    for target in targets:
        for candidate in params.get("embedding_paths", []):
            source = Path(candidate)
            if source.stem.casefold() == target.stem.casefold() and canonical(source) != canonical(target):
                raise ValueError(f"duplicate embedding name in catalog: {target.stem}")
    presets = [("", expr)] + (PRESET_EXPRESSIONS if params.get("batch_presets") else [])
    jobs = []
    warnings = list(row_warnings)
    for target, (suffix, expression) in zip(targets, presets):
        if target.exists() and not params.get("overwrite"):
            raise FileExistsError(f"embedding already exists: {target.name}")
        mixed = {key: value.clone() for key, value in original.items()}
        if expression:
            try:
                mixed = {key: evaluate_vectors(value, expression, allow_custom_code=True) for key, value in mixed.items()}
            except Exception as exc:
                warnings.append(f"{target.name}: eval failed ({type(exc).__name__}: {exc}); original vectors retained")
        if params.get("combine"):
            mixed = {key: value.sum(dim=0, keepdim=True) for key, value in mixed.items()}
        masks = {key: torch.count_nonzero(value, dim=1) > 0 for key, value in mixed.items()}
        keep = masks["clip_l"] | masks["clip_g"] if len(mixed) == 2 else masks["clip_l"]
        mixed = {key: value[keep] for key, value in mixed.items()}
        if any(not value.shape[0] for value in mixed.values()):
            if len(presets) > 1:
                warnings.append(f"{target.name}: all mixed vectors are zero; skipped")
                continue
            raise ValueError("all mixed vectors are zero")
        jobs.append((target, mixed))
    if not jobs:
        raise ValueError("all mixed vectors are zero")
    for target, mixed in jobs:
        save_embedding(target, mixed, step=params.get("step"), overwrite=bool(params.get("overwrite")))
    return {"path": output, "paths": [str(target) for target, _ in jobs], "warnings": warnings,
            "encoders": {k: {"shape": list(v.shape), "vectors": v.tolist()} for k, v in jobs[-1][1].items()}}
