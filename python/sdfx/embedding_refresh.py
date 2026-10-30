"""Transactional live refresh of saved textual-inversion encoder rows."""

import copy
import re
from pathlib import Path

import torch

from .embedding_inspector import _file_embeddings, _streams
from .resource_hashes import FileSnapshot


def validate_saved_names(engine, paths):
    """Reject built-in-token collisions before touching any output file."""
    streams = _streams(engine.pipe)
    names = [Path(p).stem for p in paths]
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("duplicate saved embedding names")
    for name in names:
        if any(other != name and re.fullmatch(re.escape(name) + r"_[0-9]+", other) for other in names):
            raise ValueError(f"embedding output names may overlap: {name}")
    for key, (tokenizer, _) in streams.items():
        added, vocab = tokenizer.get_added_vocab(), tokenizer.get_vocab()
        for name in names:
            if name in getattr(tokenizer, "all_special_tokens", ()) or (name in vocab and name not in added):
                raise ValueError(f"saved embedding conflicts with a built-in token: {name}")
            owned = getattr(engine, "embedding_token_owners", {}).get(name, {}).get(key)
            if name in added and (owned is None or name not in owned):
                raise ValueError(f"loaded embedding ownership is unknown: {name}")
            if owned is not None and any(token not in added for token in owned):
                raise ValueError(f"loaded embedding ownership is inconsistent: {name}")
            for token in vocab:
                if re.fullmatch(re.escape(name) + r"_[0-9]+", token) and token not in (owned or ()):
                    raise ValueError(f"embedding token collision: {token}")
            if any(existing.casefold() == name.casefold() and existing != name for existing in added):
                raise ValueError(f"saved embedding name differs only by case from a loaded token: {name}")


def refresh_saved_embeddings(engine, paths):
    """Replace saved tokens across encoders; restore both streams on failure.

    Called only on the engine's serial worker with offload hooks suspended.
    Never call concurrently with inference. On failure the saved files still exist; caller must report that
    they are not active. A failed restoration invalidates the whole pipeline.
    """
    paths = [Path(p) for p in paths]
    streams = _streams(engine.pipe)
    load, _, _ = _file_embeddings([str(p) for p in paths], streams)
    names = [p.stem for p in paths]
    if len({n.lower() for n in names}) != len(names):
        raise ValueError("duplicate saved embedding names")
    captured = {name: FileSnapshot.capture(str(path)) for name, path in zip(names, paths)}
    values = {name: load(name) for name in names}
    if any(snapshot.identity_of(snapshot.path) != snapshot.identity for snapshot in captured.values()):
        raise ValueError("saved embedding file changed while loading")
    snapshots = []
    for key, (tokenizer, encoder) in streams.items():
        added = tokenizer.get_added_vocab()
        vocab = tokenizer.get_vocab()
        for name in names:
            if name in getattr(tokenizer, "all_special_tokens", ()) or (name in vocab and name not in added):
                raise ValueError(f"saved embedding conflicts with a built-in token: {name}")
        module = encoder.get_input_embeddings()
        snapshots.append((tokenizer, copy.deepcopy(tokenizer.__dict__), encoder, module,
                          module.weight.detach().clone(), getattr(module, "num_embeddings", None),
                          getattr(module, "padding_idx", None), getattr(encoder.config, "vocab_size", None),
                          getattr(encoder, "vocab_size", None)))
    new_owners = {}
    try:
        for name in names:
            for key, (tokenizer, encoder) in streams.items():
                added = tokenizer.get_added_vocab()
                owned = getattr(engine, "embedding_token_owners", {}).get(name, {}).get(key)
                if name in added and (owned is None or name not in owned):
                    raise ValueError(f"loaded embedding ownership is unknown: {name}")
                for token in tokenizer.get_vocab():
                    if re.fullmatch(re.escape(name) + r"_[0-9]+", token) and token not in (owned or ()):
                        raise ValueError(f"embedding token collision: {token}")
                old_names = list(owned or ())
                if any(token not in added for token in old_names):
                    raise ValueError(f"loaded embedding ownership is inconsistent: {name}")
                count = values[name][key].shape[0]
                new_names = [name] + [f"{name}_{i}" for i in range(1, count)]
                collisions = set(new_names) & (set(tokenizer.get_vocab()) - set(old_names))
                if collisions:
                    raise ValueError(f"embedding token collision: {sorted(collisions)[0]}")
                if old_names:
                    old_names.sort(key=lambda token: added[token])
                    engine.pipe.unload_textual_inversion(tokens=old_names, tokenizer=tokenizer, text_encoder=encoder)
                before = set(tokenizer.get_added_vocab())
                engine.pipe.load_textual_inversion(values[name][key], token=name,
                                                   tokenizer=tokenizer, text_encoder=encoder)
                new_owners.setdefault(name, {})[key] = sorted(set(tokenizer.get_added_vocab()) - before)
        if any(snapshot.identity_of(snapshot.path) != snapshot.identity for snapshot in captured.values()):
            raise ValueError("saved embedding file changed during live refresh")
    except BaseException as failure:
        try:
            for tokenizer, state, encoder, module, weight, size, padding, vocab, encoder_vocab in snapshots:
                tokenizer.__dict__.clear(); tokenizer.__dict__.update(state)
                module.weight.data = weight
                if size is not None:
                    module.num_embeddings = size
                if hasattr(module, "padding_idx"):
                    module.padding_idx = padding
                encoder.set_input_embeddings(module)
                if vocab is not None:
                    encoder.config.vocab_size = vocab
                if encoder_vocab is not None:
                    encoder.vocab_size = encoder_vocab
        except BaseException as restore_error:
            try:
                engine._free()
            except BaseException as free_error:
                restore_error.add_note(f"model unload failed: {free_error}")
            raise RuntimeError("embedding refresh failed and model restoration failed; checkpoint must be reloaded") from restore_error
        raise failure
    engine.derived.clear()
    if not hasattr(engine, "embedding_snapshots"):
        engine.embedding_snapshots = {}
    if not hasattr(engine, "embedding_token_owners"):
        engine.embedding_token_owners = {}
    for name, path in zip(names, paths):
        engine.embedding_token_owners[name] = new_owners[name]
        engine.loaded_embeddings.add(name)
        engine.loaded_embedding_files[name] = str(path)
        engine.embedding_snapshots[name] = captured[name]
