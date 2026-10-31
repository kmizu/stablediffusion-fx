"""Transient editor weight scope; adapter owns model-specific state."""

from contextlib import contextmanager

from .protocol import EngineError, NO_MODEL


@contextmanager
def temporary_edit(adapter, values):
    """Restore the full snapshot on success, error and interruption.

    The adapter must make snapshot independent of mutable model storage and
    invalidate the whole model if restoration cannot be completed.
    """
    snapshot = adapter.snapshot()
    try:
        adapter.install(values)
        yield
    finally:
        engine = getattr(adapter, 'engine', None)
        if engine is not None and hasattr(engine, 'pipe') and engine.pipe is None:
            import sys
            failure = sys.exc_info()[1]
            if not isinstance(failure, EngineError) or failure.code != NO_MODEL:
                raise EngineError(NO_MODEL, 'Preview model was unloaded; reload the checkpoint') from failure
        else:
            try:
                adapter.restore(snapshot)
            except BaseException as failure:
                try:
                    adapter.invalidate()
                except BaseException as unload_error:
                    failure.add_note(f'model unload also failed: {unload_error}')
                raise EngineError(NO_MODEL, f'Could not restore preview embeddings; reload the checkpoint: {failure}') from failure


class ModelEditorScope:
    """Temporary edits of exactly the rows owned by one loaded embedding.

    The caller must hold the engine encoder-access/offload guard throughout
    snapshot, install, encoding and restore. No tokenizer or ownership changes
    are made here.
    """
    def __init__(self, engine, name):
        self.engine, self.name = engine, name
        self.pipe = None
        self.entries = None
        self.owner_mapping = None
        self.installed_owners = None

    def snapshot(self):
        from .embedding_inspector import _streams
        streams = _streams(self.engine.pipe)
        matches = [token for token in self.engine.embedding_token_owners if token.casefold() == self.name.casefold()]
        if len(matches) != 1:
            raise ValueError('loaded embedding ownership is unknown or ambiguous')
        actual = matches[0]
        owners = self.engine.embedding_token_owners[actual]
        if set(owners) != set(streams):
            raise ValueError('loaded embedding ownership is incomplete')
        entries, owner_mapping = {}, {}
        for key, (tokenizer, encoder) in streams.items():
            added = tokenizer.get_added_vocab()
            names = owners[key]
            if not names or actual not in names or len(set(names)) != len(names) or any(n not in added for n in names):
                raise ValueError('loaded embedding ownership is inconsistent')
            ids = sorted(added[n] for n in names)
            owner_mapping[key] = tuple(sorted((n, added[n]) for n in names))
            weights = encoder.get_input_embeddings().weight
            if getattr(weights, 'is_meta', False) or any(i < 0 or i >= weights.shape[0] for i in ids):
                raise ValueError('loaded embedding rows are unavailable')
            entries[key] = (encoder, weights, ids, weights[ids].detach().clone())
        self.pipe, self.entries, self.owner_mapping = self.engine.pipe, entries, owner_mapping
        return entries

    def install(self, values):
        import torch
        from .embedding_editor_state import EditorDraft
        draft = EditorDraft(values).values()
        if set(draft) != set(self.entries):
            raise ValueError('preview encoder set differs')
        prepared = {}
        for key, (_, weights, ids, _) in self.entries.items():
            rows = draft[key]
            if len(rows) != len(ids) or any(len(row) != weights.shape[1] for row in rows):
                raise ValueError('preview embedding shape differs')
            converted = torch.as_tensor(rows, dtype=weights.dtype, device=weights.device)
            if not torch.isfinite(converted).all():
                raise ValueError('preview values must remain finite in the encoder dtype')
            prepared[key] = converted
        with torch.no_grad():
            for key, (_, weights, ids, _) in self.entries.items():
                weights[ids] = prepared[key]

    def restore(self, snapshot):
        import torch
        if self.engine.pipe is not self.pipe:
            raise RuntimeError('checkpoint changed during preview')
        for encoder, weights, _, _ in snapshot.values():
            if encoder.get_input_embeddings().weight is not weights:
                raise RuntimeError('embedding matrix changed during preview')
        with torch.no_grad():
            for _, weights, ids, old_rows in snapshot.values():
                weights[ids] = old_rows

    def capture_installed(self):
        self.installed_owners = dict(self.owner_mapping)
        return {key: weights[ids].detach().clone() for key, (_, weights, ids, _) in self.entries.items()}

    def restore_rebound(self, snapshot, installed):
        """Restore after offload hooks may have replaced the weight Parameter."""
        import torch
        if self.engine.pipe is not self.pipe:
            raise RuntimeError('checkpoint changed during preview')
        current = self.snapshot()
        if set(current) != set(snapshot) or set(current) != set(installed) or self.owner_mapping != self.installed_owners:
            raise RuntimeError('embedding streams changed during preview')
        for key, (encoder, weights, ids, rows) in current.items():
            old_encoder, old_weights, old_ids, old_rows = snapshot[key]
            if (encoder is not old_encoder or ids != old_ids or weights.shape != old_weights.shape
                    or weights.dtype != old_weights.dtype or rows.shape != old_rows.shape
                    or not torch.equal(rows.to(device='cpu'), installed[key].to(device='cpu'))):
                raise RuntimeError('embedding ownership or rows changed during preview')
        with torch.no_grad():
            for key, (_, weights, ids, _) in current.items():
                weights[ids] = snapshot[key][3].to(device=weights.device, dtype=weights.dtype)

    def invalidate(self):
        try:
            self.engine._free()
        finally:
            self.engine._embedding_reload_error = 'preview embedding restoration failed'


def run_editor_encoding(engine, name, values, operation, *, transaction, scope_factory=ModelEditorScope):
    """Apply an editor draft only during prompt encoding, under one shared guard.

    `transaction` must materialize and place both encoders and compose with
    other in-flight encoding scopes; this module does not enter a second guard.
    """
    def encode():
        with temporary_edit(scope_factory(engine, name), values):
            return operation()
    return transaction(engine, encode)


def run_pipeline_editor(engine, name, values, operation, *, scope_factory=ModelEditorScope):
    """Keep edits through a pipeline which performs text encoding internally.

    Offload hooks are active during the pipeline; use the encoder guard only
    for installation and restoration, and revalidate ownership on return.
    """
    scope = scope_factory(engine, name)
    def fail_closed(failure):
        try:
            scope.invalidate()
        except BaseException as unload_error:
            failure.add_note(f'model unload also failed: {unload_error}')
        raise EngineError(NO_MODEL, f'Could not restore preview embeddings; reload the checkpoint: {failure}') from failure
    def install():
        snapshot = scope.snapshot()
        try:
            scope.install(values)
            return snapshot, scope.capture_installed()
        except BaseException:
            try:
                scope.restore(snapshot)
            except BaseException as failure:
                fail_closed(failure)
            raise
    snapshot, installed = engine._with_embedding_encoder_access(install)
    try:
        return operation()
    finally:
        if hasattr(engine, 'pipe') and engine.pipe is None:
            import sys
            failure = sys.exc_info()[1]
            if not isinstance(failure, EngineError) or failure.code != NO_MODEL:
                raise EngineError(NO_MODEL, 'Preview model was unloaded; reload the checkpoint') from failure
        else:
            try:
                engine._with_embedding_encoder_access(lambda: scope.restore_rebound(snapshot, installed))
            except BaseException as failure:
                fail_closed(failure)
