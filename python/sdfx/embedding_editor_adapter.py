"""Adapter boundary between a live engine and an editor draft.

File and checkpoint identities are independently verified; native tensor and
publication operations are injected so the boundary stays testable without torch.
"""

import hashlib
from pathlib import Path

from .resource_hashes import FileSnapshot
from .embedding_editor_state import SourceStamp


class EngineEditorAdapter:
    def __init__(self, engine, *, read_vectors, publish_vectors=None, preview_vectors=None, generate=None):
        self.engine = engine
        self.read_vectors = read_vectors
        self.publish_vectors = publish_vectors
        self.preview_vectors = preview_vectors
        self.generate_callback = generate

    def _selected(self, name):
        model = getattr(self.engine, 'model', None) or {}
        token = model.get('model_token')
        if not token:
            raise ValueError('load a checkpoint before editing')
        files = getattr(self.engine, 'loaded_embedding_files', {})
        matches = [key for key in files if key.casefold() == name.strip().casefold()]
        if len(matches) != 1:
            raise ValueError('embedding name is ambiguous' if matches else 'embedding is not loaded')
        actual = matches[0]
        owners = getattr(self.engine, 'embedding_token_owners', {}).get(actual)
        if not owners or any(not rows or actual not in rows for rows in owners.values()):
            raise ValueError('loaded embedding ownership is unknown')
        snapshot = getattr(self.engine, 'embedding_snapshots', {}).get(actual)
        if snapshot is None:
            raise ValueError('loaded embedding file identity is unknown')
        path = Path(files[actual])
        if snapshot.identity_of(path) != snapshot.identity:
            raise ValueError('loaded embedding file changed; reload it before editing')
        return token, actual, path, snapshot

    def current(self, name):
        token, _, path, snapshot = self._selected(name)
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
        if snapshot.identity_of(path) != snapshot.identity:
            raise ValueError('embedding file changed while reading')
        if (getattr(self.engine, 'model', None) or {}).get('model_token') != token:
            raise ValueError('model changed while reading')
        # Preserve nanosecond timestamps and inode values across JSON clients using binary64.
        return {'model_token': token, 'path': str(path),
                'file_identity': (*map(str, snapshot.identity), digest.hexdigest())}

    def read(self, name, guidance):
        before = self.current(name)
        values = self.read_vectors(self.engine, name, guidance)
        after = self.current(name)
        if before != after:
            raise ValueError('embedding changed while reading')
        return {**values, **after}

    def _check(self, name, expected):
        actual = self.current(name)
        SourceStamp(model_token=expected.get('model_token'), path=expected.get('path'),
                    file_identity=tuple(expected.get('file_identity') or ())).require_current(
            model_token=actual['model_token'], path=actual['path'], file_identity=actual['file_identity'])

    def publish(self, name, values, *, expected):
        self._check(name, expected)
        if self.publish_vectors is None:
            raise ValueError('embedding editor save is unavailable')
        return self.publish_vectors(self.engine, name, values, expected)

    def preview_scope(self, name, values, *, expected):
        self._check(name, expected)
        if self.preview_vectors is None:
            raise ValueError('embedding editor preview is unavailable')
        return self.preview_vectors(self.engine, name, values, expected)

    def generate(self, params):
        if self.generate_callback is None:
            raise ValueError('embedding editor preview generation is unavailable')
        return self.generate_callback(self.engine, params)
