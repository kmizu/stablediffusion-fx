"""Actual local resource fingerprints for optional generation provenance."""
from pathlib import Path
from .models import full_hash
import re


class ResourceHashes(dict):
    def __init__(self):
        super().__init__()
        self.ambiguous = set()


def embedding_hashes(loaded_files, texts):
    """Hash only successfully loaded tokens referenced by this generation."""
    text = ' '.join(texts)
    return {f'embed:{token}': file_resource_hash(path) for token, path in loaded_files.items()
            if re.search(r'(?<![\w-])' + re.escape(token) + r'(?![\w-])', text)}


def record_file(resources, kind, path, warn, snapshot=None):
    """Record a caller-confirmed loaded file, or explain why no hash is available."""
    try:
        key = f'{kind}:{Path(path).name}'
        if key in getattr(resources, 'ambiguous', set()):
            return
        value = snapshot.digest() if snapshot is not None else file_resource_hash(path)
        if key in resources and resources[key] != value:
            resources.pop(key)
            if isinstance(resources, ResourceHashes):
                resources.ambiguous.add(key)
            raise ValueError(f'ambiguous resource name {key}')
        resources[key] = value
    except (OSError, ValueError) as error:
        warn(f'Resource hash unavailable for {kind}: {error}')


def file_resource_hash(path):
    """AutoV2 prefix of whole-file SHA-256; never hash a pathname as fallback."""
    source = Path(path)
    if not source.is_file():
        raise ValueError('resource hash requires an existing weight file')
    return full_hash(source)[:10]


class FileSnapshot:
    """Lazy hash only while file identity still matches the load-time snapshot."""
    def __init__(self, path, identity):
        self.path, self.identity = Path(path), identity

    @staticmethod
    def identity_of(path):
        info = Path(path).stat()
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

    @classmethod
    def capture(cls, path):
        return cls(path, cls.identity_of(path))

    def digest(self):
        if self.identity_of(self.path) != self.identity:
            raise ValueError('resource file changed since loading; reload it before recording hashes')
        digest = file_resource_hash(self.path)
        if self.identity_of(self.path) != self.identity:
            raise ValueError('resource file changed since loading; reload it before recording hashes')
        return digest
