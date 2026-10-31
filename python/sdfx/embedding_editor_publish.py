"""Publication orchestration for a confirmed existing-embedding edit."""

from .embedding_editor_state import SourceStamp


def publish_editor(engine, name, values, expected, *, current, write, refresh):
    """Run inside an owned publication boundary; write must be atomic.

    `current` must fingerprint actual file bytes as well as model identity.
    A concurrent external writer is not excluded by this Python-level check.
    """
    stamp = SourceStamp(model_token=expected.get('model_token'), path=expected.get('path'),
                        file_identity=tuple(expected.get('file_identity') or ()))
    actual = current(name)
    stamp.require_current(model_token=actual.get('model_token'), path=actual.get('path'),
                          file_identity=tuple(actual.get('file_identity') or ()))
    try:
        write(stamp.path, values)
    except BaseException as failure:
        try:
            actual = current(name)
            stamp.require_current(model_token=actual.get('model_token'), path=actual.get('path'),
                                  file_identity=tuple(actual.get('file_identity') or ()))
        except BaseException:
            engine._embedding_reload_error = str(failure)
        raise
    try:
        refresh([stamp.path])
    except BaseException as failure:
        engine._embedding_reload_error = str(failure)
        raise RuntimeError(f'Embedding file was saved but reload failed; reload the checkpoint: {failure}') from failure
    return {'path': stamp.path, 'reloaded': True}


def publish_safetensors_editor(engine, name, values, expected, *, baseline, current, refresh):
    """Stage an edited copy and retain a verified pre-edit backup before replace.

    Caller must hold `ctx.publish` and the encoder-access guard. Publication
    verifies the atomically vacated source and installs only into a vacant path.
    """
    import json
    import struct
    from pathlib import Path
    from .embedding_editor_safetensors import _MAX_HEADER, _unique, build_safetensors_copy

    source = Path(expected.get('path', ''))
    if source.suffix.lower() != '.safetensors':
        raise ValueError('not a safetensors embedding')
    fingerprint = tuple(expected.get('file_identity') or ())
    if not fingerprint or not isinstance(fingerprint[-1], str):
        raise ValueError('source fingerprint is missing')
    stamp = SourceStamp(model_token=expected.get('model_token'), path=str(source), file_identity=fingerprint)
    def check():
        actual = current(name)
        stamp.require_current(model_token=actual.get('model_token'), path=actual.get('path'),
                              file_identity=tuple(actual.get('file_identity') or ()))
    check()
    with source.open('rb') as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError('invalid safetensors header')
        length = struct.unpack('<Q', prefix)[0]
        if length < 2 or length > _MAX_HEADER:
            raise ValueError('invalid safetensors header')
        header = json.loads(handle.read(length), object_pairs_hook=_unique)
    if not isinstance(header, dict):
        raise ValueError('invalid safetensors header')
    keys = set(header) - {'__metadata__'}
    if set(values) == {'clip_l', 'clip_g'} and {'clip_l', 'clip_g'} <= keys:
        mapping = {'clip_l': 'clip_l', 'clip_g': 'clip_g'}
    elif set(values) == {'clip_l'} and 'clip_l' in keys:
        mapping = {'clip_l': 'clip_l'}
    elif set(values) == {'clip_l'} and 'emb_params' in keys:
        mapping = {'clip_l': 'emb_params'}
    elif set(values) == {'clip_l'} and len(keys) == 1:
        mapping = {'clip_l': next(iter(keys))}
    else:
        raise ValueError('embedding tensor names are ambiguous')
    if set(baseline) != set(values):
        raise ValueError('baseline encoder set differs')
    return _publish_staged(engine, name, expected, current=current, refresh=refresh,
        build=lambda src, dst, digest: build_safetensors_copy(src, dst, {mapping[k]: v for k, v in values.items()}, digest,
                                                              baseline={mapping[k]: v for k, v in baseline.items()}))


def publish_pt_editor(engine, name, values, expected, *, baseline, current, refresh, load=None, dump=None, tensor=None):
    from pathlib import Path
    from .embedding_editor_pt import build_pt_copy
    if Path(expected.get('path', '')).suffix.lower() not in ('.pt', '.bin'):
        raise ValueError('not a PyTorch embedding')
    return _publish_staged(engine, name, expected, current=current, refresh=refresh,
        build=lambda src, dst, digest: build_pt_copy(src, dst, values, digest, baseline=baseline,
                                                     load=load, dump=dump, tensor=tensor))


def _publish_staged(engine, name, expected, *, current, refresh, build):
    import hashlib
    import os
    import uuid
    from pathlib import Path
    from .resource_hashes import FileSnapshot
    source = Path(expected.get('path', ''))
    fingerprint = tuple(expected.get('file_identity') or ())
    if not fingerprint or not isinstance(fingerprint[-1], str):
        raise ValueError('source fingerprint is missing')
    stamp = SourceStamp(model_token=expected.get('model_token'), path=str(source), file_identity=fingerprint)
    def check():
        actual = current(name)
        stamp.require_current(model_token=actual.get('model_token'), path=actual.get('path'),
                              file_identity=tuple(actual.get('file_identity') or ()))
    check()
    suffix = uuid.uuid4().hex
    stage = source.with_name('.' + source.name + '.editing-' + suffix)
    backup = source.with_name(source.name + '.before-edit-' + suffix + '.backup')
    replace_started = False
    stage_owned = False
    stage_identity = None
    backup_created = False
    try:
        build(source, stage, fingerprint[-1])
        stage_owned = True
        stage_identity = FileSnapshot.identity_of(stage)
        check()
        with backup.open('xb'):
            backup_created = True
        replace_started = True
        try:
            os.replace(source, backup)
        except BaseException as failure:
            engine._embedding_reload_error = str(failure) or type(failure).__name__
            raise RuntimeError(f'Embedding may have been saved; reload the checkpoint. Backup: {backup}: {failure}') from failure
        try:
            digest = hashlib.sha256()
            with backup.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != fingerprint[-1]:
                raise ValueError('embedding file changed before publication')
            os.link(stage, source)
        except BaseException as failure:
            try:
                os.link(backup, source)
            except FileExistsError:
                if FileSnapshot.identity_of(source) == stage_identity:
                    engine._embedding_reload_error = str(failure) or type(failure).__name__
            except BaseException as restore_failure:
                engine._embedding_reload_error = str(restore_failure) or type(restore_failure).__name__
            else:
                backup.unlink()
                backup_created = False
            raise
        try:
            refresh([str(source)])
        except BaseException as failure:
            engine._embedding_reload_error = str(failure)
            raise RuntimeError(f'Embedding saved but reload failed; reload the checkpoint. Backup: {backup}: {failure}') from failure
        return {'path': str(source), 'backup': str(backup), 'reloaded': True}
    finally:
        if stage_owned:
            try:
                if FileSnapshot.identity_of(stage) == stage_identity:
                    stage.unlink()
            except FileNotFoundError:
                pass
        if not replace_started and backup_created:
            backup.unlink(missing_ok=True)
