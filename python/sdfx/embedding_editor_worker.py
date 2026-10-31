"""Backend-neutral orchestration for an embedding editor draft.

The adapter must read stable snapshots and publish through the engine's serial
worker. Preview scopes must restore all transient state in their own finally.
"""

from .embedding_editor_state import EditorDraft, SourceStamp, parse_guidance


def _name(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('choose an embedding')
    return value.strip()


def _stamp(data):
    return SourceStamp(model_token=data.get('model_token'), path=data.get('path'),
                       file_identity=tuple(data.get('file_identity') or ()))


def _schema(vectors):
    return {key: (len(rows), len(rows[0])) for key, rows in vectors.items()}


def open_editor(adapter, name, guidance=''):
    name = _name(name)
    references = parse_guidance(guidance)
    data = adapter.read(name, references)
    values = EditorDraft(data['vectors']).values()
    _stamp(data).require_current(model_token=data.get('model_token'), path=data.get('path'),
                                 file_identity=tuple(data.get('file_identity') or ()))
    return {**data, 'vectors': values, 'name': name}


def _validate(adapter, name, draft):
    name = _name(name)
    if not isinstance(draft, dict) or draft.get('name') != name:
        raise ValueError('draft embedding changed; reopen the embedding')
    now = adapter.current(name)
    _stamp(draft).require_current(model_token=now.get('model_token'), path=now.get('path'),
                                  file_identity=tuple(now.get('file_identity') or ()))
    values = EditorDraft(draft.get('vectors')).values()
    current_data = adapter.read(name, [])
    _stamp(draft).require_current(model_token=current_data.get('model_token'), path=current_data.get('path'),
                                  file_identity=tuple(current_data.get('file_identity') or ()))
    current = EditorDraft(current_data['vectors']).values()
    if _schema(values) != _schema(current):
        raise ValueError('embedding vector dimensions changed; reopen the embedding')
    return name, values


def save_editor(adapter, name, draft, *, overwrite=False):
    if overwrite is not True:
        raise ValueError('explicit overwrite consent is required')
    name, values = _validate(adapter, name, draft)
    expected = {key: draft[key] for key in ('model_token', 'path', 'file_identity')}
    return adapter.publish(name, values, expected=expected)


def preview_editor(adapter, name, draft, generation):
    name, values = _validate(adapter, name, draft)
    expected = {key: draft[key] for key in ('model_token', 'path', 'file_identity')}
    with adapter.preview_scope(name, values, expected=expected):
        return adapter.generate(generation)


def dispatch_editor(adapter, params, ctx):
    if not isinstance(params, dict):
        raise ValueError('invalid editor request')
    operation = params.get('operation')
    if operation not in ('open', 'guidance', 'save', 'preview'):
        raise ValueError('unknown editor operation')
    if ctx is None:
        raise ValueError('editor request requires a job context')
    if operation == 'open':
        ctx.check_interrupt()
        result = open_editor(adapter, params.get('name'), params.get('guidance', ''))
        ctx.check_interrupt()
        return result
    if operation == 'guidance':
        ctx.check_interrupt()
        draft = params.get('draft')
        name, _ = _validate(adapter, params.get('name'), draft)
        data = adapter.read(name, parse_guidance(params.get('guidance', '')))
        _stamp(draft).require_current(model_token=data.get('model_token'), path=data.get('path'),
                                      file_identity=tuple(data.get('file_identity') or ()))
        ctx.check_interrupt()
        return {'ranges': data['ranges'], 'guidance': data['guidance']}
    if operation == 'save':
        if params.get('overwrite') is not True:
            raise ValueError('explicit overwrite consent is required')
        ctx.check_interrupt()
        draft = params.get('draft')
        name, values = _validate(adapter, params.get('name'), draft)
        ctx.check_interrupt()
        expected = {key: draft[key] for key in ('model_token', 'path', 'file_identity')}
        return ctx.publish(lambda: adapter.publish(name, values, expected=expected))
    ctx.check_interrupt()
    return preview_editor(adapter, params.get('name'), params.get('draft'), params.get('generation', {}))
