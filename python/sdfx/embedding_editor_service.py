"""Wire editor draft operations to the live engine's serialized worker.

Open and save are registered on the engine worker. Preview still requires an
owned generation request and the shared encoding transaction integration.
"""

from pathlib import Path
from .embedding_editor_adapter import EngineEditorAdapter
from .embedding_editor_publish import publish_pt_editor, publish_safetensors_editor
from .embedding_editor_tensor import read_editor_vectors
from .embedding_editor_worker import dispatch_editor


def run_editor_service(engine, params, ctx, *, read=read_editor_vectors,
                       publish_safetensors=publish_safetensors_editor, publish_pt=publish_pt_editor,
                       refresh=None, validate=None, preview_scope=None, generate=None):
    if not getattr(engine, 'model', None) or getattr(engine, 'pipe', None) is None:
        raise ValueError('load a checkpoint before using the embedding editor')
    if params.get('operation') in ('save', 'preview') and getattr(engine, '_embedding_reload_error', None):
        raise ValueError('reload the checkpoint before editing after an embedding failure')
    if refresh is None:
        def refresh(paths):
            from .embedding_refresh import refresh_saved_embeddings
            return refresh_saved_embeddings(engine, paths)
    if params.get('operation') == 'open' and params.get('path'):
        raw_name, raw_path = params.get('name'), params['path']
        if not isinstance(raw_name, str) or not isinstance(raw_path, str):
            raise ValueError('invalid embedding selection')
        path = Path(raw_path).resolve()
        if path.suffix.lower() not in ('.pt', '.bin', '.safetensors') or path.stem.casefold() != raw_name.strip().casefold() or not path.is_file():
            raise ValueError('selected embedding path and name do not match')
        matches = [(name, Path(p).resolve()) for name, p in getattr(engine, 'loaded_embedding_files', {}).items()
                   if name.casefold() == raw_name.strip().casefold()]
        if len(matches) > 1 or (matches and matches[0][1] != path):
            raise ValueError('another embedding with this name is already loaded; reload the checkpoint to choose this file')
        if not matches:
            if ctx is None:
                raise ValueError('editor request requires a job context')
            ctx.check_interrupt()
            def activate():
                def load():
                    check = validate
                    if check is None:
                        from .embedding_refresh import validate_saved_names
                        check = validate_saved_names
                    check(engine, [str(path)])
                    refresh([str(path)])
                return engine._with_embedding_encoder_access(load)
            try:
                ctx.publish(activate)
            except BaseException as failure:
                if getattr(engine, 'pipe', None) is None:
                    from .protocol import EngineError, NO_MODEL
                    raise EngineError(NO_MODEL, 'Embedding activation failed and the checkpoint was unloaded; reload it') from failure
                raise
            ctx.check_interrupt()
    adapter = None
    def publish(active_engine, name, values, expected):
        def edit():
            check = validate
            if check is None:
                from .embedding_refresh import validate_saved_names
                check = validate_saved_names
            check(active_engine, [expected['path']])
            baseline = read(active_engine, name, [], guarded=False)['vectors']
            source = Path(expected['path'])
            fn = publish_safetensors if source.suffix.lower() == '.safetensors' else publish_pt
            return fn(active_engine, name, values, expected, baseline=baseline,
                      current=adapter.current, refresh=refresh)
        return active_engine._with_embedding_encoder_access(edit)
    adapter = EngineEditorAdapter(engine, read_vectors=read, publish_vectors=publish,
                                  preview_vectors=preview_scope, generate=generate)
    return dispatch_editor(adapter, params, ctx)
