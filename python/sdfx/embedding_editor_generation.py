"""Per-generation editor draft validation and scoped prompt encoding."""

from .embedding_editor_worker import _validate
from .embedding_editor_preview import run_editor_encoding, run_pipeline_editor


def validate_preview(adapter, draft, ctx):
    ctx.check_interrupt()
    if not isinstance(draft, dict):
        raise ValueError('invalid embedding editor preview')
    result = _validate(adapter, draft.get('name'), draft)
    ctx.check_interrupt()
    return result


def preview_encoding(engine, operation, *, runner=run_editor_encoding, transaction=None):
    draft = getattr(engine, '_editor_preview', None)
    if draft is None or getattr(engine, '_editor_preview_active', False):
        return operation()
    if transaction is None:
        from .embedding_merge_backend import run_encoding_transaction
        transaction = run_encoding_transaction
    engine._editor_preview_active = True
    try:
        name, values = draft
        return runner(engine, name, values, operation, transaction=transaction)
    finally:
        engine._editor_preview_active = False


def preview_pipeline(engine, operation, *, runner=run_pipeline_editor):
    """Scope edits around an internal-encoding pipeline and nested merge work."""
    draft = getattr(engine, '_editor_preview', None)
    if draft is None or getattr(engine, '_editor_preview_active', False):
        return operation()
    engine._editor_preview_active = True
    try:
        name, values = draft
        return runner(engine, name, values, operation)
    finally:
        engine._editor_preview_active = False


def invoke_pipeline(engine, pipe, prompts, operation, *, check, editor_runner=run_pipeline_editor, merge_runner=None,
                    runtime_hook=None, extra_runtime_hooks=(), pre_runtime_hooks=()):
    """Reactivate derived-pipeline hooks after editor access restores base hooks."""
    if merge_runner is None:
        from .embedding_merge_pipeline import run_pipeline
        merge_runner = run_pipeline
    runtime = {}
    if runtime_hook is not None: runtime['runtime_hook'] = runtime_hook
    if extra_runtime_hooks: runtime['extra_runtime_hooks'] = extra_runtime_hooks
    if pre_runtime_hooks: runtime['pre_runtime_hooks'] = pre_runtime_hooks
    if getattr(engine, '_editor_preview', None) is None:
        return merge_runner(engine, pipe, prompts, operation, check=check, **runtime)
    hooks = list(pre_runtime_hooks) + ([runtime_hook] if runtime_hook is not None else []) + list(extra_runtime_hooks)
    rebind = bool(hooks) and bool(getattr(engine, '_offload', False))
    if rebind:
        # Encoder access rebuilds Accelerate's UNet.forward. Suspend wrappers
        # before both editor guards, including when no merge expression exists.
        for hook in reversed(hooks):
            hook.remove()
    def call():
        rebound = []
        try:
            engine._activate(pipe)
            check()
            if rebind:
                for hook in hooks:
                    hook.install()
                    rebound.append(hook)
            return merge_runner(engine, pipe, prompts, operation, check=check, **runtime)
        finally:
            for hook in reversed(rebound):
                hook.remove()
    return preview_pipeline(engine, call, runner=editor_runner)
