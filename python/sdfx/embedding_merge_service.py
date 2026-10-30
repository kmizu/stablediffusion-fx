"""Queued native inspection/export service; all tensor access uses #46's guard."""
import numpy as np

from .embedding_merge import evaluate, parse_expression
from .embedding_merge_backend import streams
from .embedding_merge_export import export_embeddings, export_plan
from .embedding_merge_inspection import Part, rows, statistics
from .embedding_merge_model import VectorSource
from .embedding_merge_prompts import expressions, restore_provenance, restore_infotext
from .protocol import EngineError, INVALID_PARAMS, NO_MODEL, UNSUPPORTED


def run(engine, params, ctx):
    if getattr(engine, 'dry_run', False):
        raise EngineError(UNSUPPORTED, 'Embedding Merge requires a real loaded model')
    if getattr(engine, 'model', None) is None or getattr(engine, 'pipe', None) is None:
        raise EngineError(NO_MODEL, 'load a checkpoint before merging embeddings')
    operation = params.get('operation', 'inspect')
    if operation not in ('inspect', 'save', 'restore'):
        raise EngineError(INVALID_PARAMS, 'Embedding Merge operation must be inspect, save or restore')
    check = ctx.check_interrupt
    check()

    def inspect():
        text = params.get('text', '')
        restored = restore_infotext(text)
        if operation == 'restore' or restored != text:
            if params.get('provenance'):
                restored = restore_provenance(restored, params['provenance'])
            return {'text': restored, 'streams': [], 'paths': [], 'active': False}
        active_streams = streams(engine.pipe)
        file_args = {}
        if params.get('embedding_paths'):
            from .embedding_merge_files import file_provider
            load_file, file_names = file_provider(params['embedding_paths'], active_streams)
            file_args = {'file_names': file_names, 'load_file': load_file}
        source = VectorSource({name: (tokenizer, encoder.get_input_embeddings().weight)
            for name, (tokenizer, encoder) in active_streams.items()}, getattr(engine, 'embedding_token_owners', {}), **file_args)

        def merge(expression):
            check()
            return evaluate(parse_expression(expression), source.resolve_text, source.resolve_token, source.stream_names)

        if not isinstance(text, str):
            raise ValueError('text must be a string')
        combined = {}
        results = []
        if text.lstrip().startswith("'"):
            values = merge(text)
            for name, value in zip(source.stream_names, values):
                if not len(value):
                    raise ValueError('expression produced zero vectors')
                combined[name] = value
                result_rows = [{'start': i + 1, 'end': i + 1, 'count': 1, 'text': '', 'tokens': [],
                                'statistics': statistics(vector)} for i, vector in enumerate(value)]
                results.append({'name': name, 'count': len(value), 'rows': result_rows, 'total': statistics(value)})
        else:
            grouping = params.get('grouping', 'parts')
            collected = {name: [] for name in source.stream_names}
            position, comma_group = 0, 0

            def plain(fragment):
                nonlocal comma_group
                check()
                for name, pieces in source.parts(fragment, split_commas=grouping == 'comma').items():
                    collected[name].extend(Part(part.vectors, part.text, part.tokens, part.pieces,
                                                part.comma_group + comma_group) for part in pieces)
                if grouping == 'comma':
                    comma_group += fragment.count(',')

            for span in expressions(text):
                plain(text[position:span.start])
                for name, value in zip(source.stream_names, merge(span.expression)):
                    if len(value):
                        collected[name].append(Part(value, span.original, None, (), comma_group))
                position = span.end
            plain(text[position:])
            for name, parts in collected.items():
                if not parts:
                    continue
                value = np.concatenate([part.vectors for part in parts])
                combined[name] = value
                results.append({'name': name, 'count': len(value), 'rows': rows(parts, grouping),
                                'total': statistics(value)})
        check()
        paths, active, warning = [], False, ''
        if operation == 'save':
            if not params.get('output_dir'):
                raise ValueError('managed embedding output directory is required')
            from .embedding_refresh import refresh_saved_embeddings, validate_saved_names
            if getattr(engine, '_embedding_reload_error', None):
                raise ValueError('Reload the checkpoint before saving after a refresh failure')
            primary = export_plan(params['output_dir'], params.get('name', ''), combined,
                                  params.get('safetensors', True))[0][0]
            validate_saved_names(engine, [primary])
            def publish(commit):
                def commit_and_activate():
                    nonlocal active, warning
                    commit()
                    try:
                        refresh_saved_embeddings(engine, [primary])
                        active = True
                    except Exception as error:
                        engine._embedding_reload_error = str(error)
                        warning = f'Files saved, but live activation failed; reload the checkpoint: {error}'
                return ctx.publish(commit_and_activate)
            paths = export_embeddings(params['output_dir'], params.get('name', ''), combined,
                params.get('safetensors', True), check=check, publish=publish)
        return {'text': text, 'streams': results, 'paths': [str(path) for path in paths],
                'active': active, 'warning': warning}
    try:
        return engine._with_embedding_encoder_access(inspect)
    except (ValueError, KeyError, OSError) as error:
        raise EngineError(INVALID_PARAMS, f'Embedding Merge: {error}') from error
