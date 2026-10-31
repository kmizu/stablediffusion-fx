"""Opt-in worker diagnostics; records metadata only, never tensor values."""
import os

path = os.environ.get('SDFX_KEYWORD_OFFLOAD_TRACE')
if path:
    import json
    import traceback
    import torch
    from sdfx.engine import Engine

    def emit(engine, method, phase, error=None):
        try:
            records = []
            pipe = getattr(engine, 'pipe', None)
            for component in ('unet', 'text_encoder', 'text_encoder_2'):
                module = getattr(pipe, component, None)
                if module is None:
                    continue
                for name, param in module.named_parameters():
                    if 'lora_' in name or 'lora_magnitude_vector' in name:
                        records.append({'component': component, 'name': name, 'shape': list(param.shape),
                                        'device': str(param.device), 'requires_grad': param.requires_grad,
                                        'is_inference': param.is_inference()})
            item = {'method': method, 'phase': phase, 'inference_mode': torch.is_inference_mode_enabled(),
                    'parameters': records}
            if error is not None:
                item['exception'] = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
            with open(path, 'a', encoding='utf8') as out:
                out.write(json.dumps(item, ensure_ascii=False) + '\n')
        except Exception:
            pass

    def wrap(name):
        original = getattr(Engine, name)
        def traced(self, *args, **kwargs):
            emit(self, name, 'before')
            try:
                result = original(self, *args, **kwargs)
            except BaseException as error:
                emit(self, name, 'error', error)
                raise
            emit(self, name, 'after')
            return result
        setattr(Engine, name, traced)
    for name in ('_apply_loras', '_place', '_activate'):
        wrap(name)
