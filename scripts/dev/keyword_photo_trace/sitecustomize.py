"""Opt-in tensor fingerprints for the isolated PhotoMaker parity diagnostic.

No tensor values or prompts are logged. This changes timing, not model math;
never enable it for ordinary user requests. SDFX_PHOTO_TRACE must name a fresh
local JSONL file in the diagnostic output directory.
"""
import hashlib
import json
import os
import threading

path = os.environ.get('SDFX_PHOTO_TRACE')
if path:
    import torch
    from sdfx.photomaker_batch import PhotoBranchCapture
    from sdfx.embedding_merge_backend import ModelBackend, streams
    from sdfx.embedding_merge_model import VectorSource
    from sdfx.neutral_runtime import RuntimeHook
    from sdfx.unet_hooks import MultiDiffusionWrapper, tile_grid
    from sdfx.clora import patches
    from sdfx import extensions
    from sdfx.photomaker_adapters import AdapterScope
    from sdfx import photomaker_adapters
    from contextlib import contextmanager
    from module_state import snapshot

    lock = threading.Lock()
    sequence = [0]
    captured = []

    previous_load = patches.load
    def load(*args, **kwargs):
        result = previous_load(*args, **kwargs)
        captured.extend((component, name, module) for component, name, module, _ in result)
        return result
    patches.load = load

    # _apply_loras tears down the preceding job before building the next one.
    # Reset the diagnostic references at the same job boundary.
    from sdfx.clora.runtime import Job
    previous_init = Job.__init__
    def init(self, *args, **kwargs):
        captured.clear()
        try:
            return previous_init(self, *args, **kwargs)
        finally:
            write('composable_hook_owners', snapshot(self.pipe.unet, captured))
    Job.__init__ = init

    previous_special_pipe = extensions.special_pipe
    def special_pipe(engine, plan):
        if getattr(plan, 'photomaker', None) and captured:
            write('unet_before_photo_load', snapshot(engine.pipe.unet, captured))
        result = previous_special_pipe(engine, plan)
        if getattr(plan, 'photomaker', None) and captured:
            write('unet_after_photo_load', snapshot(engine.pipe.unet, captured))
        return result
    extensions.special_pipe = special_pipe

    previous_activate = AdapterScope.activate
    def activate(self, pipe):
        result = previous_activate(self, pipe)
        if captured:
            write('unet_after_photo_activate', snapshot(pipe.unet, captured))
        return result
    AdapterScope.activate = activate

    previous_id_encoder_scope = photomaker_adapters.id_encoder_scope
    @contextmanager
    def id_encoder_scope(engine, pipe):
        encoder = getattr(pipe, 'id_encoder', None)
        if encoder is not None:
            layers = []
            for name, module in encoder.named_modules():
                parameters = list(module.named_parameters(recurse=False))
                if not parameters:
                    continue
                hook = getattr(module, '_hf_hook', None)
                layers.append({'name': name, 'devices': sorted({str(p.device) for _, p in parameters}),
                               'hook_type': type(hook).__name__, 'offload': getattr(hook, 'offload', None),
                               'execution_device': str(getattr(hook, 'execution_device', None)),
                               'weights_backed': getattr(hook, 'weights_map', None) is not None})
            write('photo_id_encoder_layers', layers)
        with previous_id_encoder_scope(engine, pipe):
            yield
    photomaker_adapters.id_encoder_scope = id_encoder_scope

    def fingerprint(tensor):
        value = tensor.detach().cpu().contiguous()
        payload = value.view(torch.uint8).numpy().tobytes()
        return {'sha256': hashlib.sha256(payload).hexdigest(), 'shape': list(value.shape), 'dtype': str(value.dtype)}

    def write(kind, data):
        with lock:
            sequence[0] += 1
            with open(path, 'a', encoding='utf8') as stream:
                stream.write(json.dumps({'sequence': sequence[0], 'event': kind, 'data': data}, ensure_ascii=False) + '\n')

    previous_publish = PhotoBranchCapture.publish
    def publish(self):
        result = previous_publish(self)
        report = []
        for index, branch in enumerate(self.plan.branches if self.plan is not None else self.branches()):
            for end, variant in getattr(branch, 'schedule', None) or [(None, branch)]:
                for stage, data in (('text', variant), ('identity', variant['_photomaker_id'])):
                    for key in ('prompt_embeds', 'negative_prompt_embeds', 'pooled_prompt_embeds', 'negative_pooled_prompt_embeds'):
                        if key in data and torch.is_tensor(data[key]):
                            report.append({'branch': index, 'end': end, 'stage': stage, 'key': key, **fingerprint(data[key])})
        write('photo_branches', report)
        return result
    PhotoBranchCapture.publish = publish

    previous_forward = RuntimeHook.forward
    def forward(self, sample, timestep, *args, **kwargs):
        if self.raw_step == 1 and captured:
            write('unet_first_step', snapshot(self.unet, captured))
        result = previous_forward(self, sample, timestep, *args, **kwargs)
        predicted = result[0] if isinstance(result, tuple) else result.sample
        write('neutral_unet', {'raw_step': self.raw_step, 'logical_step': self.step,
                               'input': fingerprint(sample), 'output': fingerprint(predicted)})
        return result
    RuntimeHook.forward = forward

    previous_multi = MultiDiffusionWrapper._call
    def multi(self, sample, timestep, encoder_hidden_states, *args, **kwargs):
        height, width = sample.shape[-2:]
        th = max(8, int(self.cfg['tile_height']) // 8)
        tw = max(8, int(self.cfg['tile_width']) // 8)
        overlap = max(0, int(self.cfg['tile_overlap']) // 8)
        count = 1 if height <= th and width <= tw else len(tile_grid(height, width, th, tw, overlap))
        write('multi_tiles', {'latent_height': height, 'latent_width': width, 'tile_count': count})
        return previous_multi(self, sample, timestep, encoder_hidden_states, *args, **kwargs)
    MultiDiffusionWrapper._call = multi

    # Compare the final scheduler latent and VAE output without retaining any
    # image/tensor contents. Decode may run outside the Neutral UNet scope.
    from diffusers import AutoencoderKL
    previous_decode = AutoencoderKL.decode
    def decode(self, sample, *args, **kwargs):
        param = next(self.parameters(), None)
        info = {'input': fingerprint(sample), 'vae_dtype': str(self.dtype),
                'vae_device': str(param.device) if param is not None else None,
                'force_upcast': getattr(self.config, 'force_upcast', None),
                'offload_hook': type(getattr(self, '_hf_hook', None)).__name__}
        result = previous_decode(self, sample, *args, **kwargs)
        decoded = result[0] if isinstance(result, tuple) else result.sample
        info['output'] = fingerprint(decoded)
        write('vae_decode', info)
        return result
    AutoencoderKL.decode = decode

    previous_capture, previous_restore = ModelBackend.capture, ModelBackend.restore
    def capture(self):
        snapshot = previous_capture(self)
        write('encoder_config', [{'stream': name, 'tokenizer_eos': tokenizer.eos_token_id,
                                  'encoder_eos': getattr(getattr(encoder, 'text_model', None), 'eos_token_id', None)}
                                 for name, (tokenizer, encoder) in streams(self.engine.pipe).items()])
        write('embedding_before', [fingerprint(item[4]) for item in snapshot if torch.is_tensor(item[4])])
        return snapshot
    def restore(self, snapshot):
        result = previous_restore(self, snapshot)
        active = getattr(self.engine, 'pipe', None)
        if active is not None:
            write('embedding_after', [fingerprint(encoder.get_input_embeddings().weight)
                                      for _, encoder in streams(active).values()])
        return result
    ModelBackend.capture, ModelBackend.restore = capture, restore

    previous_take = VectorSource._take
    def take(weights, ids):
        if torch.is_tensor(weights):
            info = {'type': type(weights).__module__ + '.' + type(weights).__qualname__,
                    'device': str(weights.device), 'dtype': str(weights.dtype),
                    'inference_tensor': torch.is_inference(weights), 'requires_grad': weights.requires_grad,
                    'inference_enabled': torch.is_inference_mode_enabled(), 'grad_enabled': torch.is_grad_enabled(),
                    'leaf': weights.is_leaf, 'has_grad_fn': weights.grad_fn is not None}
            try:
                info['version'] = weights._version
            except RuntimeError as error:
                info['version_error'] = str(error)
            try:
                result = previous_take(weights, ids)
            except BaseException as error:
                write('merge_weight_access_error', dict(info, error_type=type(error).__name__, message=str(error)))
                raise
            write('merge_weight_access', info)
            return result
        return previous_take(weights, ids)
    VectorSource._take = staticmethod(take)
