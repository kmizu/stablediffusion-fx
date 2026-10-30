"""Spatial K/V downsampling for SD UNet self-attention.

Behavior reference: MIT feffy380/sd-webui-token-downsampling, pinned in
docs/superpowers/plans/2026-10-30-token-downsampling.md. No upstream runtime import.
"""
import math
import re
from .protocol import EngineError, INVALID_PARAMS, UNSUPPORTED


DEFAULTS = dict(factor=1.0, factor2=1.0, disable_after=1.0)


def step_budget(base_steps, second_pass_steps, pass_name):
    """Source cutoff uses requested steps, never strength-truncated execution count."""
    return (second_pass_steps or base_steps) if pass_name == 'hires' else base_steps


def normalize(config):
    result = dict(DEFAULTS)
    for key, low, high in [('factor', 1, 5), ('factor2', 1, 5), ('disable_after', 0, 1)]:
        value = config.get(key, result[key])
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
            raise EngineError(INVALID_PARAMS, f'Token Downsampling {key} must be finite in [{low}, {high}]')
        result[key] = float(value)
    return result


def resample(states, height, width, factor):
    if states.ndim != 3 or height * width != states.shape[1]:
        raise EngineError(UNSUPPORTED, 'Token Downsampling requires an unambiguous spatial token layout')
    if factor == 1:
        return states
    from torch.nn.functional import interpolate
    batch, _, channels = states.shape
    target = max(1, int(height / factor)), max(1, int(width / factor))
    image = states.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
    return interpolate(image, size=target, mode='nearest-exact').permute(0, 2, 3, 1).reshape(batch, -1, channels)


class TokenDownsampling:
    priority = 0

    def __init__(self, unet, config, sdxl, steps, branch_mode=False):
        self.unet = unet
        self.config = normalize(config)
        self.factors = {2 if sdxl else 1: self.config['factor'], 4 if sdxl else 2: self.config['factor2']}
        self.limit = int(self.config['disable_after'] * steps)
        self.evaluations = 0
        self.branch_mode = branch_mode
        self.handles = []
        self.stats = {}  # bounded aggregate by (depth, query tokens, key tokens)

    @property
    def enabled(self):
        return self.limit > 0 and any(factor > 1 for factor in self.factors.values())

    def _tick(self, module, args, kwargs):
        self.evaluations += 1

    def before_branch(self, original, inputs, timestep, kwargs, layout):
        self.evaluations += 1

    def after_branch(self):
        pass

    def _attention(self, shape, depth, module, args, kwargs):
        if self.evaluations > self.limit:
            return None
        states = args[0] if args else kwargs.get('hidden_states')
        context = args[1] if len(args) > 1 else kwargs.get('encoder_hidden_states')
        mask = args[2] if len(args) > 2 else kwargs.get('attention_mask')
        if context is not None:
            raise EngineError(UNSUPPORTED, 'Token Downsampling cannot replace an existing self-attention context')
        processor = type(module.processor).__name__
        if processor not in ('AttnProcessor', 'AttnProcessor2_0', 'XFormersAttnProcessor', 'SlicedAttnProcessor'):
            raise EngineError(UNSUPPORTED, f'Token Downsampling does not support self-attention processor {processor}')
        if shape[0] is None or states is None or states.ndim != 3:
            raise EngineError(UNSUPPORTED, 'Token Downsampling could not determine the current spatial layout')
        height, width = shape[0]
        reduced = resample(states, height, width, self.factors[depth])
        values = dict(kwargs)
        positional = list(args)
        if len(positional) > 1:
            positional[1] = reduced
        else:
            values['encoder_hidden_states'] = reduced
        if mask is not None:
            if mask.shape[-1] != height * width:
                raise EngineError(UNSUPPORTED, 'Token Downsampling mask does not match spatial keys')
            # Sample mask keys with exactly the same nearest-exact index map.
            import torch
            indices = torch.arange(height * width, device=states.device, dtype=torch.float32).reshape(1, -1, 1)
            indices = resample(indices, height, width, self.factors[depth]).reshape(-1).long()
            mask = mask.index_select(-1, indices.to(mask.device))
            if len(positional) > 2:
                positional[2] = mask
            else:
                values['attention_mask'] = mask
        key = depth, states.shape[1], reduced.shape[1]
        self.stats[key] = self.stats.get(key, 0) + 1
        return tuple(positional), values

    def install(self):
        if self.handles or not self.enabled:
            return self
        targets = []
        stages = len(self.unet.down_blocks)
        # Capture each Transformer2D input, not a sqrt(sequence) approximation:
        # odd-size upsampling and rectangular latents preserve their actual layout.
        for name, parent in self.unet.named_modules():
            if not hasattr(parent, 'transformer_blocks'):
                continue
            match = re.match(r'(down_blocks|up_blocks)\.(\d+)\.attentions\.\d+$', name)
            if match:
                index = int(match[2])
                depth = 2 ** (index if match[1] == 'down_blocks' else stages - index - 1)
            elif re.fullmatch(r'mid_block\.attentions\.\d+', name):
                depth = 2 ** (stages - 1)
            else:
                continue
            if self.factors.get(depth, 1) <= 1:
                continue
            attentions = [block.attn1 for block in parent.transformer_blocks if not getattr(block, 'only_cross_attention', False)]
            for attn in attentions:
                if attn.group_norm is not None or attn.spatial_norm is not None or attn.norm_cross:
                    raise EngineError(UNSUPPORTED, 'Token Downsampling requires ordinary SD self-attention normalization')
            if attentions:
                targets.append((parent, depth, attentions))
        if not targets:
            raise EngineError(UNSUPPORTED, 'Token Downsampling found no selected SD self-attention layers')
        self.evaluations = 0
        self.stats.clear()
        try:
            if not self.branch_mode:
                self.handles.append(self.unet.register_forward_pre_hook(self._tick, with_kwargs=True))
            for parent, depth, attentions in targets:
                shape = [None]
                def capture(module, args, kwargs, shape=shape):
                    image = args[0] if args else kwargs.get('hidden_states')
                    shape[0] = tuple(image.shape[-2:]) if image is not None and image.ndim == 4 else None
                self.handles.append(parent.register_forward_pre_hook(capture, with_kwargs=True))
                for attn in attentions:
                    def pre(module, args, kwargs, shape=shape, depth=depth):
                        return self._attention(shape, depth, module, args, kwargs)
                    self.handles.append(attn.register_forward_pre_hook(pre, with_kwargs=True))
            return self
        except BaseException:
            self.remove()
            raise

    def remove(self):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
