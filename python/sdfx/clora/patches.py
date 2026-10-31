"""Native adapter tensor algebra and module mapping. No fused/base-weight mutation.

LoCon/LoHa/LoKr algebra follows the AGPL upstream composable_lycoris.py. Tensor
loading and diffusers routing are standalone; no A1111/reForge module is imported.
"""
from __future__ import annotations

import re


def cp(t, a, b):
    import torch
    return torch.einsum('ijhw,io,jk->okhw', t, a, b)


class Patch:
    def __init__(self, tensors, module):
        self.tensors = tensors
        self.module = module.get_base_layer() if hasattr(module, 'get_base_layer') else module
        self.cache = {}
        self.name = ''
        self.weight = 1.
        if 'lora_down.weight' in tensors:
            self.kind = 'lora'
            rank = tensors['lora_down.weight'].shape[0]
        elif 'hada_w1_a' in tensors:
            self.kind = 'loha'
            rank = tensors['hada_w1_b'].shape[0]
        elif any(k.startswith('lokr_') for k in tensors):
            self.kind = 'lokr'
            rank = next((tensors[k].shape[0] for k in ('lokr_w1_b', 'lokr_w2_b') if k in tensors), 0)
        elif 'diff' in tensors:
            self.kind, rank = 'full', 0
        elif 'weight' in tensors or 'ia3_w' in tensors:
            self.kind, rank = 'ia3', 0
        else:
            raise ValueError(f'Unsupported Composable LoRA tensor family: {sorted(tensors)}')
        if 'dora_scale' in tensors and self.kind != 'lora':
            raise ValueError('Composable DoRA currently requires conventional LoRA tensors')
        # Full LoKr factors have no low-rank normalization; upstream deliberately
        # ignores alpha when neither Kronecker factor is decomposed.
        self.alpha = (1. if self.kind == 'lokr' and not rank
                      else float(tensors.get('alpha', rank or 1)) / (rank or 1))

    def _on(self, x):
        key = (x.device, x.dtype)
        if key not in self.cache:
            self.cache[key] = {k: v.to(device=x.device, dtype=x.dtype) for k, v in self.tensors.items()}
        return self.cache[key]

    def _op(self, x, w, *, spatial=False, bias=None):
        import torch.nn.functional as f
        if self.module.weight.ndim == 2:
            return f.linear(x, w.reshape(w.shape[0], -1), bias)
        return f.conv2d(x, w, bias, stride=self.module.stride if spatial else 1,
                        padding=self.module.padding if spatial else 0,
                        dilation=self.module.dilation if spatial else 1, groups=self.module.groups)

    def delta(self, x):
        import torch
        t = self._on(x)
        dora = t.get('dora_scale')
        if self.kind == 'lora' and 'bias_indices' not in t and dora is None:
            down, up = t['lora_down.weight'], t['lora_up.weight']
            middle = t.get('lora_mid.weight')
            hidden = self._op(x, down, spatial=middle is None)
            if middle is not None:
                hidden = self._op(hidden, middle, spatial=True)
            return self._op(hidden, up) * self.alpha
        if self.kind == 'ia3':
            w = t.get('weight', t.get('ia3_w'))
            base = self.module.weight.to(device=x.device, dtype=x.dtype)
            on_input = bool(self.tensors.get('on_input', False))
            shape = [1, -1] + [1] * (base.ndim - 2) if on_input else [-1] + [1] * (base.ndim - 1)
            return self._op(x, base * w.reshape(shape), spatial=True) * self.alpha
        if '_delta' not in t:
            if self.kind == 'lora':
                down, up = t['lora_down.weight'], t['lora_up.weight']
                if 'lora_mid.weight' in t:
                    w = cp(t['lora_mid.weight'], up.flatten(1).T, down.flatten(1))
                else:
                    w = up.flatten(1) @ down.flatten(1)
            elif self.kind == 'loha':
                first = (cp(t['hada_t1'], t['hada_w1_a'], t['hada_w1_b']) if 'hada_t1' in t
                         else t['hada_w1_a'] @ t['hada_w1_b'])
                second = (cp(t['hada_t2'], t['hada_w2_a'], t['hada_w2_b']) if 'hada_t2' in t
                          else t['hada_w2_a'] @ t['hada_w2_b'])
                w = first * second
            elif self.kind == 'lokr':
                w1 = t.get('lokr_w1')
                if w1 is None:
                    w1 = t['lokr_w1_a'] @ t['lokr_w1_b']
                w2 = t.get('lokr_w2')
                if w2 is None:
                    w2 = (cp(t['lokr_t2'], t['lokr_w2_a'], t['lokr_w2_b']) if 'lokr_t2' in t
                          else t['lokr_w2_a'] @ t['lokr_w2_b'])
                if w2.ndim == 4:
                    w1 = w1[:, :, None, None]
                w = torch.kron(w1.contiguous(), w2.contiguous())
            else:
                w = t['diff']
            if 'bias_indices' in t:
                # LyCORIS sparse residual bias is a weight-matrix correction.
                size = tuple(int(v) for v in self.tensors['bias_size'])
                residual = torch.sparse_coo_tensor(self.tensors['bias_indices'].to(x.device).long(),
                    t['bias_values'], size, device=x.device).to_dense()
                w = w.reshape(size) + residual
            t['_delta'] = w.reshape(self.module.weight.shape)
        if dora is not None:
            base = self.module.weight.to(device=x.device, dtype=x.dtype)
            merged = base + t['_delta'] * self.alpha
            dims = tuple(range(1, merged.ndim))
            norm = torch.linalg.vector_norm(merged, dim=dims, keepdim=True)
            shape = (-1,) + (1,) * (merged.ndim - 1)
            change = merged * (dora.reshape(shape) / norm) - base
            return self._op(x, change, spatial=True)
        return self._op(x, t['_delta'], spatial=True, bias=t.get('diff_b')) * self.alpha


def layer_mapping(pipe):
    """Canonical diffusers paths and Kohya flattened paths -> (component, path, module)."""
    import torch
    result = {}
    for component, prefixes in (('unet', ('lora_unet_', 'lycoris_unet_')),
                                ('text_encoder', ('lora_te_', 'lora_te1_', 'lycoris_te_')),
                                ('text_encoder_2', ('lora_te2_', 'lycoris_te2_'))):
        model = getattr(pipe, component, None)
        if model is None:
            continue
        for name, module in model.named_modules():
            if any(part in name.split('.') for part in ('base_layer', 'lora_A', 'lora_B', 'lora_dropout')):
                continue
            base = module.get_base_layer() if hasattr(module, 'get_base_layer') else module
            if not isinstance(base, (torch.nn.Linear, torch.nn.Conv2d)):
                continue
            value = (component, name, module)
            result[f'{component}.{name}'] = value
            for prefix in prefixes:
                result[prefix + name.replace('.', '_')] = value
    return result


SUFFIXES = ('lora_down.weight', 'lora_up.weight', 'lora_mid.weight', 'lora_A.weight', 'lora_B.weight',
            'dora_scale', 'lora_magnitude_vector.weight',
            'hada_w1_a', 'hada_w1_b', 'hada_w2_a', 'hada_w2_b', 'hada_t1', 'hada_t2',
            'lokr_w1', 'lokr_w1_a', 'lokr_w1_b', 'lokr_w2', 'lokr_w2_a', 'lokr_w2_b', 'lokr_t2',
            'diff', 'diff_b', 'weight', 'ia3_w', 'on_input', 'alpha', 'bias_indices', 'bias_values', 'bias_size')


def resolve_layer_name(base, pipe):
    mapped = base
    roots = {'input_blocks_0_0': 'conv_in', 'time_embed_0': 'time_embedding_linear_1',
             'time_embed_2': 'time_embedding_linear_2', 'out_2': 'conv_out',
             'label_emb_0_0': 'add_embedding_linear_1', 'label_emb_0_2': 'add_embedding_linear_2'}
    prefix = next((p for p in ('lora_unet_', 'lycoris_unet_') if base.startswith(p)), None)
    if prefix and base[len(prefix):] in roots:
        mapped = prefix + roots[base[len(prefix):]]
    elif any(part in base for part in ('input_blocks', 'output_blocks', 'middle_block')):
        from diffusers.loaders.lora_conversion_utils import _maybe_map_sgm_blocks_to_diffusers
        suffix = '.lora_down.weight'
        mapped = next(iter(_maybe_map_sgm_blocks_to_diffusers({base + suffix: None}, pipe.unet.config)))[:-len(suffix)]
        mapped = mapped.replace('input_blocks', 'down_blocks').replace('output_blocks', 'up_blocks').replace('middle_block', 'mid_block')
        for old, new in (('in_layers_0', 'norm1'), ('in_layers_2', 'conv1'), ('out_layers_0', 'norm2'),
                         ('out_layers_3', 'conv2'), ('emb_layers_1', 'time_emb_proj'), ('skip_connection', 'conv_shortcut'), ('op', 'conv')):
            mapped = mapped.replace(old, new)
    return mapped


def load(path, pipe, dyn=None):
    if str(path).lower().endswith('.safetensors'):
        from safetensors.torch import load_file
        state = load_file(str(path), device='cpu')
    else:
        from ..engine import safe_torch_load
        state = safe_torch_load(str(path), 'Composable LoRA')
    if 'state_dict' in state:
        state = state['state_dict']
    if dyn is not None:
        from ..dynamic_lora import truncate_state_dict
        state = truncate_state_dict(state, dyn, allow_peft_dora=True, allow_kohya_dora=True)
    mapping, groups = layer_mapping(pipe), {}
    for key, tensor in state.items():
        suffix = next((s for s in SUFFIXES if key.endswith('.' + s)), None)
        if suffix is None:
            raise ValueError(f'Unrecognized Composable LoRA tensor: {key}')
        base = key[:-(len(suffix) + 1)]
        groups.setdefault(base, {})[{'lora_A.weight': 'lora_down.weight', 'lora_B.weight': 'lora_up.weight',
                                     'lora_magnitude_vector.weight': 'dora_scale'}.get(suffix, suffix)] = tensor
    patches = []
    for base, tensors in groups.items():
        mapped = resolve_layer_name(base, pipe)
        if mapped not in mapping:
            raise ValueError(f'Composable LoRA layer not found in loaded model: {base} -> {mapped}')
        component, name, module = mapping[mapped]
        patches.append((component, name, module, Patch(tensors, module)))
    if not patches:
        raise ValueError('Composable LoRA checkpoint contains no adapter tensors')
    return patches
