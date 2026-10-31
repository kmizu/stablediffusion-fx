"""A1111 dynamic rank for ordinary LoRA state dictionaries.

The original alpha uses the original rank, so resizing a factor pair also scales
its alpha by the same ratio before PEFT computes alpha / truncated rank.
"""


def install_dora_weight_hooks(components, adapter_weights):
    """Scale PEFT's normalized DoRA delta after normalization, as A1111 does."""
    handles = []
    found = set()
    try:
        for component, model in components:
            if model is None:
                continue
            for layer_name, layer in model.named_modules():
                vectors = getattr(layer, 'lora_magnitude_vector', None)
                if vectors is None:
                    continue
                for adapter, weights in adapter_weights.items():
                    if adapter not in vectors:
                        continue
                    if type(vectors[adapter]).__name__ == 'DoraEmbeddingLayer':
                        raise ValueError('DoRA embedding layers are not supported')
                    values = weights.get((component, layer_name), weights)
                    handles.append(vectors[adapter].register_forward_hook(
                        lambda _module, _inputs, output, values=values, key=component:
                            output * values.get(key, 1.0)))
                    found.add(adapter)
        missing = set(adapter_weights) - found
        if missing:
            raise ValueError(f'DoRA magnitude modules not found for: {", ".join(sorted(missing))}')
        return handles
    except BaseException:
        for handle in handles:
            handle.remove()
        raise


def capture_dora_block_weights(components, adapters):
    """Capture diffusers' per-layer multipliers after its normal block expansion."""
    result = {adapter: {} for adapter in adapters}
    for component, model in components:
        if model is None:
            continue
        for layer_name, layer in model.named_modules():
            vectors = getattr(layer, 'lora_magnitude_vector', None)
            if vectors is None:
                continue
            for adapter in adapters:
                if adapter not in vectors:
                    continue
                base = float(layer.lora_alpha[adapter]) / float(layer.r[adapter])
                if not base:
                    raise ValueError('DoRA block weights with zero alpha cannot be resolved safely')
                multiplier = float(layer.scaling[adapter]) / base
                result[adapter][(component, layer_name)] = {component: multiplier}
    for adapter, weights in result.items():
        if not weights:
            raise ValueError(f'DoRA magnitude modules not found for: {adapter}')
    return result


def truncate_state_dict(state, rank, *, allow_peft_dora=False, allow_kohya_dora=False):
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise ValueError('dynamic rank must be an integer')
    out = dict(state)
    magnitude = [key for key in state if 'lora_magnitude_vector' in key]
    kohya = [key for key in state if 'dora_scale' in key]
    if (kohya and not allow_kohya_dora) or (magnitude and not allow_peft_dora):
        raise ValueError('dynamic rank with this DoRA format is not supported')
    if kohya:
        for key in kohya:
            base = key.removesuffix('dora_scale')
            if (not key.endswith('.dora_scale') or
                    not base.startswith(('lora_unet_', 'lora_te_', 'lora_te1_', 'lora_te2_')) or
                    base + 'lora_down.weight' not in state or base + 'lora_up.weight' not in state):
                raise ValueError(f'unsupported A1111 DoRA format: {key}')
    if magnitude:
        for key in magnitude:
            base = key.split('lora_magnitude_vector', 1)[0]
            if (not base.startswith(('unet.', 'text_encoder.', 'text_encoder_2.')) or
                    base + 'lora_A.weight' not in state or base + 'lora_B.weight' not in state or
                    base + 'alpha' in state):
                raise ValueError(f'unsupported native DoRA format: {key}')
    pairs = [('lora_down.weight', 'lora_up.weight'), ('lora_A.weight', 'lora_B.weight'),
             ('lora_A.default.weight', 'lora_B.default.weight')]
    recognized = tuple(suffix for pair in pairs for suffix in pair)
    for key in state:
        if ('lora_A.' in key or 'lora_B.' in key or 'lora_down.' in key or 'lora_up.' in key) and not key.endswith(recognized):
            raise ValueError(f'unsupported dynamic-rank LoRA tensor name: {key}')
    for down_suffix, up_suffix in pairs:
        for key in state:
            if key.endswith(up_suffix) and key[:-len(up_suffix)] + down_suffix not in state:
                raise ValueError(f'incomplete LoRA factor pair: {key[:-len(up_suffix)]}')
        for key, down in state.items():
            if not key.endswith(down_suffix):
                continue
            base = key[:-len(down_suffix)]
            up_key = base + up_suffix
            if up_key not in state:
                raise ValueError(f'incomplete LoRA factor pair: {base}')
            up = state[up_key]
            if len(down.shape) < 2 or len(up.shape) < 2 or down.shape[0] != up.shape[1]:
                raise ValueError(f'invalid LoRA factor shapes: {base}')
            # A1111 does not apply dyn_dim to CP-decomposed LoCon.
            if base + 'lora_mid.weight' in state:
                continue
            old = down.shape[0]
            new = len(range(old)[:rank])
            if new == old:
                continue
            if new == 0:
                # PEFT rejects rank-0 tensors; a zero down factor has the same
                # additive weight delta while retaining the original valid shape.
                out[key] = down.new_zeros(down.shape)
                continue
            out[key] = down[:new].contiguous()
            out[up_key] = up[:, :new].contiguous()
            alpha_key = base + 'alpha'
            if alpha_key in state:
                out[alpha_key] = state[alpha_key] * (new / old)
    return out


def align_dora_magnitude_shapes(pipe, state):
    """Map A1111's channel vector to PEFT's actual linear/conv parameter shape."""
    out = dict(state)
    suffix = '.lora_magnitude_vector.weight'
    for key, magnitude in state.items():
        if not key.endswith(suffix):
            continue
        component, _, name = key[:-len(suffix)].partition('.')
        model = getattr(pipe, component, None)
        if model is None or not name:
            raise ValueError(f'DoRA magnitude target is unavailable: {key}')
        # Diffusers removes its attention-processor namespace and maps the
        # output projection to the first module of the to_out sequence.
        if component == 'unet':
            name = name.replace('.processor.', '.')
            if name.endswith('.to_out'):
                name += '.0'
        elif component in ('text_encoder', 'text_encoder_2'):
            for original, replacement in (('.to_k', '.k_proj'), ('.to_q', '.q_proj'),
                                          ('.to_v', '.v_proj'), ('.to_out', '.out_proj')):
                if name.endswith(original):
                    name = name[:-len(original)] + replacement
                    renamed = component + '.' + name + suffix
                    if renamed in out and renamed != key:
                        raise ValueError(f'duplicate DoRA magnitude target: {renamed}')
                    del out[key]
                    key = renamed
                    break
        try:
            weight = model.get_submodule(name).weight
        except (AttributeError, KeyError) as exc:
            raise ValueError(f'DoRA magnitude target is unavailable: {key}') from exc
        if weight.ndim not in (2, 4) or magnitude.numel() != weight.shape[0]:
            raise ValueError(f'DoRA magnitude shape does not match target: {key}')
        shape = (weight.shape[0],) if weight.ndim == 2 else (1, weight.shape[0], 1, 1)
        out[key] = magnitude.reshape(shape)
    return out


def load_kohya_dora(pipe, state, adapter_name):
    """Use diffusers' converter before its pipeline loader can discard dora_scale."""
    from diffusers.loaders.lora_conversion_utils import (
        _convert_non_diffusers_lora_to_diffusers, _maybe_map_sgm_blocks_to_diffusers)
    data = dict(state)
    if getattr(pipe, 'text_encoder_2', None) is not None:
        data = _maybe_map_sgm_blocks_to_diffusers(data, pipe.unet.config)
    converted, alphas = _convert_non_diffusers_lora_to_diffusers(data)
    if not any('lora_magnitude_vector' in key for key in converted):
        raise ValueError('DoRA magnitude vectors were lost during conversion')
    converted = align_dora_magnitude_shapes(pipe, converted)
    pipe.load_lora_into_unet(converted, network_alphas=alphas, unet=pipe.unet,
        adapter_name=adapter_name, _pipeline=pipe, low_cpu_mem_usage=False)
    for prefix, encoder in (('text_encoder', getattr(pipe, 'text_encoder', None)),
                            ('text_encoder_2', getattr(pipe, 'text_encoder_2', None))):
        if encoder is not None:
            pipe.load_lora_into_text_encoder(converted, network_alphas=alphas, text_encoder=encoder,
                prefix=prefix, lora_scale=pipe.lora_scale, adapter_name=adapter_name,
                _pipeline=pipe, low_cpu_mem_usage=False)
