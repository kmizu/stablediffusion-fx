"""Read-only PEFT/Composable topology snapshots for an opt-in diagnostic."""


def snapshot(unet, captured):
    current = dict(unet.named_modules())
    rows = []
    for component, name, hooked in captured:
        if component != 'unet':
            continue
        module = current.get(name)
        base = module.get_base_layer() if hasattr(module, 'get_base_layer') else module
        weight = getattr(base, 'weight', None)
        scales = getattr(module, 'scaling', {})
        rows.append({
            'name': name,
            'hook_owner_id': id(hooked), 'current_id': id(module) if module is not None else None,
            'base_id': id(base) if base is not None else None,
            'hook_on_current': hooked is module, 'hook_on_base': hooked is base,
            'hook_owner_type': type(hooked).__module__ + '.' + type(hooked).__qualname__,
            'current_type': type(module).__module__ + '.' + type(module).__qualname__,
            'weight_dtype': str(weight.dtype) if weight is not None else None,
            'weight_device': str(weight.device) if weight is not None else None,
            'active_adapters': list(getattr(module, 'active_adapters', [])),
            'disabled': getattr(module, 'disable_adapters', None),
            'scaling': {str(k): float(v) for k, v in scales.items()},
            'hook_count': len(getattr(hooked, '_forward_hooks', {})),
        })
    return rows
