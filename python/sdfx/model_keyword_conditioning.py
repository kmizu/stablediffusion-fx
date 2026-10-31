"""Batch-aware conditioning support for Model Keyword."""


def encode_batch(prompts, encode_one, empty, total, make_schedule):
    from .neutral_prompt import parse
    if not prompts:
        raise ValueError('at least one prompt is required')
    items = [encode_one(prompt) for prompt in prompts]
    if any('_neutral' in item for item in items):
        return collate_neutral(items, [parse(text) for text in prompts], empty, total, make_schedule)
    timelines = [getattr(item, 'schedule', None) or [(total, item)] for item in items]
    variants = collate_schedules(timelines, empty)
    return variants[0][1] if len(variants) == 1 else make_schedule(variants)


def collate_neutral(items, trees, empty, total, make_schedule):
    from .model_keyword import assemble_neutral
    from .neutral_runtime import Plan
    if not isinstance(total, int) or isinstance(total, bool) or total < 1:
        raise ValueError('conditioning schedule total must be positive')
    def combine(group):
        timelines = [getattr(item, 'schedule', None) or [(total, item)] for item in group]
        variants = collate_schedules(timelines, empty)
        return variants[0][1] if len(variants) == 1 else make_schedule(variants)
    image_trees, branches, phi = assemble_neutral(items, trees, combine)
    result = dict(branches[0])
    result['_neutral'] = Plan(image_trees[0], branches, phi, image_trees=image_trees)
    return result


def collate_schedules(timelines, empty):
    from . import negpip
    from .model_keyword import merge_schedules
    if not timelines or any(not timeline for timeline in timelines):
        raise ValueError('at least one nonempty image schedule is required')
    flat = [(end, value) for timeline in timelines for end, value in timeline]
    target_length = max(value[key].shape[1] for _, value in flat
                        for key in ('prompt_embeds', 'negative_prompt_embeds'))
    if any(value.get('_negpip') for _, value in flat):
        aligned = iter(negpip.align_variants(flat))
        timelines = [[next(aligned) for _ in timeline] for timeline in timelines]
    return merge_schedules(timelines, lambda items: collate_plain(items, empty, target_length))


def collate_plain(items, empty, target_length=None):
    import torch
    from . import negpip
    if not items or any('_neutral' in item or getattr(item, 'schedule', None) for item in items):
        raise ValueError('expected nonempty plain conditioning items')
    context = ('prompt_embeds', 'negative_prompt_embeds')
    pooled = ('pooled_prompt_embeds', 'negative_pooled_prompt_embeds')
    routing = ('_clora_branch', '_clora_prompt', '_clora_negative')
    photo_raw = ('_photo_raw_branch', '_photo_raw_prompt', '_photo_raw_negative')
    allowed = set(context + pooled + routing + photo_raw + ('_negpip',))
    if any(set(item) - allowed for item in items):
        raise ValueError('unknown conditioning keys')
    if any(not all(key in item for key in context) for item in items):
        raise ValueError('missing conditioning tensor')
    if any(any((key in item) != (key in items[0]) for item in items) for key in pooled):
        raise ValueError('inconsistent pooled conditioning')
    if any(any((key in item) != (key in items[0]) for item in items) for key in routing):
        raise ValueError('inconsistent Composable LoRA routing')
    marked = any(item.get('_negpip') for item in items)
    if marked:
        aligned = negpip.align_variants([(i + 1, item) for i, item in enumerate(items)])
        items = [item for _, item in aligned]
    else:
        length = max(target_length or 0, *(item[key].shape[1] for item in items for key in context))
        padded = []
        empties = empty if isinstance(empty, (list, tuple)) else [empty] * len(items)
        if len(empties) != len(items):
            raise ValueError('one empty conditioning tensor is required per image')
        for index, (item, item_empty) in enumerate(zip(items, empties)):
            item = dict(item)
            for key in context:
                tensor = item[key]
                missing = length - tensor.shape[1]
                if missing:
                    fill_empty = item_empty(index, item, key) if callable(item_empty) else item_empty
                    if missing % fill_empty.shape[1]:
                        raise ValueError('conditioning lengths are not compatible with empty tokens')
                    fill = torch.zeros_like(fill_empty) if not bool(tensor.abs().any()) else fill_empty
                    item[key] = torch.cat([tensor, fill.to(tensor.device, tensor.dtype).repeat(tensor.shape[0], missing // fill.shape[1], 1)], dim=1)
            padded.append(item)
        items = padded
    out = {key: torch.cat([item[key] for item in items], dim=0)
           for key in context + pooled if key in items[0]}
    if marked:
        out['_negpip'] = True
    for key in routing:
        if key in items[0]:
            out[key] = [item[key] for item in items]
    for key, fallback in zip(photo_raw, routing):
        if any(key in item for item in items):
            if any(key not in item and fallback not in item for item in items):
                raise ValueError('incomplete PhotoMaker raw routing')
            out[key] = [item[key] if key in item else item[fallback] for item in items]
    return out
