"""Scoped PhotoMaker v1 conditioning bridge for one denoising batch.

The upstream v1 pipeline's trigger-token and ID fusion code assumes one
prompt. Run those operations for each prompt while retaining the pipeline's
single batched sampler invocation. Nothing is changed on the pipeline after
the context exits, including on cancellation or error.
"""
from contextlib import contextmanager


class PhotoScheduledEmbeds(dict):
    """Conditioning mapping with a Neutral-compatible timeline, without importing torch."""


def signed_photo_conditioning(text, identity, positive_terms, negative_terms):
    """Append NegPiP value signs only after PhotoMaker has fused identity.

    Terms are already encoded per image; callers retain their own encoder and
    Composable routing scopes. The still-guarded bridge does not invoke this
    until those terms can be produced with source-compatible token alignment.
    """
    import torch
    from .negpip import _mark, _pad
    batch = text['prompt_embeds'].shape[0]
    if len(positive_terms) != batch or len(negative_terms) != batch:
        raise ValueError('PhotoMaker signed term batch dimensions disagree')
    def rows(base, terms):
        if base.ndim != 3 or base.shape[0] != batch:
            raise ValueError('PhotoMaker signed conditioning dimensions disagree')
        result = []
        for i, per_image in enumerate(terms):
            parts = [_mark(base[i:i + 1])]
            for term in per_image:
                if term.ndim != 2 or term.shape[1] != base.shape[2]:
                    raise ValueError('PhotoMaker signed term width disagrees')
                parts.append(_mark(term[None].to(device=base.device, dtype=base.dtype), -1))
            result.append(torch.cat(parts, dim=1))
        return result
    plain_pos = rows(text['prompt_embeds'], positive_terms)
    fused_pos = rows(identity['prompt_embeds'], positive_terms)
    plain_neg = rows(text['negative_prompt_embeds'], negative_terms)
    fused_neg = rows(identity['negative_prompt_embeds'], negative_terms)
    length = max(row.shape[1] for group in (plain_pos, fused_pos, plain_neg, fused_neg) for row in group)
    def join(group):
        return torch.cat([_pad(row, length) for row in group], dim=0)
    plain = dict(text, prompt_embeds=join(plain_pos), negative_prompt_embeds=join(plain_neg), _negpip=True)
    fused = dict(identity, prompt_embeds=join(fused_pos), negative_prompt_embeds=join(fused_neg), _negpip=True)
    return plain, fused


def reject_unverified_merge_composable(texts):
    """Keep the three-feature path closed until real-pipeline acceptance."""
    from .embedding_merge_prompts import expressions
    if any(expressions(text) for text in texts):
        raise ValueError('Embedding Merge with Composable PhotoMaker is not yet supported')


class PhotoBranchCapture:
    """Collect official trigger, plain and ID-fused encodings for every leaf."""
    def __init__(self, specs, groups=None):
        if not specs or any(not isinstance(s.get('prompts'), (list, tuple)) or not s['prompts']
                            or not callable(s.get('route')) or not isinstance(s.get('metadata'), dict) for s in specs):
            raise ValueError('invalid PhotoMaker branch capture specification')
        count = len(specs[0]['prompts'])
        if any(len(s['prompts']) != count for s in specs):
            raise ValueError('PhotoMaker branch batch dimensions disagree')
        self.specs = specs
        self.groups = groups
        self.trigger = self.plain = self.identity = None
        self.plan = None

    def bind(self, plan):
        if len(plan.branches) != (len(self.groups) if self.groups is not None else len(self.specs)):
            raise ValueError('PhotoMaker branch plan dimensions disagree')
        self.plan = plan

    @contextmanager
    def signed_terms(self, encode_target):
        """Temporarily clean and encode signed terms under each image's route."""
        from .negpip import extract
        missing = object()
        keys = ('prompts', 'negatives', 'signed_positive_terms', 'signed_negative_terms')
        saved = [{key: spec.get(key, missing) for key in keys} for spec in self.specs]
        try:
            any_signed = False
            for spec in self.specs:
                positive = [extract(text) for text in spec['prompts']]
                negative = [extract(text) for text in spec['negatives']]
                if len(positive) != len(negative):
                    raise ValueError('PhotoMaker signed guidance batch dimensions disagree')
                terms = []
                for items, is_positive in ((positive, True), (negative, False)):
                    group = []
                    for index, (_, targets) in enumerate(items):
                        encoded = []
                        for text, weight in targets:
                            any_signed = True
                            with spec['route'](index, is_positive):
                                encoded.append(encode_target(text, abs(weight)))
                        group.append(encoded)
                    terms.append(group)
                spec['prompts'] = [item[0] for item in positive]
                spec['negatives'] = [item[0] for item in negative]
                spec['signed_positive_terms'], spec['signed_negative_terms'] = terms
            if not any_signed:
                for spec in self.specs:
                    del spec['signed_positive_terms'], spec['signed_negative_terms']
            yield
        finally:
            for spec, original in zip(self.specs, saved):
                for key, value in original.items():
                    if value is missing:
                        spec.pop(key, None)
                    else:
                        spec[key] = value

    def publish(self):
        if self.plan is not None:
            self.plan.branches[:] = self.branches()

    def branches(self):
        if any(value is None or len(value) != len(self.specs)
               for value in (self.trigger, self.plain, self.identity)):
            raise ValueError('PhotoMaker branch encodings are incomplete')
        keys = ('prompt_embeds', 'negative_prompt_embeds', 'pooled_prompt_embeds', 'negative_pooled_prompt_embeds')
        result = []
        for spec, trigger, plain, identity in zip(self.specs, self.trigger, self.plain, self.identity):
            if len(trigger) != 5 or len(plain) != 4:
                raise ValueError('invalid PhotoMaker branch encoding result')
            if plain[0] is None or plain[1] is None or identity is None:
                raise ValueError('PhotoMaker Composable routing needs both guidance halves')
            if (plain[2] is None) != (plain[3] is None) or (trigger[2] is None) != (plain[3] is None):
                raise ValueError('PhotoMaker pooled guidance encodings disagree')
            text = dict({key: value for key, value in zip(keys, plain) if value is not None}, **spec['metadata'])
            fused = dict({key: value for key, value in zip(keys, (identity, plain[1], trigger[2], plain[3]))
                          if value is not None}, **spec['metadata'])
            if 'signed_positive_terms' in spec or 'signed_negative_terms' in spec:
                if 'signed_positive_terms' not in spec or 'signed_negative_terms' not in spec:
                    raise ValueError('PhotoMaker signed terms require both guidance halves')
                text, fused = signed_photo_conditioning(text, fused, spec['signed_positive_terms'],
                                                        spec['signed_negative_terms'])
            text['_photomaker_id'] = fused
            result.append(text)
        if self.groups is None:
            return result
        assembled = []
        for group in self.groups:
            variants = [(end, result[index]) for end, index in group]
            branch = PhotoScheduledEmbeds(variants[0][1]); branch.schedule = variants
            assembled.append(branch)
        return assembled


def select_merge_stage(text_branch, identity_branch, *, step, start_merge_step):
    """Select a branch using PhotoMaker v1's inclusive raw-iteration boundary.

    ``step`` is one-based pipeline iteration, not a scheduler's logical step.
    """
    if (isinstance(step, bool) or not isinstance(step, int) or step < 1 or
            isinstance(start_merge_step, bool) or not isinstance(start_merge_step, int) or start_merge_step < 0):
        raise ValueError('invalid PhotoMaker merge step')
    for key in ('_clora_branch', '_clora_prompt', '_clora_negative'):
        if text_branch.get(key) != identity_branch.get(key):
            raise ValueError(f'PhotoMaker branch routing disagrees: {key}')
    return text_branch if step - 1 <= start_merge_step else identity_branch


def photo_branch_transform(start_merge_step):
    """Select the prepared branch variants at each raw pipeline iteration."""
    def transform(index, current, raw_step):
        identity = current.get('_photomaker_id')
        if not isinstance(identity, dict):
            raise ValueError(f'missing PhotoMaker identity conditioning for branch {index}')
        text = {key: value for key, value in current.items() if key != '_photomaker_id'}
        return select_merge_stage(text, identity, step=raw_step, start_merge_step=start_merge_step)
    return transform


def composable_text_route(job, branch, *, count, total, pass_name='base'):
    """Bind one collated Neutral branch to PhotoMaker's per-image encoder calls."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('invalid PhotoMaker routing batch size')
    fields = []
    for key in ('_clora_branch', '_clora_prompt', '_clora_negative'):
        value = branch.get(key)
        if isinstance(value, str):
            value = [value] * count
        if not isinstance(value, (tuple, list)) or len(value) != count or any(not isinstance(v, str) for v in value):
            raise ValueError(f'invalid PhotoMaker Composable routing field: {key}')
        fields.append(tuple(value))
    def route(index, positive):
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < count or not isinstance(positive, bool):
            raise ValueError('invalid PhotoMaker Composable routing index or CFG half')
        return job.using(fields[0][index], [positive], -1, total, phase='text_encoder',
                         pass_name=pass_name, full_prompt=fields[1][index], negative_prompt=fields[2][index])
    return route


def prepare_composable_capture(plan, job, *, count, total, pass_name='base', schedule_steps=None):
    """Bind collated branch metadata to official PhotoMaker encoding calls."""
    from .clora.runtime import clean_prompt
    from . import prompt_schedule
    from .embedding_merge_prompts import substitute
    def at_endpoint(text, steps, end):
        raw = clean_prompt(text)
        protected = {}
        def placeholder(span):
            index = len(protected)
            name = f'__nui_photo_expr_{index}__'
            while name in raw or name in protected:
                index += 1; name = f'__nui_photo_expr_{index}__'
            protected[name] = span.original
            return name
        masked = substitute(raw, placeholder)
        selected = prompt_schedule.text_at(prompt_schedule.schedule(masked, steps), end)
        for name, expression in protected.items():
            selected = selected.replace(name, expression)
        return selected
    specs, groups = [], []
    for branch in plan.branches:
        timeline = getattr(branch, 'schedule', None) or [(total, branch)]
        group, previous = [], 0
        for end, variant in timeline:
            if isinstance(end, bool) or not isinstance(end, int) or end <= previous:
                raise ValueError('invalid PhotoMaker branch schedule')
            previous = end
            route = composable_text_route(job, variant, count=count, total=total, pass_name=pass_name)
            values = variant.get('_photo_raw_branch', variant['_clora_branch'])
            negatives = variant.get('_photo_raw_negative', variant['_clora_negative'])
            values = [values] * count if isinstance(values, str) else values
            negatives = [negatives] * count if isinstance(negatives, str) else negatives
            steps = schedule_steps or total
            prompts = [at_endpoint(value, steps, end) for value in values]
            negs = [at_endpoint(value, steps, end) for value in negatives]
            group.append((end, len(specs)))
            specs.append({'prompts': prompts, 'negatives': negs, 'route': route,
                          'metadata': {key: variant[key] for key in ('_clora_branch', '_clora_prompt', '_clora_negative')}})
        groups.append(group)
    capture = PhotoBranchCapture(specs, groups)
    capture.bind(plan)
    return capture


def configure_composable_photo(plan, job, kw, *, count, total, cfg, trigger, pass_name='base', schedule_steps=None):
    """Prepare the text-only PhotoMaker call and its branch capture."""
    import re
    if not isinstance(trigger, str) or not trigger:
        raise ValueError('PhotoMaker trigger must be nonempty')
    capture = prepare_composable_capture(plan, job, count=count, total=total, pass_name=pass_name, schedule_steps=schedule_steps)
    pattern = re.compile(r'\b' + re.escape(trigger) + r'\b')
    if any(not pattern.search(prompt) for spec in capture.specs for prompt in spec['prompts']):
        raise ValueError('every PhotoMaker Composable branch needs the trigger word')
    prompts = list(capture.specs[0]['prompts'])
    negative = capture.specs[0]['negatives']
    kw['prompt'] = prompts if count > 1 else prompts[0]
    kw['negative_prompt'] = negative if count > 1 else negative[0]
    kw['num_images_per_prompt'] = 1
    return capture


def validate_pipeline(pipe):
    import inspect
    try:
        source = inspect.getsource(type(pipe).__call__)
    except (OSError, TypeError) as exc:
        raise ValueError('Cannot inspect PhotoMaker pipeline compatibility') from exc
    required = ('self.tokenizer.encode(prompt,', 'self.encode_prompt_with_trigger_word(',
                'id_pixel_values.unsqueeze(0)', 'self.id_encoder(id_pixel_values, prompt_embeds, class_tokens_mask)',
                'prompt_embeds.repeat(1, num_images_per_prompt, 1)')
    if not all(marker in source for marker in required):
        raise ValueError('PhotoMaker pipeline version is not compatible with heterogeneous prompt batches')


@contextmanager
def heterogeneous_prompts(pipe, prompts, concat=None, route=None, capture=None):
    if len(prompts) < (1 if route is not None else 2) or any(not isinstance(value, str) for value in prompts):
        raise ValueError('PhotoMaker needs a nonempty compatible prompt batch')
    if concat is None:
        import torch
        concat = lambda values: torch.cat(values, dim=0)
    tokenizer, encoder = pipe.tokenizer, pipe.id_encoder
    original_trigger = pipe.encode_prompt_with_trigger_word
    original_plain = pipe.encode_prompt
    original_encode = tokenizer.encode
    original_id = encoder.forward
    patches = []

    def patch(obj, name, value):
        had = name in obj.__dict__
        previous = obj.__dict__.get(name)
        setattr(obj, name, value)
        patches.append((obj, name, had, previous))

    def element(value, index):
        return value[index] if isinstance(value, list) else value

    if capture is not None and (list(capture.specs[0]['prompts']) != list(prompts) or route is not capture.specs[0]['route']):
        raise ValueError('PhotoMaker primary branch does not match capture')

    def routed_call(method, args, index, selected_route):
        # PhotoMaker's encoders compute both CFG halves in a single call.
        # Evaluate twice so each half sees its own Composable text-encoder
        # weights, then retain the appropriate positive/negative output.
        # Neutral composition needs a reference negative even when the outer
        # PhotoMaker pipeline runs unguided. This only changes internal encoder
        # calls; the pipeline's native guidance decision remains untouched.
        args = dict(args, do_classifier_free_guidance=True)
        with selected_route(index, True):
            positive = method(**args)
        with selected_route(index, False):
            negative = method(**args)
        if len(positive) != len(negative) or len(positive) not in (4, 5):
            raise ValueError('PhotoMaker encoder returned an unsupported result')
        return tuple(negative[i] if i in (1, 3) else positive[i] for i in range(len(positive)))

    def combine(outputs):
        if not outputs or any(len(row) != len(outputs[0]) for row in outputs):
            raise ValueError('PhotoMaker encoder result lengths disagree')
        result = []
        for i in range(len(outputs[0])):
            values = [row[i] for row in outputs]
            if any(value is None for value in values) and not all(value is None for value in values):
                raise ValueError('PhotoMaker encoder result presence disagrees')
            result.append(None if values[0] is None else concat(values))
        return tuple(result)

    def trigger(**kwargs):
        if kwargs.get('num_images_per_prompt', 1) != 1:
            raise ValueError('PhotoMaker batch bridge requires one image per prompt')
        def encode(values, selected_route, negatives=None):
            outputs = []
            for index, prompt in enumerate(values):
                args = dict(kwargs, prompt=prompt)
                for name in ('prompt_2', 'negative_prompt', 'negative_prompt_2'):
                    args[name] = element(args.get(name), index)
                if negatives is not None:
                    args['negative_prompt'] = negatives[index]
                    args['negative_prompt_2'] = negatives[index]
                outputs.append(original_trigger(**args) if selected_route is None else routed_call(original_trigger, args, index, selected_route))
            return combine(outputs)
        result = encode(prompts, route)
        if capture is not None:
            capture.trigger = [result] + [encode(spec['prompts'], spec['route'], spec.get('negatives')) for spec in capture.specs[1:]]
        return result

    def text_only(**kwargs):
        token = tokenizer.convert_tokens_to_ids(pipe.trigger_word)
        def stripped(values):
            result = []
            for prompt in values:
                ids = original_encode(prompt, add_special_tokens=False)
                ids.remove(token)
                result.append(tokenizer.decode(ids, add_special_tokens=False))
            return result
        values = stripped(prompts)
        args = dict(kwargs, prompt=values)
        for name in ('negative_prompt', 'negative_prompt_2'):
            if isinstance(args.get(name), str):
                args[name] = [args[name]] * len(prompts)
        def encode_routed(values, selected_route, negatives=None):
            outputs = []
            for index, value in enumerate(values):
                item = dict(args, prompt=value)
                # Official PhotoMaker passes the primary trigger encoding's
                # negatives back into encode_prompt. Recompute them inside
                # this branch's negative routing scope instead of reusing it.
                item['negative_prompt_embeds'] = None
                item['negative_pooled_prompt_embeds'] = None
                for name in ('prompt_2', 'negative_prompt', 'negative_prompt_2'):
                    item[name] = element(item.get(name), index)
                if negatives is not None:
                    item['negative_prompt'] = negatives[index]
                    item['negative_prompt_2'] = negatives[index]
                outputs.append(routed_call(original_plain, item, index, selected_route))
            return combine(outputs)
        if route is not None:
            result = encode_routed(values, route)
            if capture is not None:
                capture.plain = [result] + [encode_routed(stripped(spec['prompts']), spec['route'], spec.get('negatives'))
                                            for spec in capture.specs[1:]]
            return result
        return original_plain(**args)

    def encode(value, *args, **kwargs):
        return original_encode(value[0] if isinstance(value, list) else value, *args, **kwargs)

    def id_forward(images, embeds, mask, *args, **kwargs):
        if len(embeds) != len(prompts) or len(mask) != len(prompts):
            raise ValueError('PhotoMaker prompt and identity batch dimensions disagree')
        def fuse(current, current_mask):
            return concat([original_id(images, current[i:i + 1], current_mask[i:i + 1], *args, **kwargs)
                           for i in range(len(prompts))])
        result = fuse(embeds, mask)
        if capture is not None:
            if capture.trigger is None:
                raise ValueError('PhotoMaker trigger encodings must precede identity fusion')
            capture.identity = [result] + [fuse(value[0], value[4]) for value in capture.trigger[1:]]
            capture.publish()
        return result

    try:
        patch(pipe, 'encode_prompt_with_trigger_word', trigger)
        patch(pipe, 'encode_prompt', text_only)
        patch(tokenizer, 'encode', encode)
        patch(encoder, 'forward', id_forward)
        yield
    finally:
        for obj, name, had, previous in reversed(patches):
            if had:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)
