"""Per-job, per-branch native adapter routing; all installed hooks are reversible."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

from ..hooks import hook_config, coerce_bool
from ..protocol import EngineError, INVALID_PARAMS, UNSUPPORTED
from . import controllers as ctl, patches, tags


def clean_prompt(text):
    # The source controller parser already protects escaped colons/brackets. Remove
    # the metadata there, then restore those escapes for native prompt processing.
    escaped = ctl.escape_prompt(text)
    escaped = ctl.re_super_cmd.sub('', escaped)
    for code, char in (('005B', '['), ('005D', ']'), ('003A', ':'), ('005C', '\\')):
        escaped = escaped.replace('\\u' + code, '\\' + char)
    return tags.TAG.sub('', escaped)


def clock_total(gp, config, actual):
    if coerce_bool(config.get('hires_global_step', False)) and gp.hires.enabled and gp.img2img is None:
        return gp.steps + (gp.hires.second_pass_steps or gp.steps)
    return actual


def branch_weights(layout, texts, prompts, negatives, calculate):
    """Weights in guidance-group-major, then per-image order."""
    fields = (texts, prompts, negatives)
    sizes = {len(value) for value in fields if isinstance(value, (list, tuple))}
    if len(sizes) > 1 or 0 in sizes:
        raise ValueError('per-image Composable LoRA branches must have equal nonzero lengths')
    count = next(iter(sizes), 1)
    def at(value, index):
        return value[index] if isinstance(value, (list, tuple)) else value
    return [calculate(at(texts, i), not positive, at(prompts, i), at(negatives, i))
            for positive in layout for i in range(count)]


def merge_signed_contexts(positive, negative, align):
    """Combine independently routed encodings without losing NegPiP markers."""
    if positive.get('_negpip') or negative.get('_negpip'):
        (_, positive), (_, negative) = align([(0, positive), (0, negative)])
    result = {**positive, **{key: value for key, value in negative.items() if key.startswith('negative_')}}
    if positive.get('_negpip') or negative.get('_negpip'):
        result['_negpip'] = True
    return result


def _photo_hook_target(module, photomaker):
    # PhotoMaker can inject a PEFT wrapper after this job attaches its hooks.
    # Keep their order identical when a later job sees an existing wrapper.
    if photomaker and hasattr(module, 'get_base_layer'):
        return module.get_base_layer()
    return module


class Job:
    def __init__(self, pipe, gp, req, ctx):
        self.pipe, self.gp, self.req, self.ctx = pipe, gp, req, ctx
        self.config = hook_config(gp.hooks, 'composable_lora', {}) or {}
        self.state = None
        self.handles, self.adapters, self.compiled, self.trace, self.pass_info = [], {}, {}, {}, {}
        self.with_step = coerce_bool(self.config.get('with_step', False))
        from .. import neutral_prompt
        self.manual = getattr(req, 'manual_loras', None)
        self.explicit = ({spec.name for spec in gp.loras} - set(self._tag_weights(gp.prompt + ' ' + gp.hires.prompt))) if self.manual is None else set()
        try:
            for spec in self.manual or []:
                if spec.lbw or spec.start is not None or spec.stop is not None:
                    raise EngineError(UNSUPPORTED, 'Composable LoRA manual Block Weight/start/stop fields are not supported')
                if not all(math.isfinite(v) for v in (spec.weight, spec.unet if spec.unet is not None else spec.weight)):
                    raise ValueError('Composable LoRA manual weights must be finite')
                if spec.name not in {item.name for item in gp.loras}:
                    raise ValueError(f'Composable LoRA manual adapter is not resolved: {spec.name}')
            for spec in gp.loras:
                key = (spec.name, spec.dyn)
                if key in self.adapters:
                    continue
                if spec.lbw or spec.start is not None or spec.stop is not None:
                    raise EngineError(UNSUPPORTED, 'Composable LoRA with native LoRA Block Weight/start/stop fields is not supported; use its step controllers or cmd layer_name')
                if self.manual is None and spec.unet is not None and spec.weight == 0 and spec.unet != 0:
                    raise EngineError(UNSUPPORTED, 'Composable LoRA requires nonzero TE weight when using a separate UNet multiplier')
                path = req.lora_paths.get(spec.name)
                if not path or not Path(path).is_file():
                    raise ValueError(f'Composable LoRA file not found: {spec.name}')
                loaded = patches.load(path, pipe, spec.dyn) if spec.dyn is not None else patches.load(path, pipe)
                adapter = SimpleNamespace(name=spec.name, path=path, multiplier=spec.weight,
                    te_multiplier=spec.weight, unet_multiplier=spec.unet if spec.unet is not None else spec.weight,
                    modules={name: patch for _, name, _, patch in loaded}, spec=spec)
                self.adapters[key] = adapter
                photo = hook_config(gp.hooks, 'photomaker', {}) is not None
                for component, name, module, patch in loaded:
                    patch.name = spec.name
                    target = _photo_hook_target(module, photo and component == 'unet')
                    self.handles.append(target.register_forward_hook(self._hook(component, name, adapter, patch)))
            self._validate_tags(gp.prompt + gp.hires.prompt)
            if self.adapters and hook_config(gp.hooks, 'photomaker', {}) is not None:
                raise EngineError(UNSUPPORTED, 'Composable LoRA integration is not available with photomaker (pending real pipeline parity)')
            # Parse/validate code before any text encoding or denoising starts.
            if self.with_step:
                for branch in neutral_prompt.leaves(neutral_prompt.parse(gp.prompt)):
                    self._controllers(branch.text)
        except BaseException:
            self.remove()
            raise

    def _validate_tags(self, text):
        self._tag_weights(text)
        for match in tags.TAG.finditer(text):
            name = match[2].split(':')[0].strip()
            if not any(adapter.name == name for adapter in self.adapters.values()):
                raise ValueError(f'Composable LoRA tag has no resolved adapter: {name}')

    def _controllers(self, text):
        if self.manual is not None:
            text = tags.component_prompt(text)
        if text not in self.compiled:
            self.compiled[text] = ctl.parse_step_rendering_syntax(text)[0]
        return self.compiled[text]

    def _tag_weights(self, text):
        if self.manual is not None:
            text = tags.component_prompt(text)
        values = {}
        for match in tags.TAG.finditer(text):
            parts = match[2].split(':')
            value = float(parts[1]) if len(parts) > 1 and parts[1] else 1.
            if not math.isfinite(value):
                raise ValueError('Composable LoRA tag weights must be finite')
            values[parts[0].strip()] = (match[1], value)
        return values

    @contextmanager
    def using(self, text, layout, step, total, *, phase='unet', pass_name='base', full_prompt=None, negative_prompt=None):
        previous = self.state
        full_prompt = full_prompt if full_prompt is not None else ((self.gp.hires.prompt or self.gp.prompt) if pass_name == 'hires' else self.gp.prompt)
        negative_prompt = negative_prompt if negative_prompt is not None else ((self.gp.hires.negative_prompt or self.gp.negative_prompt) if pass_name == 'hires' else self.gp.negative_prompt)
        self.state = (text, layout, step, total, phase, pass_name, full_prompt, negative_prompt)
        try:
            yield
        finally:
            self.state = previous

    def _weight(self, adapter, patch, layer, text, negative, step, total, phase, full_prompt, negative_prompt):
        # New hosts identify manual references independently of adapter discovery.
        # Select the tag's component weight before parsing controllers so zero TE
        # and nonzero UNet weights need no division and schedules stay scoped.
        variants = [a for a in self.adapters.values() if a.name == adapter.name]
        if len(variants) > 1:
            from ..model_keyword import parse_lora_activation
            def select_rank(match):
                if match[2].split(':', 1)[0].strip() != adapter.name:
                    return match[0]
                parsed = parse_lora_activation(match[2].split(':'))
                return '' if parsed['dyn'] != adapter.spec.dyn else match[0]
            text, full_prompt = tags.TAG.sub(select_rank, text), tags.TAG.sub(select_rank, full_prompt)
        if self.manual is not None:
            text, full_prompt = (tags.component_prompt(prompt, unet=phase == 'unet') for prompt in (text, full_prompt))
        spec = adapter.spec
        if full_prompt not in self.pass_info:
            from .. import neutral_prompt
            single = len(neutral_prompt.leaves(neutral_prompt.parse(full_prompt))) == 1
            self.pass_info[full_prompt] = (single, self._tag_weights(full_prompt))
        single, global_weights = self.pass_info[full_prompt]
        use_negative = coerce_bool(self.config.get('uc_text_model_encoder' if phase != 'unet' else 'uc_diffusion_model', False))
        use_negative = use_negative or (single and not coerce_bool(self.config.get('single_no_uc', False)))
        kind, global_weight = global_weights.get(adapter.name, ('lora', spec.weight if adapter.name in self.explicit else 0.))
        if negative and not use_negative:
            return 0.
        if negative and (phase != 'unet' or not single):
            value = global_weight
        else:
            owner = full_prompt if negative else text
            weights = self._tag_weights(owner)
            if self.with_step:
                context = dict(is_negative=negative, lora=adapter, lora_module=patch,
                    lora_type=kind, lora_name=adapter.name, lora_count=len(self.adapters),
                    block_lora_count=len(weights), layer_name=layer,
                    current_prompt=negative_prompt if negative else text, sd_processing=self.gp)
                values = [ctl.check_lora_weight(self._controllers(owner), f'{prefix}:{adapter.name}', step, total, context)
                          for prefix in ('lora', 'lyco')]
                value = max(values, key=abs)
            else:
                value = weights.get(adapter.name, ('lora', 0.))[1]
            if adapter.name in self.explicit and adapter.name not in weights:
                value = spec.weight
        if self.manual is None and phase == 'unet' and spec.unet is not None:
            value *= spec.unet / spec.weight if spec.weight else 0.
        if self.manual is not None:
            value += sum(item.unet if phase == 'unet' and item.unet is not None else item.weight
                         for item in self.manual if item.name == adapter.name and item.dyn == spec.dyn)
        if not math.isfinite(value):
            raise ValueError('Composable LoRA controller weight must be finite')
        return value

    def _hook(self, component, layer, adapter, patch):
        def apply(module, args, output):
            import torch
            if self.state is None:
                return output
            text, layout, step, total, phase, pass_name, full_prompt, negative_prompt = self.state
            if (component == 'unet') != (phase == 'unet'):
                return output
            self.ctx.check_interrupt()
            try:
                weights = branch_weights(layout, text, full_prompt, negative_prompt,
                    lambda branch, negative, full, neg: self._weight(adapter, patch, f'{component}.{layer}',
                        branch, negative, step, total, phase, full, neg))
            except (ValueError, OverflowError, ZeroDivisionError, KeyError) as exc:
                raise EngineError(INVALID_PARAMS, f'Composable LoRA {adapter.name}: {exc}') from exc
            if not weights or output.shape[0] % len(weights):
                raise EngineError(UNSUPPORTED, f'Composable LoRA unsupported layer batch at {layer}')
            multiplier = output.new_tensor(weights).repeat_interleave(output.shape[0] // len(weights))
            multiplier = multiplier.view(-1, *([1] * (output.ndim - 1)))
            if phase == 'unet' and coerce_bool(self.config.get('plot_weights', False)):
                count = len(weights) // len(layout)
                for group, positive in enumerate(layout):
                    for item in range(count):
                        value = weights[group * count + item]
                        branch = text[item] if isinstance(text, (list, tuple)) else text
                        variants = sum(a.name == adapter.name for a in self.adapters.values())
                        label = f'{adapter.name} (dyn={adapter.spec.dyn})' if variants > 1 else adapter.name
                        key = (pass_name, step, branch, not positive, label)
                        record = self.trace.setdefault(key, [0., 0, value, value])
                        record[0] += value
                        record[1] += 1
                        record[2], record[3] = min(record[2], value), max(record[3], value)
            if not any(weights):
                return output
            delta = patch.delta(args[0].to(output.dtype)).to(output.dtype)
            return output + delta * multiplier
        return apply

    def encode(self, engine, pos, neg, gp, bs, steps, hires_total=None, gradient_total=None):
        import torch
        from .. import neutral_prompt, neutral_runtime
        from ..engine import ScheduledEmbeds
        tree = neutral_prompt.parse(pos)
        if hook_config(gp.hooks, 'neutral_prompt', {}) is None and any(n.kind != 'AND' for n in neutral_prompt.leaves(tree)):
            raise EngineError(UNSUPPORTED, 'Enable Neutral Prompt to use its extended AND operators with Composable LoRA')
        nodes = neutral_prompt.leaves(tree)
        if not self.adapters and len(nodes) == 1:
            return None
        from .. import negpip
        sub = replace(gp, hooks={k: v for k, v in gp.hooks.items() if k not in ('composable_lora', 'neutral_prompt')})
        total = max(1, gradient_total or hires_total or steps)
        prepare_total = clock_total(gp, self.config, total)
        pass_name = 'hires' if hires_total is not None else 'base'
        branches = []
        for node in nodes:
            self._validate_tags(node.text)
            source = node.text
            with self.using(source, [True], -1, prepare_total, phase='text_encoder', pass_name=pass_name, full_prompt=pos, negative_prompt=neg):
                positive = engine._encode_scheduled(clean_prompt(source), '', sub, bs, steps, hires_total, gradient_total)
            with self.using(source, [False], -1, prepare_total, phase='text_encoder', pass_name=pass_name, full_prompt=pos, negative_prompt=neg):
                negative = engine._encode_scheduled('', clean_prompt(neg), sub, bs, steps, hires_total, gradient_total)
            ps, ns = getattr(positive, 'schedule', [(total, positive)]), getattr(negative, 'schedule', [(total, negative)])
            variants = []
            for end in sorted({e for e, _ in ps} | {e for e, _ in ns}):
                p = next((d for e, d in ps if end <= e), ps[-1][1])
                n = next((d for e, d in ns if end <= e), ns[-1][1])
                d = merge_signed_contexts(p, n, negpip.align_variants)
                # Long chunks are padded separately under the correct text-encoder routing.
                pe, ne = d['prompt_embeds'], d['negative_prompt_embeds']
                if not d.get('_negpip') and pe.shape[1] != ne.shape[1]:
                    longest = max(pe.shape[1], ne.shape[1])
                    for key, x, positive_flag in (('prompt_embeds', pe, True), ('negative_prompt_embeds', ne, False)):
                        if x.shape[1] < longest:
                            with self.using(source, [positive_flag], -1, prepare_total, phase='text_encoder', pass_name=pass_name, full_prompt=pos, negative_prompt=neg):
                                empty = engine._encode('', '', sub, bs)[key]
                            fill = torch.zeros_like(empty) if not bool(x.abs().any()) else empty
                            d[key] = torch.cat([x] + [fill] * ((longest - x.shape[1]) // fill.shape[1]), dim=1)
                d['_clora_branch'] = source
                d['_clora_prompt'], d['_clora_negative'] = pos, neg
                variants.append((end, d))
            if any(value.get('_negpip') for _, value in variants):
                variants = negpip.align_variants(variants)
            branch = ScheduledEmbeds(variants[0][1])
            branch.schedule = variants
            branches.append(branch)
        out = dict(branches[0])
        config = hook_config(gp.hooks, 'neutral_prompt', {'cfg_rescale': 0}) or {'cfg_rescale': 0}
        out['_neutral'] = neutral_runtime.Plan(tree, branches, float(config['cfg_rescale']))
        return out

    def remove(self):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.state = None
        for adapter in self.adapters.values():
            for patch in adapter.modules.values():
                patch.cache.clear()

    def finish(self, result, output_dir):
        if not coerce_bool(self.config.get('plot_weights', False)):
            return
        from PIL import Image, ImageDraw, ImageFont
        from ..images import next_output_path, save_png
        records = [dict(pass_name=k[0], step=k[1], prompt=k[2], is_negative=k[3], lora=k[4],
                        mean_weight=v[0] / v[1], min_weight=v[2], max_weight=v[3], layer_evaluations=v[1])
                   for k, v in self.trace.items()]
        path = next_output_path(output_dir, 'composable-lora-weights')
        image = Image.new('RGB', (1200, 680), '#17212b')
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=15)
        for candidate in ('C:/Windows/Fonts/meiryo.ttc', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'):
            if Path(candidate).is_file():
                font = ImageFont.truetype(candidate, 15)
                break
        draw.text((24, 16), 'Composable LoRA: effective UNet weights per sampling step', fill='white', font=font)
        draw.text((24, 42), 'Mean over patched layers; exact layer ranges and branch prompts are in the saved trace JSON.', fill='#aebdcc', font=font)
        grouped = {}
        for r in records:
            grouped.setdefault((r['pass_name'], r['prompt'], r['is_negative'], r['lora']), []).append(r)
        colors = ('#80c7ff', '#efad72', '#9ed792', '#dba8eb', '#ed819a', '#f3dc83')
        minimum = min([0.] + [r['mean_weight'] for r in records])
        maximum = max([1.] + [r['mean_weight'] for r in records])
        def y(value):
            return 480 - (value - minimum) / max(maximum - minimum, 1e-9) * 390
        draw.line((60, 80, 60, 480), fill='#6a7b8c')
        draw.line((60, y(0), 1160, y(0)), fill='#6a7b8c')
        for i in range(6):
            value = minimum + (maximum - minimum) * i / 5
            draw.text((8, y(value) - 6), f'{value:.2f}', fill='#aebdcc', font=font)
        for index, (key, data) in enumerate(grouped.items()):
            data = sorted(data, key=lambda r: r['step'])
            last = max(r['step'] for r in data)
            points = [(60 + (r['step'] - 1) / max(last - 1, 1) * 1100, y(r['mean_weight'])) for r in data]
            color = colors[index % len(colors)]
            if len(points) > 1:
                draw.line(points, fill=color, width=2)
            for x, yy in points:
                draw.ellipse((x - 2, yy - 2, x + 2, yy + 2), fill=color)
            if index < 8:
                label = f'{key[0]} / {key[3]} / {"negative" if key[2] else "positive"} / {key[1][:65]}'
                while draw.textlength(label, font=font) > 555:
                    label = label[:-2]
                draw.text((24 + (index % 2) * 585, 510 + (index // 2) * 30), label, fill=color, font=font)
        if not records:
            draw.text((60, 130), 'No adapter layer evaluations were recorded.', fill='white', font=font)
        save_png(image, path)
        # Native callers supply their persistent outputs root independently of image temp files.
        # Direct protocol callers retain the same persistent subdirectory under their output root.
        import uuid
        trace_dir = Path(self.config.get('output_dir') or output_dir) / 'intermediates' / 'composable-lora'
        trace_dir.mkdir(parents=True, exist_ok=True)
        manifest = str(trace_dir / (uuid.uuid4().hex + '.json'))
        Path(manifest).write_text(json.dumps(dict(description='Actual effective adapter weights; step restarts are separated by pass_name',
                                              records=records), indent=2), encoding='utf-8')
        result['composable_lora_trace'] = dict(path=manifest, plot=path, records=records)
        first = result['images'][0] if result['images'] else dict(seed=0, subseed=0)
        result['images'].append(dict(path=path, width=image.width, height=image.height,
                                     seed=first['seed'], subseed=first.get('subseed', 0)))


class BranchHook:
    priority = 10

    def __init__(self, job, scheduler, total, offset=0, pass_name='base'):
        self.job, self.scheduler = job, scheduler
        self.total, self.offset, self.pass_name = total, offset, pass_name
        self.branch, self.step, self.context = None, 1, None

    def set_branch(self, index, branch, step):
        self.branch, self.step = branch['_clora_branch'], step
        self.full_prompt, self.negative_prompt = branch.get('_clora_prompt'), branch.get('_clora_negative')

    def before_branch(self, unet, inputs, timestep, kwargs, layout):
        self.context = self.job.using(self.branch, layout, self.step + self.offset,
                                      self.total, pass_name=self.pass_name, full_prompt=self.full_prompt,
                                      negative_prompt=self.negative_prompt)
        self.context.__enter__()

    def after_branch(self):
        if self.context is not None:
            self.context.__exit__(None, None, None)
            self.context = None


def inversion_branch_hook(job, plan, scheduler, gp, steps):
    """Apply the same routed adapters to each branch of DDIM inversion."""
    if job is None or not plan.branches or not all('_clora_branch' in branch for branch in plan.branches):
        return None
    return BranchHook(job, scheduler, clock_total(gp, job.config, steps), pass_name='inversion')
