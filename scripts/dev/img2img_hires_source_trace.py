"""Isolated upstream call-order regression. No torch/Gradio/models are imported.

Reads retained licensed source bytes; executes only the selected extension methods
and actual A1111 v1.10.1 extra-network dispatch/LoRA methods with model calls stubbed.
This is a host-specific behavioral fixture, not GPU equivalence.
"""
import ast
from contextlib import nullcontext
from copy import copy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace, MethodType


def main():
    root = Path(__file__).resolve().parents[2]
    source = root / 'outputs/img2img-hires-source'
    hashes = {
        'img2img_hires_fix.py': '23350ab43d5008c06d5dc156c9e612bd40ae91d651f9ebe04af1b048a84963ad',
        'a1111-extra_networks.py': '437ae9fda77131addbfb2f375a46f6f818a841c3ac45edd41982f5817ce1e69d',
        'a1111-extra_networks_lora.py': 'e7760213744594e87a1b906ce3bfd123abc1a5dcc2b0f9f07c37a6c629080fb8',
    }
    for name, expected in hashes.items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected, name
    modules = ModuleType('modules')
    shared = SimpleNamespace(opts=SimpleNamespace(sd_lora='None', lora_add_hashes_to_infotext=False,
        extra_networks_default_multiplier=.2), sd_model=object(), state=SimpleNamespace(nextjob=lambda: None))
    modules.shared = shared
    errors = []
    modules.errors = SimpleNamespace(display=lambda error, message: errors.append((str(error), message)))
    sys.modules['modules'] = modules
    extra = ModuleType('modules.extra_networks')
    exec(compile((source / 'a1111-extra_networks.py').read_text(), str(source / 'a1111-extra_networks.py'), 'exec'), extra.__dict__)
    modules.extra_networks = extra
    sys.modules['modules.extra_networks'] = extra
    networks = ModuleType('networks')
    networks.available_networks = {'A': object()}
    active, trace = [], []
    def load(names, te, unet, dyn):
        active[:] = names
        trace.append(dict(event='load', names=list(names), te=list(te), unet=list(unet)))
    networks.load_networks = load
    sys.modules['networks'] = networks
    lora_module = {}
    exec(compile((source / 'a1111-extra_networks_lora.py').read_text(), str(source / 'a1111-extra_networks_lora.py'), 'exec'), lora_module)
    lora = lora_module['ExtraNetworkLora']()
    original_deactivate = lora.deactivate
    def deactivate(p):
        trace.append(dict(event='deactivate-before', active=list(active)))
        original_deactivate(p)
        trace.append(dict(event='deactivate-after', active=list(active)))
    lora.deactivate = deactivate
    extra.register_extra_network(lora)
    extra.register_extra_network_alias(lora, 'lyco')
    tree = ast.parse((source / 'img2img_hires_fix.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'I2IHiresFix')
    generate = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_generate_image')
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == '_process_prompt'
               for n in ast.walk(generate))
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and
                n.name in ('__init__', 'postprocess_image', '_process_prompt')]
    def condition(model, prompts, steps):
        trace.append(dict(event='conditioning', active=list(active), prompts=prompts))
        return None
    namespace = dict(scripts=SimpleNamespace(Script=object), copy=copy, extra_networks=extra, shared=shared,
        devices=SimpleNamespace(autocast=nullcontext), sd_models=SimpleNamespace(apply_token_merging=lambda *a: None),
        prompt_parser=SimpleNamespace(get_multicond_learned_conditioning=condition, get_learned_conditioning=condition))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source / 'img2img_hires_fix.py'), 'exec'), namespace)
    cases = {}
    for name, pos, neg, base, default, expected in (
        ('same-positive-negative', 'cat <lora:A:.5>', 'bad <lora:A:1>', '', 'None', ['A']),
        ('negative-only-name', 'cat <lora:B:.5>', 'bad <lora:A:1>', '', 'None', ['B']),
        ('inherited-positive', '', 'bad <lora:A:1>', 'cat <lora:A:.5>', 'None', ['A']),
        ('default-manual-analogue', 'cat <lora:B:.5>', 'bad <lora:A:1>', '', 'A', ['B', 'A']),
        ('malformed-negative-weight', 'cat <lora:A:.5>', 'bad <lora:A:not-a-number>', '', 'None', ['A']),
        ('unclosed-negative-tag', 'cat <lora:A:.5>', 'bad <lora:A', '', 'None', ['A']),
        ('negative-lyco-alias', 'cat <lora:A:.5>', 'bad <lyco:A:1>', '', 'None', ['A']),
    ):
        trace.clear(); active.clear(); errors.clear()
        shared.opts.sd_lora = default
        instance = namespace['I2IHiresFix']()
        instance.prompt, instance.negative_prompt, instance.steps = pos, neg, 10
        sampled = []
        def generate_image(self, image):
            # Exact _generate_image calls _process_prompt before VAE/model sampling;
            # only expensive tensor/image work is substituted here.
            self._process_prompt(512, 512)
            sampled[:] = active
            trace.append(dict(event='sample', active=list(active)))
            return image
        instance._generate_image = MethodType(generate_image, instance)
        p = SimpleNamespace(prompt=base, negative_prompt='base negative', disable_extra_networks=False,
            scripts=None, all_prompts=[base], sd_model=shared.sd_model, get_token_merging_ratio=lambda **kw: 0)
        instance.postprocess_image(p, SimpleNamespace(image='fixture'), True, *([None] * 11))
        assert sampled == expected, (name, sampled, expected)
        assert not errors, (name, errors)
        cases[name] = list(trace)
    report = dict(status='passed', extension='b8930527a8cc606dd79dd54ed23bc967314d1859',
        host='AUTOMATIC1111/stable-diffusion-webui@82a973c04367123ae98bd9abdf80d9eda9b910e2 (v1.10.1)',
        boundary='Extension does not pin a host; this fixture uses a specified compatible host.',
        hashes=hashes, cases=cases)
    (root / 'outputs/img2img-hires-source-trace.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PASS 7 source call traces: negative tags do not suppress named LoRAs on A1111 v1.10.1')


if __name__ == '__main__':
    main()
