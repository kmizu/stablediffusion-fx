"""Parent-only real-model acceptance: PY SCRIPT ENGINE_PY MODEL OUT [--sdxl] [--allow-code].

Uses local checkpoints/adapters, the standalone JSONL engine and no reForge imports.
Does not install dependencies, edit settings or download adapters. Run from the integrated
repository; one model family per process. Advanced Python is enabled only by --allow-code.
"""
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

from prompt_fusion_check import pixel_record, image_difference, contact_sheet

LORA_DIR = Path(r'D:\stable-diffusion-webui-reForge\models\Lora')
BASE = 'a young woman in a garden, detailed portrait'


def build_cases(sdxl=False, allow_code=False):
    size = 768 if sdxl else 512
    base = dict(prompt=BASE, negative_prompt='low quality, blurry', seed=173, subseed=23,
                steps=8, width=size, height=size, cfg_scale=6, sampler_name='DPM++ 2M',
                scheduler='Automatic', batch_size=1, n_iter=1)
    cases = []
    def add(name, prompt=BASE, adapters=(), config=None, equal_to=None, **overrides):
        hooks = {} if config is None else {'composable_lora': config}
        hooks.update(overrides.pop('hooks', {}))
        params = dict(base, prompt=prompt, hooks=hooks,
                      loras=[dict(name=n, weight=1.) for n in adapters], **overrides)
        cases.append(dict(name=name, params=params, equal_to=equal_to))
    add('baseline')
    add('enabled-no-adapter', config={}, equal_to='baseline')
    add('disabled-no-adapter', config={'enabled': False}, equal_to='baseline')
    and_text = 'a woman with red hair AND a woman with blue hair'
    add('and-no-adapter', and_text, config={})
    add('and-zero-weights', 'a woman with red hair <lora:A:0> AND a woman with blue hair <lora:B:0>',
        ('A', 'B'), {}, equal_to='and-no-adapter')
    add('single-source-default', BASE + ' <lora:A:.8>', ('A',), {})
    add('single-negative-off', BASE + ' <lora:A:.8>', ('A',), {'single_no_uc': True})
    add('single-negative-text', BASE + ' <lora:A:.8>', ('A',), {'single_no_uc': True, 'uc_text_model_encoder': True})
    add('single-negative-denoiser', BASE + ' <lora:A:.8>', ('A',), {'single_no_uc': True, 'uc_diffusion_model': True})
    add('separate-te-unet', BASE + ' <lora:A:1>', ('A',), {'single_no_uc': True})
    cases[-1]['params']['loras'][0]['unet'] = .5
    pair = 'a woman with red hair <lora:A:.8> AND a woman with blue hair <lora:B:.7>'
    add('separate-branches', pair, ('A', 'B'), {})
    add('negative-both', pair, ('A', 'B'), {'uc_text_model_encoder': True, 'uc_diffusion_model': True})
    add('shared-adapter-plot', 'a woman with red hair <lora:A:.25> AND a woman with blue hair <lora:A:.75>',
        ('A',), {'plot_weights': True})
    for name, syntax, adapters in (
        ('stop', '[<lora:A:1>::4]', ('A',)), ('start', '[<lora:A:1>:.5]', ('A',)),
        ('switch', '[<lora:A:1>:<lora:B:1>:4]', ('A', 'B')),
        ('nested-window', '[[<lora:A:1>::6]:2]', ('A',)),
        ('increment', '[<lora:A:1> #increment:2]', ('A',)),
        ('decrease', '[<lora:A:1> #decrease:2]', ('A',)),
        ('math-command', r'[<lora:A:1> #cmd\(sin\(life\) * warmup\(0.25\)\):2]', ('A',)),
        ('escaped-hash', r'[<lora:A:1> \u0023increment:2]', ('A',)),
        ('alternate', '[<lora:A:1>|<lora:B:1>]', ('A', 'B'))):
        add(name, BASE + ' ' + syntax, adapters, {'with_step': True, 'single_no_uc': True})
    if allow_code:
        code = '[<lora:A:1> #cmd\\(\ndef curve\\(\\)\\:\n    return sin\\(life\\)\ncurve\\(\\)\n\\):2]'
        add('python-function', BASE + ' ' + code, ('A',), {'with_step': True, 'single_no_uc': True})
    add('batch', pair, ('A', 'B'), {}, batch_size=2)
    add('heun', BASE + ' [<lora:A:1> #increment]', ('A',), {'with_step': True}, sampler_name='Heun')
    for global_clock in (False, True):
        add('hires-global' if global_clock else 'hires-restart',
            BASE + ' [<lora:A:1> #increment]', ('A',),
            dict(with_step=True, hires_global_step=global_clock, plot_weights=global_clock),
            hires=dict(enabled=True, scale=1.25, upscaler='Latent', second_pass_steps=4, denoising_strength=.6,
                       prompt=BASE + ' [<lora:A:.7> #increment]'))
    add('img2img', BASE + ' [<lora:A:1> #decrease]', ('A',), {'with_step': True},
        img2img=dict(init_image='@baseline', denoising_strength=.6))
    add('fusion', 'a [young woman:woman with blue hair: , 5] in a garden <lora:A:.8>', ('A',), {},
        hooks={'prompt_fusion': {}})
    add('neutral', 'a woman in a garden <lora:A:.8> AND_PERP red flowers <lora:B:.5>', ('A', 'B'), {},
        hooks={'neutral_prompt': {'cfg_rescale': 0}})
    if sdxl:
        add('locon', BASE + ' <lyco:C:.7>', ('C',), {'single_no_uc': True})
        add('lokr', BASE + ' <lyco:K:.7>', ('K',), {'single_no_uc': True})
    add('restored', equal_to='baseline')
    return cases


def build_guards(init_image, allow_code=False):
    plain = dict(prompt=BASE + ' <lora:A:1>', loras=[dict(name='A', weight=1.)])
    guards = [
        dict(name='photomaker', params=dict(plain, hooks={'composable_lora': {}, 'photomaker': {}}),
             code='unsupported', message='photomaker'),
        dict(name='alternative', params=dict(plain, hooks={'composable_lora': {}, 'img2imgalt': {}},
             img2img=dict(init_image=init_image, denoising_strength=.6)), code='unsupported', message='img2imgalt'),
        dict(name='negpip-signed', params=dict(plain, prompt=BASE + ' (hair:-1) <lora:A:1>',
             hooks={'composable_lora': {}, 'negpip': {}}), code='unsupported', message='NegPiP'),
        dict(name='native-lbw', params=dict(plain, loras=[dict(name='A', weight=1., lbw='ALL')],
             hooks={'composable_lora': {}}), code='unsupported', message='Block Weight'),
    ]
    if not allow_code:
        guards.append(dict(name='python-opt-in', params=dict(plain,
            prompt=BASE + ' [<lora:A:1> #cmd\\(enable_prepare_step=True\n0.5\\)]',
            hooks={'composable_lora': {'with_step': True}}), code='unsupported', message='Allow custom code'))
    return guards


class Client:
    def __init__(self, executable, root, output, allow_code):
        self.counter, self.messages = 0, queue.Queue()
        self.stderr = (output / 'stderr.log').open('w', encoding='utf-8')
        self.journal = (output / 'protocol.jsonl').open('w', encoding='utf-8')
        env = dict(os.environ, PYTHONPATH=str(root / 'python'), PYTHONUNBUFFERED='1')
        if allow_code:
            env['SDFX_ALLOW_CODE'] = '1'
        else:
            env.pop('SDFX_ALLOW_CODE', None)
        self.process = subprocess.Popen([str(Path(executable).resolve()), '-m', 'sdfx'], cwd=root / 'python', env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, text=True, encoding='utf-8')
        def reader():
            try:
                for line in self.process.stdout:
                    self.messages.put(json.loads(line))
            except Exception as exc:
                self.messages.put(dict(type='reader-error', message=str(exc)))
            finally:
                self.messages.put(dict(type='eof'))
        threading.Thread(target=reader, daemon=True).start()

    def receive(self, deadline):
        if time.monotonic() >= deadline:
            raise TimeoutError('engine response deadline exceeded; see protocol.jsonl')
        try:
            message = self.messages.get(timeout=max(.001, deadline - time.monotonic()))
        except queue.Empty as exc:
            raise TimeoutError('engine response deadline exceeded; see stderr.log') from exc
        self.journal.write(json.dumps(dict(direction='response', message=message)) + '\n')
        self.journal.flush()
        if message['type'] in ('eof', 'reader-error'):
            raise RuntimeError(message)
        return message

    def call(self, kind, params):
        self.counter += 1
        request = dict(id=f'clora-{self.counter}', type=kind, params=params)
        self.journal.write(json.dumps(dict(direction='request', message=request)) + '\n')
        self.journal.flush()
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        deadline = time.monotonic() + 1200
        while True:
            message = self.receive(deadline)
            if message.get('id') == request['id'] and message['type'] in ('result', 'error'):
                return message

    def close(self):
        try:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        finally:
            self.stderr.close()
            self.journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('engine_python')
    parser.add_argument('model')
    parser.add_argument('out')
    parser.add_argument('--sdxl', action='store_true')
    parser.add_argument('--allow-code', action='store_true')
    parser.add_argument('--lora-a')
    parser.add_argument('--lora-b')
    parser.add_argument('--locon')
    parser.add_argument('--lokr')
    args = parser.parse_args()
    root, output = Path(__file__).resolve().parents[2], Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    defaults = ('alraune_v1_illustrious_b-000007.safetensors', 'sdxl-vbs-SeifukuKh03.safetensors') if args.sdxl else ('20_yui.safetensors', 'slimegirl11.safetensors')
    paths = dict(A=str(Path(args.lora_a or LORA_DIR / defaults[0]).resolve(strict=True)),
                 B=str(Path(args.lora_b or LORA_DIR / defaults[1]).resolve(strict=True)))
    if args.sdxl:
        paths.update(C=str(Path(args.locon or LORA_DIR / 'sdxl-vbs-BNSeifukuKh03.safetensors').resolve(strict=True)),
                     K=str(Path(args.lokr or LORA_DIR / 'MonsterGirlXL_ANI31_lokr_V43P1NF.safetensors').resolve(strict=True)))
    model = str(Path(args.model).resolve(strict=True))
    report = dict(status='running', family='sdxl' if args.sdxl else 'sd15', model=model, adapters=paths,
                  cases={}, comparisons={}, guards={}, cpu_only_formats=['LoHa', 'IA3', 'full-rank', 'LoCon CP'],
                  scope='Standalone protocol GPU checks; native UI/PNG replay and visual quality require parent review.')
    def save():
        (output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    client = Client(args.engine_python, root, output, args.allow_code)
    try:
        hello = client.receive(time.monotonic() + 120)
        assert hello['type'] == 'hello' and hello.get('cuda') and not hello.get('dry_run'), hello
        assert 'composable_lora' in hello['features'], hello
        report['hello'] = hello
        loaded = client.call('load_model', dict(path=model, dtype='fp16', vram_mode='gpu'))
        assert loaded['type'] == 'result' and loaded['result']['kind'] == ('sdxl' if args.sdxl else 'sd15'), loaded
        def check_guards():
            baseline = report['cases']['baseline']['images'][0]['path']
            base = build_cases(args.sdxl)[0]['params']
            for guard in build_guards(baseline, args.allow_code):
                params = dict(base, **guard['params'])
                response = client.call('img2img' if params.get('img2img') else 'txt2img',
                    dict(params=params, lora_paths=paths, output_dir=str(output / ('guard-' + guard['name'])), preview_every=0))
                report['guards'][guard['name']] = response
                save()
                assert response['type'] == 'error' and response.get('code') == guard['code'], response
                assert guard['message'].lower() in response.get('message', '').lower(), response
                assert not list((output / ('guard-' + guard['name'])).glob('*.png'))
                print('guard', guard['name'], 'OK', flush=True)
        for case in build_cases(args.sdxl, args.allow_code):
            name, params = case['name'], case['params']
            if name == 'restored':
                check_guards()
            if params.get('img2img'):
                params['img2img']['init_image'] = report['cases']['baseline']['images'][0]['path']
            started = time.monotonic()
            response = client.call('img2img' if params.get('img2img') else 'txt2img',
                dict(params=params, lora_paths=paths, output_dir=str(output / name), preview_every=0))
            report['cases'][name] = dict(response=response, seconds=round(time.monotonic() - started, 3))
            save()
            assert response['type'] == 'result', response
            result = response['result']
            plot = params['hooks'].get('composable_lora', {}).get('plot_weights', False)
            assert len(result['images']) == params['batch_size'] + int(plot), result
            records = [pixel_record(im['path']) for im in result['images']]
            report['cases'][name]['images'] = records
            scale = 1.25 if name.startswith('hires-') else 1
            assert all((r['width'], r['height']) == (int(params['width'] * scale), int(params['height'] * scale))
                       for r in records[:params['batch_size']]), records
            assert all(max(r['stddev']) > 1 for r in records), records
            assert set(result['lora_hashes']) == {lr['name'] for lr in params['loras']}, result
            if case['equal_to']:
                comparison = image_difference(report['cases'][case['equal_to']]['images'][0]['path'], records[0]['path'])
                report['comparisons'][name] = comparison
                assert comparison['exact'], comparison
            if plot:
                trace = result['composable_lora_trace']
                assert Path(trace['path']).is_file() and trace['records'], trace
                if name == 'shared-adapter-plot':
                    assert {r['mean_weight'] for r in trace['records'] if not r['is_negative']} == {.25, .75}, trace
                if name == 'hires-global':
                    first = [r['step'] for r in trace['records'] if r['pass_name'] == 'base']
                    second = [r['step'] for r in trace['records'] if r['pass_name'] == 'hires']
                    assert first and second and min(second) > max(first), trace
            print(name, 'OK', round(time.monotonic() - started, 1), 'seconds', flush=True)
            save()
        baseline = report['cases']['baseline']['images'][0]['path']
        for name in ('single-source-default', 'separate-branches', 'increment', 'math-command'):
            report['comparisons'][name + '-baseline'] = image_difference(baseline, report['cases'][name]['images'][0]['path'])
        assert not report['comparisons']['single-source-default-baseline']['exact'], 'adapter had no pixel effect'
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', failure=repr(exc))
        raise
    finally:
        try:
            try:
                contact_sheet(output, report['cases'])
            except BaseException as exc:
                report.update(status='failed', artifact_failure=repr(exc))
                raise
            finally:
                save()
        finally:
            client.close()


if __name__ == '__main__':
    main()
