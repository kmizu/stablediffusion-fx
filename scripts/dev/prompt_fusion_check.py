"""Real-model Prompt Fusion acceptance: PYTHON script.py ENGINE_PY MODEL OUT [--sdxl].

Run once per family, using its local checkpoint. Only starts the standalone sdfx engine.
Writes raw requests/responses, incremental results.json and contact.png for visual review.
No model download, dependency installation or settings mutation is performed by this script.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time


PROMPT = 'a red fox in a forest'
LINEAR = 'a [red fox:white owl: , 5] in a forest'


def build_cases(sdxl=False):
    size = 768 if sdxl else 512
    base = dict(prompt=PROMPT, negative_prompt='low quality', seed=812, subseed=123,
                steps=8, width=size, height=size, cfg_scale=6, sampler_name='DPM++ 2M',
                scheduler='Automatic', batch_size=1, n_iter=1)
    cases = []

    def add(name, hook=None, equal_to=None, **overrides):
        p = dict(base, hooks={} if hook is None else {'prompt_fusion': hook}, **overrides)
        cases.append(dict(name=name, params=p, equal_to=equal_to))

    add('baseline')
    add('ordinary-enabled', {}, equal_to='baseline')
    add('ordinary-disabled', {'enabled': False}, equal_to='baseline')
    add('variables-functions', {}, equal_to='baseline',
        prompt='$subject($color) = a $color fox in a forest\n$subject(red)')
    add('constant-curve', {}, equal_to='baseline', prompt=f'[{PROMPT}:{PROMPT}: , 5]')
    add('linear', {}, prompt=LINEAR)
    add('bezier', {}, prompt='a [red fox:blue butterfly:white owl: , 2, 5:bezier] in a forest')
    add('catmull', {}, prompt='a [red fox:blue butterfly:white owl: , 2, 5:catmull] in a forest')
    add('weighted-mean', {}, prompt='a [red fox:white owl:1,3:mean] in a forest')
    add('alternation', {}, prompt='a [red fox|white owl:0.5] in a forest')
    add('contextual-attention', {}, prompt='a fox in a forest [(red flowers:0.5,2)::4]')
    add('slerp-negative-origin', {'slerp_scale': 1, 'negative_origin': True}, prompt=LINEAR)
    add('slerp-empty-origin', {'slerp_scale': 1, 'negative_origin': False, 'slerp_epsilon': .0001}, prompt=LINEAR)
    add('negative-curve', {}, negative_prompt='[low quality:blurry: , 5]')
    add('long-conditioning', {},
        prompt='[' + PROMPT + ':' + ('mossy trees, ancient forest, ' * 18) + 'owl: , 5]')
    add('batch', {}, prompt=LINEAR, batch_size=2)
    add('heun', {}, prompt=LINEAR, sampler_name='Heun')
    add('hires', {}, prompt=LINEAR,
        hires=dict(enabled=True, scale=1.25, upscaler='Latent', second_pass_steps=4,
                   denoising_strength=.6, prompt='a [white owl:red fox: , ] in a forest',
                   negative_prompt='[low quality:blurry: , ]'))
    add('img2img', {}, prompt=LINEAR, img2img=dict(init_image='@baseline', denoising_strength=.6))
    add('restored', equal_to='baseline')
    return cases


def build_guards(init_image):
    fusion = {'prompt_fusion': {}}
    return [
        dict(name='photomaker', params=dict(prompt=LINEAR, hooks={**fusion, 'photomaker': {}}),
             expected_code='unsupported', message='PhotoMaker'),
        dict(name='negpip', params=dict(prompt='[(fox:-1):owl::mean]', hooks={**fusion, 'negpip': {}}),
             expected_code='unsupported', message='NegPiP'),
        dict(name='alternative-main', params=dict(prompt=LINEAR, hooks={**fusion, 'img2imgalt': {}},
             img2img=dict(init_image=init_image, denoising_strength=.6)),
             expected_code='unsupported', message='inversion'),
        dict(name='alternative-override', params=dict(prompt=PROMPT,
             hooks={**fusion, 'img2imgalt': {'override_prompt': LINEAR}},
             img2img=dict(init_image=init_image, denoising_strength=.6)),
             expected_code='unsupported', message='inversion'),
        dict(name='nonfinite-position', params=dict(prompt='$n = 1e309\n[a:b: , $n]', hooks=fusion),
             expected_code='invalid_params', message='finite'),
        dict(name='nonfinite-attention', params=dict(prompt='$n = 1e309\n(a:1,$n)', hooks=fusion),
             expected_code='invalid_params', message='finite'),
    ]


def pixel_record(path):
    from PIL import Image, ImageStat
    with Image.open(path) as source:
        source.load()
        image = source.convert('RGB')
    stats = ImageStat.Stat(image)
    return dict(path=str(Path(path).resolve()), width=image.width, height=image.height,
                pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                mean=stats.mean, stddev=stats.stddev, extrema=image.getextrema())


def image_difference(a, b):
    from PIL import Image, ImageChops, ImageStat
    with Image.open(a) as left, Image.open(b) as right:
        assert left.size == right.size, (left.size, right.size)
        difference = ImageChops.difference(left.convert('RGB'), right.convert('RGB'))
    return dict(exact=difference.getbbox() is None,
                mean_absolute_difference=sum(ImageStat.Stat(difference).mean) / 3)


def contact_sheet(output, reports):
    from PIL import Image, ImageDraw, ImageOps
    thumbs = []
    for name, report in reports.items():
        for i, record in enumerate(report.get('images', [])):
            with Image.open(record['path']) as image:
                thumb = ImageOps.pad(image.convert('RGB'), (200, 200))
            card = Image.new('RGB', (200, 224), '#18212b')
            card.paste(thumb, (0, 24))
            ImageDraw.Draw(card).text((5, 6), f'{name} [{i}]', fill='white')
            thumbs.append(card)
    if thumbs:
        sheet = Image.new('RGB', (200 * 4, 224 * ((len(thumbs) + 3) // 4)), '#18212b')
        for i, card in enumerate(thumbs):
            sheet.paste(card, (i % 4 * 200, i // 4 * 224))
        sheet.save(output / 'contact.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('engine_python')
    parser.add_argument('model')
    parser.add_argument('out')
    parser.add_argument('--sdxl', action='store_true')
    parser.add_argument('--timeout', type=float, default=1200, help='Deadline in seconds for each request')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = Path(args.model).resolve(strict=True)
    report = dict(status='running', family='sdxl' if args.sdxl else 'sd15', model=str(model),
                  model_bytes=model.stat().st_size, cases={}, guards={}, comparisons={},
                  scope='Standalone engine acceptance; native Scala UI/MCP and artistic quality require separate review.')

    def save_report():
        (output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    env = dict(os.environ, PYTHONPATH=str(root / 'python'), PYTHONUNBUFFERED='1')
    with (output / 'stderr.log').open('w', encoding='utf-8') as stderr, \
         (output / 'protocol.jsonl').open('w', encoding='utf-8') as journal:
        process = subprocess.Popen([str(Path(args.engine_python).resolve()), '-m', 'sdfx'],
            cwd=root / 'python', env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stderr, text=True, encoding='utf-8')
        messages = queue.Queue()

        def reader():
            try:
                for line in process.stdout:
                    messages.put(json.loads(line))
            except Exception as exc:
                messages.put(dict(type='reader_error', message=str(exc)))
            finally:
                messages.put(dict(type='eof'))

        threading.Thread(target=reader, daemon=True).start()
        counter = 0

        def receive(deadline):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('engine response deadline exceeded; see protocol.jsonl and stderr.log')
            try:
                message = messages.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError('engine response deadline exceeded; see stderr.log') from exc
            journal.write(json.dumps(dict(direction='response', message=message)) + '\n')
            journal.flush()
            if message['type'] in ('eof', 'reader_error'):
                raise RuntimeError(f'engine terminated: {message}; see stderr.log')
            return message

        def call(kind, params):
            nonlocal counter
            counter += 1
            request = dict(id=f'fusion-{counter}', type=kind, params=params)
            journal.write(json.dumps(dict(direction='request', message=request)) + '\n')
            journal.flush()
            process.stdin.write(json.dumps(request) + '\n')
            process.stdin.flush()
            deadline = time.monotonic() + args.timeout
            while True:
                message = receive(deadline)
                if message.get('id') == request['id'] and message['type'] in ('result', 'error'):
                    return message

        try:
            hello = receive(time.monotonic() + 120)
            assert hello['type'] == 'hello' and hello.get('cuda') and not hello.get('dry_run'), hello
            assert 'prompt_fusion' in hello['features'], hello
            report['hello'] = hello
            loaded = call('load_model', dict(path=str(model), dtype='fp16', vram_mode='gpu'))
            assert loaded['type'] == 'result', loaded
            assert loaded['result']['kind'] == ('sdxl' if args.sdxl else 'sd15'), loaded
            report['load_model'] = loaded
            cases = build_cases(args.sdxl)

            def check_guards():
                baseline = report['cases']['baseline']['images'][0]['path']
                base = build_cases(args.sdxl)[0]['params']
                for guard in build_guards(baseline):
                    params = dict(base, **guard['params'])
                    result = call('img2img' if params.get('img2img') else 'txt2img',
                                  dict(params=params, output_dir=str(output / ('guard-' + guard['name'])), preview_every=0))
                    report['guards'][guard['name']] = result
                    save_report()
                    assert result['type'] == 'error' and result.get('code') == guard['expected_code'], result
                    assert guard['message'].lower() in result.get('message', '').lower(), result
                    assert not list((output / ('guard-' + guard['name'])).glob('*.png')), result
                    print('guard', guard['name'], result['code'], flush=True)

            for case in cases:
                name, params = case['name'], case['params']
                if name == 'restored':
                    # Exact same-seed rendering after both successful hooks and rejected requests.
                    check_guards()
                if params.get('img2img'):
                    params['img2img']['init_image'] = report['cases']['baseline']['images'][0]['path']
                started = time.monotonic()
                result = call('img2img' if params.get('img2img') else 'txt2img',
                              dict(params=params, output_dir=str(output / name), preview_every=0))
                report['cases'][name] = dict(response=result, seconds=round(time.monotonic() - started, 3))
                save_report()
                assert result['type'] == 'result', result
                images = result['result']['images']
                assert len(images) == params['batch_size'] * params['n_iter'], result
                records = [pixel_record(image['path']) for image in images]
                report['cases'][name]['images'] = records
                scale = 1.25 if name == 'hires' else 1
                expected_size = (int(params['width'] * scale), int(params['height'] * scale))
                assert all((r['width'], r['height']) == expected_size for r in records), records
                assert all(max(r['stddev']) > 1 for r in records), f'{name}: nearly flat output'
                if case['equal_to']:
                    reference = report['cases'][case['equal_to']]['images'][0]['path']
                    comparison = image_difference(reference, records[0]['path'])
                    report['comparisons'][name] = comparison
                    assert comparison['exact'], f'{name}: expected exact ordinary/expanded-prompt identity: {comparison}'
                print(name, 'OK', round(time.monotonic() - started, 2), 'seconds', flush=True)
                save_report()

            baseline = report['cases']['baseline']['images'][0]['path']
            for name in ('linear', 'bezier', 'catmull', 'weighted-mean', 'alternation', 'contextual-attention',
                         'slerp-negative-origin', 'slerp-empty-origin', 'negative-curve', 'long-conditioning'):
                report['comparisons'][name + '-baseline'] = image_difference(
                    baseline, report['cases'][name]['images'][0]['path'])
            assert not report['comparisons']['linear-baseline']['exact'], 'active interpolation had no pixel effect'
            report['comparisons']['slerp-origins'] = image_difference(
                report['cases']['slerp-negative-origin']['images'][0]['path'],
                report['cases']['slerp-empty-origin']['images'][0]['path'])

            report['status'] = 'passed'
        except BaseException as exc:
            report['status'] = 'failed'
            report['failure'] = repr(exc)
            raise
        finally:
            try:
                try:
                    contact_sheet(output, report['cases'])
                except BaseException as exc:
                    report['status'] = 'failed'
                    report['artifact_failure'] = repr(exc)
                    raise
                finally:
                    save_report()
            finally:
                process.stdin.close()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == '__main__':
    main()
