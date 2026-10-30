"""Real official PhotoMaker v1 acceptance; run only in the coordinated GPU slot."""
import argparse
import json
import os
from pathlib import Path
import time

from composable_lora_check import Client
from prompt_fusion_check import pixel_record, image_difference


def main():
    parser = argparse.ArgumentParser()
    for name in ('python', 'model', 'adapter', 'reference', 'output'):
        parser.add_argument(name)
    parser.add_argument('--python-path', action='append', required=True)
    parser.add_argument('--vram-mode', default='gpu', choices=['gpu', 'model_offload', 'sequential_offload'])
    parser.add_argument('--plain-control', action='store_true')
    parser.add_argument('--window-lora', help='Read-only SDXL LoRA for adapter ownership/window recovery')
    parser.add_argument('--engine-root', help='Extracted package root; defaults to this source checkout')
    parser.add_argument('--cache-boundaries', action='store_true')
    parser.add_argument('--fail-first-load', action='store_true', help='Test-only post-injection failure in genuine loader')
    args = parser.parse_args()
    root = Path(args.engine_root).resolve() if args.engine_root else Path(__file__).resolve().parents[2]
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    python_paths = list(args.python_path)
    if args.fail_first_load:
        os.environ['SDFX_TEST_PHOTOMAKER_LOAD_FAILURE'] = '1'
        python_paths.insert(0, str(Path(__file__).resolve().parent / 'photomaker_fault_fixture'))
    client = Client(args.python, root, output, False, python_paths=python_paths)
    report = dict(status='running', mode=args.vram_mode, cases={}, comparisons={})
    def record():
        (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    def call(kind, params):
        result = client.call(kind, params)
        assert result['type'] == 'result', result
        return result['result']
    def send(rid, kind, params):
        request = dict(id=rid, type=kind, params=params)
        client.journal.write(json.dumps(dict(direction='request', message=request)) + '\n'); client.journal.flush()
        client.process.stdin.write(json.dumps(request) + '\n'); client.process.stdin.flush()
    try:
        assert client.receive(time.monotonic() + 180)['type'] == 'hello'
        call('load_model', dict(path=str(Path(args.model).resolve()), dtype='fp16', vram_mode=args.vram_mode))
        expression = "'red hair' * 0.7 + 'blue hair' * 0.3"
        saved = call('embedding_merge', dict(operation='save', text=expression, name='photo47',
                                             output_dir=str(output / 'embeddings')))
        assert saved['active'], saved
        base = dict(prompt='a woman with red hair in a garden, detailed portrait',
                    negative_prompt='blurry', seed=173, steps=8, width=768, height=768,
                    cfg_scale=5, sampler_name='DPM++ 2M', scheduler='Automatic')
        lora_paths = {}
        if args.window_lora:
            lora_paths['ordinary'] = str(Path(args.window_lora).resolve(strict=True))
            base['loras'] = [dict(name='ordinary', weight=.4, start=0, stop=.5)]
        def generate(name, params):
            result = call('txt2img', dict(params=params, output_dir=str(output / name),
                embedding_paths=[saved['paths'][0]], lora_paths=lora_paths, preview_every=0))
            image = pixel_record(result['images'][0]['path'])
            report['cases'][name] = image; record()
            print(name, image['path'], flush=True)
            return image
        baseline = generate('baseline', base)
        pm = dict(model=str(Path(args.adapter).resolve()), images=[str(Path(args.reference).resolve())],
                  trigger_word='img', start_merge_step=2)
        actual = dict(base, prompt=f'a woman img with <{expression}> in a garden, detailed portrait',
                      negative_prompt=f'blurry, <{expression}>', hooks={'photomaker': pm})
        equivalent = dict(actual, prompt='a woman img with photo47 in a garden, detailed portrait',
                          negative_prompt='blurry, photo47')
        if args.plain_control:
            actual = dict(base, prompt='a woman img with red hair in a garden, detailed portrait', hooks={'photomaker': pm})
            equivalent = dict(actual)
        if args.fail_first_load:
            failure = client.call('txt2img', dict(params=actual, output_dir=str(output / 'injected-failure'),
                embedding_paths=[saved['paths'][0]], lora_paths=lora_paths, preview_every=0))
            assert failure['type'] == 'error' and failure['code'] == 'no_model', failure
            blocked = client.call('txt2img', dict(params=base, lora_paths=lora_paths,
                                                 output_dir=str(output / 'must-reload')))
            assert blocked['type'] == 'error' and blocked['code'] == 'no_model', blocked
            report['partial-load-failure'] = failure
            call('load_model', dict(path=str(Path(args.model).resolve()), dtype='fp16', vram_mode=args.vram_mode))
            reloaded = generate('after-failure-reload', base)
            diff = image_difference(baseline['path'], reloaded['path'])
            report['comparisons']['partial-failure-reload'] = diff
            assert diff['exact'], diff
        left = generate('photomaker-expression', actual)
        right = generate('photomaker-saved', equivalent)
        diff = image_difference(left['path'], right['path'])
        report['comparisons']['expression-vs-saved'] = diff
        assert diff['exact'], diff
        if args.cache_boundaries:
            call('set_vae', {'path': None})
            after_vae = generate('photomaker-after-vae-reset', actual)
            diff = image_difference(left['path'], after_vae['path'])
            report['comparisons']['vae-reset'] = diff
            assert diff['exact'], diff
            alias = output / 'alternate-photomaker-v1.bin'
            os.link(Path(args.adapter).resolve(), alias)
            alternate = dict(actual, hooks={'photomaker': dict(pm, model=str(alias))})
            switched = generate('photomaker-alternate-path', alternate)
            returned = generate('photomaker-original-path', actual)
            for name, result in (('alternate-path', switched), ('original-path-return', returned)):
                diff = image_difference(left['path'], result['path'])
                report['comparisons'][name] = diff
                assert diff['exact'], diff
        send('photo-merge-cancel', 'txt2img', dict(params=dict(actual, steps=40),
            embedding_paths=[saved['paths'][0]], lora_paths=lora_paths, output_dir=str(output / 'cancelled'), preview_every=0))
        deadline = time.monotonic() + 300
        interrupted, decision, final = False, None, None
        while final is None or decision is None:
            message = client.receive(deadline)
            if message.get('id') == 'photo-merge-cancel' and message.get('event') == 'progress' and not interrupted:
                send('photo-merge-interrupt', 'interrupt', {'target_id': 'photo-merge-cancel'})
                interrupted = True
            if message.get('id') == 'photo-merge-interrupt' and message['type'] in ('result', 'error'):
                decision = message
            if message.get('id') == 'photo-merge-cancel' and message['type'] in ('result', 'error'):
                final = message
                assert interrupted, message
        assert final['type'] == 'error' and final['code'] == 'interrupted', final
        assert decision['type'] == 'result' and decision['result']['accepted'], decision
        report['cancellation'] = dict(job=final, decision=decision)
        recovered = generate('recovered', base)
        diff = image_difference(baseline['path'], recovered['path'])
        report['comparisons']['off-recovery'] = diff
        assert diff['exact'], diff
        report['status'] = 'passed-pending-visual-inspection'
    except BaseException as error:
        report.update(status='failed', failure=repr(error))
        raise
    finally:
        record(); client.close()


if __name__ == '__main__':
    main()
