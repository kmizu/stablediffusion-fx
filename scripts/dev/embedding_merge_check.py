"""Coordinated real-model acceptance, never imported by ordinary tests.

ENGINE_PY MODEL OUT [--sdxl] [--vram-mode gpu|offload|sequential]
Uses the serial JSONL worker, local read-only checkpoints, and private exports.
"""
import argparse
import copy
import json
from pathlib import Path
import time

from composable_lora_check import Client
from prompt_fusion_check import pixel_record, image_difference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('python')
    parser.add_argument('model')
    parser.add_argument('output')
    parser.add_argument('--sdxl', action='store_true')
    parser.add_argument('--vram-mode', default='gpu')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    client = Client(args.python, root, output, False)
    report = {'status': 'running', 'cases': {}, 'comparisons': {}}
    def save_report():
        (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    def request(kind, params):
        result = client.call(kind, params)
        assert result['type'] == 'result', result
        return result['result']
    try:
        hello = client.receive(time.monotonic() + 180)
        assert hello['type'] == 'hello', hello
        assert 'embedding_merge' in hello.get('features', hello.get('result', {}).get('features', [])), hello
        loaded = request('load_model', {'path': str(Path(args.model).resolve()), 'dtype': 'fp16', 'vram_mode': args.vram_mode})
        report['model'] = loaded
        expression = "'red hair' * 0.7 + 'blue hair' * 0.3"
        inspected = request('embedding_merge', {'text': expression})
        assert len(inspected['streams']) == (2 if args.sdxl else 1), inspected
        assert all(stream['count'] == 2 for stream in inspected['streams']), inspected
        saved = request('embedding_merge', {'operation': 'save', 'text': expression, 'name': 'merged47',
                                            'output_dir': str(output / 'embeddings')})
        assert saved['active'], saved
        report['save'] = saved
        primary = saved['paths'][0]
        width = 768 if args.sdxl else 512
        base = dict(prompt='a woman with red hair in a garden, detailed portrait',
                    negative_prompt='low quality, blurry', seed=173, steps=8, width=width, height=width,
                    cfg_scale=6, sampler_name='DPM++ 2M', scheduler='Automatic')
        def generate(name, params):
            result = request('txt2img', {'params': params, 'output_dir': str(output / name),
                                         'embedding_paths': [primary], 'preview_every': 0})
            records = [pixel_record(image['path']) for image in result['images']]
            assert records, result
            report['cases'][name] = records
            save_report()
            print(name, 'generated', flush=True)
            return records
        baseline = generate('baseline', base)
        for mode in ('plain', 'negative', 'hires', 'neutral', 'composable'):
            actual = copy.deepcopy(base)
            reference = copy.deepcopy(base)
            actual['prompt'] = f'a woman with <{expression}> in a garden, detailed portrait'
            reference['prompt'] = 'a woman with merged47 in a garden, detailed portrait'
            if mode == 'negative':
                actual['negative_prompt'] = f'low quality, <{expression}>'
                reference['negative_prompt'] = 'low quality, merged47'
            if mode == 'hires':
                for params in (actual, reference):
                    params['hires'] = dict(enabled=True, scale=1.25, upscaler='Latent', second_pass_steps=4,
                        denoising_strength=.6, prompt=params['prompt'], negative_prompt=params['negative_prompt'])
            if mode == 'neutral':
                for params in (actual, reference):
                    params['prompt'] += ' AND_PERP blue flowers'
                    params['hooks'] = {'neutral_prompt': {'cfg_rescale': 0}}
            if mode == 'composable':
                for params in (actual, reference):
                    params['hooks'] = {'composable_lora': {}}
            a = generate(mode + '-expression', actual)
            b = generate(mode + '-saved', reference)
            assert len(a) == len(b)
            for index, (left, right) in enumerate(zip(a, b)):
                diff = image_difference(left['path'], right['path'])
                report['comparisons'][f'{mode}-{index}'] = diff
                assert diff['exact'], diff
        recovered = generate('recovered', base)
        diff = image_difference(baseline[0]['path'], recovered[0]['path'])
        report['comparisons']['off-recovery'] = diff
        assert diff['exact'], diff
        report['status'] = 'passed-pending-visual-inspection'
    except BaseException as error:
        report.update(status='failed', failure=repr(error))
        raise
    finally:
        save_report()
        client.close()


if __name__ == '__main__':
    main()
