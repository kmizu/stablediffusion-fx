"""Real-model boundary acceptance: PT, repeated expressions, Composable branches,
owned cancellation, exact recovery and model reload/conversion catalog selection.
Run only in the coordinated GPU slot; checkpoints are read-only.
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
    parser.add_argument('python'); parser.add_argument('model'); parser.add_argument('output')
    parser.add_argument('--sdxl', action='store_true')
    parser.add_argument('--vram-mode', default='gpu', choices=['gpu', 'model_offload', 'sequential_offload'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    client = Client(args.python, root, output, False)
    report = {'status': 'running', 'mode': args.vram_mode, 'cases': {}, 'comparisons': {}}
    def record(): (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
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
        load = dict(path=str(Path(args.model).resolve()), dtype='fp16', vram_mode=args.vram_mode)
        call('load_model', load)
        expression = "'red hair' * 0.7 + 'blue hair' * 0.3"
        exported = call('embedding_merge', dict(operation='save', text=expression, name='boundary47',
            output_dir=str(output / 'embeddings'), safetensors=False))
        assert exported['active'] and len(exported['paths']) == 1, exported
        converted = call('embedding_merge', dict(operation='save', text=expression, name='converted47',
            output_dir=str(output / 'embeddings'), safetensors=True))
        assert converted['active'] and len(converted['paths']) == 2, converted
        paths = exported['paths'] + converted['paths']
        size = 768 if args.sdxl else 512
        base = dict(prompt='a woman in a garden, detailed portrait', negative_prompt='blurry, low quality',
                    seed=173, steps=8, width=size, height=size, cfg_scale=4, sampler_name='DPM++ 2M', scheduler='Automatic')
        def generate(name, params):
            result = call('txt2img', dict(params=params, embedding_paths=paths, output_dir=str(output / name), preview_every=0))
            images = [pixel_record(image['path']) for image in result['images']]
            assert len(images) == 1, images
            report['cases'][name] = images[0]; record(); print(name, 'generated', flush=True)
            return images[0]
        baseline = generate('baseline', base)
        actual = dict(base, prompt=f"a woman with <{expression}> AND a woman with {{{expression}}} in a garden",
            negative_prompt=f"blurry <{expression}>", hooks={'composable_lora': {}}, cfg_scale=3)
        equivalent = dict(actual, prompt='a woman with boundary47 AND a woman with boundary47 in a garden',
                          negative_prompt='blurry boundary47')
        left = generate('branches-expression', actual); right = generate('branches-pt', equivalent)
        diff = image_difference(left['path'], right['path']); report['comparisons']['branches'] = diff
        assert diff['exact'], diff
        # Cancel a real running image job after its first denoising progress event.
        cancel_params = copy.deepcopy(actual); cancel_params['steps'] = 40
        send('merge-cancel-job', 'txt2img', dict(params=cancel_params, embedding_paths=paths,
            output_dir=str(output / 'cancelled'), preview_every=0))
        deadline = time.monotonic() + 300
        interrupted, decisions, final = False, None, None
        while final is None or decisions is None:
            message = client.receive(deadline)
            if message.get('id') == 'merge-cancel-job' and message.get('event') == 'progress' and not interrupted:
                send('merge-cancel-request', 'interrupt', {'target_id': 'merge-cancel-job'})
                interrupted = True
            if message.get('id') == 'merge-cancel-request' and message['type'] in ('result', 'error'):
                decisions = message
            if message.get('id') == 'merge-cancel-job' and message['type'] in ('result', 'error'):
                final = message
                assert interrupted, message
        assert final['type'] == 'error' and final['code'] == 'interrupted', final
        assert decisions['type'] == 'result' and decisions['result']['accepted'], decisions
        report['cancellation'] = dict(job=final, decision=decisions)
        recovered = generate('recovered', base)
        diff = image_difference(baseline['path'], recovered['path']); report['comparisons']['recovery'] = diff
        assert diff['exact'], diff
        call('unload', {})
        call('load_model', load)
        # Both incompatible conversion siblings are deliberately passed to native loader.
        file_params = dict(base, prompt='a woman with converted47 in a garden')
        restored = generate('after-reload-converted', file_params)
        expected = generate('after-reload-expression', dict(base, prompt=f'a woman with <{expression}> in a garden'))
        diff = image_difference(restored['path'], expected['path']); report['comparisons']['reloaded-catalog'] = diff
        assert diff['exact'], diff
        report['status'] = 'passed-pending-visual-inspection'
    except BaseException as error:
        report.update(status='failed', failure=repr(error)); raise
    finally:
        record(); client.close()


if __name__ == '__main__': main()
