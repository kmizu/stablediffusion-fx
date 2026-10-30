"""Real detector/inpaint Hires batch cancellation acceptance (local assets only)."""
import copy
import json
from pathlib import Path
import sys
import time

from composable_lora_check import Client
from prompt_fusion_check import pixel_record, image_difference


class ObservedClient(Client):
    cancel_second = False
    cancel_sent = False

    def receive(self, deadline):
        message = super().receive(deadline)
        if message.get('event') == 'progress':
            self.progress.append(message)
            if (self.cancel_second and not self.cancel_sent and message.get('pass') == 'adetailer'
                    and message.get('image') == 1 and message.get('step', 0) >= 1):
                request = dict(id='cancel-second-ad', type='interrupt', params={})
                self.process.stdin.write(json.dumps(request) + '\n')
                self.process.stdin.flush()
                self.cancel_sent = True
        return message


def main():
    executable, model, detector, destination = sys.argv[1:5]
    sdxl = '--sdxl' in sys.argv[5:]
    root, output = Path(__file__).resolve().parents[2], Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=True)
    client = ObservedClient(executable, root, output, False)
    client.progress = []
    report = dict(status='running', cases={})
    def save():
        (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    try:
        assert client.receive(time.monotonic() + 180)['type'] == 'hello'
        loaded = client.call('load_model', dict(path=str(Path(model).resolve()), dtype='fp16', vram_mode='gpu'))
        assert loaded['type'] == 'result', loaded
        base = dict(prompt='portrait of a woman, close-up face, detailed eyes, flowers in background',
            negative_prompt='low quality, blurry', seed=173, steps=12, width=512, height=512,
            cfg_scale=6, sampler_name='DPM++ 2M', scheduler='Automatic', batch_size=1,
            hires=dict(enabled=True, scale=1.5, second_pass_steps=8, denoising_strength=.55))
        if sdxl:
            base.update(prompt='a woman with red hair in a garden, portrait, detailed illustration',
                width=768, height=768, steps=20, cfg_scale=4)
            base['hires'].update(scale=1.25, second_pass_steps=12)
        def run(name, params):
            client.progress = []
            response = client.call('txt2img', dict(params=params, output_dir=str(output / name), preview_every=0))
            assert response['type'] == 'result', response
            result = response['result']
            records = [dict(im, **pixel_record(im['path'])) for im in result['images']]
            report['cases'][name] = dict(images=records, progress=client.progress,
                interrupted=result.get('interrupted', False))
            save()
            print(name, len(records), 'images', flush=True)
            return records
        baseline = run('baseline', base)
        batch = copy.deepcopy(base)
        batch['batch_size'] = 2
        batch['hooks'] = dict(hires_fix_tweaks=dict(batch_count=2, seed_enabled=True, seed=100),
            adetailer=dict(units=[dict(model=str(Path(detector).resolve()), confidence=.15,
                mask_k_largest=1, use_steps=True, steps=12)]))
        complete = run('ad-batch', batch)
        assert len(complete) == 4
        # A repeat advances each member's Hires seed by one, as upstream does.
        assert [im['hires_seed'] for im in complete] == [100, 101, 101, 102]
        observed = {event['image'] for event in client.progress if event.get('pass') == 'adetailer'}
        assert observed == {0, 1, 2, 3}, f'No actual face inpaint for some images: {observed}'
        client.cancel_second = True
        partial = run('ad-partial', batch)
        assert client.cancel_sent and len(partial) == 1
        assert report['cases']['ad-partial']['interrupted']
        comparison = image_difference(complete[0]['path'], partial[0]['path'])
        report['completed_before_interrupt'] = comparison
        assert comparison['exact'], comparison
        client.cancel_second = False
        recovery = run('recovery', base)
        comparison = image_difference(baseline[0]['path'], recovery[0]['path'])
        report['recovery'] = comparison
        assert comparison['exact'], comparison
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', failure=repr(exc))
        raise
    finally:
        save()
        client.close()


if __name__ == '__main__':
    main()
