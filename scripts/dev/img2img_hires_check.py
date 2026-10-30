"""Coordinated real-model #61 acceptance; local models only, never downloads.

ENGINE_PY MODEL OUTPUT [--sdxl] [--lora FILE]. Run one GPU process at a time.
"""
import argparse
import copy
import json
from pathlib import Path
import time

from PIL import Image, ImageDraw
from composable_lora_check import Client
from prompt_fusion_check import pixel_record, image_difference


class ObservedClient(Client):
    cancel_second = False
    cancel_sent = False

    def receive(self, deadline):
        message = super().receive(deadline)
        if message.get('event') == 'progress':
            self.progress.append(message)
            if (self.cancel_second and not self.cancel_sent and message.get('pass') == 'img2img_hires'
                    and message.get('image') == 1 and message.get('step', 0) >= 1):
                request = dict(id='cancel-refinement', type='interrupt', params={})
                self.journal.write(json.dumps(dict(direction='request', message=request)) + '\n')
                self.process.stdin.write(json.dumps(request) + '\n')
                self.process.stdin.flush()
                self.cancel_sent = True
        return message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('python')
    parser.add_argument('model')
    parser.add_argument('output')
    parser.add_argument('--sdxl', action='store_true')
    parser.add_argument('--lora')
    args = parser.parse_args()
    root, output = Path(__file__).resolve().parents[2], Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    client = ObservedClient(args.python, root, output, False)
    client.progress = []
    report = dict(status='running', cases={}, comparisons={})
    def save():
        (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    def compare(name, first, second):
        result = image_difference(first['path'], second['path'])
        report['comparisons'][name] = result
        save()
        assert result['exact'], (name, result)
    def run(name, params, **request):
        client.progress = []
        response = client.call('img2img' if params.get('img2img') else 'txt2img', dict(params=params,
            output_dir=str(output / name), preview_every=0, **request))
        assert response['type'] == 'result', response
        result = response['result']
        images = [dict(im, **pixel_record(im['path'])) for im in result['images']]
        report['cases'][name] = dict(images=images, progress=client.progress,
            interrupted=result.get('interrupted', False))
        save()
        print(name, len(images), 'images', flush=True)
        return images
    try:
        assert client.receive(time.monotonic() + 180)['type'] == 'hello'
        loaded = client.call('load_model', dict(path=str(Path(args.model).resolve()), dtype='fp16', vram_mode='gpu'))
        assert loaded['type'] == 'result', loaded
        side = 768 if args.sdxl else 512
        base = dict(prompt='a woman with red hair in a garden, portrait, detailed illustration',
            negative_prompt='low quality, blurry', seed=173, steps=16, width=side, height=side,
            cfg_scale=4 if args.sdxl else 6, sampler_name='DPM++ 2M', scheduler='Automatic')
        config = dict(upscaler='Lanczos', ratio=1.25, steps=12, denoise=.5)
        baseline = run('baseline', base)[0]
        disabled = run('disabled', dict(base, hooks={'img2img_hires_fix': dict(config, enabled=False)}))[0]
        compare('disabled-exact', baseline, disabled)
        enabled = dict(base, hooks={'img2img_hires_fix': config})
        refined = run('refined', enabled)[0]
        assert refined['width'] == side * 5 // 4
        compare('deterministic', refined, run('replay', enabled)[0])
        resized = run('resize-only', dict(base, hooks={'img2img_hires_fix': dict(config, denoise=0)}))[0]
        assert not any(p.get('pass') == 'img2img_hires' for p in client.progress)
        assert resized['width'] == side * 5 // 4
        explicit = run('explicit-controls', dict(base, hooks={'img2img_hires_fix': dict(config,
            width=side, height=side + 64, prompt='a woman with blue hair in a garden', negative_prompt='blurry',
            cfg=3, sampler='Euler', scheduler='Karras')}))[0]
        assert (explicit['width'], explicit['height']) == (side, side + 64)
        run('img2img', dict(enabled, img2img=dict(init_image=baseline['path'], denoising_strength=.4)))
        mask = Image.new('L', (side, side), 0)
        ImageDraw.Draw(mask).rectangle((side // 4, side // 4, side * 3 // 4, side * 3 // 4), fill=255)
        mask.save(output / 'mask.png')
        run('inpaint', dict(enabled, img2img=dict(init_image=baseline['path'], mask=str(output / 'mask.png'),
            denoising_strength=.5)))
        native = run('native-latent-first', dict(enabled, hires=dict(enabled=True, scale=1.25,
            upscaler='Latent', second_pass_steps=8, denoising_strength=.5)))[0]
        assert native['width'] == (side * 1.25 * 1.25 // 8) * 8
        batch = dict(enabled, batch_size=2, n_iter=2)
        complete = run('batch', batch)
        assert [im['seed'] for im in complete] == [173, 174, 175, 176]
        client.cancel_second = True
        partial = run('cancel-second', batch)
        assert client.cancel_sent and len(partial) == 1 and report['cases']['cancel-second']['interrupted']
        compare('completed-before-cancel', complete[0], partial[0])
        client.cancel_second = False
        if args.lora:
            path = str(Path(args.lora).resolve(strict=True))
            refs = [dict(name='A', weight=.25)]
            run('refinement-only-lora', enabled, img2img_hires_loras=refs, lora_paths={'A': path})
        compare('off-recovery', baseline, run('recovery', base)[0])
        report['status'] = 'passed-awaiting-image-review'
    except BaseException as error:
        report.update(status='failed', failure=repr(error))
        raise
    finally:
        save()
        client.close()


if __name__ == '__main__':
    main()
