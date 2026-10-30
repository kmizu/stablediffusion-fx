"""Real Hires-only custom LBW and optional ControlNet repeat equivalence."""
import copy
import json
from pathlib import Path
import sys
import time

from composable_lora_check import Client
from prompt_fusion_check import pixel_record, image_difference


def main():
    executable, model, lora, destination = sys.argv[1:5]
    cn = next((arg for arg in sys.argv[5:] if not arg.startswith('--')), None)
    sdxl = '--sdxl' in sys.argv[5:]
    root, output = Path(__file__).resolve().parents[2], Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=True)
    client = Client(executable, root, output, False)
    report = dict(status='running', cases={}, comparisons={})
    def save():
        (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    try:
        assert client.receive(time.monotonic() + 180)['type'] == 'hello'
        loaded = client.call('load_model', dict(path=str(Path(model).resolve()), dtype='fp16', vram_mode='gpu'))
        assert loaded['type'] == 'result', loaded
        params = dict(prompt='portrait of a woman, flowers in background, detailed illustration',
            negative_prompt='low quality, blurry', seed=173, steps=12, width=512, height=512,
            cfg_scale=6, sampler_name='DPM++ 2M', scheduler='Automatic',
            hires=dict(enabled=True, scale=1.5, second_pass_steps=8, denoising_strength=.55),
            hooks=dict(hires_fix_tweaks=dict(batch_count=2, seed_enabled=True, seed=100)))
        if sdxl:
            params.update(prompt='a woman with red hair in a garden, portrait, detailed illustration',
                width=768, height=768, steps=20, cfg_scale=4)
            params['hires'].update(scale=1.25, second_pass_steps=12)
        request = dict(params=params, preview_every=0, lora_paths={'A': str(Path(lora).resolve())},
            hires_loras=[dict(name='A', weight=.3, unet=.5, lbw='HIRES_CUSTOM')],
            lbw_presets={'HIRES_CUSTOM': 'ALL0.5'})
        def run(name, req):
            response = client.call('txt2img', dict(req, output_dir=str(output / name)))
            assert response['type'] == 'result', response
            records = [dict(im, **pixel_record(im['path'])) for im in response['result']['images']]
            report['cases'][name] = dict(images=records)
            save()
            print(name, len(records), 'images', flush=True)
            return records
        custom = run('custom-hires-lbw', request)
        reference = copy.deepcopy(request)
        reference['hires_loras'] = [dict(name='A', weight=.15, unet=.25)]
        plain = run('equivalent-hires-weights', reference)
        assert len(custom) == len(plain) == 2
        for i in range(2):
            diff = image_difference(custom[i]['path'], plain[i]['path'])
            report['comparisons'][f'lbw-{i}'] = diff
            assert diff['exact'], diff
        off = copy.deepcopy(request)
        off['hires_loras'] = []
        baseline = run('hires-no-lora', off)
        assert not image_difference(custom[0]['path'], baseline[0]['path'])['exact'], 'Hires-only LoRA had no effect'
        if cn:
            hint = client.call('preprocess', dict(preprocessor='canny', image_path=baseline[0]['path'], output_dir=str(output), args={}))
            assert hint['type'] == 'result', hint
            conditioned = copy.deepcopy(off)
            conditioned['params']['controlnet'] = [dict(enabled=True, model='canny', image=hint['result']['path'],
                preprocessor='none', weight=.5, guidance_start=0., guidance_end=1.)]
            conditioned['controlnet_paths'] = {'canny': str(Path(cn).resolve())}
            repeated = run('controlnet-repeats', conditioned)
            assert len(repeated) == 2
            for i in range(2):
                single = copy.deepcopy(conditioned)
                single['params']['hooks']['hires_fix_tweaks'].update(batch_count=1, seed=100+i)
                independent = run(f'controlnet-single-{i}', single)
                diff = image_difference(repeated[i]['path'], independent[0]['path'])
                report['comparisons'][f'controlnet-{i}'] = diff
                assert diff['exact'], diff
            assert not image_difference(repeated[0]['path'], baseline[0]['path'])['exact'], 'ControlNet had no effect'
        recovery = run('recovery', off)
        report['comparisons']['recovery'] = image_difference(baseline[0]['path'], recovery[0]['path'])
        assert report['comparisons']['recovery']['exact']
        large = copy.deepcopy(off)
        large['params']['hooks']['hires_fix_tweaks'].update(batch_count=1, seed='9007199254740993')
        exact_text = run('large-seed-text', large)
        assert exact_text[0]['hires_seed'] == '9007199254740993'
        large['params']['hooks']['hires_fix_tweaks']['seed'] = 9007199254740993
        exact_number = run('large-seed-integer', large)
        report['comparisons']['large-seed'] = image_difference(exact_text[0]['path'], exact_number[0]['path'])
        assert report['comparisons']['large-seed']['exact']
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', failure=repr(exc))
        raise
    finally:
        save()
        client.close()


if __name__ == '__main__':
    main()
