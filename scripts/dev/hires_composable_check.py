"""Real SD1.5/SDXL manual/tag provenance and Hires repeat equivalence check."""
import copy
import json
from pathlib import Path
import sys
import time

from composable_lora_check import Client
from prompt_fusion_check import pixel_record, image_difference


def main():
    executable, model, lora, destination = sys.argv[1:5]
    visual_only = '--visual-only' in sys.argv[5:]
    root = Path(__file__).resolve().parents[2]
    output = Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=True)
    client = Client(executable, root, output, False)
    report = dict(status='running', cases={}, comparisons={})
    def save():
        (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    try:
        assert client.receive(time.monotonic() + 180)['type'] == 'hello'
        result = client.call('load_model', dict(path=str(Path(model).resolve()), dtype='fp16', vram_mode='gpu'))
        assert result['type'] == 'result', result
        base = dict(prompt='a woman with red hair <lora:A:.1:.2> AND a woman in a garden',
            negative_prompt='low quality, blurry', seed=173, steps=8, width=512, height=512,
            cfg_scale=6, sampler_name='DPM++ 2M', scheduler='Automatic',
            loras=[dict(name='A', weight=.2, unet=.3)],
            hires=dict(enabled=True, scale=1.5, second_pass_steps=6, denoising_strength=.55,
                prompt='a woman with red hair AND a woman in a garden', upscaler='Latent'),
            hooks=dict(composable_lora=dict(plot_weights=True),
                hires_fix_tweaks=dict(batch_count=2, seed_enabled=True, seed=100, prompts_resolved=True)))
        manual = [dict(name='A', weight=.2, unet=.3)]
        controls = ('composable-balanced', 'composable-balanced-manual') if visual_only else (
            'plain-and', 'composable-and', 'plain-single', 'native-lora-single', 'composable-lora-single')
        for control in controls:
            params = copy.deepcopy(base)
            params['loras'] = []
            params['prompt'] = params['hires']['prompt']
            if control.endswith('single'):
                params['prompt'] = params['hires']['prompt'] = 'a woman with red hair in a garden'
            if not control.startswith('composable-'):
                del params['hooks']['composable_lora']
            refs = []
            if 'balanced' in control:
                params['cfg_scale'] = 3
            if 'lora-single' in control or control.endswith('manual'):
                params['loras'] = copy.deepcopy(manual)
                refs = manual
            response = client.call('txt2img', dict(params=params, manual_loras=refs, lora_paths={'A': str(Path(lora).resolve())},
                output_dir=str(output / control), preview_every=0))
            assert response['type'] == 'result', response
            report['cases'][control] = dict(images=[pixel_record(im['path']) for im in response['result']['images']])
            save()
            print(control, 'generated', flush=True)
        if visual_only:
            report['status'] = 'generated-for-visual-review'
            return
        def run(name, params, refs):
            response = client.call('txt2img', dict(params=params, manual_loras=refs,
                lora_paths={'A': str(Path(lora).resolve())}, output_dir=str(output / name), preview_every=0))
            assert response['type'] == 'result', response
            result = response['result']
            images = [im for im in result['images'] if im.get('hires_seed') is not None]
            assert len(images) == 2, result
            assert [im['hires_seed'] for im in images] == [100, 101]
            records = [pixel_record(im['path']) for im in images]
            assert all((im['width'], im['height']) == (768, 768) for im in records)
            trace = result['composable_lora_trace']['records']
            hires = [r for r in trace if r['pass_name'] == 'hires' and not r['is_negative']]
            assert hires and all(abs(r['mean_weight'] - .3) < 1e-6 for r in hires), hires
            report['cases'][name] = dict(images=records, trace=trace)
            save()
            print(name, 'passed', flush=True)
            return records
        for stepping in (False, True):
            params = copy.deepcopy(base)
            params['hooks']['composable_lora']['with_step'] = stepping
            name = 'step' if stepping else 'plain'
            actual = run(name + '-manual', params, manual)
            equivalent = copy.deepcopy(params)
            equivalent['prompt'] = 'a woman with red hair <lora:A:.3:.5> AND a woman in a garden <lora:A:.2:.3>'
            equivalent['hires']['prompt'] = 'a woman with red hair <lora:A:.2:.3> AND a woman in a garden <lora:A:.2:.3>'
            reference = run(name + '-tags', equivalent, [])
            for i, (a, b) in enumerate(zip(actual, reference)):
                diff = image_difference(a['path'], b['path'])
                report['comparisons'][f'{name}-{i}'] = diff
                save()
                assert diff['exact'], diff
        params = copy.deepcopy(base)
        params['hooks']['composable_lora'].update(uc_text_model_encoder=True, uc_diffusion_model=True)
        run('negative-both', params, manual)
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', failure=repr(exc))
        raise
    finally:
        save()
        client.close()


if __name__ == '__main__':
    main()
