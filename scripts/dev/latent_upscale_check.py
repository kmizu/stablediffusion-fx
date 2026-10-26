"""MANUAL real-model acceptance, run once per checkpoint. No downloads or source generation.

Usage: <engine-python> scripts/dev/latent_upscale_check.py checkpoint.safetensors init.png output-dir [--full]
Set PYTHONPATH to this checkout's python directory. Requires actual SD1.5/SDXL model and GPU-capable environment.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time

from PIL import Image, ImageDraw
from sdfx.engine import Engine
from sdfx.protocol import Emitter, JobContext
from sdfx.latent_upscale import METHODS, SCHEDULES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('source')
    parser.add_argument('output')
    parser.add_argument('--full', action='store_true', help='exercise each meaningful interpolation and each scheduler')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(args.source) as im:
        source = im.convert('RGB')
    width, height = [max(64, int(x * 1.25) // 8 * 8) for x in source.size]
    mask = Image.new('L', source.size)
    ImageDraw.Draw(mask).ellipse((source.width // 4, source.height // 4, source.width * 3 // 4, source.height * 3 // 4), fill=255)
    mask_path = output / 'mask.png'
    mask.save(mask_path)
    engine = Engine()
    report = {'source': str(Path(args.source).resolve()), 'target_size': [width, height], 'cases': []}
    context = lambda name: JobContext(Emitter(io.BytesIO()), name)
    report['model'] = engine.load_model({'path': str(Path(args.checkpoint).resolve()), 'dtype': 'fp16', 'vram_mode': 'gpu'}, context('load'))
    trace = []
    original_encode = engine.pipe.vae.encode
    def encode(image, *a, **kw):
        trace.append(list(image.shape))
        return original_encode(image, *a, **kw)
    engine.pipe.vae.encode = encode
    cases = [('baseline', None, None, False, False, 1),
             ('nearest-simple', 'nearest-exact', 'simple', False, False, 1),
             ('area-normal-batch', 'area', 'normal', False, False, 2),
             ('bicubic-aa-karras', 'bicubic-antialiased', 'karras', False, False, 1),
             ('masked-nearest', 'nearest-exact', 'simple', True, False, 1),
             ('masked-full-res', 'bilinear', 'automatic', True, True, 1)]
    if args.full:
        cases += [(f'method-{method}', method, 'simple', False, False, 1) for method in METHODS]
        cases += [(f'schedule-{scheduler}', 'nearest-exact', scheduler, False, False, 1) for scheduler in SCHEDULES]
    cases += [('disabled', None, None, False, False, 1), ('restored-baseline', None, None, False, False, 1)]
    try:
        for name, method, scheduler, masked, full_res, batch in cases:
            case_dir = output / name
            case_dir.mkdir(exist_ok=True)
            trace.clear()
            events = io.BytesIO()
            ctx = JobContext(Emitter(events), name)
            params = dict(prompt='detailed texture, crisp edges', negative_prompt='blurry', steps=7,
                sampler_name='Euler', scheduler='Uniform', cfg_scale=3, seed=321,
                width=width, height=height, batch_size=batch,
                img2img=dict(init_image=str(Path(args.source).resolve()), denoising_strength=.4,
                             mask=str(mask_path) if masked else None, mask_blur=2, inpaint_full_res=full_res),
                hooks={'latent_upscale': dict(method=method, scheduler=scheduler)} if method else {})
            if name == 'disabled':
                params['hooks'] = {'latent_upscale': {'enabled': False, 'method': 'nearest-exact', 'scheduler': 'simple'}}
            started = time.monotonic()
            result = engine.img2img({'params': params, 'output_dir': str(case_dir), 'preview_method': 'none'}, ctx)
            log = [json.loads(line) for line in events.getvalue().splitlines()]
            progress = [e for e in log if e.get('event') == 'progress']
            actual = max((e['step'] for e in progress), default=0)
            assert actual == (7 if method else 2), (name, actual)
            hashes = [hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() for item in result['images']]
            rgb_hashes = []
            if method and not full_res:
                assert all(shape[-2:] == [source.height // 8 * 8, source.width // 8 * 8] for shape in trace), (name, trace)
            assert len(result['images']) == batch
            for image in result['images']:
                with Image.open(image['path']) as im:
                    assert im.size == (source.size if full_res else (width, height))
                    rgb_hashes.append(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
            row = dict(name=name, request=params, result=result, actual_steps=actual,
                vae_encode_shapes=list(trace), png_sha256=hashes, rgb_sha256=rgb_hashes, elapsed_s=time.monotonic()-started)
            report['cases'].append(row)
            (case_dir / 'events.json').write_text(json.dumps(log, indent=2), encoding='utf-8')
            (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(name, 'steps', actual, 'VAE', trace, flush=True)
        baseline = report['cases'][0]['rgb_sha256']
        assert report['cases'][-1]['rgb_sha256'] == baseline, 'no-hook baseline changed after Latent Upscale'
        assert report['cases'][-2]['rgb_sha256'] == baseline, 'disabled hook changed ordinary generation'
        preview = Image.new('RGB', (320 * 3, 260 * 2))
        draw = ImageDraw.Draw(preview)
        for index, row in enumerate(report['cases'][:6]):
            with Image.open(row['result']['images'][0]['path']) as im:
                im.thumbnail((320, 240))
                preview.paste(im, ((index % 3) * 320, (index // 3) * 260 + 20))
            draw.text(((index % 3) * 320, (index // 3) * 260), row['name'], fill='white')
        preview.save(output / 'comparison.png')
        report['success'] = True
        (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    finally:
        engine.pipe.vae.encode = original_encode
        engine.unload({}, context('unload'))


if __name__ == '__main__':
    main()
