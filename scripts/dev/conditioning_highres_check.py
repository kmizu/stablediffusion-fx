"""Coordinated real-model acceptance; never run concurrently with another GPU job."""
import argparse
import io
import json
from pathlib import Path
import sys
import threading


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('photo')
    ap.add_argument('output')
    ap.add_argument('--engine-root', default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument('--vram-mode', default='model_offload')
    ap.add_argument('--four-channel', action='store_true')
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.engine_root).resolve() / 'python'))
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from sdfx.engine import Engine
    from sdfx.protocol import Emitter, JobContext, EngineError

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Image.open(args.photo).convert('RGB')
    source.save(output / 'source.png')
    mask = Image.new('L', source.size)
    ImageDraw.Draw(mask).ellipse((source.width * .25, source.height * .12,
                                 source.width * .72, source.height * .7), fill=255)
    mask.save(output / 'mask.png')
    events = io.BytesIO()
    interrupt = threading.Event()
    class TraceEmitter(Emitter):
        cancel_next = False
        def emit(self, obj):
            super().emit(obj)
            if self.cancel_next and obj.get('event') == 'progress':
                self.cancel_next = False
                interrupt.set()
    emitter = TraceEmitter(events)
    ctx = JobContext(emitter, 'conditioning-real', interrupt)
    engine = Engine()
    report = {'status': 'running', 'cases': {}, 'conditioning_traces': []}
    def record():
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    def pixels(result):
        return np.asarray(Image.open(result['images'][-1]['path']).convert('RGB'))
    # Independent transcription of the pinned source's seven branches.
    formulas = {'empty': lambda d: 1., 'lowest': lambda d: 1. - d / 4,
                'low': lambda d: 1. - d / 2, 'medium': lambda d: .5,
                'high': lambda d: d / 2, 'highest': lambda d: d / 4, 'full': lambda d: 0.}
    try:
        info = engine.load_model({'path': str(Path(args.model).resolve()), 'dtype': 'fp16',
                                  'vram_mode': args.vram_mode}, ctx)
        channels = engine.pipe.unet.config.in_channels
        assert channels == (4 if args.four_channel else 9), (info, channels)
        report['model'] = info
        report['channels'] = channels
        side = 768 if engine.sdxl else 512
        run = engine._run
        def traced(task, gp, req, context, **kw):
            expected = None
            if channels == 9 and task == 'inpaint' and kw.get('image') is not None:
                pipe = engine._get_pipe(task)
                d = kw.get('conditioning_denoise')
                if d is None:
                    d = gp.hires.denoising_strength if kw['pass_'] == 'hires' else (
                        (gp.img2img.denoising_strength if gp.img2img else None)
                        if kw['pass_'] == 'base' else kw['strength'])
                mode = gp.hooks.get('conditioning_highres_fix', {}).get('mode', 'disabled')
                weight = formulas[mode](d) if mode in formulas and channels == 9 and d is not None else 1.
                latent = kw.get('latent_source')
                image = latent[0] if latent else kw['image']
                actual_mask = latent[1] if latent else kw['mask']
                h, w = (image.height, image.width) if latent else (kw['h'], kw['w'])
                original = pipe.image_processor.preprocess(image, height=h, width=w).float()
                if latent:
                    m = actual_mask.convert('L').resize((w, h), Image.Resampling.NEAREST)
                    m = torch.from_numpy(np.asarray(m).copy())[None, None].float() / 255
                else:
                    m = pipe.mask_processor.preprocess(actual_mask, height=h, width=w).float()
                expected = original * (1. - (m >= .5).float() * weight)
            captured = []
            vae = engine.pipe.vae
            encode = vae.encode
            def capture(x, *a, **k):
                captured.append(x.detach().float().cpu().clone())
                return encode(x, *a, **k)
            vae.encode = capture
            try:
                result = run(task, gp, req, context, **kw)
                if expected is not None:
                    assert captured, 'masked conditioning was not encoded'
                    error = float((captured[-1] - expected).abs().max())
                    assert error < .003, (kw['pass_'], error)
                    report['conditioning_traces'].append({'pass': kw['pass_'], 'max_error': error,
                                                         'latent_source': kw.get('latent_source') is not None})
                return result
            finally:
                vae.encode = encode
        engine._run = traced
        base = dict(prompt='portrait photograph of a woman astronaut in an orange flight suit, detailed, natural light',
                    negative_prompt='cartoon, illustration, blurry, distorted', steps=24, seed=173,
                    width=side, height=side, cfg_scale=5, sampler_name='Euler', scheduler='Simple',
                    img2img=dict(init_image=str(output / 'source.png'), mask=str(output / 'mask.png'),
                                 denoising_strength=.5, mask_blur=4, inpainting_fill=1))
        def generate(name, params, operation='img2img'):
            result = getattr(engine, operation)({'params': params, 'output_dir': str(output / name),
                                                'preview_method': 'none'}, ctx)
            report['cases'][name] = result['images'][-1]['path']
            record()
            print(name, result['images'][-1]['path'], flush=True)
            return result
        baseline = generate('disabled', base)
        for strength in (.35, .7):
            reference = generate(f'disabled-{strength}', dict(base, img2img=dict(base['img2img'], denoising_strength=strength)))
            for mode in formulas:
                params = dict(base, img2img=dict(base['img2img'], denoising_strength=strength),
                              hooks={'conditioning_highres_fix': {'mode': mode}})
                actual = generate(f'{mode}-{strength}', params)
                if args.four_channel or mode == 'empty':
                    assert np.array_equal(pixels(reference), pixels(actual)), (mode, 'no-op changed RGB')
        active = dict(base, hooks={'conditioning_highres_fix': {'mode': 'high'}})
        generate('inverted-crop', dict(active, img2img=dict(base['img2img'],
                 inpainting_mask_invert=True, inpaint_full_res=True, inpaint_full_res_padding=24)))
        generate('latent-source', dict(active, hooks=dict(active['hooks'],
                 latent_upscale={'method': 'nearest-exact', 'scheduler': 'simple'})))
        generate('img2img-unmasked', dict(active, img2img=dict(base['img2img'], mask=None)))
        hires_params = dict(active, img2img=None, width=512, height=512,
                 hires=dict(enabled=True, scale=1.5, upscaler='Lanczos', second_pass_steps=16,
                            denoising_strength=.5))
        hires = generate('hires', hires_params, 'txt2img')
        if args.four_channel:
            hires_disabled = generate('hires-disabled', dict(hires_params, hooks={}), 'txt2img')
            assert np.array_equal(pixels(hires), pixels(hires_disabled)), '4ch Hires changed RGB'
            report['hires_noop_exact'] = True
        emitter.cancel_next = True
        try:
            generate('cancelled', active)
            raise AssertionError('running interruption did not stop generation')
        except EngineError as e:
            assert e.code == 'interrupted', e
            report['interrupted'] = True
        finally:
            interrupt.clear()
            emitter.cancel_next = False
        recovered = generate('disabled-recovery', base)
        assert np.array_equal(pixels(baseline), pixels(recovered)), 'disabled recovery RGB changed'
        if args.four_channel:
            assert 'no effect without a 9-channel' in events.getvalue().decode('utf-8')
        report.update(status='passed-pending-visual-inspection', disabled_recovery_exact=True)
        record()
    except BaseException as e:
        report.update(status='failed', failure=repr(e)); record(); raise
    finally:
        (output / 'events.jsonl').write_bytes(events.getvalue())
        engine._free()


if __name__ == '__main__':
    main()
