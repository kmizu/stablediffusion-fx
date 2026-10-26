"""Parent-owned real-model acceptance: PYTHON script.py ENGINE_PY MODEL OUT [--sdxl] [--video].

Starts only the standalone sdfx engine. Saves raw responses, equality checks, manifests,
and contact sheets for human visual review. No model download or environment mutation.
"""
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('engine_python')
    parser.add_argument('model')
    parser.add_argument('out')
    parser.add_argument('--sdxl', action='store_true')
    parser.add_argument('--video', action='store_true', help='Also verify ffmpeg GIF, WebM and MP4 export')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(root / 'python'), PYTHONUNBUFFERED='1', USE_TF='0')
    # GPU execution is intentionally left to the person invoking this acceptance script.
    with (output / 'stderr.log').open('w', encoding='utf-8') as stderr:
        process = subprocess.Popen([str(Path(args.engine_python).resolve()), '-m', 'sdfx'],
            cwd=root / 'python', env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
            text=True, encoding='utf-8')
        messages = queue.Queue()
        def reader():
            for line in process.stdout:
                messages.put(json.loads(line))
            messages.put({'type': 'eof'})
        threading.Thread(target=reader, daemon=True).start()
        counter = 0
        def call(kind, params):
            nonlocal counter
            counter += 1
            rid = f'check-{counter}'
            process.stdin.write(json.dumps(dict(id=rid, type=kind, params=params)) + '\n')
            process.stdin.flush()
            while True:
                result = messages.get(timeout=1200)
                if result['type'] == 'eof':
                    raise RuntimeError('engine exited; see stderr.log')
                if result.get('id') == rid and result['type'] in ('result', 'error'):
                    if result['type'] == 'error':
                        raise RuntimeError(result)
                    return result['result']
        reports = {}
        try:
            hello = messages.get(timeout=120)
            assert hello['type'] == 'hello' and 'save_intermediate' in hello['features'], hello
            call('load_model', dict(path=str(Path(args.model).resolve()), dtype='fp16', vram_mode='gpu'))
            size = 768 if args.sdxl else 512
            base = dict(prompt='a red lighthouse on a rocky island at sunset, detailed landscape',
                        negative_prompt='low quality', seed=812, subseed=123, steps=8, width=size, height=size,
                        cfg_scale=6, sampler_name='DPM++ 2M', batch_size=1, n_iter=1)
            def generate(name, hook=None, **overrides):
                p = dict(base, **overrides)
                p['hooks'] = {} if hook is None else {'save_intermediate': hook}
                result = call('img2img' if p.get('img2img') else 'txt2img',
                    dict(params=p, output_dir=str(output / name), preview_every=0, preview_method='approx'))
                reports[name] = result
                (output / f'{name}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
                print(name, 'images=', len(result['images']), 'auxiliaries=', len(result.get('intermediate_artifacts', [])), flush=True)
                return result
            from PIL import Image, ImageChops, ImageOps, ImageDraw
            baseline = generate('baseline')
            def equal(result):
                with Image.open(baseline['images'][0]['path']) as a, Image.open(result['images'][0]['path']) as b:
                    return ImageChops.difference(a.convert('RGB'), b.convert('RGB')).getbbox() is None
            equality = {}
            for subject in ('Denoised', 'Noisy', 'Preview'):
                result = generate(subject.lower(), dict(every_n=2, image_type=subject, final_save=True))
                equality[subject] = equal(result)
                frames = [a for a in result['intermediate_artifacts'] if a['type'] == 'image']
                assert len(frames) == 5 and [f['step'] for f in frames] == [2, 4, 6, 8, None]
            equality['disabled'] = equal(generate('disabled', dict(enabled=False)))
            generate('batch', dict(every_n=2, start_at=3, stop_at=7, mode='Suffix'), batch_size=2, n_iter=2,
                     width=384, height=384)
            hires = generate('hires', dict(every_n=2, final_save=True),
                hires=dict(enabled=True, scale=1.25, upscaler='Latent', second_pass_steps=4, denoising_strength=.5))
            assert {a.get('pass_name') for a in hires['intermediate_artifacts'] if a['type'] == 'image'} == {'base', 'hires', 'final'}
            generate('img2img', dict(every_n=1, stop_at=3),
                img2img=dict(init_image=baseline['images'][0]['path'], denoising_strength=.5))
            generate('heun', dict(every_n=2), sampler_name='Heun')
            if args.video:
                for fmt in ('gif', 'webm', 'mp4'):
                    result = generate('video-' + fmt, dict(every_n=2, mode='Video', format=fmt,
                        final_save=True, add_first_frames=2, add_last_frames=2, ffmpeg_bat=True, fps=6, seconds=3))
                    assert any(a['type'] == 'video' for a in result['intermediate_artifacts'])
                    assert not list((output / ('video-' + fmt)).rglob('.video-*'))
            for name, result in reports.items():
                frames = [a for a in result.get('intermediate_artifacts', []) if a['type'] == 'image']
                if not frames:
                    continue
                thumbs = []
                for f in frames[:32]:
                    with Image.open(f['path']) as im:
                        assert 'sdfx_intermediate' in im.info
                        thumb = ImageOps.pad(im.convert('RGB'), (192, 192))
                    ImageDraw.Draw(thumb).text((3, 3), f"{f['pass_name']} {f['step']} seed={f['seed']}", fill='white', stroke_fill='black', stroke_width=1)
                    thumbs.append(thumb)
                sheet = Image.new('RGB', (192 * 4, 192 * ((len(thumbs) + 3) // 4)))
                for i, thumb in enumerate(thumbs):
                    sheet.paste(thumb, (i % 4 * 192, i // 4 * 192))
                sheet.save(output / f'{name}-contact.png')
            (output / 'equality.json').write_text(json.dumps(equality, indent=2), encoding='utf-8')
            assert all(equality.values()), equality
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    main()
