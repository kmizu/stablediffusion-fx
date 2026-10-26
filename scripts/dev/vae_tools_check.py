"""MANUAL real-model acceptance. Root runs GPU; never imported by tests.

python vae_tools_check.py CHECKPOINT SOURCE_IMAGE OUTPUT [--approx-nn WEIGHTS] [--taesd] [--allow-download] [--vae PATH]
SOURCE_IMAGE may be an existing verify-intermediate baseline final image. No model generation is needed for codec checks.
"""
import argparse
from contextlib import nullcontext
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from sdfx.engine import Engine
from sdfx.protocol import Emitter, JobContext
from sdfx import vae_tools


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('source')
    parser.add_argument('output')
    parser.add_argument('--approx-nn')
    parser.add_argument('--taesd', action='store_true')
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--vae')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = Path(args.source).resolve()
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    engine = Engine()
    context = lambda: JobContext(Emitter(io.BytesIO()), 'vae-tools-acceptance')
    loaded = engine.load_model(dict(path=str(Path(args.checkpoint).resolve()), dtype='fp16', vram_mode='gpu'), context())
    reports, checks = [], []
    def run(**changes):
        options = dict(input_paths=[str(source)], output_dir=str(root), operation='encode', representation='L',
                       approximation='Full', seed=71, tiled=False, allow_download=args.allow_download)
        options.update(changes)
        report = engine.vae_tools(options, context())
        assert all(item['status'] == 'complete' for item in report['results']), report
        assert hashlib.sha256(source.read_bytes()).hexdigest() == before
        reports.append(report)
        return report
    originals = {}
    for rep in ('L', 'RGBA', 'CMYK'):
        encoded = run(representation=rep)
        item = encoded['results'][0]
        originals[rep] = item
        decoded = run(operation='decode', representation=rep, input_paths=[item['path']])
        raw = run(operation='decode', input_paths=[item['latent_path']])
        roundtrip = run(operation='encode+decode', representation=rep)
        if rep == 'L':
            assert Image.open(raw['results'][0]['path']).tobytes() == Image.open(roundtrip['results'][0]['path']).tobytes()
        assert Image.open(decoded['results'][0]['path']).size == Image.open(source).size
    checks.append('Full all three representations, saved-file decode, raw safetensors and encode+decode; input bytes retained')
    run(operation='encode+decode', tiled=True)
    assert engine.pipe.vae.use_tiling == engine.load_opts.vae_tiling
    run(representation='CMYK', cmyk_format='JPEG100')
    jpeg = reports[-1]['results'][0]['path']
    run(operation='decode', representation='CMYK', input_paths=[jpeg])
    run(input_paths=[str(source), str(source)])
    checks.append('native tiling restored, CMYK JPEG100 saved-file decode, ordered batch')
    run(operation='decode', representation='RGBA', approximation='Approx cheap', input_paths=[originals['RGBA']['path']])
    if args.approx_nn:
        run(operation='decode', representation='RGBA', approximation='Approx NN', input_paths=[originals['RGBA']['path']], approx_nn_path=args.approx_nn)
        checks.append('real official Approx NN weights safely loaded and decoded')
    if args.taesd:
        run(operation='encode+decode', approximation='TAESD')
        checks.append('real TAESD encode+decode with diffusers [-1,1] pixel convention')
    if args.vae:
        engine.set_vae({'path': str(Path(args.vae).resolve())}, context())
        run(operation='decode', representation='RGBA', input_paths=[originals['RGBA']['path']])
        engine.set_vae({'path': None}, context())
        checks.append('explicit selected external VAE decode and checkpoint VAE restoration')
    # A real sampler run verifies that the catcher does not change successful RGB.
    request = dict(params=dict(prompt='a cat', steps=2, width=256, height=256, seed=71, sampler_name='Euler'),
                   output_dir=str(root / 'generation'), vae_recovery_dir=str(root), preview_method='none')
    original_catcher = vae_tools.catch_decode
    vae_tools.catch_decode = lambda *a: nullcontext()
    try:
        baseline = engine.txt2img(request, context())
    finally:
        vae_tools.catch_decode = original_catcher
    guarded = engine.txt2img(request, context())
    a, b = baseline['images'][0]['path'], guarded['images'][0]['path']
    assert Image.open(a).convert('RGB').tobytes() == Image.open(b).convert('RGB').tobytes()
    original_decode = engine.pipe.vae.decode
    def fail(*a, **kw):
        raise RuntimeError('acceptance injected final VAE decode error')
    engine.pipe.vae.decode = fail
    try:
        caught = engine.txt2img(request, context())
    finally:
        engine.pipe.vae.decode = original_decode
    assert caught['vae_recovery'] and Path(caught['vae_recovery'][0]['path']).suffix == '.png'
    run(operation='decode', input_paths=[caught['vae_recovery'][0]['latent_path']])
    restored = engine.txt2img(request, context())
    assert not restored.get('vae_recovery')
    assert Image.open(a).convert('RGB').tobytes() == Image.open(restored['images'][0]['path']).convert('RGB').tobytes()
    checks.append('real generation RGB invariance, injected final decode error recovery, exact latent decode and ordinary generation restoration')
    summary = dict(success=True, model=loaded, source_sha256=before, checks=checks, reports=reports,
        baseline_png_sha256=hashlib.sha256(Path(a).read_bytes()).hexdigest(),
        baseline_rgb_sha256=hashlib.sha256(Image.open(a).convert('RGB').tobytes()).hexdigest(), recovery=caught['vae_recovery'])
    (root / 'acceptance.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(dict(success=True, model=loaded['kind'], checks=len(checks), reports=len(reports))))


if __name__ == '__main__':
    main()
