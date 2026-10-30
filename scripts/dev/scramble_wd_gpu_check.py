"""Real WD v3 and diffusion worker check for protected batch snapshots."""
import io, json, os, subprocess, sys
from pathlib import Path
from PIL import Image
from sdfx.encrypt_image import encode_png, decode_png

python, checkpoint, output, model, tags, source = sys.argv[1:]
out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=True)
wdtemp = out / 'wd-temp'; wdtemp.mkdir()
source_bytes = Path(source).read_bytes()
for variant in ('plain', 'protected'):
    folder = out / variant; folder.mkdir()
    (folder / 'source.png').write_bytes(encode_png(source_bytes, 'test-wd-password') if variant == 'protected' else source_bytes)
(out / 'results').mkdir()
with (out / 'worker-stderr.log').open('w', encoding='utf-8') as err:
    proc = subprocess.Popen([python, '-m', 'sdfx', '--wd-v3-model', str(Path(model).resolve()),
                             '--wd-v3-tags', str(Path(tags).resolve()), '--wd-v3-temp', str(wdtemp)],
                            cwd=Path('python').resolve(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=err, text=True, encoding='utf-8', env=dict(os.environ, PYTHONUNBUFFERED='1'))
    try:
        hello = json.loads(proc.stdout.readline())
        assert 'batch_interrogate_wd_v3_partial' in hello.get('features', []), hello
        seq = 0
        def call(kind, params, secret=False):
            global seq
            seq += 1; rid = str(seq)
            request = {'id': rid, 'type': kind, 'owner_id': 'test-owner', 'params': params}
            if secret: request['image_password'] = 'test-wd-password'
            proc.stdin.write(json.dumps(request) + '\n'); proc.stdin.flush()
            for line in proc.stdout:
                msg = json.loads(line)
                if msg.get('id') == rid and msg.get('type') in ('result', 'error'):
                    if msg['type'] != 'result': raise RuntimeError(json.dumps(msg))
                    return msg['result']
            raise RuntimeError('worker exited')
        model_result = call('load_model', {'path': str(Path(checkpoint).resolve()), 'dtype': 'fp16', 'vram_mode': 'gpu'})
        token = model_result['model_token']
        results = []
        for variant in ('plain', 'protected'):
            gp = {'prompt': 'portrait, detailed', 'negative_prompt': 'low quality', 'seed': 42,
                  'steps': 3, 'cfg_scale': 6.0, 'width': 256, 'height': 256, 'batch_size': 1, 'n_iter': 1,
                  'sampler_name': 'Euler', 'scheduler': 'Automatic',
                  'img2img': {'init_image': '', 'denoising_strength': .65}}
            request = {'images_root': str(out / variant), 'generation': {'params': gp, 'output_dir': str(out),
                       'preview_every': 0, 'preview_method': 'none'}, 'output_parent': str(out / 'results'),
                       'output_name': variant, 'checkpoint_token': token}
            prep = call('batch_interrogate_prepare', request)
            result = call('batch_interrogate_execute', {'token': prep['token']}, variant == 'protected')
            row = result['rows'][0]
            assert row['status'] == 'complete', row
            assert 'test-wd-password' not in json.dumps(result)
            results.append(row)
        assert results[0]['positive'] == results[1]['positive']
        plain = Path(results[0]['result']['images'][0]['path'])
        scrambled = Path(results[1]['result']['images'][0]['path'])
        decoded, verified = decode_png(scrambled.read_bytes(), 'test-wd-password')
        with Image.open(plain) as a, Image.open(io.BytesIO(decoded)) as b, Image.open(scrambled) as c:
            assert verified is True and c.info['Encrypt'] == 'pixel_shuffle_3'
            assert a.convert('RGB').tobytes() == b.convert('RGB').tobytes()
            assert 'WD interrogators:' in c.info['parameters'] and 'test-wd-password' not in str(c.info)
            b.save(out / 'decoded.png')
        assert (out / 'plain' / 'source.png').read_bytes() == source_bytes
        assert decode_png((out / 'protected' / 'source.png').read_bytes(), 'test-wd-password')[1]
        assert not list((out / 'results').rglob('.inputs'))
        print('PASS real WD v3 and diffusion: caption, protected PNG and decoded RGB parity', flush=True)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
