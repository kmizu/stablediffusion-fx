"""Real worker JSONL check for protected intermediate PNG frames; not release acceptance."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from PIL import Image
from sdfx.encrypt_image import decode_png

python, checkpoint, out = sys.argv[1:4]
steps_animation = len(sys.argv) > 4 and sys.argv[4] == '--steps-animation'
video = len(sys.argv) > 5 and sys.argv[5] == '--video'
out = Path(out).resolve()
out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONUNBUFFERED='1')
with (out / 'worker-stderr.log').open('w', encoding='utf-8') as err:
    proc = subprocess.Popen([python, '-m', 'sdfx'], cwd=Path('python').resolve(), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=err, text=True, encoding='utf-8', env=env)
    try:
        assert json.loads(proc.stdout.readline())['type'] == 'hello'
        seq = 0
        def call(kind, params, protected=False):
            global seq
            seq += 1
            rid = str(seq)
            envelope = {'id': rid, 'type': kind, 'params': params}
            if protected:
                envelope['image_password'] = 'intermediate-test-password'
                if video:
                    envelope['allow_plaintext_video'] = True
            proc.stdin.write(json.dumps(envelope) + '\n')
            proc.stdin.flush()
            for line in proc.stdout:
                msg = json.loads(line)
                if msg.get('id') == rid and msg.get('type') in ('result', 'error'):
                    if msg['type'] != 'result':
                        raise RuntimeError(json.dumps(msg))
                    return msg
            raise RuntimeError('worker exited before response')
        call('load_model', {'path': str(Path(checkpoint).resolve()), 'dtype': 'fp16', 'vram_mode': 'gpu'})
        outputs = []
        for protected in (False, True):
            folder = out / ('protected' if protected else 'plain')
            folder.mkdir()
            gp = {'prompt': 'portrait of a girl, detailed, soft light', 'negative_prompt': 'low quality',
                  'seed': 42, 'subseed': 0, 'steps': 3, 'cfg_scale': 6.0, 'width': 384, 'height': 384,
                  'batch_size': 1, 'n_iter': 1, 'sampler_name': 'Euler', 'scheduler': 'Automatic',
                  'hooks': ({'steps_animation': {'create_output': video, 'duration': .5, 'fps': 10,
                                                'interpolation': 'none'}} if steps_animation else
                            {'save_intermediate': {'every_n': 1, 'image_type': 'Denoised'}})}
            call('txt2img', {'params': gp, 'output_dir': str(folder), 'preview_every': 0, 'preview_method': 'none'}, protected)
            pattern = 'steps-animation/frames/generation-*/*.png' if steps_animation else 'intermediates/generation-*/*.png'
            frames = sorted(folder.glob(pattern))
            assert frames, 'no intermediate frames'
            outputs.append(frames)
        assert len(outputs[0]) == len(outputs[1])
        for plain, scrambled in zip(*outputs):
            decoded, verified = decode_png(scrambled.read_bytes(), 'intermediate-test-password')
            with Image.open(plain) as a, Image.open(io.BytesIO(decoded)) as b, Image.open(scrambled) as c:
                assert verified is True and c.info['Encrypt'] == 'pixel_shuffle_3'
                assert a.convert('RGB').tobytes() == b.convert('RGB').tobytes()
                assert a.convert('RGB').tobytes() != c.convert('RGB').tobytes()
                assert 'intermediate-test-password' not in str(c.info)
        if video:
            videos = [next((out / name / 'steps-animation/videos').glob('*.mp4')) for name in ('plain', 'protected')]
            def rgb(path):
                return subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path),
                                                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
            assert rgb(videos[0]) == rgb(videos[1])
            assert not list((out / 'protected').rglob('.steps-encode-*'))
        print('PASS real worker intermediate protected frame pixel parity:', len(outputs[0]), flush=True)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
