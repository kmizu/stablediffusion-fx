"""Manual GPU check of VAE Tools session handling; pass checkpoint, source, output."""
import hashlib, io, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from sdfx.engine import Engine
from sdfx.protocol import Emitter, JobContext, EngineError
from sdfx.encrypt_image import encode_png, decode_png
from sdfx.vae_tools import read_latent

checkpoint, source, destination = map(Path, sys.argv[1:])
out = destination.resolve(); out.mkdir(parents=True, exist_ok=True)
source = source.resolve(); before = hashlib.sha256(source.read_bytes()).hexdigest()
protected = out / 'protected-input.png'; protected.write_bytes(encode_png(source.read_bytes(), 'check-password'))
protected_hash = hashlib.sha256(protected.read_bytes()).hexdigest()
def context(password=None):
    c = JobContext(Emitter(io.BytesIO()), 'vae-protected-gpu-check'); c.image_password = password; return c
engine = Engine(); assert engine.cuda
engine.load_model({'path': str(checkpoint.resolve()), 'dtype': 'fp16', 'vram_mode': 'gpu'}, context())
def run(path, rep, operation, password=None, consent=False):
    params = dict(input_paths=[str(path)], output_dir=str(out), operation=operation, representation=rep, seed=71, allow_plaintext_latent=consent)
    return engine.vae_tools(params, context(password))
def pixels(path, password=None):
    data = Path(path).read_bytes()
    if password: data, verified = decode_png(data, password); assert verified is True
    with Image.open(io.BytesIO(data)) as image: return image.mode, image.size, image.tobytes()
for rep in ('L', 'RGBA', 'CMYK'):
    plain = run(source, rep, 'encode')['results'][0]
    secured_report = run(protected, rep, 'encode', 'check-password', True)
    secured = secured_report['results'][0]
    assert plain['status'] == secured['status'] == 'complete', (plain, secured)
    assert 'plaintext' in secured_report['warnings'][0]
    assert json.loads(Path(secured_report['report_path']).read_text(encoding='utf8'))['warnings'] == secured_report['warnings']
    assert np.array_equal(read_latent(plain['latent_path'], 'L', 4), read_latent(secured['latent_path'], 'L', 4))
    assert np.array_equal(read_latent(plain['path'], rep, 4), read_latent(secured['path'], rep, 4, 'check-password'))
    with Image.open(secured['path']) as im:
        assert im.mode == rep and ('Encrypt' in im.info) == (rep == 'RGBA')
    ordinary_dec = run(plain['path'], rep, 'decode')['results'][0]
    secured_dec = run(secured['path'], rep, 'decode', 'check-password')['results'][0]
    assert ordinary_dec['status'] == secured_dec['status'] == 'complete', (ordinary_dec, secured_dec)
    assert pixels(ordinary_dec['path']) == pixels(secured_dec['path'], 'check-password')
    assert pixels(plain['preview_path']) == pixels(secured['preview_path'], 'check-password')
    if rep == 'RGBA':
        wrong = run(secured['path'], rep, 'decode', 'wrong')['results'][0]
        assert wrong['status'] == 'error' and 'password' in wrong['error']
    decoded, _ = decode_png(Path(secured_dec['path']).read_bytes(), 'check-password')
    (out / f'{rep}-decoded-for-inspection.png').write_bytes(decoded)
    print(f'PASS {rep}: latent parity, retained mode, protected preview/decoded pixels, readable report', flush=True)
wrong = run(protected, 'L', 'encode+decode', 'wrong')['results'][0]
assert wrong['status'] == 'error' and 'password' in wrong['error']
try: run(protected, 'L', 'encode', 'check-password')
except EngineError as e: assert 'consent' in str(e)
else: raise AssertionError('missing plaintext consent should fail')
assert hashlib.sha256(source.read_bytes()).hexdigest() == before
assert hashlib.sha256(protected.read_bytes()).hexdigest() == protected_hash
print('PASS wrong password, plaintext consent, source hashes', flush=True)
