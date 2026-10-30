"""Real-model injected decode failure; verifies protected recovery PNG and exact tensor."""
import io, sys
from pathlib import Path
import torch
from PIL import Image
from safetensors.torch import load_file
from sdfx.engine import Engine
from sdfx.protocol import Emitter, JobContext
from sdfx.encrypt_image import decode_png

checkpoint, output = sys.argv[1:]
out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=True)
engine = Engine()
context = JobContext(Emitter(io.BytesIO()), 'vae-recovery')
engine.load_model({'path': str(Path(checkpoint).resolve()), 'dtype': 'fp16', 'vram_mode': 'gpu'}, context)
context.image_password = 'test-recovery-password'
captured = []
vae = engine.pipe.vae
original = vae.decode
def fail(z, *args, **kwargs):
    captured.append(z.detach().cpu().clone())
    raise RuntimeError('injected decoder failure')
vae.decode = fail
request = {'params': {'prompt': 'portrait', 'negative_prompt': 'blurry', 'seed': 17, 'steps': 3,
                      'width': 256, 'height': 256, 'sampler_name': 'Euler'},
           'output_dir': str(out / 'images'), 'vae_recovery_dir': str(out / 'recovery'), 'preview_method': 'none'}
try:
    result = engine.txt2img(request, context)
finally:
    vae.decode = original
assert result.get('vae_recovery') and captured
artifact = result['vae_recovery'][0]
plain, verified = decode_png(Path(artifact['path']).read_bytes(), 'test-recovery-password')
with Image.open(io.BytesIO(plain)) as image:
    assert verified is True and image.mode == 'RGBA'
    image.save(out / 'recovery-decoded.png')
assert torch.equal(load_file(artifact['latent_path'])['tensor'].view(torch.uint8), captured[0].view(torch.uint8))
assert b'remains readable' in context.emitter._stream.getvalue()
with Image.open(result['images'][0]['path']) as image:
    image.save(out / 'broken-preview.png')
print('PASS real model recovery: scrambled PNG, exact readable tensor, warning and broken preview')
