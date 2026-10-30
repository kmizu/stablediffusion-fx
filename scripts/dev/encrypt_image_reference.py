"""Generate pixel fixtures by running the pinned upstream standalone core, never the host UI."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/encrypt-image-source/scripts/core/core.py'
EXPECTED = '31b06b74b6cf8ebe7d294469eebe588cd767e4675cbcdb00106e19fb5bd61065'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
spec = importlib.util.spec_from_file_location('upstream_encrypt_core', SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
import numpy as np
from PIL import Image

cases = []
for version in (1, 2, 3):
    for width, height, mode, password in [(1, 1, 'RGB', 'a'), (2, 5, 'RGB', 'abc'), (7, 3, 'RGBA', '日本語🔑'), (9, 11, 'RGB', 'pass word'), (67, 3, 'RGB', 'wrap'), (3, 67, 'RGBA', '鍵')]:
        channels = len(mode)
        pixels = np.array([(i * 73 + 19) % 256 for i in range(width * height * channels)], dtype=np.uint8).reshape(height, width, channels)
        image = Image.fromarray(pixels, mode).copy()
        key = mod.get_sha256(password)
        encrypt = getattr(mod, 'encrypt_image' + (f'_v{version}' if version > 1 else ''))
        decrypt = getattr(mod, 'decrypt_image' + (f'_v{version}' if version > 1 else ''))
        result = encrypt(image, key)
        if version == 3:
            image = Image.fromarray(result)
        scrambled = np.asarray(image).tobytes().hex()
        result = decrypt(image, key)
        if version == 3:
            image = Image.fromarray(result)
        assert image.tobytes() == pixels.tobytes()
        cases.append(dict(version=version,width=width,height=height,mode=mode,password=password,plain=pixels.tobytes().hex(),scrambled=scrambled))
output = ROOT / 'python/tests/fixtures/encrypt_image_reference.json'
output.write_text(json.dumps(dict(source_sha256=EXPECTED,cases=cases),ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(f'{len(cases)} source cases written to {output}')
