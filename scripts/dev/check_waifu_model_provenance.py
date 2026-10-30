"""Research gate: compare extension mirrors with licensed model distributions."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
import zipfile

root = Path('outputs/waifu2x-models')
root.mkdir(parents=True, exist_ok=True)
original = [json.loads(line) for line in Path('outputs/waifu2x-original-model-tree.jsonl').read_text().splitlines()]
mirror = [json.loads(line) for line in Path('outputs/waifu2x-wrapper-assets.jsonl').read_text().splitlines()]
revision = 'cc385f97a9debfe611316aabfd5d8bb30ba2dbeb'
wrapper_revision = 'feede81c2b15ebcd5ff881b3f5ad4c3dbf017368'

def fetch(item):
    style, noise = item
    relative = f'models/upconv_7/{"art" if style == "anime" else style}/noise{noise}_scale2.0x_model.json'
    origin = next(entry for entry in original if entry['path'] == relative)
    mirrored = next(entry for entry in mirror if entry['path'].endswith(f'Upconv_7/{style}/noise{noise}_scale2.0x_model.json'))
    assert origin['sha'] == mirrored['sha'] and origin['size'] == mirrored['size']
    url = f'https://raw.githubusercontent.com/nagadomi/waifu2x/{revision}/{relative}'
    path = root / f'{style}-noise{noise}.json'
    if not path.exists():
        with urlopen(url, timeout=60) as response:
            data = response.read(origin['size'] + 1)
        path.write_bytes(data)
    data = path.read_bytes()
    assert len(data) == origin['size']
    assert hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == origin['sha']
    return dict(id=f'{style}-{noise}', filename=path.name, url=url, bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(), git_blob=origin['sha'], license='MIT')

with ThreadPoolExecutor(max_workers=4) as pool:
    assets = list(pool.map(fetch, [(style, noise) for style in ('anime', 'photo') for noise in range(4)]))
with zipfile.ZipFile('outputs/waifu2x-carn-original.zip') as archive:
    data = archive.read('CARN_model_checkpoint.pt')
    assert data == Path('outputs/waifu2x-carn-mirror.pt').read_bytes()
    (root / 'carn-v2.pt').write_bytes(data)
    assets.append(dict(id='carn-v2', filename='carn-v2.pt',
        url=f'https://raw.githubusercontent.com/light-and-ray/sd-webui-waifu2x-upscaler/{wrapper_revision}/waifu2x/yu45020/model_check_points/CARN_V2/CARN_model_checkpoint.pt',
        bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), license='GPL-3.0',
        provenance='Byte-identical to CARN_model_checkpoint.pt inside pinned GPL yu45020/Waifu2x archive',
        original_archive='https://raw.githubusercontent.com/yu45020/Waifu2x/1e6c456e9769bb8f418024c0782ef965d8dfd026/model_check_points/CRAN_V2/CRAN_V2_02_28_2019.zip'))
Path('outputs/waifu2x-assets-verified.json').write_text(json.dumps(assets, indent=2), encoding='utf-8')
print('PASS eight UpConv7 JSONs identical to pinned MIT original Git blobs; CARN V2 identical to pinned GPL archive member')
