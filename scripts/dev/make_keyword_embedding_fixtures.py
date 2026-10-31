"""Write labelled, synthetic SD1.5/SDXL textual inversion fixtures without torch."""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def create(directory):
    directory = Path(directory).absolute()
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for name, widths in [('synthetic_keyword_sd15', {'clip_l': 768}),
                         ('synthetic_keyword_sdxl', {'clip_l': 768, 'clip_g': 1280})]:
        header = {'__metadata__': {'synthetic': 'true', 'purpose': 'model-keyword acceptance fixture'}}
        body = bytearray()
        for key, width in widths.items():
            start = len(body)
            body.extend(struct.pack('<f', .01) * width)
            header[key] = {'dtype': 'F32', 'shape': [1, width], 'data_offsets': [start, len(body)]}
        raw = json.dumps(header, separators=(',', ':')).encode('utf8')
        raw += b' ' * (-len(raw) % 8)
        path = directory / f'{name}.safetensors'
        with path.open('xb') as stream:
            stream.write(struct.pack('<Q', len(raw))); stream.write(raw); stream.write(body)
        data = path.read_bytes()
        n = struct.unpack('<Q', data[:8])[0]
        parsed = json.loads(data[8:8+n])
        for key, width in widths.items():
            item = parsed[key]
            assert item['shape'] == [1, width] and item['dtype'] == 'F32'
            assert item['data_offsets'][1] - item['data_offsets'][0] == 4 * width
        manifest[name] = {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(),
                          'dimensions': widths, 'synthetic': True}
    (directory / 'synthetic-embeddings.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('new_output_directory')
    print(json.dumps(create(parser.parse_args().new_output_directory), indent=2))
