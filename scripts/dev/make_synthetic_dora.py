"""Create a synthetic DoRA fixture from a conventional Kohya safetensors LoRA.

For loader/conditioning tests only: magnitude vectors are artificial ones, not
trained parameters. The original adapter is read-only and destination must not
exist. This uses no torch and preserves all original tensor bytes.
"""
import argparse
import json
import os
import shutil
import struct
from pathlib import Path


def make_fixture(source, destination):
    source, destination = Path(source), Path(destination)
    with source.open('rb') as inp:
        prefix = inp.read(8)
        if len(prefix) != 8:
            raise ValueError('truncated safetensors header')
        size = struct.unpack('<Q', prefix)[0]
        if not 1 <= size <= 100_000_000:
            raise ValueError('invalid safetensors header size')
        header = json.loads(inp.read(size))
        origin_data = 8 + size
        data_size = source.stat().st_size - origin_data
        extras = []
        for key, value in list(header.items()):
            if not key.endswith('.lora_up.weight'):
                continue
            base = key[:-len('lora_up.weight')]
            if (not base.startswith(('lora_unet_', 'lora_te_', 'lora_te1_', 'lora_te2_')) or
                    base + 'lora_down.weight' not in header or base + 'dora_scale' in header):
                raise ValueError(f'unsupported or already DoRA factor: {base}')
            rows = value['shape'][0]
            if not isinstance(rows, int) or rows <= 0:
                raise ValueError(f'invalid output size for {key}')
            end = data_size + rows * 2
            header[base + 'dora_scale'] = {'dtype': 'F16', 'shape': [rows], 'data_offsets': [data_size, end]}
            extras.append(rows)
            data_size = end
        if not extras:
            raise ValueError('no conventional Kohya LoRA factors found')
        raw = json.dumps(header, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        raw += b' ' * (-len(raw) % 8)
        with destination.open('xb') as out:
            out.write(struct.pack('<Q', len(raw))); out.write(raw)
            inp.seek(origin_data)
            shutil.copyfileobj(inp, out, length=1024 * 1024)
            for rows in extras:
                out.write(struct.pack('<e', 1.) * rows)
            out.flush(); os.fsync(out.fileno())
    return len(extras)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source'); parser.add_argument('destination')
    args = parser.parse_args()
    print(f'Created synthetic DoRA fixture with {make_fixture(args.source, args.destination)} layers')
