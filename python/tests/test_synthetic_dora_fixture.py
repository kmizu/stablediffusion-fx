import importlib.util
import json
import struct
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / 'scripts/dev/make_synthetic_dora.py'
spec = importlib.util.spec_from_file_location('make_synthetic_dora', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_fixture_preserves_tensor_bytes_and_never_overwrites(tmp_path):
    source = tmp_path / 'source.safetensors'
    destination = tmp_path / 'synthetic.safetensors'
    header = {'lora_unet_x.lora_down.weight': {'dtype': 'F16', 'shape': [2, 1], 'data_offsets': [0, 4]},
              'lora_unet_x.lora_up.weight': {'dtype': 'F16', 'shape': [3, 2], 'data_offsets': [4, 16]}}
    raw = json.dumps(header).encode(); raw += b' ' * (-len(raw) % 8)
    original = struct.pack('<Q', len(raw)) + raw + bytes(range(16))
    source.write_bytes(original)
    assert module.make_fixture(source, destination) == 1
    data = destination.read_bytes(); size = struct.unpack('<Q', data[:8])[0]
    result = json.loads(data[8:8 + size]); body = data[8 + size:]
    assert body[:16] == bytes(range(16))
    assert result['lora_unet_x.dora_scale'] == {'dtype': 'F16', 'shape': [3], 'data_offsets': [16, 22]}
    assert body[16:] == struct.pack('<eee', 1., 1., 1.)
    with pytest.raises(FileExistsError): module.make_fixture(source, destination)
    assert source.read_bytes() == original and destination.read_bytes() == data
