import hashlib
import io
import struct
import numpy as np
import pytest
from PIL import Image
from sdfx.protocol import EngineError, Emitter, JobContext
from sdfx import vae_tools as tools


@pytest.mark.parametrize('channels', [4, 16])
def test_source_l_container_is_exact_and_safe(channels):
    pytest.importorskip('safetensors')
    array = np.linspace(-12, 12, channels * 3 * 5, dtype=np.float32).reshape(1, channels, 3, 5)
    image, stats = tools.pack(array, 'L')
    assert image.mode == 'L'
    data = image.tobytes()
    size = struct.unpack('>I', data[:4])[0]
    from safetensors.numpy import load
    assert np.array_equal(load(data[4:4 + size])['tensor'], array)
    assert np.array_equal(tools.unpack(image, 'L', channels), array)
    assert stats['quantized'] is False


@pytest.mark.parametrize('representation', ['RGBA', 'CMYK'])
@pytest.mark.parametrize('channels', [4, 16])
def test_quantization_grid_order_and_clipping(representation, channels):
    array = np.linspace(-12, 12, channels * 3 * 5, dtype=np.float32).reshape(1, channels, 3, 5)
    image, stats = tools.pack(array, representation)
    assert image.mode == representation
    assert image.size == ((5, 3) if channels == 4 else (10, 6))
    actual = tools.unpack(image, representation, channels)
    assert actual.shape == array.shape
    assert np.max(np.abs(actual - np.clip(array, -10, 10))) <= 10 / 255 + 1e-5
    assert stats['clipped_values'] == int(np.count_nonzero(np.abs(array) > 10))
    assert stats['quantized'] is True


def test_rejects_unsafe_or_wrong_mode_container():
    with pytest.raises(ValueError, match='RGBA'):
        tools.unpack(Image.new('RGB', (8, 8)), 'RGBA', 4)
    with pytest.raises(ValueError, match='length'):
        tools.unpack(Image.frombytes('L', (2, 2), b'\xff' * 4), 'L', 4)
    with pytest.raises(ValueError):
        tools.unpack(Image.frombytes('L', (3, 3), b'\0\0\0\x04evil\0'), 'L', 4)
    with pytest.raises(ValueError, match='finite'):
        tools.pack(np.full((1, 4, 2, 2), np.nan), 'L')
    with pytest.raises(ValueError):
        tools.pack(np.zeros((2, 4, 2, 2)), 'L')


def test_cmyk_file_preserves_mode_and_input_bytes(tmp_path):
    original = np.linspace(-3, 3, 4 * 3 * 5, dtype=np.float32).reshape(1, 4, 3, 5)
    image, _ = tools.pack(original, 'CMYK')
    for suffix, fmt in [('tiff', 'TIFF'), ('jpg', 'JPEG')]:
        path = tmp_path / f'encoded.{suffix}'
        image.save(path, format=fmt, quality=100)
        before = path.read_bytes()
        actual = tools.read_latent(path, 'CMYK', 4)
        assert actual.shape == original.shape
        assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(before).digest()
        assert np.mean(abs(actual - original)) < .12


def test_invalid_raw_requests_are_protocol_errors_without_torch(tmp_path):
    from sdfx.dryrun import FakeEngine
    from sdfx.base import BaseEngine
    ctx = JobContext(Emitter(io.BytesIO()), 'vae')
    with pytest.raises(EngineError) as exc:
        FakeEngine().vae_tools({}, ctx)
    assert exc.value.code == 'unsupported'
    engine = BaseEngine()
    with pytest.raises(EngineError) as exc:
        engine.vae_tools({}, ctx)
    assert exc.value.code == 'no_model'
