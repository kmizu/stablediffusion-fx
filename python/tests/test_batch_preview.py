import base64
import io
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from sdfx.preview import latent_preview_b64, TaesdPreviewer


def decoded(data):
    return np.array(Image.open(io.BytesIO(base64.b64decode(data))))


def levels(channels=4):
    lat = np.zeros((4, channels, 8, 8), dtype=np.float32)
    lat[:, 0] = np.array([-3, -1, 1, 3])[:, None, None]
    return lat


@pytest.mark.parametrize('sdxl', [False, True])
def test_cheap_batch_shows_four_distinct_tiles_and_updates(sdxl):
    first = decoded(latent_preview_b64(levels(), sdxl))
    second = decoded(latent_preview_b64(levels() * .5, sdxl))
    assert first.shape == (512, 512, 3)
    centers = [(128, 128), (128, 384), (384, 128), (384, 384)]
    red = [int(first[y, x, 0]) for y, x in centers]
    assert all(b - a > 25 for a, b in zip(red, red[1:]))
    assert all(not np.array_equal(first[y, x], second[y, x]) for y, x in centers)


def test_cheap_single_and_extra_channels_keep_same_pixels():
    single = levels()[1]
    assert np.array_equal(decoded(latent_preview_b64(single)), decoded(latent_preview_b64(single[None])))
    extra = np.pad(levels(), ((0, 0), (0, 5), (0, 0), (0, 0)), constant_values=999)
    assert np.array_equal(decoded(latent_preview_b64(levels(), True)), decoded(latent_preview_b64(extra, True)))


def test_taesd_batch_decodes_all_members_with_bounded_output():
    torch = pytest.importorskip('torch')
    pytest.importorskip('diffusers')
    class Decoder:
        config = SimpleNamespace(scaling_factor=2.)
        def decode(self, value):
            assert value.shape[0] == 1
            return SimpleNamespace(sample=value[:, :3].expand(-1, -1, 512, 512))
    previewer = TaesdPreviewer()
    previewer._models['sdxl'] = Decoder()
    lat = torch.tensor([-2., -1., 1., 2.])[:, None, None, None].expand(4, 4, 1, 1)
    actual = decoded(previewer.decode_b64(lat, True, 'cpu', torch.float32))
    assert actual.shape == (512, 512, 3)
    assert np.allclose([int(actual[y, x, 0]) for y, x in [(128, 128), (128, 384), (384, 128), (384, 384)]], [0, 63, 191, 255], atol=2)


@pytest.mark.parametrize('method', ['approx', 'taesd'])
def test_callback_preserves_batch_including_taesd_exception_fallback(method):
    torch = pytest.importorskip('torch')
    pytest.importorskip('diffusers')
    from sdfx.engine import Engine
    from sdfx.params import GenerationRequest
    engine = Engine()
    engine.kind = 'sdxl'
    events, progress = [], []
    context = SimpleNamespace(interrupted=False, start_pass=lambda: None,
                              progress=lambda *a, **kw: progress.append(kw),
                              preview=lambda data, step: events.append(decoded(data)))
    def broken(*args):
        raise RuntimeError('decoder unavailable')
    engine.taesd.decode_b64 = broken
    callback = engine._callback(context, GenerationRequest(preview_every=1, preview_method=method), 'base', 4, 20, 3)
    values = {'latents': torch.from_numpy(levels(channels=9))}
    assert callback(SimpleNamespace(_num_timesteps=3), 0, 0, values) is values
    assert progress[0]['batch_size'] == 4
    assert len(events) == 1
    assert events[0][128, 128, 0] < events[0][128, 384, 0] < events[0][384, 128, 0] < events[0][384, 384, 0]


def test_large_grid_is_bounded_without_dropping_last_member():
    batch = np.zeros((64, 4, 8, 8), dtype=np.float32)
    batch[-1, 0] = 3
    result = decoded(latent_preview_b64(batch))
    assert result.shape == (1024, 1024, 3)
    assert result[960, 960, 0] > 230
    assert abs(int(result[64, 64, 0]) - 128) < 3


def test_landscape_tiles_use_compact_layout_without_distortion():
    batch = np.zeros((2, 4, 8, 32), dtype=np.float32)
    batch[1, 0] = 3
    result = decoded(latent_preview_b64(batch))
    assert result.shape == (128, 256, 3)
    assert result[96, 128, 0] > result[32, 128, 0] + 80
