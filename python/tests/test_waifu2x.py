import hashlib
import io
import json
from pathlib import Path
import pytest
from sdfx.protocol import EngineError


def test_nine_pinned_model_choices_and_carn_noise_semantics():
    from sdfx.waifu2x import NAMES, options, parse_name, assets
    assert len(NAMES) == len(assets()) == 9
    assert {parse_name(name) for name in NAMES} == {asset['id'] for asset in assets()}
    assert options('Anime', 2, 4)['model_id'] == 'anime-2'
    assert options('Photo', 0, 16)['model_id'] == 'photo-0'
    assert options('CarnV2', 3, 2)['model_id'] == 'carn-v2'
    assert all(len(asset['sha256']) == 64 and asset['bytes'] > 0 and 'https://' in asset['url'] for asset in assets())


@pytest.mark.parametrize('style,noise,scale,tile', [('Other', 0, 2, 64), ('Anime', True, 2, 64),
    ('Anime', 4, 2, 64), ('Anime', 1, float('nan'), 64), ('Anime', 1, 17, 64),
    ('Anime', 1, 0, 64), ('Anime', 1, 2, 0), ('Anime', 1, 2, 3.5)])
def test_invalid_options_fail_before_model_loading(style, noise, scale, tile):
    from sdfx.waifu2x import options
    with pytest.raises(EngineError):
        options(style, noise, scale, tile)


def asset(data=b'weights'):
    return dict(filename='fixed.bin', bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), url='https://example.invalid/fixed')


def test_model_download_verified_before_atomic_publication_and_reused_offline(tmp_path):
    from sdfx.waifu2x import materialize
    calls = []
    def opener(url, **kwargs):
        calls.append(url)
        return io.BytesIO(b'weights')
    path = materialize(asset(), tmp_path, True, lambda: None, opener)
    assert path.read_bytes() == b'weights'
    assert materialize(asset(), tmp_path, False, lambda: None, opener) == path
    assert len(calls) == 1 and [p.name for p in tmp_path.iterdir()] == ['fixed.bin']


@pytest.mark.parametrize('payload', [b'corrupt', b'weights-plus-extra', b''])
def test_corrupt_download_is_never_promoted(tmp_path, payload):
    from sdfx.waifu2x import materialize
    with pytest.raises(EngineError):
        materialize(asset(), tmp_path, True, lambda: None, lambda *args, **kwargs: io.BytesIO(payload))
    assert not list(tmp_path.iterdir())


def test_existing_corrupt_model_is_rejected_without_deletion_or_network(tmp_path):
    from sdfx.waifu2x import materialize
    path = tmp_path / 'fixed.bin'
    path.write_bytes(b'bad')
    with pytest.raises(EngineError):
        materialize(asset(), tmp_path, True, lambda: None, lambda *args, **kwargs: pytest.fail('network called'))
    assert path.read_bytes() == b'bad'


def test_download_cancel_removes_only_its_temporary_file(tmp_path):
    from sdfx.waifu2x import materialize
    from sdfx.protocol import INTERRUPTED
    unrelated = tmp_path / 'unrelated.part'
    unrelated.write_bytes(b'keep')
    checks = 0
    def stop():
        nonlocal checks
        checks += 1
        if checks > 2:
            raise EngineError(INTERRUPTED, 'cancelled')
    with pytest.raises(EngineError) as error:
        materialize(asset(), tmp_path, True, stop, lambda *args, **kwargs: io.BytesIO(b'weights'))
    assert error.value.code == INTERRUPTED
    assert list(tmp_path.iterdir()) == [unrelated]


@pytest.mark.parametrize('width,height', [(1, 1), (2, 7), (64, 64), (65, 63), (121, 122), (129, 130)])
def test_streaming_tiles_cover_odd_and_tiny_images_exactly_once(width, height):
    from sdfx.waifu2x import tile_plan
    seen = [[0] * width for _ in range(height)]
    for x, y, w, h in tile_plan(width, height, 64):
        assert w > 0 and h > 0
        for row in range(y, y+h):
            for column in range(x, x+w):
                seen[row][column] += 1
    assert all(value == 1 for row in seen for value in row)


def test_tile_remainder_geometry_matches_legacy_three_pixel_border_adjustment():
    from sdfx.waifu2x import tile_plan
    # Padded 58+6 reaches 64, so source raises its shared core step by six.
    assert list(tile_plan(58, 135, 64)) == [(0, 0, 58, 70), (0, 70, 58, 65)]


def test_dimensions_and_output_budget_rejected_before_allocating_pixels():
    from sdfx.waifu2x import tile_plan, validate_dimensions
    for size in ((0, 1), (-1, 2), (1.5, 2), (True, 1), (100000, 100000)):
        with pytest.raises(EngineError):
            validate_dimensions(*size)
    with pytest.raises(EngineError):
        list(tile_plan(1, 1, 0))


def test_upconv_loader_rejects_incomplete_weights(tmp_path):
    pytest.importorskip('torch')
    from sdfx.waifu2x import load_model
    path = tmp_path / 'weights.json'
    path.write_text('[]')
    with pytest.raises(EngineError):
        load_model('anime-0', path, 'cpu')


def test_carn_loader_uses_restricted_weights_and_rejects_non_state_payload(tmp_path):
    torch = pytest.importorskip('torch')
    from sdfx.waifu2x import load_model
    path = tmp_path / 'weights.pt'
    torch.save({'wrong': torch.ones(1)}, path)
    with pytest.raises(EngineError):
        load_model('carn-v2', path, 'cpu')


@pytest.mark.parametrize('size', [(1, 1), (7, 2), (58, 135), (65, 63), (129, 130)])
def test_streamed_tile_crops_have_no_gaps_shifts_or_quantization_change(size):
    torch = pytest.importorskip('torch')
    np = pytest.importorskip('numpy')
    from PIL import Image
    from sdfx.waifu2x import run_tiles
    width, height = size
    pixels = np.random.default_rng(613).integers(0, 256, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    output = run_tiles(image, lambda patch: patch.repeat_interleave(2, 2).repeat_interleave(2, 3))
    assert np.array_equal(np.asarray(output), pixels.repeat(2, 0).repeat(2, 1))


def test_streaming_cancellation_does_not_return_a_partially_filled_image():
    pytest.importorskip('torch')
    from PIL import Image
    from sdfx.waifu2x import run_tiles
    from sdfx.protocol import INTERRUPTED
    image = Image.new('RGB', (129, 130), 'red')
    checks = 0
    def stop():
        nonlocal checks
        checks += 1
        if checks > 3:
            raise EngineError(INTERRUPTED, 'cancelled')
    with pytest.raises(EngineError) as error:
        run_tiles(image, lambda patch: patch.repeat_interleave(2, 2).repeat_interleave(2, 3), check_interrupt=stop)
    assert error.value.code == INTERRUPTED and image.getpixel((128, 129)) == (255, 0, 0)


def test_malformed_model_outputs_fail_instead_of_becoming_black_tile_holes():
    torch = pytest.importorskip('torch')
    from PIL import Image
    from sdfx.waifu2x import run_tiles
    for model in (lambda x: x, lambda x: torch.full((1, 3, x.shape[2]*2, x.shape[3]*2), float('nan'))):
        with pytest.raises(EngineError):
            run_tiles(Image.new('RGB', (3, 5)), model)


@pytest.mark.parametrize('scale,passes', [(0.5, 1), (1, 1), (2, 1), (3, 2), (4, 2), (8, 3), (16, 4)])
def test_requested_scale_runs_neural_pass_even_for_denoise_at_one_x(monkeypatch, scale, passes):
    from PIL import Image
    import sdfx.waifu2x as waifu
    calls = []
    monkeypatch.setattr(waifu, 'model_for', lambda *args, **kwargs: None)
    def inference(image, model, tile, check_interrupt):
        calls.append(image.size)
        return image.resize((image.width*2, image.height*2))
    monkeypatch.setattr(waifu, 'run_tiles', inference)
    output = waifu.upscale(Image.new('RGB', (5, 7)), max(1, round(5*scale)), max(1, round(7*scale)), 'anime-1')
    assert len(calls) == passes
    assert output.size == (max(1, round(5*scale)), max(1, round(7*scale)))


def test_excessive_output_and_intermediate_size_fail_before_model_download(monkeypatch):
    from PIL import Image
    import sdfx.waifu2x as waifu
    monkeypatch.setattr(waifu, 'model_for', lambda *args, **kwargs: pytest.fail('loaded model'))
    for dimensions in ((0, 2), (10000, 10000), (1024, 2)):
        with pytest.raises(EngineError):
            waifu.upscale(Image.new('RGB', (8, 8)), *dimensions, 'anime-1')


def test_builtin_choices_pass_normal_upscale_validation_without_local_model_mapping(tmp_path):
    from PIL import Image
    from sdfx.base import BaseEngine
    from sdfx.waifu2x import NAMES
    path = tmp_path / 'input.png'
    Image.new('RGB', (5, 7)).save(path)
    engine = BaseEngine()
    for name in NAMES:
        request = engine.parse_upscale(dict(image_path=str(path), output_dir=str(tmp_path), scale=2, upscaler=name))
        assert request.upscaler == name


def test_builtin_upscaler_routes_to_neural_implementation_and_propagates_cancel_check(monkeypatch):
    from PIL import Image
    from sdfx import upscalers, waifu2x
    calls = []
    def upscale(image, width, height, model_id, **kwargs):
        calls.append((model_id, width, height, kwargs['device']))
        kwargs['check_interrupt']()
        return image.resize((width, height))
    monkeypatch.setattr(waifu2x, 'upscale', upscale)
    checks = []
    output = upscalers.upscale_to(Image.new('RGB', (3, 5)), 6, 10, waifu2x.NAMES[7], device='cpu', check_interrupt=lambda: checks.append(True))
    assert output.size == (6, 10) and calls == [('photo-3', 6, 10, 'cpu')] and checks == [True]


def test_dedicated_image_tool_routes_options_and_interrupt_callback(monkeypatch):
    from PIL import Image
    from sdfx import image_tools, waifu2x
    calls = []
    def upscale(image, width, height, model_id, **kwargs):
        calls.append((model_id, width, height))
        kwargs['check_interrupt']()
        return image.resize((width, height))
    monkeypatch.setattr(waifu2x, 'upscale', upscale)
    checks = []
    output, extra = image_tools.run('waifu2x', Image.new('RGB', (3, 5)), dict(style='Photo', noise=3, scale=4),
                                    'cpu', check_interrupt=lambda: checks.append(True))
    assert output.size == (12, 20) and calls == [('photo-3', 12, 20)] and checks == [True]
    assert extra['model'] == 'photo-3'


def test_dry_run_does_not_download_or_run_a_neural_model(tmp_path, monkeypatch):
    from PIL import Image
    from sdfx.dryrun import FakeEngine
    from sdfx.protocol import Emitter, JobContext
    from sdfx import waifu2x
    path = tmp_path / 'input.png'
    Image.new('RGB', (5, 7)).save(path)
    monkeypatch.setattr(waifu2x, 'model_for', lambda *args, **kwargs: pytest.fail('loaded neural model'))
    with pytest.raises(EngineError) as error:
        FakeEngine().image_tool(dict(tool='waifu2x', image_path=str(path), output_dir=str(tmp_path/'output')),
                                JobContext(Emitter(io.BytesIO()), 'dry'))
    assert error.value.code == 'unsupported' and not (tmp_path/'output').exists()


def test_actual_image_tool_cancellation_cannot_save_an_output(tmp_path, monkeypatch):
    from PIL import Image
    from sdfx.base import BaseEngine
    from sdfx.protocol import Emitter, JobContext, INTERRUPTED
    from sdfx import waifu2x
    path = tmp_path / 'input.png'
    Image.new('RGB', (5, 7)).save(path)
    def cancel(*args, **kwargs):
        raise EngineError(INTERRUPTED, 'cancelled')
    monkeypatch.setattr(waifu2x, 'upscale', cancel)
    with pytest.raises(EngineError) as error:
        BaseEngine().image_tool(dict(tool='waifu2x', image_path=str(path), output_dir=str(tmp_path/'output')),
                               JobContext(Emitter(io.BytesIO()), 'cancel'))
    assert error.value.code == INTERRUPTED and not (tmp_path/'output').exists()


def test_streaming_progress_counts_actual_tiles_monotonically():
    pytest.importorskip('torch')
    from PIL import Image
    from sdfx.waifu2x import run_tiles
    progress = []
    run_tiles(Image.new('RGB', (129, 130)), lambda patch: patch.repeat_interleave(2, 2).repeat_interleave(2, 3),
              progress=lambda done, total: progress.append((done, total)))
    assert progress == [(index, 9) for index in range(1, 10)]


def test_cache_transitions_keep_one_model_and_revalidate_bytes(tmp_path, monkeypatch):
    import sdfx.waifu2x as waifu
    entries = [dict(asset(), id=name, filename=name+'.bin') for name in ('anime-0', 'photo-0')]
    for entry in entries:
        (tmp_path / entry['filename']).write_bytes(b'weights')
    monkeypatch.setattr(waifu, 'assets', lambda: entries)
    monkeypatch.setattr(waifu, '_model_cache', None)
    calls = []
    def load(model_id, path, device):
        calls.append((model_id, device))
        return object()
    monkeypatch.setattr(waifu, 'load_model', load)
    def get(model='anime-0', device='cpu'):
        return waifu.model_for(model, device, tmp_path, allow_download=False)
    first = get()
    assert get() is first and len(calls) == 1
    assert get(device='cuda:0') is not first
    assert get('photo-0') is not first
    assert get() is not first and len(calls) == 4
    (tmp_path / entries[0]['filename']).write_bytes(b'corrupt')
    with pytest.raises(EngineError):
        get()
    assert len(calls) == 4  # A cached callable must not bypass disk integrity.
    waifu.unload()
    assert waifu._model_cache is None


def test_rgba_tiles_discard_alpha_without_compositing_rgb():
    pytest.importorskip('torch')
    from PIL import Image
    from sdfx.waifu2x import run_tiles
    source = Image.new('RGBA', (2, 1))
    source.putdata([(255, 0, 0, 0), (0, 255, 0, 128)])
    output = run_tiles(source, lambda patch: patch.repeat_interleave(2, 2).repeat_interleave(2, 3))
    assert output.mode == 'RGB' and output.size == (4, 2)
    assert list(output.getdata()) == [(255, 0, 0), (255, 0, 0), (0, 255, 0), (0, 255, 0)] * 2
