from types import SimpleNamespace
import pytest
from sdfx.protocol import EngineError


def test_full_denoise_steps_are_exact_at_low_strength_and_one():
    from sdfx import latent_upscale as lu
    assert lu.full_steps(7, 7 / 10000)[0] == 10000
    for requested in (1, 7, 30):
        for strength in (.01, .17, .4, .99, 1):
            n, effective, actual = lu.full_steps(requested, strength)
            assert actual == requested
            assert int(n * effective) == requested
            assert 0 < effective <= 1
            assert n >= requested
            assert min(int(n * strength + 1e-8), n) == requested


def test_configuration_and_schedule_are_validated_without_importing_torch():
    from sdfx import latent_upscale as lu
    assert lu.normalize({}) == {'method': 'nearest-exact', 'scheduler': 'simple'}
    assert lu.schedule({'scheduler': 'automatic'}, 'Karras') == 'Karras'
    assert lu.schedule({'scheduler': 'normal'}, 'Karras') == 'Normal'
    for method in ('linear', 'trilinear'):
        with pytest.raises(EngineError, match='4-D'):
            lu.normalize({'method': method})
    for cfg in ({'method': 'bogus'}, {'scheduler': 'bogus'}, {'scheduler': []}):
        with pytest.raises(EngineError):
            lu.normalize(cfg)
    with pytest.raises(EngineError, match='greater than zero'):
        lu.full_steps(20, 0)


@pytest.mark.parametrize('mode', ['bilinear', 'bilinear-antialiased', 'bicubic', 'bicubic-antialiased', 'area', 'nearest', 'nearest-exact'])
def test_interpolation_matches_real_torch_and_keeps_batch_channels_and_dtype(mode):
    torch = pytest.importorskip('torch')
    from sdfx import latent_upscale as lu
    x = torch.arange(2 * 4 * 3 * 5, dtype=torch.float32).reshape(2, 4, 3, 5) / 99
    x = x.to(torch.float16)
    original = x.clone()
    result = lu.resize(x, 80, 64, mode)
    actual_mode = mode.split('-antialiased')[0]
    kwargs = {'antialias': mode.endswith('-antialiased'), 'align_corners': False} if actual_mode in ('bilinear', 'bicubic') else {}
    expected = torch.nn.functional.interpolate(x.float(), size=(8, 10), mode=actual_mode, **kwargs).to(x.dtype)
    assert torch.equal(result, expected)
    assert result.shape == (2, 4, 8, 10) and result.dtype == x.dtype
    assert torch.equal(x, original)


def test_vae_encoding_uses_source_pixels_before_interpolation_and_restores_dtype():
    torch = pytest.importorskip('torch')
    from sdfx import latent_upscale as lu
    class Vae:
        dtype = torch.float16
        device = torch.device('cpu')
        config = SimpleNamespace(force_upcast=True, scaling_factor=.5)
        def to(self, dtype):
            self.dtype = dtype
        def encode(self, image):
            assert image.shape[-2:] == (24, 40), 'source must not be resized to target before encoding'
            assert self.dtype == torch.float32
            return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda generator: torch.ones(1, 4, 3, 5)))
    vae = Vae()
    pipe = SimpleNamespace(vae=vae, image_processor=SimpleNamespace(preprocess=lambda im: torch.zeros(1, 3, 24, 40)))
    generators = [torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)]
    lat = lu.encode(pipe, object(), 80, 64, 'nearest-exact', generators, torch.device('cpu'), torch.float16)
    assert lat.shape == (2, 4, 8, 10) and torch.all(lat == .5)
    assert vae.dtype == torch.float16
    def broken(image):
        raise RuntimeError('encode failed')
    vae.encode = broken
    with pytest.raises(RuntimeError, match='encode failed'):
        lu.encode(pipe, object(), 80, 64, 'nearest-exact', generators, torch.device('cpu'), torch.float16)
    assert vae.dtype == torch.float16


def test_seeded_vae_sampling_and_masked_conditioning_are_repeatable():
    torch = pytest.importorskip('torch')
    from PIL import Image
    from sdfx import latent_upscale as lu
    calls = []
    class Vae:
        dtype = torch.float32
        config = SimpleNamespace(force_upcast=False, scaling_factor=1.)
        def encode(self, image):
            calls.append(image.clone())
            return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda generator: torch.randn((1, 4, 2, 3), generator=generator)))
    pipe = SimpleNamespace(vae=Vae(), image_processor=SimpleNamespace(preprocess=lambda im: torch.ones(1, 3, 16, 24)))
    def generators():
        return [torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)]
    a = lu.encode(pipe, object(), 48, 32, 'nearest', generators(), 'cpu', torch.float32)
    b = lu.encode(pipe, object(), 48, 32, 'nearest', generators(), 'cpu', torch.float32)
    assert torch.equal(a, b) and not torch.equal(a[0], a[1])
    lu.encode(pipe, object(), 48, 32, 'nearest', generators(), 'cpu', torch.float32,
              mask=Image.new('L', (24, 16), 255))
    assert torch.all(calls[-1] == 0)


@pytest.mark.parametrize('sampler', ['Euler', 'Heun', 'DPM++ 2M', 'DDIM'])
def test_real_diffusers_timestep_slicing_runs_requested_logical_steps(sampler):
    pytest.importorskip('torch')
    pytest.importorskip('diffusers')
    from diffusers import StableDiffusionImg2ImgPipeline
    from sdfx import schedulers, latent_upscale as lu
    scheduler, _ = schedulers.build_scheduler(schedulers.plan(sampler, 'Simple'), {})
    for strength in (.17, .4, 1):
        n, effective, expected = lu.full_steps(7, strength)
        scheduler.set_timesteps(n, device='cpu')
        timesteps, actual = StableDiffusionImg2ImgPipeline.get_timesteps(SimpleNamespace(scheduler=scheduler), n, effective, 'cpu')
        assert actual == expected == 7
        assert len(timesteps) == (7 * scheduler.order - (scheduler.order - 1))


def test_request_validation_and_disabled_noop(tmp_path):
    from PIL import Image
    from sdfx.base import BaseEngine
    src = tmp_path / 'source.png'
    Image.new('RGB', (64, 64)).save(src)
    params = {'params': {'steps': 7, 'width': 64, 'height': 64, 'img2img': {'init_image': str(src), 'denoising_strength': .4},
                         'hooks': {'latent_upscale': {'method': 'area', 'scheduler': 'simple'}}}, 'output_dir': str(tmp_path)}
    req = BaseEngine().parse_generation(params, 'img2img', None)
    assert req.params.steps == 7
    with pytest.raises(EngineError, match='img2img'):
        BaseEngine().parse_generation(params, 'txt2img', None)
    params['params']['hooks']['img2imgalt'] = {}
    with pytest.raises(EngineError, match='Alternative'):
        BaseEngine().parse_generation(params, 'img2img', None)
    params['params']['hooks'] = {'latent_upscale': {'enabled': False, 'method': 'nonsense'}}
    BaseEngine().parse_generation(params, 'img2img', None)


@pytest.mark.parametrize('strength', [1e-308, 5e-324])
def test_tiny_positive_strength_raw_request_returns_invalid_params_before_schedule_overflow(tmp_path, strength):
    from PIL import Image
    from sdfx.base import BaseEngine
    from sdfx import latent_upscale as lu
    source = tmp_path / 'source.png'
    Image.new('RGB', (64, 64)).save(source)
    params = {'params': {'steps': 7, 'width': 64, 'height': 64,
                        'img2img': {'init_image': str(source), 'denoising_strength': strength},
                        'hooks': {'latent_upscale': {}}}, 'output_dir': str(tmp_path)}
    with pytest.raises(EngineError) as direct:
        lu.full_steps(7, strength)
    assert direct.value.code == 'invalid_params'
    with pytest.raises(EngineError) as raw:
        BaseEngine().parse_generation(params, 'img2img', None)
    assert raw.value.code == 'invalid_params'
    assert 'schedule limit 10000' in raw.value.message


def test_dryrun_json_lines_reports_full_steps_and_disabled_restores_original_cadence(tmp_path):
    from PIL import Image
    from test_server_dryrun import Engine
    source = tmp_path / 'source.png'
    Image.new('RGB', (64, 64)).save(source)
    e = Engine()
    try:
        assert 'latent_upscale' in e.next()['features']
        params = {'params': {'steps': 7, 'width': 64, 'height': 64,
                            'img2img': {'init_image': str(source), 'denoising_strength': .4},
                            'hooks': {'latent_upscale': {'method': 'area'}}}, 'output_dir': str(tmp_path)}
        result, events = e.call('img2img', params)
        assert result['type'] == 'result', result
        assert max(event['step'] for event in events if event.get('event') == 'progress') == 7
        params['params']['hooks']['latent_upscale']['enabled'] = False
        result, events = e.call('img2img', params)
        assert result['type'] == 'result', result
        assert max(event['step'] for event in events if event.get('event') == 'progress') == 2
    finally:
        e.close()
