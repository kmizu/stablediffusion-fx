import io
import hashlib
from types import SimpleNamespace
import numpy as np
import pytest
torch = pytest.importorskip('torch')
diffusers = pytest.importorskip('diffusers')
from PIL import Image
from sdfx import vae_tools as tools
from sdfx.engine import Engine
from sdfx.protocol import JobContext, Emitter, EngineError


@pytest.fixture
def engine():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(5)
    vae = diffusers.AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4,
        down_block_types=('DownEncoderBlock2D',) * 4, up_block_types=('UpDecoderBlock2D',) * 4,
        block_out_channels=(8, 8, 8, 8), layers_per_block=1, norm_num_groups=4)
    unet = diffusers.UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
        down_block_types=('CrossAttnDownBlock2D',), up_block_types=('CrossAttnUpBlock2D',),
        block_out_channels=(8,), layers_per_block=1, norm_num_groups=4, cross_attention_dim=8, attention_head_dim=2)
    pipe = diffusers.StableDiffusionPipeline(vae=vae, text_encoder=None, tokenizer=None, unet=unet,
        scheduler=diffusers.EulerDiscreteScheduler(steps_offset=1), safety_checker=None,
        feature_extractor=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    eng = Engine()
    eng.pipe, eng.kind, eng.dtype = pipe, 'sd15', torch.float32
    eng.model = {'name': 'tiny', 'kind': 'sd15', 'hash': '1234'}
    eng.base_sched_config = dict(pipe.scheduler.config)
    eng._encode_scheduled = lambda *a, **kw: {'prompt_embeds': torch.zeros(1, 4, 8), 'negative_prompt_embeds': torch.zeros(1, 4, 8)}
    yield eng
    torch.set_num_threads(previous)


def ctx():
    return JobContext(Emitter(io.BytesIO()), 'vae-test')


@pytest.mark.parametrize('representation', ['L', 'RGBA', 'CMYK'])
def test_actual_tiny_vae_file_roundtrip_and_batch(engine, tmp_path, representation):
    source = tmp_path / 'source.png'
    Image.fromarray(np.random.default_rng(2).integers(0, 256, (64, 64, 3), dtype='uint8')).save(source)
    before = source.read_bytes()
    options = dict(input_paths=[str(source), str(source)], output_dir=str(tmp_path), representation=representation,
                   operation='encode', seed=7, tiled=True)
    result = engine.vae_tools(options, ctx())
    assert all(r['status'] == 'complete' for r in result['results']), result
    assert engine.pipe.vae.use_tiling is False
    assert source.read_bytes() == before
    assert result['results'][0]['input_sha256'] == hashlib.sha256(before).hexdigest()
    for item in result['results']:
        decoded = engine.vae_tools(dict(options, input_paths=[item['path']], operation='decode'), ctx())
        assert decoded['results'][0]['status'] == 'complete', decoded
        assert Image.open(decoded['results'][0]['path']).size == (64, 64)
    second = engine.vae_tools(dict(options, input_paths=[str(source)]), ctx())
    assert np.array_equal(tools.read_latent(result['results'][0]['latent_path'], 'L', 4), tools.read_latent(second['results'][0]['latent_path'], 'L', 4))
    roundtrip = engine.vae_tools(dict(options, input_paths=[str(source)], operation='encode+decode'), ctx())
    assert roundtrip['results'][0]['status'] == 'complete', roundtrip


def test_generation_decode_failure_recovers_batch_and_restores(engine, tmp_path, monkeypatch):
    vae = engine.pipe.vae
    original = vae.decode
    def broken(*a, **kw):
        raise RuntimeError('injected VAE decode failure')
    monkeypatch.setattr(vae, 'decode', broken)
    with tools.catch_decode(engine, vae, tmp_path, ctx()):
        output = vae.decode(torch.zeros(2, 4, 8, 8), return_dict=False)[0]
    assert vae.decode is broken
    assert output.shape == (2, 3, 64, 64)
    assert len(engine._vae_recovery) == 2
    for artifact in engine._vae_recovery:
        assert Image.open(artifact['path']).mode == 'RGBA'
        assert tools.read_latent(artifact['latent_path'], 'L', 4).shape == (1, 4, 8, 8)
    # The standalone generation helper used before latent hires/inpaint also recovers;
    # intermediate-recorder calls without a generation context retain their own error policy.
    assert engine._decode(torch.ones(1, 4, 8, 8), ctx(), str(tmp_path))[0].size == (64, 64)
    monkeypatch.setattr(vae, 'decode', original)


def test_successful_generation_rgb_unchanged_and_recovery_not_sticky(engine, tmp_path):
    request = {'params': {'steps': 1, 'width': 64, 'height': 64, 'seed': 3, 'cfg_scale': 1, 'sampler_name': 'Euler'},
               'output_dir': str(tmp_path), 'preview_method': 'none'}
    first = engine.txt2img(request, ctx())
    assert not first.get('vae_recovery')
    second = engine.txt2img(request, ctx())
    assert Image.open(first['images'][0]['path']).tobytes() == Image.open(second['images'][0]['path']).tobytes()
    assert not second.get('vae_recovery')


@pytest.mark.parametrize('upscaler', ['Latent', 'Lanczos'])
def test_hires_repeats_use_real_diffusion_and_disabled_recovery(engine, tmp_path, upscaler):
    request = {'params': {'steps': 2, 'width': 64, 'height': 64, 'seed': 3, 'subseed': 4,
        'cfg_scale': 1, 'sampler_name': 'Euler',
        'hires': {'enabled': True, 'scale': 1.5, 'upscaler': upscaler, 'second_pass_steps': 2}},
        'output_dir': str(tmp_path), 'preview_method': 'none'}
    before = engine.txt2img(request, ctx())
    request['params']['hooks'] = {'hires_fix_tweaks': {'batch_count': 2, 'seed_enabled': True,
        'seed': 30, 'extras': True, 'subseed': 50, 'strength': .5,
        'resize_width': 64, 'resize_height': 64}}
    repeated = engine.txt2img(request, ctx())
    assert [x['hires_seed'] for x in repeated['images']] == [30, 30]
    assert [x['hires_subseed'] for x in repeated['images']] == [50, 51]
    assert all((x['width'], x['height']) == (96, 96) for x in repeated['images'])
    pixels = [Image.open(x['path']).tobytes() for x in repeated['images']]
    assert pixels[0] != pixels[1]
    del request['params']['hooks']
    after = engine.txt2img(request, ctx())
    assert Image.open(before['images'][0]['path']).tobytes() == Image.open(after['images'][0]['path']).tobytes()


def test_actual_adapter_can_be_disabled_for_hires_and_following_job_recovers(engine, tmp_path, monkeypatch):
    from safetensors.torch import save_file
    layer = 'lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q'
    path = tmp_path / 'small.safetensors'
    save_file({layer + '.lora_down.weight': torch.ones(2, 8) * .05,
               layer + '.lora_up.weight': torch.ones(8, 2) * .05}, str(path))
    request = dict(output_dir=str(tmp_path), preview_method='none', params=dict(seed=3, steps=2,
        width=64, height=64, cfg_scale=1, sampler_name='Euler',
        hires=dict(enabled=True, scale=1.5, upscaler='Latent', second_pass_steps=2)))
    before = engine.txt2img(request, ctx())
    request['params']['loras'] = [dict(name='small')]
    request['lora_paths'] = {'small': str(path)}
    request['hires_loras'] = []
    passes = []
    original = engine._run
    def run(*a, **kw):
        passes.append((kw['pass_'], len(engine._lora_state[0])))
        return original(*a, **kw)
    monkeypatch.setattr(engine, '_run', run)
    result = engine.txt2img(request, ctx())
    assert 'small' in result['lora_hashes']
    assert passes == [('base', 1), ('hires', 0)]
    assert len(engine._lora_state[0]) == 1
    request['params']['loras'] = []
    request.pop('hires_loras')
    after = engine.txt2img(request, ctx())
    assert Image.open(before['images'][0]['path']).tobytes() == Image.open(after['images'][0]['path']).tobytes()


def test_real_generation_error_paths_persist_and_next_run_recovers(engine, tmp_path, monkeypatch):
    request = {'params': {'steps': 1, 'width': 64, 'height': 64, 'seed': 3, 'cfg_scale': 1, 'sampler_name': 'Euler'},
               'output_dir': str(tmp_path / 'temporary'), 'vae_recovery_dir': str(tmp_path / 'persistent'), 'preview_method': 'none'}
    original = engine.pipe.vae.decode
    def broken(*a, **kw):
        raise RuntimeError('injected final decode')
    monkeypatch.setattr(engine.pipe.vae, 'decode', broken)
    result = engine.txt2img(request, ctx())
    assert result['vae_recovery'][0]['error'] == 'injected final decode'
    from pathlib import Path
    assert Path(result['vae_recovery'][0]['latent_path']).is_relative_to(tmp_path / 'persistent')
    assert engine.pipe.vae.decode is broken
    monkeypatch.setattr(engine.pipe.vae, 'decode', original)
    again = engine.txt2img(request, ctx())
    assert 'vae_recovery' not in again


def test_odd_source_not_resized_and_failed_explicit_decode_is_not_caught(engine, tmp_path, monkeypatch):
    source = tmp_path / 'odd.png'
    Image.new('RGB', (65, 67), '#a65388').save(source)
    before = source.read_bytes()
    options = dict(input_paths=[str(source)], output_dir=str(tmp_path), operation='encode+decode')
    result = engine.vae_tools(options, ctx())
    assert result['results'][0]['status'] == 'complete', result
    assert result['results'][0]['input_dimensions'] == [65, 67]
    assert Image.open(result['results'][0]['path']).size == (64, 64)
    assert source.read_bytes() == before
    def broken(*a, **kw):
        raise RuntimeError('explicit decode failure')
    monkeypatch.setattr(engine.pipe.vae, 'decode', broken)
    failed = engine.vae_tools(options, ctx())
    assert failed['results'][0]['status'] == 'error'
    assert 'explicit decode failure' in failed['results'][0]['error']
    assert not getattr(engine, '_vae_recovery', [])


def test_approx_modes_have_real_resolution_and_encoding_fallback(engine, tmp_path):
    source = tmp_path / 'source.png'
    Image.new('RGB', (64, 64)).save(source)
    result = engine.vae_tools(dict(input_paths=[str(source)], output_dir=str(tmp_path), operation='encode+decode', approximation='Approx cheap'), ctx())
    assert result['warnings'] and result['encoding_approximation'] == 'Full'
    assert result['results'][0]['width'] == 8
    # A controlled tensor-only checkpoint exercises all eight learned layers with known output.
    weights = {}
    for i, (cin, cout, kernel) in enumerate(zip((4, 8, 16, 32, 64, 32, 16, 8), (8, 16, 32, 64, 32, 16, 8, 3), (7, 5, 3, 3, 3, 3, 3, 3)), 1):
        weights[f'conv{i}.weight'] = torch.zeros(cout, cin, kernel, kernel)
        weights[f'conv{i}.bias'] = torch.zeros(cout)
    path = tmp_path / 'approx.pt'
    torch.save(weights, path)
    result = engine.vae_tools(dict(input_paths=[str(source)], output_dir=str(tmp_path), operation='encode+decode', approximation='Approx NN', approx_nn_path=str(path)), ctx())
    assert result['results'][0]['status'] == 'complete', result
    image = Image.open(result['results'][0]['path'])
    assert image.size == (16, 16) and np.array(image).min() == np.array(image).max() == 127


def test_taesd_diffusers_pixel_convention_without_model_download(engine):
    tiny = diffusers.AutoencoderTiny(encoder_block_out_channels=(8, 8, 8, 8), decoder_block_out_channels=(8, 8, 8, 8),
        num_encoder_blocks=(1, 1, 1, 1), num_decoder_blocks=(1, 1, 1, 1))
    engine._vae_tools_tiny = {'madebyollin/taesd': tiny}
    image = Image.new('RGB', (64, 64), (21, 109, 230))
    options = dict(allow_download=False, tiled=False)
    with torch.inference_mode():
        pixels = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1)[None] / 127.5 - 1)
        expected = tiny.encode(pixels).latents
        encoded = tools.encode(engine, image, 'TAESD', options, 0)
        assert np.array_equal(encoded, expected.numpy())
        decoded = tools.decode(engine, encoded, 'TAESD', options)
        expected_rgb = ((tiny.decode(expected).sample[0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).numpy().astype('uint8')
        assert np.array_equal(np.array(decoded), expected_rgb)


def test_live_preview_exception_falls_back_to_cheap(engine, monkeypatch):
    from sdfx.params import GenerationRequest
    from sdfx import engine as module
    events = []
    context = ctx()
    context.preview = lambda data, step: events.append((data, step))
    def broken(*args):
        raise RuntimeError('injected preview error')
    monkeypatch.setattr(engine.taesd, 'decode_b64', broken)
    monkeypatch.setattr(module.preview, 'latent_preview_b64', lambda *a: 'cheap-real-projection')
    callback = engine._callback(context, GenerationRequest(preview_every=1, preview_method='taesd'), 'base', 0, 1, 3)
    callback(SimpleNamespace(_num_timesteps=3), 0, 0, {'latents': torch.zeros(1, 4, 8, 8)})
    assert events == [('cheap-real-projection', 1)]


def test_nonfinite_retry_uses_fp32_and_restores_user_vae_state(engine, tmp_path, monkeypatch):
    vae = engine.pipe.vae
    vae.to(dtype=torch.float16)
    vae.register_to_config(force_upcast=False)  # exercise the retry, not the preventative SDXL upcast
    vae.enable_tiling()
    original = vae.encode
    seen = []
    def encode(pixels):
        seen.append(vae.dtype)
        if vae.dtype == torch.float16:
            return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda **kw: torch.full((1, 4, 8, 8), float('nan'))))
        return original(pixels)
    monkeypatch.setattr(vae, 'encode', encode)
    path = tmp_path / 'source.png'
    Image.new('RGB', (64, 64)).save(path)
    result = engine.vae_tools(dict(input_paths=[str(path)], operation='encode', output_dir=str(tmp_path)), ctx())
    assert result['results'][0]['status'] == 'complete', result
    assert seen == [torch.float16, torch.float32]
    assert vae.dtype == torch.float16 and vae.use_tiling


def test_catcher_never_hides_interruption(engine, tmp_path, monkeypatch):
    def interrupted(*a, **kw):
        raise EngineError('interrupted', 'cancelled')
    monkeypatch.setattr(engine.pipe.vae, 'decode', interrupted)
    with tools.catch_decode(engine, engine.pipe.vae, tmp_path, ctx()), pytest.raises(EngineError) as exc:
        engine.pipe.vae.decode(torch.zeros(1, 4, 8, 8))
    assert exc.value.code == 'interrupted'
    assert not getattr(engine, '_vae_recovery', [])


def test_source_bfloat_safetensors_values_load_without_unpickling(tmp_path):
    from safetensors.torch import save_file
    value = torch.linspace(-2, 2, 4 * 3 * 5).reshape(1, 4, 3, 5).bfloat16()
    path = tmp_path / 'tensor.safetensors'
    save_file({'tensor': value}, str(path))
    assert np.array_equal(tools.read_latent(path, 'L', 4), value.float().numpy())


def test_batch_missing_unreadable_oversized_inputs_do_not_abort_or_read_whole_files(engine, tmp_path, monkeypatch):
    from pathlib import Path
    missing, unreadable, huge, valid = [tmp_path / name for name in ('missing.png', 'unreadable.png', 'huge.safetensors', 'valid.png')]
    unreadable.write_bytes(b'not readable')
    with huge.open('wb') as stream:
        stream.truncate(tools.MAX_BYTES + 1)
    Image.frombytes('RGBA', (2, 2), bytes([128] * 16)).save(valid)
    original_open = Path.open
    def guarded_open(path, *args, **kw):
        if path == unreadable:
            raise PermissionError('injected unreadable queued input')
        if path == huge:
            pytest.fail('oversized decode input must be rejected before opening/hash')
        return original_open(path, *args, **kw)
    monkeypatch.setattr(Path, 'open', guarded_open)
    monkeypatch.setattr(Path, 'read_bytes', lambda path: pytest.fail('input hashing must stream, never read_bytes'))
    result = engine.vae_tools(dict(input_paths=list(map(str, (missing, unreadable, huge, valid))), operation='decode',
                                   representation='RGBA', approximation='Approx cheap', output_dir=str(tmp_path)), ctx())
    assert [r['status'] for r in result['results']] == ['error', 'error', 'error', 'complete']
    assert 'size limit' in result['results'][2]['error']
    assert result['results'][3]['input_sha256']


@pytest.mark.parametrize('operation', ['encode', 'encode+decode', 'decode'])
def test_png_provenance_is_portable_with_selected_external_vae_and_scale(engine, tmp_path, operation):
    engine.vae_info = {'vae_name': 'selected-external.safetensors', 'vae_hash': 'abc123'}
    source = tmp_path / 'input.png'
    Image.new('RGB', (64, 64)).save(source)
    if operation == 'decode':
        packed, _ = tools.pack(np.zeros((1, 4, 8, 8), dtype='float32'), 'L')
        packed.save(source)
    result = engine.vae_tools(dict(input_paths=[str(source)], operation=operation, output_dir=str(tmp_path)), ctx())
    assert result['results'][0]['status'] == 'complete'
    import json
    for key in ('path', 'preview_path'):
        metadata = json.loads(Image.open(result['results'][0][key]).info['VAE Tools'])
        assert metadata['vae'] == engine.vae_info
        assert metadata['family'] == 'sd15'
        assert metadata['scaling_factor'] == engine.pipe.vae.config.scaling_factor


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_exact_recovery_restores_actual_tiny_vae_decoder_input_bits(engine, tmp_path, monkeypatch, dtype):
    vae = engine.pipe.vae
    vae.to(dtype=dtype)
    vae.register_to_config(force_upcast=False)
    original_decode = vae.decode
    value = torch.linspace(-10, 10, 10000).reshape(1, 4, 50, 50).to(dtype)
    value.reshape(-1)[0] = -7.99609375
    def broken(*a, **kw):
        raise RuntimeError('capture decoder input')
    monkeypatch.setattr(vae, 'decode', broken)
    with tools.catch_decode(engine, vae, tmp_path, ctx()):
        vae.decode(value, return_dict=False)
    captured = []
    def capture(z, *a, **kw):
        captured.append(z.detach().clone())
        return original_decode(z, *a, **kw)
    monkeypatch.setattr(vae, 'decode', capture)
    result = engine.vae_tools(dict(input_paths=[engine._vae_recovery[0]['latent_path']], operation='decode', output_dir=str(tmp_path)), ctx())
    assert result['results'][0]['status'] == 'complete', result
    assert torch.equal(captured[0].view(torch.uint8), value.view(torch.uint8)), 'Exact recovery changed the failed decoder input bits'
    import json
    metadata = json.loads(Image.open(engine._vae_recovery[0]['path']).info['VAE Tools'])
    assert metadata['scaling_factor'] == vae.config.scaling_factor
    assert metadata['coordinate'] == 'diffusion_scaled'


def test_standard_diffusion_coordinates_invert_before_fp16_cast(engine, tmp_path, monkeypatch):
    from safetensors.numpy import save_file
    vae = engine.pipe.vae
    vae.to(dtype=torch.float16)
    vae.register_to_config(force_upcast=False)
    decoder_input = torch.linspace(-10, 10, 10000).reshape(1, 4, 50, 50).half()
    decoder_input.reshape(-1)[0] = -7.99609375
    # Independently construct the public diffusion-scaled tensor format in fp32.
    scaled = decoder_input.float().numpy() * np.float32(vae.config.scaling_factor)
    path = tmp_path / 'source-format.safetensors'
    save_file({'tensor': scaled}, str(path))
    original = vae.decode
    captured = []
    def capture(z, *a, **kw):
        captured.append(z.clone())
        return original(z, *a, **kw)
    monkeypatch.setattr(vae, 'decode', capture)
    result = engine.vae_tools(dict(input_paths=[str(path)], operation='decode', output_dir=str(tmp_path)), ctx())
    assert result['results'][0]['status'] == 'complete', result
    assert torch.equal(captured[0].view(torch.uint8), decoder_input.view(torch.uint8))


@pytest.mark.parametrize('params', [dict(seed=1e20), dict(seed=True), dict(input_paths=[]), dict(tiled='false'), dict(representation='RGB')])
def test_bad_requests_are_invalid_params(engine, tmp_path, params):
    source = tmp_path / 'source.png'
    Image.new('RGB', (64, 64)).save(source)
    with pytest.raises(EngineError) as exc:
        engine.vae_tools(dict(input_paths=[str(source)], **params) if 'input_paths' not in params else params, ctx())
    assert exc.value.code == 'invalid_params'
