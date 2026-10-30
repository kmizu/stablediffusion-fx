import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from sdfx import save_intermediate as si
from sdfx.protocol import EngineError


def test_cadence_and_rounding():
    c = si.normalize({'every_n': 3, 'start_at': 5, 'stop_at': 11})
    assert (c['start_at'], c['stop_at']) == (3, 9)
    assert [s for s in range(15) if si.selected(c, s)] == [3, 6, 9]
    assert [s for s in range(8) if si.selected(si.normalize({}), s)] == [5]


@pytest.mark.parametrize('cfg', [{'every_n': 0}, {'every_n': 1.5}, {'stop_at': -1},
    {'image_type': 'garbage'}, {'mode': 'garbage'}, {'fps': float('nan')}, {'format': 'exe'}])
def test_invalid_configuration(cfg):
    with pytest.raises(EngineError, match='save intermediate'):
        si.normalize(cfg)


class Context:
    def check_interrupt(self):
        pass
    def warn(self, message):
        self.warning = message


def test_recorder_batch_metadata_final_and_manifest(tmp_path):
    rec = si.Recorder(tmp_path, {'every_n': 1, 'final_save': True}, Context(), {'prompt': 'cat'})
    frames = [Image.new('RGB', (16, 8), 'red'), Image.new('RGB', (16, 8), 'blue')]
    rec.capture(frames, [42, 43], 'base', 1, 1, 0)
    rec.capture(frames, [42, 43], 'hires', 1, 4, 0)
    rec.final(frames[0], 42, 0)
    artifacts = rec.finish()
    assert len(artifacts) == 6  # 4 captures + final + manifest
    manifest = json.loads(Path(artifacts[-1]['path']).read_text())
    assert [f['seed'] for f in manifest['frames']] == [42, 43, 42, 43, 42]
    assert [f['step'] for f in manifest['frames']] == [1, 1, 4, 4, None]
    with Image.open(manifest['frames'][0]['path']) as im:
        metadata = json.loads(im.info['sdfx_intermediate'])
        assert metadata['seed'] == 42 and metadata['prompt'] == 'cat'
        assert im.info['parameters'].startswith('cat\n')
        assert 'Seed: 42' in im.info['parameters'] and 'Size: 16x8' in im.info['parameters']
    rec2 = si.Recorder(tmp_path, {}, Context(), {})
    assert rec.directory != rec2.directory


def test_empty_capture_no_video_and_stop_never_interrupts(tmp_path):
    rec = si.Recorder(tmp_path, {'every_n': 1, 'stop_at': 1}, Context(), {})
    rec.capture([Image.new('RGB', (8, 8))], [1], 'base', 2, 2, 0)
    assert rec.finish() == []


def test_video_timeline_holds_final_and_filters_passes():
    frames = [dict(path='base', pass_name='base'), dict(path='hires', pass_name='hires'),
              dict(path='final', pass_name='final')]
    c = si.normalize({'add_first_frames': 2, 'add_last_frames': 1, 'lores': False})
    assert [f['path'] for f in si.timeline(frames, c)] == ['final', 'final', 'hires', 'final', 'final']


def test_observer_captures_original_without_mutation_and_restores():
    torch = pytest.importorskip('torch')
    diffusers = pytest.importorskip('diffusers')
    scheduler = diffusers.DDIMScheduler(clip_sample=False)
    scheduler.set_timesteps(4)
    original = scheduler.step
    observer = si.observe_scheduler(scheduler)
    x = torch.ones(2, 4, 2, 2)
    noise = torch.ones_like(x) * .25
    out = scheduler.step(noise, scheduler.timesteps[0], x)
    expected = original(noise, scheduler.timesteps[0], x)
    assert torch.equal(out.prev_sample, expected.prev_sample)
    assert torch.equal(observer.denoised, out.pred_original_sample)
    assert torch.equal(observer.noisy, x)
    assert torch.equal(x, torch.ones_like(x))
    observer.remove()
    assert scheduler.step == original


def test_logical_steps_skip_second_order_substages():
    pipe = SimpleNamespace(scheduler=SimpleNamespace(order=2), _num_timesteps=7)
    assert [si.logical_step(pipe, i, 4) for i in range(7)] == [None, 1, None, 2, None, 3, 4]


@pytest.mark.parametrize('cls,prediction', [
    ('DPMSolverMultistepScheduler', 'epsilon'), ('DPMSolverMultistepScheduler', 'v_prediction'),
    ('UniPCMultistepScheduler', 'epsilon'), ('EulerDiscreteScheduler', 'epsilon'),
    ('EulerDiscreteScheduler', 'v_prediction'), ('HeunDiscreteScheduler', 'epsilon'),
    ('KDPM2DiscreteScheduler', 'epsilon'), ('LCMScheduler', 'epsilon'),
    ('LMSDiscreteScheduler', 'epsilon'), ('EulerAncestralDiscreteScheduler', 'epsilon'),
    ('KDPM2AncestralDiscreteScheduler', 'epsilon'), ('DPMSolverSinglestepScheduler', 'epsilon'),
    ('DPMSolverSDEScheduler', 'epsilon')])
def test_observer_real_schedulers_tuple_outputs_exact_and_finite(cls, prediction):
    torch = pytest.importorskip('torch')
    d = pytest.importorskip('diffusers')
    opts = dict(prediction_type=prediction)
    if cls == 'LCMScheduler':
        opts['clip_sample'] = False
    a, b = getattr(d, cls)(**opts), getattr(d, cls)(**opts)
    a.set_timesteps(4)
    b.set_timesteps(4)
    observer = si.observe_scheduler(a)
    x, y = torch.full((1, 4, 2, 2), .25), torch.full((1, 4, 2, 2), .25)
    for t in a.timesteps:
        noise = torch.full_like(x, .01)
        a.scale_model_input(x, t)
        b.scale_model_input(y, t)
        torch.manual_seed(42)
        out = a.step(noise, t, x, return_dict=False)
        torch.manual_seed(42)
        expected = b.step(noise, t, y, return_dict=False)
        assert torch.equal(out[0], expected[0])
        assert observer.denoised is not None and torch.isfinite(observer.denoised).all()
        if len(expected) > 1:
            assert torch.equal(observer.denoised, expected[1])
        x, y = out[0], expected[0]
    observer.remove()


def test_dpm_prediction_uses_normalized_alpha_sigma():
    torch = pytest.importorskip('torch')
    d = pytest.importorskip('diffusers')
    scheduler = d.DPMSolverMultistepScheduler()
    scheduler.set_timesteps(4)
    observer = si.observe_scheduler(scheduler)
    sample = torch.ones(1, 4, 2, 2)
    model = torch.ones_like(sample) * .25
    alpha, sigma = scheduler._sigma_to_alpha_sigma_t(scheduler.sigmas[0])
    expected = (sample - sigma * model) / alpha
    scheduler.step(model, scheduler.timesteps[0], sample)
    assert torch.allclose(observer.denoised, expected)
    observer.remove()


def test_callback_chaining_hires_batch_clock_and_restore(tmp_path):
    torch = pytest.importorskip('torch')
    d = pytest.importorskip('diffusers')
    scheduler = d.DDIMScheduler(clip_sample=False)
    scheduler.set_timesteps(3)
    pipe = SimpleNamespace(scheduler=scheduler, _num_timesteps=3)
    rec = si.Recorder(tmp_path, {'every_n': 2}, Context(), {})
    class Engine:
        def _decode(self, lat):
            return [Image.new('RGB', (8, 8), (int(t.mean() * 10) % 255, 0, 0)) for t in lat]
    called = []
    kw = {'callback_on_step_end': lambda p, i, t, data: called.append(i) or data}
    original = scheduler.step
    remove = si.install(Engine(), pipe, kw, rec, [50, 51], 'hires', 3, 3, 1)
    sample = torch.ones(2, 4, 1, 1)
    for i, t in enumerate(scheduler.timesteps):
        sample = scheduler.step(torch.zeros_like(sample), t, sample).prev_sample
        data = {'latents': sample}
        assert kw['callback_on_step_end'](pipe, i, t, data) is data
    remove()
    assert called == [0, 1, 2]
    assert scheduler.step == original
    assert [f['step'] for f in rec.frames] == [4, 4, 6, 6]
    assert [f['seed'] for f in rec.frames] == [50, 51, 50, 51]
    rec.close()
    assert json.loads((rec.directory / 'manifest.json').read_text())['status'] == 'incomplete'


def test_failed_save_removes_placeholder_and_missing_ffmpeg_creates_no_directory(tmp_path, monkeypatch):
    rec = si.Recorder(tmp_path, {'every_n': 1}, Context(), {})
    class Broken:
        width = height = 8
        def save(self, *args, **kwargs):
            raise OSError('disk full')
    with pytest.raises(OSError, match='disk full'):
        rec.capture([Broken()], [1], 'base', 1, 1, 0)
    assert list(rec.directory.iterdir()) == []
    rec.close()
    monkeypatch.setattr(si.shutil, 'which', lambda name: None)
    with pytest.raises(EngineError, match='ffmpeg'):
        si.Recorder(tmp_path, {'mode': 'Video'}, Context(), {})
    assert list((tmp_path / 'intermediates').iterdir()) == []


def test_recipes_only_keep_originals_and_have_no_temporary_frames(tmp_path):
    rec = si.Recorder(tmp_path, {'mode': 'Video', 'bat_only': True, 'every_n': 1}, Context(), {})
    rec.capture([Image.new('RGB', (8, 8))], [123], 'base', 1, 1, 0)
    artifacts = rec.finish()
    recipes = [a for a in artifacts if a['type'] == 'ffmpeg_recipe']
    assert len(recipes) == 1
    recipe = Path(recipes[0]['path']).read_text()
    assert 'cd /d "%~dp0"' in recipe and '-f concat' in recipe
    assert not list(rec.directory.glob('.video-*'))


def test_callback_cancellation_prevents_decode(tmp_path):
    torch = pytest.importorskip('torch')
    d = pytest.importorskip('diffusers')
    class Cancelled:
        def check_interrupt(self):
            raise RuntimeError('cancelled')
    rec = si.Recorder(tmp_path, {'every_n': 1}, Cancelled(), {})
    scheduler = d.DDIMScheduler()
    scheduler.set_timesteps(1)
    pipe = SimpleNamespace(scheduler=scheduler, _num_timesteps=1)
    kw = {}
    remove = si.install(None, pipe, kw, rec, [1], 'base', 1, 0, 0)
    with pytest.raises(RuntimeError, match='cancelled'):
        kw['callback_on_step_end'](pipe, 0, scheduler.timesteps[0], {})
    remove()
    rec.close()


def test_lcm_dict_prediction_matches_its_boundary_condition():
    torch = pytest.importorskip('torch')
    d = pytest.importorskip('diffusers')
    scheduler = d.LCMScheduler(clip_sample=False, timestep_scaling=.01)
    scheduler.set_timesteps(4)
    observer = si.observe_scheduler(scheduler)
    sample = torch.ones(1, 4, 1, 1)
    result = scheduler.step(torch.zeros_like(sample), scheduler.timesteps[0], sample)
    assert torch.equal(observer.denoised, result.denoised)
    observer.remove()


def test_final_copy_does_not_create_a_nonexistent_hires_recipe(tmp_path):
    rec = si.Recorder(tmp_path, {'mode': 'Video', 'bat_only': True, 'every_n': 1, 'final_save': True}, Context(), {})
    image = Image.new('RGB', (8, 8))
    rec.capture([image], [123], 'base', 1, 1, 0)
    rec.final(image, 123, 0)
    artifacts = rec.finish()
    assert len([a for a in artifacts if a['type'] == 'ffmpeg_recipe']) == 1


def test_encoding_cancellation_removes_partial_video_and_temporary_frames(tmp_path, monkeypatch):
    class CancelContext(Context):
        cancelled = False
        def check_interrupt(self):
            if self.cancelled:
                raise RuntimeError('cancelled while encoding')
    ctx = CancelContext()
    monkeypatch.setattr(si.shutil, 'which', lambda name: 'ffmpeg')
    rec = si.Recorder(tmp_path, {'mode': 'Video', 'every_n': 1}, ctx, {})
    rec.capture([Image.new('RGB', (8, 8))], [1], 'base', 1, 1, 0)
    class Running:
        killed = False
        waited = False
        def kill(self):
            self.killed = True
        def wait(self, **kwargs):
            self.waited = True
            return 0
    running = Running()
    def start(args, **kwargs):
        (Path(kwargs['cwd']) / args[-1]).write_bytes(b'partial')
        assert kwargs['stdout'] == si.subprocess.DEVNULL
        assert kwargs['stderr'] is not None
        ctx.cancelled = True
        return running
    monkeypatch.setattr(si.subprocess, 'Popen', start)
    with pytest.raises(RuntimeError, match='cancelled while encoding'):
        rec.finish()
    assert running.killed and running.waited
    assert not list(rec.directory.glob('*.mp4'))
    assert not list(rec.directory.glob('.video-*'))
    rec.close()
    assert json.loads((rec.directory / 'manifest.json').read_text())['status'] == 'incomplete'


def test_encoder_error_retains_frames_and_returns_manifest_with_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(si.shutil, 'which', lambda name: 'ffmpeg')
    ctx = Context()
    rec = si.Recorder(tmp_path, {'mode': 'Video', 'every_n': 1}, ctx, {})
    rec.capture([Image.new('RGB', (8, 8))], [1], 'base', 1, 1, 0)
    class Failed:
        def kill(self):
            pass
        def wait(self, **kwargs):
            return 1
    def start(args, **kwargs):
        (Path(kwargs['cwd']) / args[-1]).write_bytes(b'partial')
        kwargs['stderr'].write(b'Unknown encoder libx264')
        return Failed()
    monkeypatch.setattr(si.subprocess, 'Popen', start)
    artifacts = rec.finish()
    assert [a['type'] for a in artifacts] == ['image', 'manifest']
    assert 'ffmpeg export failed' in ctx.warning
    assert not list(rec.directory.glob('*.mp4'))
    assert not list(rec.directory.glob('.video-*'))
    manifest = json.loads((rec.directory / 'manifest.json').read_text())
    assert manifest['status'] == 'complete_with_export_errors'
    assert 'Unknown encoder' in manifest['export_errors'][0]


@pytest.mark.parametrize('smooth,seconds,fps,expected_rate,expected_duration', [
    (True, 0, 30, 1, 4), (True, 8, 30, .5, 8), (True, 2, 30, 1, 4),
    (False, 0, 2, 2, 2), (False, 8, 30, .5, 8)])
def test_video_source_clock_and_duration(smooth, seconds, fps, expected_rate, expected_duration):
    c = si.normalize(dict(smooth=smooth, seconds=seconds, fps=fps))
    rate, duration = si.video_timing(c, 4)
    assert rate == expected_rate and duration == expected_duration
    if smooth:
        assert rate < fps


@pytest.mark.parametrize('seconds,rate,duration,padding', [(0, 1, 4, 2), (8, .5, 8, 4)])
def test_smoothing_command_and_recipe_preserve_endpoints_and_duration(tmp_path, monkeypatch, seconds, rate, duration, padding):
    monkeypatch.setattr(si.shutil, 'which', lambda name: 'ffmpeg')
    rec = si.Recorder(tmp_path, dict(mode='Video', every_n=1, smooth=True, seconds=seconds,
        fps=30, ffmpeg_bat=True), Context(), {})
    for step in range(1, 5):
        rec.capture([Image.new('RGB', (8, 8), (step * 50, 0, 0))], [1], 'base', step, step, 0)
    commands = []
    class Finished:
        def wait(self, **kwargs):
            return 0
    def start(args, **kwargs):
        commands.append(args)
        Path(args[-1]).write_bytes(b'video')
        return Finished()
    monkeypatch.setattr(si.subprocess, 'Popen', start)
    artifacts = rec.finish()
    command = commands[0]
    assert float(command[command.index('-framerate') + 1]) == rate
    assert float(command[command.index('-t') + 1]) == duration
    filters = command[command.index('-vf') + 1]
    assert f'fps={rate:g},tpad=' in filters
    assert f'tpad=stop_mode=clone:stop_duration={padding}' in filters
    assert filters.index('tpad=') < filters.index('minterpolate=fps=30') < filters.index(f'trim=duration={duration}')
    assert filters.endswith('setpts=PTS-STARTPTS')
    recipe = Path(next(a['path'] for a in artifacts if a['type'] == 'ffmpeg_recipe')).read_text()
    assert filters in recipe and f'-t {duration}' in recipe
    playlist = Path(next(a['path'] for a in artifacts if a['type'] == 'ffmpeg_playlist')).read_text()
    assert playlist.count(f'duration {1 / rate:.9f}') == 4


def test_smoothing_requires_output_fps_above_one():
    with pytest.raises(EngineError, match='smoothing.*fps'):
        si.normalize(dict(smooth=True, fps=1))


@pytest.mark.parametrize('fail', [False, True])
def test_vae_upcast_restores_dtype_even_when_decode_raises(fail):
    torch = pytest.importorskip('torch')
    pytest.importorskip('diffusers')
    from sdfx.engine import Engine
    class Vae:
        dtype = torch.float16
        device = 'cpu'
        config = SimpleNamespace(force_upcast=True, scaling_factor=1)
        def to(self, dtype):
            self.dtype = dtype
        def decode(self, latent, **kwargs):
            assert self.dtype == torch.float32 and latent.dtype == torch.float32
            if fail:
                raise RuntimeError('injected VAE decode failure')
            return (latent,)
    vae = Vae()
    engine = Engine.__new__(Engine)
    engine.pipe = SimpleNamespace(vae=vae, image_processor=SimpleNamespace(postprocess=lambda image, **kw: image))
    latent = torch.ones(1, 4, 2, 2, dtype=torch.float16)
    if fail:
        with pytest.raises(RuntimeError, match='injected VAE decode failure'):
            engine._decode(latent)
    else:
        assert engine._decode(latent).dtype == torch.float32
    assert vae.dtype == torch.float16


def test_real_gif_retains_source_endpoint_colors(tmp_path):
    if not si.shutil.which('ffmpeg'):
        pytest.skip('optional real ffmpeg executable unavailable')
    rec = si.Recorder(tmp_path, dict(mode='Video', format='gif', every_n=1, fps=2), Context(), {})
    colors = [(20, 30, 60), (60, 50, 60), (100, 70, 60)]
    for step, color in enumerate(colors, 1):
        rec.capture([Image.new('RGB', (16, 16), color)], [1], 'base', step, step, 0)
    artifacts = rec.finish()
    video = next(a['path'] for a in artifacts if a['type'] == 'video')
    with Image.open(video) as image:
        for index in (0, len(colors) - 1):
            image.seek(index)
            actual = image.convert('RGB').getpixel((8, 8))
            assert max(abs(a - b) for a, b in zip(actual, colors[index])) <= 2


def test_protected_intermediate_frames_keep_pixels_and_metadata_without_plaintext_file(tmp_path):
    from sdfx.encrypt_image import decode_png
    rec = si.Recorder(tmp_path, {'every_n': 1}, Context(), {'prompt': 'cat'}, image_password='secret')
    image = Image.new('RGB', (9, 7))
    image.putdata([(x * 20, y * 30, x + y) for y in range(7) for x in range(9)])
    rec.capture([image], [42], 'base', 1, 1, 0)
    path = Path(rec.frames[0]['path'])
    with Image.open(path) as stored:
        assert stored.info['Encrypt'] == 'pixel_shuffle_3'
        assert list(stored.getdata()) != list(image.getdata())
        assert stored.info['parameters'].startswith('cat\n')
    plain, verified = decode_png(path.read_bytes(), 'secret')
    from io import BytesIO
    with Image.open(BytesIO(plain)) as decoded:
        assert list(decoded.getdata()) == list(image.getdata())
        assert json.loads(decoded.info['sdfx_intermediate'])['seed'] == 42
    assert verified is True
    assert not any(p.name.startswith('.sdfx-frame-') for p in rec.directory.iterdir())


def test_protected_intermediate_write_failure_removes_reserved_file(tmp_path, monkeypatch):
    from sdfx import encrypt_image
    rec = si.Recorder(tmp_path, {'every_n': 1}, Context(), {}, image_password='secret')
    def fail(*args):
        raise RuntimeError('encoding failed')
    monkeypatch.setattr(encrypt_image, 'encode_png', fail)
    with pytest.raises(RuntimeError, match='encoding failed'):
        rec.capture([Image.new('RGB', (4, 4))], [1], 'base', 1, 1, 0)
    assert list(rec.directory.iterdir()) == []
    assert rec.frames == []


def test_protected_intermediate_video_is_explicitly_rejected_until_supported(tmp_path):
    with pytest.raises(EngineError, match='protected video export is not yet supported'):
        si.Recorder(tmp_path, {'mode': 'Video'}, Context(), {}, image_password='secret')
    assert list(tmp_path.iterdir()) == []


def test_protected_video_decodes_frames_only_with_explicit_plaintext_export(tmp_path, monkeypatch):
    import shutil
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg unavailable')
    rec = si.Recorder(tmp_path, {'mode': 'Video', 'format': 'gif', 'every_n': 1, 'fps': 2},
                      Context(), {}, image_password='secret', allow_plaintext_video=True)
    frames = [Image.new('RGB', (16, 16)), Image.new('RGB', (16, 16))]
    for n, frame in enumerate(frames):
        for y in range(16):
            for x in range(16):
                frame.putpixel((x, y), ((x * 17) % 256, (y * 17) % 256, n * 200))
    plain = si.Recorder(tmp_path, {'mode': 'Video', 'format': 'gif', 'every_n': 1, 'fps': 2}, Context(), {})
    for i, frame in enumerate(frames):
        plain.capture([frame], [1], 'base', i + 1, i + 1, 0)
    plain_gif = next(Path(a['path']) for a in plain.finish() if a['type'] == 'video')
    rec.capture([frames[0]], [1], 'base', 1, 1, 0)
    rec.capture([frames[1]], [1], 'base', 2, 2, 0)
    artifacts = rec.finish()
    gif = next(Path(a['path']) for a in artifacts if a['type'] == 'video')
    with Image.open(gif) as actual, Image.open(plain_gif) as expected:
        assert actual.n_frames == expected.n_frames == 2
        for index in range(actual.n_frames):
            actual.seek(index)
            expected.seek(index)
            assert actual.convert('RGB').tobytes() == expected.convert('RGB').tobytes()
    for index, record in enumerate(rec.frames):
        with Image.open(record['path']) as stored:
            assert stored.info['Encrypt'] == 'pixel_shuffle_3'
            assert stored.convert('RGB').tobytes() != frames[index].tobytes()
    assert not list(rec.directory.glob('.video-*'))


@pytest.mark.parametrize('option', ['bat_only', 'ffmpeg_bat'])
def test_protected_video_recipe_options_generate_recipe(tmp_path, option):
    import shutil
    if option == 'ffmpeg_bat' and not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg required')
    rec = si.Recorder(tmp_path, {'mode': 'Video', option: True, 'every_n': 1}, Context(), {},
                      image_password='secret', allow_plaintext_video=True)
    rec.capture([Image.new('RGB', (8, 8), 'red')], [1], 'base', 1, 1, 0)
    artifacts = rec.finish()
    assert any(item['type'] == 'ffmpeg_recipe' for item in artifacts)
    assert any(item['type'] == 'video' for item in artifacts) == (option == 'ffmpeg_bat')


def test_protected_video_interruption_removes_plaintext_temp_frames(tmp_path):
    import shutil
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg unavailable')
    class CancelDuringVideo(Context):
        def __init__(self):
            self.cancel = False
            self.observed_plain_frame = False
        def check_interrupt(self):
            if self.cancel:
                self.observed_plain_frame = any(tmp_path.rglob('.video-*/*.png'))
                if self.observed_plain_frame:
                    raise RuntimeError('stopped')
    ctx = CancelDuringVideo()
    rec = si.Recorder(tmp_path, {'mode': 'Video', 'every_n': 1}, ctx, {},
                      image_password='secret', allow_plaintext_video=True)
    rec.capture([Image.new('RGB', (8, 8), 'red')], [1], 'base', 1, 1, 0)
    rec.capture([Image.new('RGB', (8, 8), 'blue')], [1], 'base', 2, 2, 0)
    ctx.cancel = True
    with pytest.raises(RuntimeError, match='stopped'):
        rec.finish()
    assert ctx.observed_plain_frame
    assert not list(rec.directory.glob('.video-*'))
    assert not list(rec.directory.glob('*.mp4'))
    assert Path(rec.frames[0]['path']).is_file()


def test_generation_forgets_recorder_when_close_raises():
    pytest.importorskip('torch')
    from sdfx.engine import Engine
    class Recorder:
        def close(self):
            raise RuntimeError('close failed')
    class EngineStub:
        _generate = Engine._generate
        def _generate_recorded(self, params, ctx, mode):
            self._intermediate_recorder = Recorder()
            return {}
    engine = EngineStub()
    with pytest.raises(RuntimeError, match='close failed'):
        engine._generate({}, object(), 'txt2img')
    assert engine._intermediate_recorder is None


def test_protected_recipe_can_export_later_without_storing_password(tmp_path, monkeypatch):
    import shutil
    from sdfx import protected_recipe
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg required')
    rec = si.Recorder(tmp_path, {'mode': 'Video', 'bat_only': True, 'every_n': 1, 'format': 'gif'},
                      Context(), {}, image_password='private-password', allow_plaintext_video=True)
    for i in range(2):
        image = Image.new('RGB', (16, 16), ('red', 'blue')[i])
        rec.capture([image], [1], 'base', i + 1, i + 1, 0)
    artifacts = rec.finish()
    spec = Path(next(a['path'] for a in artifacts if a['type'] == 'ffmpeg_playlist'))
    recipe = Path(next(a['path'] for a in artifacts if a['type'] == 'ffmpeg_recipe'))
    assert 'private-password' not in spec.read_text(encoding='utf-8')
    assert 'private-password' not in recipe.read_text(encoding='utf-8')
    target = protected_recipe.export(spec, 'private-password')
    assert target.is_file()
    with Image.open(target) as result:
        assert result.n_frames >= 2
        assert result.convert('RGB').getpixel((8, 8))[0] > 200
        result.seek(result.n_frames - 1)
        assert result.convert('RGB').getpixel((8, 8))[2] > 200
    assert not list(rec.directory.glob('.video-*'))
    with pytest.raises(FileExistsError):
        protected_recipe.export(spec, 'private-password')


def test_protected_recipe_rejects_wrong_password_without_partial_output(tmp_path):
    import shutil
    from sdfx import protected_recipe
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg required')
    rec = si.Recorder(tmp_path, {'mode': 'Video', 'bat_only': True, 'every_n': 1},
                      Context(), {}, image_password='secret', allow_plaintext_video=True)
    rec.capture([Image.new('RGB', (8, 8), 'red')], [1], 'base', 1, 1, 0)
    spec = Path(next(a['path'] for a in rec.finish() if a['type'] == 'ffmpeg_playlist'))
    with pytest.raises(ValueError, match='incorrect password'):
        protected_recipe.export(spec, 'wrong')
    assert not list(rec.directory.glob('*.mp4')) and not list(rec.directory.glob('.video-*'))


def test_protected_recipe_windows_batch_executes_and_hides_password(tmp_path):
    import os, shutil, subprocess
    if os.name != 'nt' or not shutil.which('ffmpeg'):
        pytest.skip('Windows and ffmpeg required')
    folder = tmp_path / '日本語 % folder'
    folder.mkdir()
    rec = si.Recorder(folder, {'mode': 'Video', 'bat_only': True, 'every_n': 1, 'format': 'gif'},
                      Context(), {}, image_password='private-password', allow_plaintext_video=True)
    rec.capture([Image.new('RGB', (8, 8), 'red')], [1], 'base', 1, 1, 0)
    artifacts = rec.finish()
    bat = next(a['path'] for a in artifacts if a['type'] == 'ffmpeg_recipe')
    result = subprocess.run(['cmd.exe', '/d', '/c', bat], input='private-password\n',
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert 'private-password' not in result.stdout + result.stderr
    assert len(list(rec.directory.glob('*.gif'))) == 1


def test_protected_recipe_concurrent_output_is_not_overwritten(tmp_path, monkeypatch):
    import shutil
    from sdfx import protected_recipe
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg required')
    rec = si.Recorder(tmp_path, {'mode': 'Video', 'bat_only': True, 'every_n': 1},
                      Context(), {}, image_password='secret', allow_plaintext_video=True)
    rec.capture([Image.new('RGB', (8, 8), 'red')], [1], 'base', 1, 1, 0)
    spec = Path(next(a['path'] for a in rec.finish() if a['type'] == 'ffmpeg_playlist'))
    link = protected_recipe.os.link
    def concurrent_target(source, target):
        Path(target).write_bytes(b'existing output')
        return link(source, target)
    monkeypatch.setattr(protected_recipe.os, 'link', concurrent_target)
    with pytest.raises(FileExistsError):
        protected_recipe.export(spec, 'secret')
    assert next(rec.directory.glob('*.mp4')).read_bytes() == b'existing output'
    assert not list(rec.directory.glob('.protected-recipe-*'))
