import pytest
from sdfx.steps_animation import normalize
from sdfx.protocol import EngineError


def test_defaults_preserve_full_step_sequence():
    settings = normalize({})
    assert settings['duration'] == 5 and settings['fps'] == 30
    assert settings['interpolation'] == 'blend'
    assert settings['skip_initial'] == 0
    assert settings['create_output'] and not settings['delete_intermediate']


@pytest.mark.parametrize('config', [
    {'duration': 0}, {'duration': float('nan')}, {'fps': 0}, {'fps': 241},
    {'skip_initial': -1}, {'skip_initial': 1.5}, {'codec': 'shell'},
    {'interpolation': 'unknown'}, {'frames_dir': '../outside'},
    {'output_dir': 'C:/outside'}, {'output_dir': '/absolute'},
    {'frames_dir': 'frames/../../escape'}, {'frames_dir': 'a:stream'},
])
def test_invalid_configuration_fails_before_generation(config):
    with pytest.raises(EngineError):
        normalize(config)


def test_nested_relative_output_and_explicit_false():
    config = normalize({'frames_dir': 'journey/frames', 'create_output': 'false', 'codec': 'prores'})
    assert config['frames_dir'] == 'journey/frames'
    assert config['create_output'] is False
    assert config['codec'] == 'prores'


@pytest.mark.parametrize('interpolation', ['none', 'blend', 'mci'])
@pytest.mark.parametrize('frame_count', [2, 3])
def test_real_video_duration_and_final_frame(tmp_path, interpolation, frame_count):
    import shutil, subprocess, json
    from PIL import Image
    from sdfx.steps_animation import encode_frames
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg required')
    frames = []
    for index, color in enumerate(['red', 'blue'] if frame_count == 2 else ['red', 'green', 'blue']):
        path = tmp_path / f'{index}.png'
        Image.new('RGB', (64, 64), color).save(path)
        frames.append(str(path))
    target = tmp_path / 'result.mp4'
    class Context:
        def check_interrupt(self): pass
    encode_frames(frames, target, normalize({'duration': 0.6, 'fps': 20, 'interpolation': interpolation}),
                  Context(), {'prompt': 'teapot', 'seed': 7})
    report = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(target)]))
    assert abs(float(report['format']['duration']) - .6) < .06
    assert int(report['streams'][0]['nb_frames']) == 12
    assert report['format']['tags']['title'] == 'teapot'
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(target), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    final = raw[-64*64*3:]
    assert sum(final[2::3]) / 4096 > 230


@pytest.mark.parametrize('create_output,delete_intermediate', [(False, True), (True, False), (True, True)])
def test_recorder_cleanup_only_after_successful_video(tmp_path, create_output, delete_intermediate):
    import shutil, json
    from pathlib import Path
    from PIL import Image
    from sdfx.steps_animation import Recorder
    if create_output and not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg required')
    class Context:
        def check_interrupt(self): pass
        def warn(self, message): raise AssertionError(message)
    recorder = Recorder(tmp_path, {'create_output': create_output, 'delete_intermediate': delete_intermediate,
                                  'duration': .5, 'fps': 10, 'skip_initial': 1}, Context(), {'prompt': 'test'})
    for step in range(1, 4):
        recorder.capture([Image.new('RGB', (64, 64), 'blue')], [7], 'base', step, step, 0)
    assert len(recorder.frames) == 2
    paths = [entry['path'] for entry in recorder.frames]
    artifacts = recorder.finish()
    assert bool([item for item in artifacts if item['type'] == 'video']) == create_output
    assert all(Path(path).exists() for path in paths) == (not(create_output and delete_intermediate))
    assert json.loads((recorder.directory / 'manifest.json').read_text(encoding='utf-8'))['status'] == 'complete'


@pytest.mark.parametrize('changes', [{'batch_size': 2}, {'n_iter': 2}, {'hooks': {'save_intermediate': {}}}])
def test_generation_conflicts_rejected(changes):
    from types import SimpleNamespace
    from sdfx.steps_animation import validate_generation
    params = dict(batch_size=1, n_iter=1, hooks={})
    params.update(changes)
    with pytest.raises(EngineError):
        validate_generation(SimpleNamespace(**params))



def test_failed_export_preserves_frames_even_when_delete_requested(tmp_path, monkeypatch):
    from pathlib import Path
    from PIL import Image
    from sdfx import steps_animation
    from sdfx.protocol import UNSUPPORTED
    warnings = []
    class Context:
        def check_interrupt(self): pass
        def warn(self, message): warnings.append(message)
    monkeypatch.setattr('shutil.which', lambda _: 'ffmpeg')
    def fail(*args):
        raise EngineError(UNSUPPORTED, 'injected encoder failure')
    monkeypatch.setattr(steps_animation, 'encode_frames', fail)
    recorder = steps_animation.Recorder(tmp_path, {'delete_intermediate': True}, Context(), {})
    recorder.capture([Image.new('RGB', (64, 64))], [1], 'base', 1, 1, 0)
    artifacts = recorder.finish()
    assert warnings and 'frames retained' in warnings[0]
    assert not any(item['type'] == 'video' for item in artifacts)
    assert all(Path(item['path']).exists() for item in artifacts)


@pytest.mark.parametrize('run_incomplete', [False, True])
def test_incomplete_frames_export_only_when_requested(tmp_path, run_incomplete):
    import json, threading, shutil
    from PIL import Image
    from sdfx.steps_animation import Recorder
    from sdfx.protocol import INTERRUPTED
    if run_incomplete and not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg required')
    class Context:
        interrupt_event = threading.Event()
        def check_interrupt(self):
            if self.interrupt_event.is_set():
                raise EngineError(INTERRUPTED, 'interrupted')
        def warn(self, message): raise AssertionError(message)
    context = Context()
    recorder = Recorder(tmp_path, {'run_incomplete': run_incomplete, 'duration': .5}, context, {})
    recorder.capture([Image.new('RGB', (64, 64), 'blue')], [1], 'base', 1, 1, 0)
    context.interrupt_event.set()
    recorder.close()
    manifest = json.loads((recorder.directory / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'].startswith('incomplete')
    assert any(item['type'] == 'video' for item in manifest['artifacts']) == run_incomplete
    assert context.interrupt_event.is_set(), 'original cancellation must remain observable'



def test_encoding_interrupt_kills_process_and_removes_partial_target(tmp_path, monkeypatch):
    from PIL import Image
    from sdfx.steps_animation import encode_frames
    from sdfx.protocol import INTERRUPTED
    frame = tmp_path / 'frame.png'
    Image.new('RGB', (64, 64)).save(frame)
    target = tmp_path / 'partial.mp4'
    state = {'started': False, 'killed': False, 'waited': False}
    class Process:
        def __init__(self, *args, **kwargs):
            state['started'] = True
            target.write_bytes(b'incomplete')
        def poll(self): return None
        def kill(self): state['killed'] = True
        def wait(self, *args, **kwargs): state['waited'] = True; return 0
    class Context:
        def check_interrupt(self):
            if state['started']:
                raise EngineError(INTERRUPTED, 'cancel encoding')
    monkeypatch.setattr('shutil.which', lambda _: 'ffmpeg')
    monkeypatch.setattr('subprocess.Popen', Process)
    with pytest.raises(EngineError, match='cancel encoding'):
        encode_frames([frame], target, normalize({}), Context(), {})
    assert state['killed'] and state['waited'] and not target.exists()
    assert frame.exists()


@pytest.mark.parametrize('codec', ['h264', 'h265', 'vp9', 'av1', 'prores'])
def test_real_codec_export_decodes(tmp_path, codec):
    import json, shutil, subprocess
    from PIL import Image
    from sdfx.steps_animation import encode_frames
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg required')
    image = tmp_path / 'frame.png'
    Image.new('RGB', (64, 64), 'blue').save(image)
    extension = {'vp9': 'webm', 'prores': 'mov'}.get(codec, 'mp4')
    target = tmp_path / ('codec.' + extension)
    class Context:
        def check_interrupt(self): pass
    encode_frames([image], target, normalize({'codec': codec, 'duration': .4, 'fps': 10}), Context(), {})
    report = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-show_streams',
                                                 '-show_format', '-of', 'json', str(target)]))
    assert int(report['streams'][0]['nb_read_frames']) == 4
    assert abs(float(report['format']['duration']) - .4) < .06


@pytest.mark.parametrize('reason', ['second_interrupt', 'deadline'])
def test_partial_export_stops_without_hiding_original_cancel(tmp_path, monkeypatch, reason):
    import json, threading
    from pathlib import Path
    from PIL import Image
    from sdfx import steps_animation
    from sdfx.protocol import INTERRUPTED
    clock = {'now': 100.0}
    monkeypatch.setattr('time.monotonic', lambda: clock['now'])
    monkeypatch.setattr('shutil.which', lambda _: 'ffmpeg')
    class Context:
        interrupt_event = threading.Event()
        def check_interrupt(self):
            if self.interrupt_event.is_set():
                raise EngineError(INTERRUPTED, 'second interrupt')
        def warn(self, message): pass
    context = Context()
    recorder = steps_animation.Recorder(tmp_path, {'run_incomplete': True}, context, {})
    recorder.capture([Image.new('RGB', (64, 64))], [1], 'base', 1, 1, 0)
    context.interrupt_event.set()
    def stopped(frames, target, settings, partial_context, metadata):
        if reason == 'second_interrupt':
            context.interrupt_event.set()
        else:
            clock['now'] = 161.0
        partial_context.check_interrupt()
        pytest.fail('partial export must stop')
    monkeypatch.setattr(steps_animation, 'encode_frames', stopped)
    recorder.close()
    manifest = json.loads((recorder.directory / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'incomplete_with_export_errors'
    assert manifest['export_errors']
    assert context.interrupt_event.is_set()
    assert all(Path(frame['path']).exists() for frame in recorder.frames)



def test_random_seed_metadata_records_resolved_frame_seed(tmp_path):
    import json
    from PIL import Image
    from sdfx.steps_animation import Recorder
    class Context:
        def check_interrupt(self): pass
    recorder = Recorder(tmp_path, {'create_output': False}, Context(), {'seed': -1})
    recorder.capture([Image.new('RGB', (64, 64))], [123456], 'base', 1, 1, 0)
    recorder.finish()
    manifest = json.loads((recorder.directory / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['metadata']['seed'] == 123456


def test_protected_frames_are_saved_without_plaintext(tmp_path):
    import io
    from pathlib import Path
    from PIL import Image
    from sdfx.encrypt_image import decode_png
    from sdfx.steps_animation import Recorder
    class Context:
        def check_interrupt(self): pass
        def warn(self, message): raise AssertionError(message)
    recorder = Recorder(tmp_path, {'create_output': False}, Context(), {}, image_password='secret')
    image = Image.new('RGB', (16, 16))
    for y in range(16):
        for x in range(16):
            image.putpixel((x, y), (x * 15, y * 15, (x + y) * 7))
    recorder.capture([image], [1], 'base', 1, 1, 0)
    artifacts = recorder.finish()
    path = Path(next(a['path'] for a in artifacts if a['type'] == 'image'))
    with Image.open(path) as scrambled:
        assert scrambled.info['Encrypt'] == 'pixel_shuffle_3'
        assert scrambled.tobytes() != image.tobytes()
    decoded, verified = decode_png(path.read_bytes(), 'secret')
    with Image.open(io.BytesIO(decoded)) as restored:
        assert restored.tobytes() == image.tobytes() and verified is True


def test_protected_video_requires_explicit_plaintext_export(tmp_path):
    from sdfx.steps_animation import Recorder
    class Context:
        def check_interrupt(self): pass
    with pytest.raises(EngineError, match='plaintext video export'):
        Recorder(tmp_path, {}, Context(), {}, image_password='secret')
    assert list(tmp_path.iterdir()) == []


def test_protected_animation_video_matches_plain_frames(tmp_path):
    import shutil, subprocess
    from pathlib import Path
    from PIL import Image
    from sdfx.steps_animation import Recorder
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg required')
    class Context:
        def check_interrupt(self): pass
        def warn(self, message): raise AssertionError(message)
    opts = {'duration': .5, 'fps': 10, 'interpolation': 'none'}
    regular = Recorder(tmp_path / 'plain', opts, Context(), {})
    protected = Recorder(tmp_path / 'protected', opts, Context(), {}, image_password='secret', allow_plaintext_video=True)
    for i in range(2):
        image = Image.new('RGB', (32, 32))
        for y in range(32):
            for x in range(32):
                image.putpixel((x, y), (x * 7, y * 7, i * 180))
        regular.capture([image], [1], 'base', i + 1, i + 1, 0)
        protected.capture([image], [1], 'base', i + 1, i + 1, 0)
    plain_video = next(a['path'] for a in regular.finish() if a['type'] == 'video')
    protected_video = next(a['path'] for a in protected.finish() if a['type'] == 'video')
    def rgb(path):
        return subprocess.check_output(['ffmpeg', '-v', 'error', '-i', path, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    assert rgb(plain_video) == rgb(protected_video)
    assert not list(Path(protected.video_directory).glob('.steps-encode-*'))


def test_protected_animation_cancellation_removes_plaintext_staging(tmp_path):
    import shutil
    from pathlib import Path
    from PIL import Image
    from sdfx.steps_animation import Recorder
    from sdfx.protocol import INTERRUPTED
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg required')
    class Context:
        cancel = False
        saw_staging = False
        def check_interrupt(self):
            if self.cancel and any(tmp_path.rglob('.steps-encode-*/*.png')):
                self.saw_staging = True
                raise EngineError(INTERRUPTED, 'stopped')
        def warn(self, message): raise AssertionError(message)
    ctx = Context()
    rec = Recorder(tmp_path, {'duration': .5}, ctx, {}, image_password='secret', allow_plaintext_video=True)
    for i in range(2):
        rec.capture([Image.new('RGB', (32, 32), (i * 200, 0, 0))], [1], 'base', i + 1, i + 1, 0)
    ctx.cancel = True
    with pytest.raises(EngineError, match='stopped'):
        rec.finish()
    assert ctx.saw_staging
    assert not list(tmp_path.rglob('.steps-encode-*'))
    assert not list(Path(rec.video_directory).glob('*.mp4'))
    assert all(Path(f['path']).is_file() for f in rec.frames)
