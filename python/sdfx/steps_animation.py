"""Steps Animation capture and video export, independently implemented from documented behavior."""
import math
from pathlib import PurePosixPath, PureWindowsPath
from .hooks import coerce_bool
from .protocol import EngineError, INVALID_PARAMS

DEFAULTS = dict(duration=5.0, fps=30, codec='h264', interpolation='blend', skip_initial=0,
                debug=False, run_incomplete=False, delete_intermediate=False, create_output=True,
                frames_dir='steps-animation/frames', output_dir='steps-animation/videos')


def normalize(raw):
    settings = dict(DEFAULTS, **raw)
    def invalid(key):
        raise EngineError(INVALID_PARAMS, f'steps animation: invalid {key}')
    for key, low, high, integer in [('duration', 0.01, 3600, False), ('fps', 1, 240, True),
                                    ('skip_initial', 0, 10000, True)]:
        try:
            value = float(settings[key])
            if not math.isfinite(value) or not low <= value <= high or (integer and value != int(value)):
                invalid(key)
            settings[key] = int(value) if integer else value
        except (ValueError, TypeError, OverflowError):
            invalid(key)
    for key, choices in [('codec', ('h264', 'h265', 'vp9', 'av1', 'prores')),
                         ('interpolation', ('none', 'blend', 'mci'))]:
        if settings[key] not in choices:
            invalid(key)
    for key in ('debug', 'run_incomplete', 'delete_intermediate', 'create_output'):
        settings[key] = coerce_bool(settings[key], DEFAULTS[key])
    for key in ('frames_dir', 'output_dir'):
        value = settings[key]
        if not isinstance(value, str) or not value or any(ord(char) < 32 for char in value):
            invalid(key)
        path = PureWindowsPath(value)
        parts = PurePosixPath(value.replace(chr(92), '/')).parts
        if path.drive or path.root or ':' in value or '..' in parts:
            invalid(key)
        settings[key] = '/'.join(parts)
        if settings[key] in ('', '.'):
            invalid(key)
    return settings


def encode_frames(frames, target, settings, ctx, metadata, image_password=None):
    """Encode owned captured frames; preserve source PNGs on any export failure."""
    import json
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path
    from PIL import Image, ImageOps
    from .protocol import UNSUPPORTED
    executable = shutil.which('ffmpeg')
    if not executable:
        raise EngineError(UNSUPPORTED, 'steps animation: ffmpeg is required')
    if not frames:
        raise EngineError(INVALID_PARAMS, 'steps animation: no frames to encode')
    target = Path(target)
    if target.exists():
        raise EngineError(INVALID_PARAMS, 'steps animation: output already exists')
    target.parent.mkdir(parents=True, exist_ok=True)
    duration, fps = settings['duration'], settings['fps']
    rate = len(frames) / duration
    dimensions = []
    for source in frames:
        with Image.open(source) as image:
            dimensions.append(image.size)
    width = (max(w for w, h in dimensions) + 1) // 2 * 2
    height = (max(h for w, h in dimensions) + 1) // 2 * 2
    filters = [f'tpad=stop_mode=clone:stop_duration={2 / rate:g}']
    if settings['interpolation'] != 'none' and len(frames) >= 2:
        mode = 'blend' if settings['interpolation'] == 'blend' else 'mci'
        filters.append(f'minterpolate=fps={fps}:mi_mode={mode}')
    else:
        filters.append(f'fps={fps}')
    filters += [f'trim=duration={duration:g}', 'setpts=PTS-STARTPTS']
    codec, pixel = {'h264': ('libx264', 'yuv420p'), 'h265': ('libx265', 'yuv420p'),
                    'vp9': ('libvpx-vp9', 'yuv420p'), 'av1': ('libaom-av1', 'yuv420p'),
                    'prores': ('prores_ks', 'yuv422p10le')}[settings['codec']]
    with tempfile.TemporaryDirectory(prefix='.steps-encode-', dir=target.parent) as temporary:
        for index, source in enumerate(frames):
            ctx.check_interrupt()
            if image_password:
                import io
                from .encrypt_image import decode_png
                decoded, _ = decode_png(Path(source).read_bytes(), image_password)
                source = io.BytesIO(decoded)
            with Image.open(source) as image:
                ImageOps.pad(image.convert('RGB'), (width, height), color='black').save(Path(temporary) / f'{index:06d}.png')
        args = [executable, '-nostdin', '-n', '-v', 'error', '-framerate', f'{rate:g}', '-i',
                str(Path(temporary) / '%06d.png'), '-vf', ','.join(filters), '-t', f'{duration:g}',
                '-c:v', codec, '-pix_fmt', pixel, '-metadata', 'title=' + str(metadata.get('prompt', '')),
                '-metadata', 'description=' + json.dumps(metadata, ensure_ascii=False),
                '-metadata', 'artist=Stable Diffusion NUI', str(target)]
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=errors,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            try:
                while True:
                    ctx.check_interrupt()
                    try:
                        result = process.wait(timeout=.2)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                if result:
                    errors.seek(0)
                    raise EngineError(UNSUPPORTED, 'steps animation: video export failed',
                                      errors.read(8192).decode('utf-8', errors='replace'))
            except BaseException:
                if process.poll() is None:
                    process.kill()
                process.wait()
                target.unlink(missing_ok=True)
                raise
    return str(target)


from .save_intermediate import Recorder as IntermediateRecorder


def scoped_directory(root, relative):
    from pathlib import Path
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise EngineError(INVALID_PARAMS, 'steps animation: output directory escapes generation folder')
    target.mkdir(parents=True, exist_ok=True)
    return target


class Recorder(IntermediateRecorder):
    def __init__(self, output_dir, raw, ctx, metadata, image_password=None, allow_plaintext_video=False):
        import shutil, tempfile
        from pathlib import Path
        from . import save_intermediate
        from .protocol import UNSUPPORTED
        self.settings = normalize(raw)
        if image_password and self.settings['create_output'] and not allow_plaintext_video:
            raise EngineError(UNSUPPORTED, 'steps animation: plaintext video export requires explicit permission')
        self.image_password = image_password
        if self.settings['create_output'] and not shutil.which('ffmpeg'):
            raise EngineError(UNSUPPORTED, 'steps animation: ffmpeg is required')
        self.c = save_intermediate.normalize(dict(every_n=1, start_at=self.settings['skip_initial'] + 1,
                                                  image_type='Denoised', mode='Standard'))
        self.ctx, self.metadata = ctx, metadata
        self.frames, self.artifacts, self.finished = [], [], False
        self.directory = Path(tempfile.mkdtemp(prefix='generation-', dir=scoped_directory(output_dir, self.settings['frames_dir'])))
        self.video_directory = scoped_directory(output_dir, self.settings['output_dir'])

    def _save(self, image, seed, pass_name, step, absolute_step, iteration, batch_index):
        super()._save(image, seed, pass_name, step, absolute_step, iteration, batch_index)
        if len(self.frames) == 1:
            self.metadata["seed"] = seed
        if self.settings['debug'] and hasattr(self.ctx, 'log'):
            self.ctx.log('info', f'Steps Animation captured {pass_name} step {absolute_step}')

    def close(self):
        import json, time
        if self.finished:
            return
        context = self.ctx
        event = getattr(context, 'interrupt_event', None)
        was_interrupted = event is not None and event.is_set()
        errors = []
        if self.frames and self.settings['run_incomplete'] and not getattr(getattr(context, 'emitter', None), 'broken', False):
            # The original exception remains in Engine._generate's finally. Consume its first
            # cancellation only during this explicitly requested export; a second interrupt cancels it.
            deadline = time.monotonic() + 60
            class PartialExportContext:
                def check_interrupt(self):
                    context.check_interrupt()
                    if time.monotonic() >= deadline:
                        from .protocol import INTERRUPTED
                        raise EngineError(INTERRUPTED, 'partial animation export timed out after 60 seconds')
                def warn(self, message):
                    context.warn(message)
            try:
                if was_interrupted:
                    event.clear()
                self.ctx = PartialExportContext()
                self.finish()
            except (EngineError, OSError) as error:
                errors.append(str(error))
            finally:
                self.ctx = context
                if was_interrupted:
                    event.set()
        manifest = self.directory / 'manifest.json'
        manifest.write_text(json.dumps(dict(status='incomplete_with_export_errors' if errors else 'incomplete',
            settings=self.settings, metadata=self.metadata, frames=self.frames, artifacts=self.artifacts,
            export_errors=errors), ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
        self.finished = True

    def finish(self):
        import json
        from pathlib import Path
        if self.finished:
            return self.artifacts
        errors = []
        if self.frames and self.settings['create_output']:
            extension = {'vp9': 'webm', 'prores': 'mov'}.get(self.settings['codec'], 'mp4')
            target = self.video_directory / f'{self.directory.name}.{extension}'
            try:
                args = ([frame['path'] for frame in self.frames], target, self.settings, self.ctx, self.metadata)
                if self.image_password:
                    encode_frames(*args, image_password=self.image_password)
                else:
                    encode_frames(*args)
                self.artifacts.append(dict(type='video', path=str(target), frames=len(self.frames), format=extension))
                if self.settings['delete_intermediate']:
                    for frame in self.frames:
                        Path(frame['path']).unlink(missing_ok=True)
                    self.artifacts = [item for item in self.artifacts if item['type'] != 'image']
            except (EngineError, OSError) as error:
                # An interruption must continue to propagate through normal job cancellation.
                from .protocol import UNSUPPORTED
                if isinstance(error, EngineError) and error.code != UNSUPPORTED:
                    raise
                message = f'steps animation: export failed; frames retained: {error}'
                errors.append(message)
                self.ctx.warn(message)
        manifest = self.directory / 'manifest.json'
        manifest.write_text(json.dumps(dict(status='complete_with_export_errors' if errors else 'complete',
            settings=self.settings, metadata=self.metadata, frames=self.frames, artifacts=self.artifacts,
            export_errors=errors), ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
        self.artifacts.append(dict(type='manifest', path=str(manifest)))
        self.finished = True
        return self.artifacts



def validate_generation(params):
    from .hooks import hook_config
    if params.batch_size != 1 or params.n_iter != 1:
        raise EngineError(INVALID_PARAMS, 'steps animation requires batch size and count equal to one')
    if hook_config(params.hooks, 'save_intermediate', {}) is not None:
        raise EngineError(INVALID_PARAMS, 'steps animation cannot run together with Save intermediate images')
