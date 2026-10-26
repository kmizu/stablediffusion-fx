"""Standalone intermediate sampling recorder (AlUlkesh behaviour; upstream Unlicense).

No scheduler return values or sampling tensors are changed. Intermediate artifacts are
auxiliary results, never additional final generation images.
"""
from __future__ import annotations

import inspect
import functools
import json
import math
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from .hooks import coerce_bool
from .images import next_output_path
from .protocol import EngineError, INVALID_PARAMS, UNSUPPORTED

DEFAULTS = dict(every_n=5, start_at=0, stop_at=0, image_type='Denoised', mode='Standard',
                final_save=False, format='mp4', codec='h264', fps=2, add_first_frames=0,
                add_last_frames=0, smooth=False, seconds=0, lores=True, hires=True,
                ffmpeg_bat=False, bat_only=False)


def normalize(raw):
    c = dict(DEFAULTS, **raw)
    for key in ('every_n', 'start_at', 'stop_at', 'add_first_frames', 'add_last_frames'):
        try:
            value = float(c[key])
            if not math.isfinite(value) or value != int(value) or value < (1 if key == 'every_n' else 0):
                raise ValueError()
            c[key] = int(value)
        except (ValueError, TypeError, OverflowError):
            raise EngineError(INVALID_PARAMS, f'save intermediate: {key} must be a positive integer' if key == 'every_n'
                              else f'save intermediate: {key} must be a nonnegative integer')
    for key in ('fps', 'seconds'):
        try:
            c[key] = float(c[key])
            if not math.isfinite(c[key]) or c[key] < (0.01 if key == 'fps' else 0):
                raise ValueError()
        except (ValueError, TypeError):
            raise EngineError(INVALID_PARAMS, f'save intermediate: invalid {key}')
    for key, choices in dict(image_type=('Denoised', 'Noisy', 'Preview'), mode=('Standard', 'Suffix', 'Video'),
                             format=('gif', 'webm', 'mp4'), codec=('h264', 'h265/hevc', 'av1')).items():
        if c[key] not in choices:
            raise EngineError(INVALID_PARAMS, f'save intermediate: invalid {key}')
    for key in ('final_save', 'smooth', 'lores', 'hires', 'ffmpeg_bat', 'bat_only'):
        c[key] = coerce_bool(c[key], DEFAULTS[key])
    if c['smooth'] and c['fps'] <= 1:
        raise EngineError(INVALID_PARAMS, 'save intermediate: smoothing requires output fps greater than 1')
    for key in ('start_at', 'stop_at'):
        c[key] = c[key] // c['every_n'] * c['every_n']
    if c['mode'] == 'Video' and not (c['lores'] or c['hires']):
        raise EngineError(INVALID_PARAMS, 'save intermediate: select base or hires frames for video')
    return c


def selected(c, step):
    return (step > 0 and step % c['every_n'] == 0 and step >= c['start_at']
            and (not c['stop_at'] or step <= c['stop_at']))


def logical_step(pipe, i, expected):
    """Diffusers progress-bar boundary: do not count Heun's predictor/corrector twice."""
    count = int(getattr(pipe, '_num_timesteps', 0) or expected)
    order = max(1, int(getattr(pipe.scheduler, 'order', 1)))
    warmup = max(count - expected * order, 0)
    if i == count - 1:
        return expected
    if i + 1 > warmup and (i + 1) % order == 0:
        return min(expected, (i + 1 - warmup) // order)
    return None


def timeline(frames, c):
    selected_frames = [f for f in frames if f['pass_name'] == 'final'
                       or (f['pass_name'] == 'hires' and c['hires'])
                       or (f['pass_name'] == 'base' and c['lores'])]
    if not selected_frames:
        return []
    # Upstream explicitly holds the LAST frame at both ends.
    return ([selected_frames[-1]] * c['add_first_frames'] + selected_frames
            + [selected_frames[-1]] * c['add_last_frames'])


def video_timing(c, count):
    """Each selected/held frame occupies one source interval; duration includes all holds.

    Like upstream, smoothing starts at 1 source fps and only stretches time: D=max(N, requested seconds).
    Its configured fps is the higher output rate. Ordinary mode defaults to N/output_fps seconds.
    """
    duration = (max(float(count), c['seconds']) if c['smooth'] else
                c['seconds'] or count / c['fps'])
    return count / duration, duration


def observe_scheduler(scheduler):
    """Capture guided predictions at scheduler.step, after CFG and sampling hooks."""
    original = scheduler.step
    signature = inspect.signature(original)
    state = SimpleNamespace(noisy=None, denoised=None)

    @functools.wraps(original)
    def step(*args, **kwargs):
        values = signature.bind(*args, **kwargs).arguments
        sample, model = values['sample'], values['model_output']
        state.noisy = sample.detach().clone()
        prediction = scheduler.config.get('prediction_type', 'epsilon')
        state.denoised = None
        # Sigma must be read BEFORE step increments the scheduler index.
        sigmas = getattr(scheduler, 'sigmas', None)
        if sigmas is not None:
            idx = getattr(scheduler, 'step_index', None)
            if idx is None:
                t = values['timestep']
                idx = scheduler.index_for_timestep(t)
            sigma = sigmas[idx].to(sample.device, sample.dtype)
            if hasattr(scheduler, 'sigmas_interpol') and not scheduler.state_in_first_order:
                sigma = scheduler.sigmas_interpol[idx].to(sample.device, sample.dtype)
            normalized = getattr(scheduler, '_sigma_to_alpha_sigma_t', None)
            if normalized is not None:
                alpha, sigma_t = normalized(sigma)
                if prediction == 'epsilon':
                    state.denoised = (sample - sigma_t * model) / alpha
                elif prediction == 'v_prediction':
                    state.denoised = alpha * sample - sigma_t * model
                elif prediction == 'sample':
                    state.denoised = model
            elif prediction == 'epsilon':
                state.denoised = sample - sigma * model
            elif prediction == 'v_prediction':
                state.denoised = model * (-sigma / (sigma * sigma + 1).sqrt()) + sample / (sigma * sigma + 1)
            elif prediction in ('sample', 'original_sample'):
                state.denoised = model
        elif hasattr(scheduler, 'alphas_cumprod'):
            alpha = scheduler.alphas_cumprod[int(values['timestep'])].to(sample.device, sample.dtype)
            if prediction == 'epsilon':
                state.denoised = (sample - (1 - alpha).sqrt() * model) / alpha.sqrt()
            elif prediction == 'v_prediction':
                state.denoised = alpha.sqrt() * sample - (1 - alpha).sqrt() * model
            elif prediction == 'sample':
                state.denoised = model
        result = original(*args, **kwargs)
        predicted = getattr(result, 'pred_original_sample', None)
        if predicted is None:
            predicted = getattr(result, 'denoised', None)
        if predicted is None and isinstance(result, tuple) and len(result) > 1:
            predicted = result[1]
        if predicted is not None:
            state.denoised = predicted
        if state.denoised is not None:
            state.denoised = state.denoised.detach().clone()
        return result

    scheduler.step = step

    def remove():
        scheduler.step = original
        state.noisy = state.denoised = None
    state.remove = remove
    return state


class Recorder:
    def __init__(self, output_dir, raw, ctx, metadata):
        self.c = normalize(raw)
        self.ctx, self.metadata = ctx, metadata
        self.frames, self.artifacts = [], []
        self.finished = False
        self.ffmpeg = None
        if self.c['mode'] == 'Video' and not self.c['bat_only']:
            self.ffmpeg = shutil.which('ffmpeg')
            if not self.ffmpeg:
                raise EngineError(UNSUPPORTED, 'save intermediate: video export requires ffmpeg on PATH')
        root = Path(output_dir) / 'intermediates'
        root.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix='generation-', dir=root))

    def _save(self, image, seed, pass_name, step, absolute_step, iteration, batch_index):
        from PIL.PngImagePlugin import PngInfo
        self.ctx.check_interrupt()
        suffix = (f'{seed}-{pass_name}-{absolute_step if absolute_step is not None else "final"}'
                  if self.c['mode'] == 'Suffix' else f'{pass_name}-{absolute_step}-{seed}')
        path = next_output_path(str(self.directory), suffix)
        record = dict(path=path, seed=seed, pass_name=pass_name, step=absolute_step, pass_step=step,
                      iteration=iteration, batch_index=batch_index, width=image.width, height=image.height,
                      image_type='Final' if pass_name == 'final' else self.c['image_type'])
        info = PngInfo()
        info.add_text('sdfx_intermediate', json.dumps(dict(self.metadata, **record), ensure_ascii=False))
        prompt, negative = self.metadata.get('prompt', ''), self.metadata.get('negative_prompt', '')
        fields = dict(Steps=self.metadata.get('steps', 0), Sampler=self.metadata.get('sampler', ''),
                      **{'CFG scale': self.metadata.get('cfg_scale', 0), 'Seed': seed,
                         'Size': f'{image.width}x{image.height}', 'Intermediate pass': pass_name,
                         'Intermediate step': absolute_step, 'Intermediate subject': record['image_type']})
        # The native PNG importer understands A1111's parameters key; the complete request stays in JSON.
        parameters = prompt + ('\nNegative prompt: ' + negative if negative else '') + '\n'
        text_fields = []
        for k, v in fields.items():
            if v is None:
                continue
            if isinstance(v, str) and any(ch in v for ch in (',', '\n', '"')):
                v = json.dumps(v, ensure_ascii=False)
            text_fields.append(f'{k}: {v}')
        parameters += ', '.join(text_fields)
        info.add_text('parameters', parameters)
        try:
            image.save(path, format='PNG', pnginfo=info, compress_level=4)
        except BaseException:
            Path(path).unlink(missing_ok=True)
            raise
        self.frames.append(record)
        self.artifacts.append(dict(type='image', **record))

    def capture(self, images, seeds, pass_name, step, absolute_step, iteration):
        if not selected(self.c, absolute_step):
            return
        for j, (image, seed) in enumerate(zip(images, seeds)):
            self._save(image, seed, pass_name, step, absolute_step, iteration, j)

    def final(self, image, seed, iteration, batch_index=0):
        if self.c['final_save']:
            self._save(image, seed, 'final', None, None, iteration, batch_index)

    def _filters(self, frames):
        w, h = max(f['width'] for f in frames), max(f['height'] for f in frames)
        w, h = (w + 1) // 2 * 2, (h + 1) // 2 * 2
        rate, duration = video_timing(self.c, len(frames))
        filters = [f'scale={w}:{h}:force_original_aspect_ratio=decrease:eval=frame',
                   f'pad={w}:{h}:(ow-iw)/2:(oh-ih)/2',
                   'settb=AVTB', f'setpts=N/{rate:g}/TB', f'fps={rate:g}']
        if self.c['smooth'] and len(frames) >= 3:
            # minterpolate needs future frames. Clone two final source intervals, then trim the
            # lookahead away so the last original/held image survives through the requested endpoint.
            filters += [f'tpad=stop_mode=clone:stop_duration={2 / rate:g}',
                        f'minterpolate=fps={self.c["fps"]:g}']
        filters += [f'trim=duration={duration:g}', 'setpts=PTS-STARTPTS']
        graph = ','.join(filters)
        if self.c['format'] == 'gif':
            # Match upstream's adaptive palette instead of ffmpeg's coarse default RGB palette.
            graph += ',split[colors][frames];[colors]palettegen=stats_mode=full[palette];[frames][palette]paletteuse=dither=sierra2_4a'
        return graph

    def _codec(self):
        if self.c['format'] == 'gif':
            return []
        codec = ('libvpx-vp9' if self.c['format'] == 'webm' else
                 {'h264': 'libx264', 'h265/hevc': 'libx265', 'av1': 'libaom-av1'}[self.c['codec']])
        return ['-c:v', codec, '-pix_fmt', 'yuv420p']

    def _recipe(self, frames, name):
        """Only generated relative file names enter the batch recipe; no user prompt/path command text."""
        listing = self.directory / f'{name}.ffconcat'
        rate, duration = video_timing(self.c, len(frames))
        lines = ['ffconcat version 1.0']
        for f in frames:
            lines += [f"file '{Path(f['path']).name}'", f'duration {1 / rate:.9f}']
        lines += [f"file '{Path(frames[-1]['path']).name}'"]
        listing.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
        target = f'{name}.{self.c["format"]}'
        args = ['ffmpeg', '-n', '-f', 'concat', '-safe', '1', '-i', listing.name,
                '-vf', self._filters(frames), '-t', f'{duration:g}', *self._codec(), target]
        bat = self.directory / f'{name}.bat'
        # Quote the filter explicitly: cmd.exe treats its parentheses as syntax otherwise.
        command = subprocess.list2cmdline(args).replace(self._filters(frames), f'"{self._filters(frames)}"')
        bat.write_text('@echo off\ncd /d "%~dp0"\n' + command + '\n',
                       encoding='utf-8', newline='\n')
        self.artifacts.extend([dict(type='ffmpeg_recipe', path=str(bat)),
                               dict(type='ffmpeg_playlist', path=str(listing))])

    def _video(self, frames, name):
        from PIL import Image, ImageOps
        target = self.directory / f'{name}.{self.c["format"]}'
        rate, duration = video_timing(self.c, len(frames))
        w, h = max(f['width'] for f in frames), max(f['height'] for f in frames)
        w, h = (w + 1) // 2 * 2, (h + 1) // 2 * 2
        with tempfile.TemporaryDirectory(prefix='.video-', dir=self.directory) as temp:
            for i, f in enumerate(frames):
                self.ctx.check_interrupt()
                if hasattr(self.ctx, 'progress'):
                    self.ctx.progress(i + 1, len(frames), 'intermediate video')
                with Image.open(f['path']) as im:
                    image = ImageOps.pad(im.convert('RGB'), (w, h), color='black')
                    image.save(Path(temp) / f'{i:06d}.png')
            args = [self.ffmpeg, '-nostdin', '-n', '-loglevel', 'error', '-framerate', str(rate),
                    '-i', str(Path(temp) / '%06d.png'), '-vf', self._filters(frames), '-t', f'{duration:g}',
                    *self._codec(), str(target)]
            with tempfile.TemporaryFile() as errors:
                proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=errors,
                                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                try:
                    last_log = 0
                    while True:
                        self.ctx.check_interrupt()
                        if time.monotonic() - last_log >= 5 and hasattr(self.ctx, 'log'):
                            self.ctx.log('info', f'Encoding intermediate video: {target.name}')
                            last_log = time.monotonic()
                        try:
                            code = proc.wait(timeout=.2)
                            break
                        except subprocess.TimeoutExpired:
                            pass
                    if code:
                        errors.seek(0)
                        raise EngineError(UNSUPPORTED, 'save intermediate: ffmpeg export failed',
                                          errors.read(8192).decode('utf-8', errors='replace'))
                except BaseException:
                    proc.kill()
                    proc.wait()
                    target.unlink(missing_ok=True)
                    raise
        self.artifacts.append(dict(type='video', path=str(target), frames=len(frames),
                                   format=self.c['format']))

    def finish(self):
        if not self.frames:
            self.directory.rmdir()
            self.finished = True
            return []
        export_errors = []
        if self.c['mode'] == 'Video':
            groups = sorted({(f['iteration'], f['batch_index'], f['seed']) for f in self.frames})
            for iteration, j, seed in groups:
                frames = [f for f in self.frames if (f['iteration'], f['batch_index'], f['seed']) == (iteration, j, seed)]
                selected_frames = timeline(frames, self.c)
                name = f'iteration-{iteration}-image-{j}-seed-{seed}'
                if selected_frames and not self.c['bat_only']:
                    try:
                        self._video(selected_frames, name)
                    except EngineError as e:
                        if e.code != UNSUPPORTED:
                            raise
                        message = f'{e.message}; PNG frames retained. {e.detail}'
                        export_errors.append(message)
                        self.ctx.warn(message)
                    except OSError as e:
                        message = f'save intermediate: video export failed; PNG frames retained. {e}'
                        export_errors.append(message)
                        self.ctx.warn(message)
                if self.c['ffmpeg_bat'] or self.c['bat_only']:
                    for pass_name in ('base', 'hires'):
                        if not any(f['pass_name'] == pass_name for f in frames):
                            continue
                        part = [f for f in frames if f['pass_name'] in (pass_name, 'final')]
                        part = timeline(part, dict(self.c, lores=True, hires=True))
                        if part:
                            self._recipe(part, f'{name}-{pass_name}')
        manifest = self.directory / 'manifest.json'
        manifest.write_text(json.dumps(dict(status='complete_with_export_errors' if export_errors else 'complete',
                                            settings=self.c, metadata=self.metadata, frames=self.frames,
                                            artifacts=self.artifacts, export_errors=export_errors), indent=2, ensure_ascii=False),
                            encoding='utf-8', newline='\n')
        self.artifacts.append(dict(type='manifest', path=str(manifest)))
        self.finished = True
        return self.artifacts

    def close(self):
        """Preserve captured PNGs with an incomplete manifest when a job is interrupted or fails."""
        if self.finished:
            return
        if not self.frames:
            self.directory.rmdir()
            return
        (self.directory / 'manifest.json').write_text(json.dumps(dict(status='incomplete', settings=self.c,
            metadata=self.metadata, frames=self.frames, artifacts=self.artifacts), indent=2, ensure_ascii=False),
            encoding='utf-8', newline='\n')


def install(engine, pipe, kw, recorder, seeds, pass_name, expected, offset, iteration):
    """Install after sampling extensions, return restoration function for Engine._run's finally."""
    from .extensions import _chain_callback
    observer = observe_scheduler(pipe.scheduler)

    def callback(p, i, t, data):
        step = logical_step(p, i, expected)
        if step is None or not selected(recorder.c, offset + step):
            return data
        recorder.ctx.check_interrupt()
        image_type = recorder.c['image_type']
        latents = observer.noisy if image_type == 'Noisy' else observer.denoised
        if image_type == 'Preview':
            latents = data.get('latents')
        if latents is None:
            raise EngineError(UNSUPPORTED, 'save intermediate: scheduler did not expose a denoised prediction')
        if image_type == 'Preview':
            import base64
            import io
            from PIL import Image
            from .preview import latent_preview_b64
            images = []
            for lat in latents:
                b64 = None
                if recorder.metadata.get('preview_method') == 'taesd':
                    b64 = engine.taesd.decode_b64(lat[None], engine.sdxl, engine.device, engine.dtype)
                if b64 is None:
                    b64 = latent_preview_b64(lat[None, :4].float().cpu().numpy(), engine.sdxl)
                with Image.open(io.BytesIO(base64.b64decode(b64))) as im:
                    images.append(im.convert('RGB'))
        else:
            images = engine._decode(latents)
        recorder.capture(images, seeds, pass_name, step, offset + step, iteration)
        return data

    _chain_callback(kw, callback)
    return observer.remove
