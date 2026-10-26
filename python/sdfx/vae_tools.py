"""VAE Tools: independently implemented file contract and scoped model operations.

Latent PNGs are data, not ordinary pictures: never orient, resize or convert their mode.
Source-compatible tensors are diffusion-scaled BCHW. Exact recovery safetensors explicitly
tag unchanged VAE decoder-input coordinates, which need no lossy scale/inverse round trip.
No pickle input.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import struct
import time
import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin
from .protocol import EngineError, INVALID_PARAMS, INTERRUPTED, UNSUPPORTED

MAX_ELEMENTS = 16 * 1024 * 1024
MAX_BYTES = MAX_ELEMENTS * 4 + 65536
APPROXIMATIONS = ('Full', 'Approx NN', 'Approx cheap', 'TAESD')
TINY_REVISIONS = {'madebyollin/taesd': '614f76814bbe30edbe2e627ace1c2234c81a2c0e',
                  'madebyollin/taesdxl': 'b20258aaef75ef61e659c1e0f14f251cf0ad153e'}


def checked(array, channels=None):
    a = np.asarray(array)
    if a.ndim != 4 or a.shape[0] != 1 or a.shape[1] not in (4, 16) or min(a.shape[2:]) < 1:
        raise ValueError('latent must be one BCHW image with 4 or 16 channels')
    if channels is not None and a.shape[1] != channels:
        raise ValueError(f'latent has {a.shape[1]} channels; selected VAE requires {channels}')
    if a.size > MAX_ELEMENTS or a.dtype.kind != 'f' or not np.isfinite(a).all():
        raise ValueError('latent must be finite floating point data within the 16M element limit')
    return np.ascontiguousarray(a)


def pack(array, representation):
    a = checked(array)
    stats = dict(shape=list(a.shape), dtype=str(a.dtype), minimum=float(a.min()), maximum=float(a.max()),
                 quantized=representation != 'L', clipped_values=int(np.count_nonzero(np.abs(a) > 10)) if representation != 'L' else 0)
    if representation == 'L':
        from safetensors.numpy import save
        payload = save({'tensor': a})
        data = struct.pack('>I', len(payload)) + payload
        side = math.ceil(math.sqrt(len(data)))
        return Image.frombytes('L', (side, side), data.ljust(side * side, b'\0')), stats
    if representation not in ('RGBA', 'CMYK'):
        raise ValueError('representation must be L, RGBA or CMYK')
    pixels = np.clip(np.rint((a[0].transpose(1, 2, 0) + 10) * (255 / 20)), 0, 255).astype('uint8')
    height, width = a.shape[2:]
    grid = Image.new(representation, (width, height) if a.shape[1] == 4 else (2 * width, 2 * height))
    for tile in range(a.shape[1] // 4):
        grid.paste(Image.frombytes(representation, (width, height), pixels[:, :, tile * 4:tile * 4 + 4].tobytes()),
                   ((tile % 2) * width, (tile // 2) * height))
    return grid, stats


def _load_tensor(payload):
    from safetensors.numpy import load
    try:
        data = load(payload)
    except (TypeError, KeyError):
        # numpy has no native bfloat16 dtype. Safe torch loading widens those values
        # exactly to fp32, matching the source decoder's selected-VAE dtype conversion.
        from safetensors.torch import load as torch_load
        data = {key: value.float().numpy() for key, value in torch_load(payload).items()}
    if set(data) != {'tensor'}:
        raise ValueError('expected a single tensor key')
    return data['tensor']


def unpack(image, representation, channels):
    if image.mode != representation:
        raise ValueError(f'encoded input must retain {representation} mode; got {image.mode}. Do not convert its channels.')
    if channels not in (4, 16):
        raise ValueError('only 4/16 channel latent formats are supported')
    if image.width * image.height * len(image.getbands()) > MAX_BYTES:
        raise ValueError('encoded image exceeds latent size limit')
    if representation == 'L':
        data = image.tobytes()
        if len(data) < 4:
            raise ValueError('invalid latent length header')
        length = struct.unpack('>I', data[:4])[0]
        if length < 8 or length > MAX_BYTES or length > len(data) - 4:
            raise ValueError('invalid latent payload length')
        try:
            return checked(_load_tensor(data[4:4 + length]), channels)
        except Exception as exc:
            raise ValueError(f'invalid safetensors latent: {exc}') from exc
    if representation not in ('RGBA', 'CMYK'):
        raise ValueError('unknown latent representation')
    pixels = np.asarray(image, dtype=np.float32) / 255 * 20 - 10
    if channels == 16:
        if image.width % 2 or image.height % 2:
            raise ValueError('16-channel latent grid must have even dimensions')
        h, w = image.height // 2, image.width // 2
        pixels = np.concatenate([pixels[y:y + h, x:x + w] for y, x in ((0, 0), (0, w), (h, 0), (h, w))], axis=2)
    return checked(pixels.transpose(2, 0, 1)[None], channels)


def read_latent(path, representation, channels):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('latent file exceeds size limit')
    if path.suffix.lower() == '.safetensors':
        try:
            return checked(_load_tensor(path.read_bytes()), channels)
        except Exception as exc:
            raise ValueError(f'invalid safetensors latent: {exc}') from exc
    with Image.open(path) as image:
        return unpack(image, representation, channels)


def latent_metadata(path):
    if Path(path).suffix.lower() != '.safetensors':
        return {}
    from safetensors import safe_open
    with safe_open(str(path), framework='np') as handle:
        raw = (handle.metadata() or {}).get('sdfx_vae_tools')
    data = json.loads(raw) if raw else {}
    if data.get('coordinate', 'diffusion_scaled') not in ('diffusion_scaled', 'vae_decoder_input'):
        raise ValueError('unknown latent coordinate convention')
    return data


def provenance(engine, coordinate='diffusion_scaled', **details):
    vae = engine.pipe.vae
    config = vae.config
    return dict(model=engine.model, vae=getattr(engine, 'vae_info', None),
                family=getattr(engine, 'kind', None) or engine.model.get('kind'),
                scaling_factor=float(config.scaling_factor), shift_factor=getattr(config, 'shift_factor', None),
                latents_mean=getattr(config, 'latents_mean', None), latents_std=getattr(config, 'latents_std', None),
                coordinate=coordinate, **details)


def png_info(metadata):
    info = PngImagePlugin.PngInfo()
    info.add_text('VAE Tools', json.dumps(metadata, default=str))
    return info


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def vae_scope(vae, tiled=False):
    original_dtype = vae.dtype
    original_tiling = getattr(vae, 'use_tiling', False)
    try:
        if original_dtype.__str__() == 'torch.float16' and getattr(vae.config, 'force_upcast', False):
            import torch
            vae.to(dtype=torch.float32)
        if hasattr(vae, 'enable_tiling'):
            vae.enable_tiling() if tiled else vae.disable_tiling()
        yield
    finally:
        if hasattr(vae, 'enable_tiling'):
            vae.enable_tiling() if original_tiling else vae.disable_tiling()
        if vae.dtype != original_dtype:
            vae.to(dtype=original_dtype)


def _retry_finite(vae, function):
    import torch
    value = function()
    if torch.isfinite(value).all():
        return value
    if vae.dtype != torch.float32:
        vae.to(dtype=torch.float32)
        value = function()
        if torch.isfinite(value).all():
            return value
    raise ValueError('VAE produced nonfinite values, including after the fp32 retry')


def _tiny(engine, allow_download, cache_dir=None):
    from diffusers import AutoencoderTiny
    repo = 'madebyollin/taesdxl' if engine.sdxl else 'madebyollin/taesd'
    cache = getattr(engine, '_vae_tools_tiny', {})
    if repo not in cache:
        try:
            cache[repo] = AutoencoderTiny.from_pretrained(repo, local_files_only=not allow_download,
                revision=TINY_REVISIONS[repo], use_safetensors=True, cache_dir=cache_dir).to(engine.device)
        except Exception as exc:
            raise EngineError(UNSUPPORTED, f'TAESD requires {repo} safe weights. Allow optional downloads or cache the model first: {exc}') from exc
        engine._vae_tools_tiny = cache
    engine._vae_tools_weights = dict(repo=repo, revision=TINY_REVISIONS[repo], license='MIT')
    return cache[repo]


def _approx_nn(engine, array, options):
    import torch
    from torch import nn
    path = str(options.get('approx_nn_path') or '')
    if not path:
        name = 'vaeapprox-sdxl.pt' if engine.sdxl else 'model.pt'
        path = str(Path(options['cache_dir']) / 'vae-approx' / name)
        if not Path(path).is_file() and options['allow_download']:
            from urllib.request import urlopen
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            url = 'https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/download/v1.0.0-pre/' + name
            with urlopen(url, timeout=30) as response:
                data = response.read(16 * 1024 * 1024 + 1)
            if len(data) > 16 * 1024 * 1024:
                raise ValueError('Approx NN weight download exceeds size limit')
            target.write_bytes(data)
    if not Path(path).is_file():
        raise EngineError(UNSUPPORTED, 'Approx NN requires official A1111 model.pt / vaeapprox-sdxl.pt; select a local weights file or allow optional downloads')
    # Tensor-only loader: no arbitrary object/global unpickling. Eight public layer shapes
    # reconstruct the checkpoint interface; no Web UI module is imported.
    if Path(path).stat().st_size > 16 * 1024 * 1024:
        raise ValueError('Approx NN weights exceed size limit')
    weights = torch.load(path, map_location='cpu', weights_only=True)
    engine._vae_tools_weights = dict(path=path, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                                    format='tensor-only torch checkpoint; weights_only=True')
    if not isinstance(weights, dict):
        raise ValueError('Approx NN weights must be a tensor dictionary')
    x = torch.from_numpy(array).float()
    x = nn.functional.pad(nn.functional.interpolate(x, scale_factor=2, mode='nearest'), (11, 11, 11, 11))
    for index, (cin, cout, kernel) in enumerate(zip((4, 8, 16, 32, 64, 32, 16, 8), (8, 16, 32, 64, 32, 16, 8, 3), (7, 5, 3, 3, 3, 3, 3, 3)), 1):
        weight, bias = weights[f'conv{index}.weight'], weights[f'conv{index}.bias']
        if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor) or tuple(weight.shape) != (cout, cin, kernel, kernel) or tuple(bias.shape) != (cout,):
            raise ValueError('Approx NN weight shape mismatch')
        x = nn.functional.leaky_relu(nn.functional.conv2d(x, weight.float(), bias.float()), .1)
    return x


def _scale(vae, tensor, inverse=False):
    import torch
    scale = float(vae.config.scaling_factor)
    mean, std = getattr(vae.config, 'latents_mean', None), getattr(vae.config, 'latents_std', None)
    if mean is not None and std is not None:
        mean = torch.tensor(mean, device=tensor.device, dtype=tensor.dtype)[None, :, None, None]
        std = torch.tensor(std, device=tensor.device, dtype=tensor.dtype)[None, :, None, None]
        return tensor / scale * std + mean if inverse else (tensor - mean) / std * scale
    shift = float(getattr(vae.config, 'shift_factor', None) or 0)
    return tensor / scale + shift if inverse else (tensor - shift) * scale


def encode(engine, image, approximation, options, seed):
    import torch
    if approximation == 'TAESD':
        vae = _tiny(engine, options['allow_download'], options.get('cache_dir'))
        pixels = torch.from_numpy(np.array(image.convert('RGB'), dtype=np.float32).transpose(2, 0, 1)[None] / 127.5 - 1).to(engine.device, vae.dtype)
        with vae_scope(vae, options['tiled']):
            return vae.encode(pixels).latents.float().cpu().numpy()
    vae = engine.pipe.vae
    with vae_scope(vae, options['tiled']):
        pixels = torch.from_numpy(np.array(image.convert('RGB'), dtype=np.float32).transpose(2, 0, 1)[None] / 127.5 - 1)
        def run():
            generator = torch.Generator(device=engine.device).manual_seed(seed)
            return _scale(vae, vae.encode(pixels.to(engine.device, vae.dtype)).latent_dist.sample(generator=generator).float())
        return _retry_finite(vae, run).float().cpu().numpy()


def decode(engine, array, approximation, options):
    import torch
    metadata = options.get('latent_metadata', {})
    raw_decoder_input = metadata.get('coordinate') == 'vae_decoder_input'
    if raw_decoder_input and approximation != 'Full':
        from types import SimpleNamespace
        if not isinstance(metadata.get('scaling_factor'), (int, float)) or not 0 < metadata['scaling_factor'] < float('inf'):
            raise ValueError('decoder-input latent metadata needs a finite positive scaling factor')
        config = SimpleNamespace(**metadata)
        array = _scale(SimpleNamespace(config=config), torch.from_numpy(array).float()).numpy()
    if approximation == 'Approx cheap':
        from .preview import latent_to_rgb
        return Image.fromarray(latent_to_rgb(array, engine.sdxl))
    if approximation == 'Approx NN':
        value = _approx_nn(engine, array, options)
    elif approximation == 'TAESD':
        vae = _tiny(engine, options['allow_download'], options.get('cache_dir'))
        with vae_scope(vae, options['tiled']):
            value = vae.decode(torch.from_numpy(array).to(engine.device, vae.dtype)).sample
    else:
        vae = engine.pipe.vae
        with vae_scope(vae, options['tiled']):
            def run():
                tensor = torch.from_numpy(array).to(engine.device)
                # Inverse scaling must precede the low-precision cast. Exact decoder-input
                # recovery bypasses scaling altogether and retains the saved input bits.
                z = tensor if raw_decoder_input else _scale(vae, tensor.float(), True)
                return vae.decode(z.to(vae.dtype), return_dict=False)[0]
            value = _retry_finite(vae, run)
    if not torch.isfinite(value).all():
        raise ValueError('decoder produced nonfinite values')
    pixels = ((value[0].float().cpu().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).numpy().astype('uint8')
    return Image.fromarray(pixels)


def run(engine, params, ctx):
    engine._vae_tools_weights = None
    operation = params.get('operation', 'encode')
    representation = params.get('representation', 'L')
    approximation = params.get('approximation', 'Full')
    if operation not in ('encode', 'decode', 'encode+decode') or representation not in ('L', 'RGBA', 'CMYK') or approximation not in APPROXIMATIONS:
        raise ValueError('unknown VAE operation, representation or approximation')
    paths = params.get('input_paths')
    if not isinstance(paths, list) or not 1 <= len(paths) <= 64 or any(not isinstance(p, str) for p in paths):
        raise ValueError('input_paths must contain 1..64 file paths')
    seed = params.get('seed', 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError('seed must be an integer in 0..4294967295')
    for key in ('tiled', 'allow_download'):
        if key in params and not isinstance(params[key], bool):
            raise ValueError(f'{key} must be boolean')
    options = dict(tiled=params.get('tiled', False), allow_download=params.get('allow_download', False), approx_nn_path=params.get('approx_nn_path', ''),
                   cache_dir=params.get('cache_dir') or str(Path(params.get('output_dir') or '.') / '.vae-tools-models'))
    cmyk_format = params.get('cmyk_format', 'TIFF')
    if cmyk_format not in ('TIFF', 'JPEG100'):
        raise ValueError('CMYK output must be TIFF or JPEG100')
    output = Path(params.get('output_dir') or '.') / 'vae-tools'
    # A unique session directory prevents output collisions and any overwrite of input data.
    import tempfile
    output.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='run-', dir=output))
    warnings = []
    encode_approx = approximation
    if operation != 'decode' and approximation not in ('Full', 'TAESD'):
        warnings.append(f'{approximation} cannot encode; Full is used for encoding')
        encode_approx = 'Full'
    channels = int(getattr(engine.pipe.vae.config, 'latent_channels', 4))
    results = []
    import torch
    for index, path in enumerate(paths):
        ctx.check_interrupt()
        started = time.monotonic()
        item = dict(input_path=path, status='complete')
        try:
            if operation == 'decode' and Path(path).stat().st_size > MAX_BYTES:
                raise ValueError('latent file exceeds size limit')
            item['input_sha256'] = hash_file(path)
            item_options = dict(options, latent_metadata={})
            with torch.inference_mode():
                if operation == 'decode':
                    array = read_latent(path, representation, channels)
                    item_options['latent_metadata'] = latent_metadata(path)
                    item['input_coordinate'] = item_options['latent_metadata'].get('coordinate', 'diffusion_scaled')
                    stats = dict(shape=list(array.shape), quantized=representation != 'L')
                else:
                    with Image.open(path) as image:
                        # Preserve pixels as supplied. Odd sizes can round down in the VAE; do not
                        # resize source bytes or silently force the generation form's multiple-of-8 rule.
                        if min(image.size) < 8 or image.width * image.height > MAX_ELEMENTS:
                            raise ValueError('encode input must be at least 8 pixels per side and at most 16M pixels')
                        item['input_dimensions'] = list(image.size)
                        array = checked(encode(engine, image, encode_approx, options, seed + index), channels)
                    encoded, stats = pack(array, representation)
                    if operation == 'encode+decode':
                        array = unpack(encoded, representation, channels)
                if operation == 'encode':
                    suffix = '.tiff' if representation == 'CMYK' and cmyk_format == 'TIFF' else '.jpg' if representation == 'CMYK' else '.png'
                    dst = output / f'{index + 1:03d}-encoded{suffix}'
                    if suffix == '.png':
                        encoded.save(dst, pnginfo=png_info(provenance(engine, operation=operation, representation=representation,
                            approximation=approximation, seed=seed + index, statistics=stats)))
                    else:
                        encoded.save(dst, quality=100, compression='raw') if suffix == '.tiff' else encoded.save(dst, quality=100, subsampling=0)
                    from safetensors.numpy import save_file
                    raw = output / f'{index + 1:03d}-latent.safetensors'
                    save_file({'tensor': array}, str(raw), metadata={'sdfx_vae_tools': json.dumps(provenance(engine), default=str)})
                    item['latent_path'] = str(raw)
                    # A separate RGB preview is for display only; the original encoded file is untouched.
                    display = encoded.convert('RGB')
                else:
                    display = decode(engine, array, approximation, item_options)
                    dst = output / f'{index + 1:03d}-decoded.png'
                    display.save(dst, pnginfo=png_info(provenance(engine, operation=operation, representation=representation,
                        approximation=approximation, seed=seed + index, statistics=stats)))
                preview = output / f'{index + 1:03d}-preview.png'
                display.save(preview, pnginfo=png_info(provenance(engine, operation=operation, representation=representation,
                    approximation=approximation, seed=seed + index, statistics=stats, display_only=True)))
                item.update(path=str(dst), preview_path=str(preview), statistics=stats, width=display.width, height=display.height)
        except EngineError as exc:
            if exc.code == INTERRUPTED:
                raise
            item.update(status='error', code=exc.code, error=exc.message)
        except Exception as exc:
            item.update(status='error', code=INVALID_PARAMS if isinstance(exc, (ValueError, OSError)) else 'internal', error=str(exc))
        item['elapsed_seconds'] = time.monotonic() - started
        results.append(item)
        ctx.progress(index + 1, len(paths), 'vae_tools')
    report = dict(operation=operation, representation=representation, approximation=approximation,
                  encoding_approximation=encode_approx, tiled=options['tiled'], seed=seed,
                  model=engine.model, vae=getattr(engine, 'vae_info', None), channels=channels,
                  scaling_factor=float(engine.pipe.vae.config.scaling_factor), weights=engine._vae_tools_weights, warnings=warnings,
                  note='L/safetensors preserve tensor values. RGBA/CMYK quantize [-10,10] to 8-bit and clip outside that range; JPEG adds loss. Encoding/decoding itself is lossy. CMYK preview is RGB for display only.', results=results)
    report_path = output / 'report.json'
    report['report_path'] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    return report


@contextmanager
def catch_decode(engine, vae, output_dir, ctx):
    """Scoped final decode recovery; explicit VAE tool operations never enter this scope."""
    original = vae.decode
    had_own = 'decode' in vae.__dict__
    own = vae.__dict__.get('decode')
    def guarded(z, *args, **kwargs):
        import torch
        try:
            result = original(z, *args, **kwargs)
            sample = result[0] if isinstance(result, tuple) else result.sample
            if not torch.isfinite(sample).all():
                raise ValueError('VAE decoding returned nonfinite values')
            return result
        except Exception as exc:
            if isinstance(exc, EngineError) and exc.code == INTERRUPTED:
                raise
            ctx.check_interrupt()
            from .preview import latent_to_rgb
            from .images import next_output_path
            array = _scale(vae, z.detach().float()).cpu().numpy()
            folder = Path(output_dir) / 'vae-recovery'
            folder.mkdir(parents=True, exist_ok=True)
            previews = []
            for part in range(z.shape[0]):
                latent = checked(array[part:part + 1])
                rgba, stats = pack(latent, 'RGBA')
                path = Path(next_output_path(str(folder), 'caught-latent'))
                rgba.save(path, pnginfo=png_info(provenance(engine, representation='RGBA', statistics=stats, error=str(exc))))
                raw = path.with_suffix('.safetensors')
                # The pre-division denoising tensor is no longer available here. Retain the
                # actual failed decoder input, rather than claim inverse half arithmetic is exact.
                decoder_input = z[part:part + 1].detach().cpu()
                from safetensors.torch import save_file as save_torch_file
                exact = provenance(engine, coordinate='vae_decoder_input', dtype=str(decoder_input.dtype))
                save_torch_file({'tensor': decoder_input.contiguous()}, str(raw), metadata={'sdfx_vae_tools': json.dumps(exact, default=str)})
                artifacts = getattr(engine, '_vae_recovery', [])
                artifacts.append(dict(path=str(path), latent_path=str(raw), error=str(exc), statistics=stats,
                                      exact_coordinate='vae_decoder_input', png_coordinate='diffusion_scaled'))
                engine._vae_recovery = artifacts
                image = Image.fromarray(latent_to_rgb(latent, engine.sdxl)).resize((z.shape[3] * 8, z.shape[2] * 8), Image.Resampling.NEAREST)
                ImageDraw.Draw(image).text((8, 8), 'Broken - VAE decode failed', fill='red', stroke_width=1)
                previews.append(torch.from_numpy(np.array(image).transpose(2, 0, 1)).float() / 127.5 - 1)
            ctx.warn(f'VAE decode failed ({type(exc).__name__}: {exc}); Broken preview is approximate. Recover exact VAE decoder input from {raw}')
            sample = torch.stack(previews).to(z.device, z.dtype)
            if kwargs.get('return_dict', args[0] if args else True):
                from diffusers.models.autoencoders.vae import DecoderOutput
                return DecoderOutput(sample=sample)
            return (sample,)
    vae.decode = guarded
    try:
        yield
    finally:
        if had_own:
            vae.decode = own
        else:
            del vae.decode
