"""Pinned Waifu2x model assets and native inference, without A1111 imports.

UpConv7 weights originate in MIT nagadomi/waifu2x. CARN V2 weights originate
in GPL-3.0 yu45020/Waifu2x; the mirror is byte-identical to its original zip.
The unlicensed extension wrapper is not copied or imported.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

from .protocol import EngineError, INVALID_PARAMS, MODEL_LOAD_FAILED, UNSUPPORTED


NAMES = tuple(f'Waifu2x {style} denoise {noise} ({label})'
              for style in ('anime', 'photo') for noise, label in enumerate(('none', 'low', 'medium', 'high'))) + ('Waifu2x+ model CarnV2',)


def assets():
    return json.loads(Path(__file__).with_name('waifu2x_assets.json').read_text(encoding='utf-8'))


def parse_name(name):
    try:
        index = NAMES.index(name)
    except ValueError:
        raise EngineError(INVALID_PARAMS, f'Unknown Waifu2x upscaler: {name}')
    return ('anime' if index < 4 else 'photo') + f'-{index % 4}' if index < 8 else 'carn-v2'


def options(style='Anime', noise=2, scale=2, tile=64):
    if style not in ('Anime', 'Photo', 'CarnV2'):
        raise EngineError(INVALID_PARAMS, 'Waifu2x style must be Anime, Photo or CarnV2')
    if type(noise) is not int or not 0 <= noise <= 3:
        raise EngineError(INVALID_PARAMS, 'Waifu2x noise must be 0..3')
    if type(scale) not in (int, float) or not math.isfinite(scale) or not 0 < scale <= 16:
        raise EngineError(INVALID_PARAMS, 'Waifu2x scale must be finite and in (0, 16]')
    if type(tile) is not int or not 16 <= tile <= 1024:
        raise EngineError(INVALID_PARAMS, 'Waifu2x tile must be 16..1024')
    return dict(model_id='carn-v2' if style == 'CarnV2' else f'{style.lower()}-{noise}', scale=scale, tile=tile)


def validate_dimensions(width, height):
    if type(width) is not int or type(height) is not int or width < 1 or height < 1 or width * height > 64 * 1024**2:
        raise EngineError(INVALID_PARAMS, 'Waifu2x dimensions must be positive integers within 64 megapixels')


def tile_plan(width, height, tile=64):
    """Core rectangles with legacy replicated border/remainder geometry, streamed."""
    validate_dimensions(width, height)
    options(tile=tile)
    step = tile + (6 if (width + 6) % tile < 3 or (height + 6) % tile < 3 else 0)
    for y in range(0, height, step):
        for x in range(0, width, step):
            yield x, y, min(step, width-x), min(step, height-y)


def _tile_count(width, height, tile):
    step = tile + (6 if (width + 6) % tile < 3 or (height + 6) % tile < 3 else 0)
    return math.ceil(width / step) * math.ceil(height / step)


def _verified(path, asset, check_interrupt):
    if path.stat().st_size != asset['bytes']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(65536):
            check_interrupt()
            digest.update(data)
    return digest.hexdigest() == asset['sha256']


def materialize(asset, cache_dir, allow_download=True, check_interrupt=lambda: None, opener=None):
    """Verify existing bytes; atomically publish only a complete verified download."""
    check_interrupt()
    directory = Path(cache_dir)
    filename = asset['filename']
    if Path(filename).name != filename or filename in ('', '.', '..'):
        raise EngineError(INVALID_PARAMS, 'Invalid Waifu2x cache filename')
    path = directory / filename
    if path.exists():
        if not _verified(path, asset, check_interrupt):
            raise EngineError(MODEL_LOAD_FAILED, f'Waifu2x cached model failed integrity verification: {path}. Remove this file to download it again.')
        return path
    if not allow_download:
        raise EngineError(UNSUPPORTED, f'Waifu2x model is not cached: {path}; enable model downloads or provide the pinned model file')
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.waifu2x-', suffix='.part', dir=directory)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            check_interrupt()
            with (opener or urlopen)(asset['url'], timeout=30) as response:
                count = 0
                while chunk := response.read(65536):
                    check_interrupt()
                    count += len(chunk)
                    if count > asset['bytes']:
                        raise EngineError(MODEL_LOAD_FAILED, 'Waifu2x model download exceeded the pinned size')
                    stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if not _verified(temporary, asset, check_interrupt):
            raise EngineError(MODEL_LOAD_FAILED, 'Waifu2x model download failed pinned size/SHA256 verification')
        check_interrupt()
        os.replace(temporary, path)
        return path
    except EngineError:
        raise
    except Exception as error:
        raise EngineError(MODEL_LOAD_FAILED, f'Waifu2x model download failed: {error}') from error
    finally:
        temporary.unlink(missing_ok=True)


def _carn_shapes():
    shapes = {}
    def affine(prefix, weight, bias=None):
        shapes[prefix + '.weight'] = weight
        if bias is not None:
            shapes[prefix + '.bias'] = (bias,)
    affine('1.entry_block.m.0', (64, 3, 3, 3), 64)
    for outer in range(3):
        prefix = f'1.blocks.{outer}'
        for inner in range(6):
            block = f'{prefix}.blocks.{inner}'
            if inner % 2 == 0:
                for layer in (0, 2):
                    affine(f'{block}.m.{layer}', (64, 64, 3, 3), 64)
            else:
                affine(block + '.channel_excite.0', (4, 64), 4)
                affine(block + '.channel_excite.2', (64, 4), 64)
                affine(block + '.spatial_excite.0', (1, 64, 1, 1))
        for inner in range(3):
            affine(f'{prefix}.singles.{inner}.m.0', (64, 64 * (inner+2), 3, 3), 64)
    for inner in range(3):
        affine(f'1.singles.{inner}.m.0', (64, 64 * (inner+2), 3, 3), 64)
    affine('1.upsampler.m.0.0', (256, 64, 3, 3), 256)
    affine('1.exit_conv', (3, 64, 3, 3), 3)
    return shapes


def load_model(model_id, path, device='cpu'):
    """Independently expressed inference graph, with strict pinned tensor schemas.

    CARN's legacy forward zips six alternating residual/excitation blocks with
    three compressors. Only the first three blocks execute. Retain that trained
    behavior; changing it to three residual/excitation pairs changes the model.
    """
    import torch
    import torch.nn.functional as functional
    try:
        if model_id not in {asset['id'] for asset in assets()}:
            raise ValueError('unknown model identifier')
        if model_id != 'carn-v2':
            layers = json.loads(Path(path).read_text(encoding='utf-8'))
            if not isinstance(layers, list) or len(layers) != 7:
                raise ValueError('UpConv7 requires seven layers')
            channels = (3, 16, 32, 64, 128, 128, 256)
            parameters = []
            for index, layer in enumerate(layers):
                weight = torch.tensor(layer['weight'], dtype=torch.float32)
                bias = torch.tensor(layer['bias'], dtype=torch.float32)
                expected = (channels[index+1], channels[index], 3, 3) if index < 6 else (256, 3, 4, 4)
                if tuple(weight.shape) != expected or tuple(bias.shape) != (expected[0] if index < 6 else 3,):
                    raise ValueError('UpConv7 tensor shape mismatch')
                if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
                    raise ValueError('nonfinite UpConv7 weights')
                if index == 6 and torch.count_nonzero(bias):
                    raise ValueError('unexpected transposed-convolution bias')
                parameters.append((weight.to(device), bias.to(device)))
            def forward(pixels):
                value = functional.pad(pixels.to(device=device, dtype=torch.float32), (7, 7, 7, 7))
                for weight, bias in parameters[:6]:
                    value = functional.leaky_relu(functional.conv2d(value, weight, bias), .1)
                return functional.conv_transpose2d(value, parameters[6][0], stride=2, padding=3)
            return forward
        tensors = torch.load(path, map_location='cpu', weights_only=True)
        shapes = _carn_shapes()
        if not isinstance(tensors, dict) or set(tensors) != set(shapes):
            raise ValueError('CARN V2 state keys differ from the pinned architecture')
        for name, shape in shapes.items():
            value = tensors[name]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f'invalid CARN V2 tensor: {name}')
        precision = torch.float16 if torch.device(device).type == 'cuda' else torch.float32
        tensors = {name: value.to(device=device, dtype=precision) for name, value in tensors.items()}
        def conv(value, prefix, padding=1):
            return functional.conv2d(value, tensors[prefix+'.weight'], tensors.get(prefix+'.bias'), padding=padding)
        def activated(value, prefix):
            return functional.leaky_relu(conv(value, prefix), .1)
        def excitation(value, prefix):
            pooled = value.mean(dim=(2, 3))
            hidden = functional.relu(functional.linear(pooled, tensors[prefix+'.channel_excite.0.weight'], tensors[prefix+'.channel_excite.0.bias']))
            channel = functional.linear(hidden, tensors[prefix+'.channel_excite.2.weight'], tensors[prefix+'.channel_excite.2.bias']).sigmoid()[:, :, None, None]
            spatial = conv(value, prefix+'.spatial_excite.0', 0).sigmoid()
            return value * channel + value * spatial
        def forward(pixels):
            value = activated(pixels.to(device=device, dtype=precision), '1.entry_block.m.0')
            residual, outer_history = value, [value]
            for outer in range(3):
                inner_history = [value]
                prefix = f'1.blocks.{outer}'
                for inner in range(3):
                    block = f'{prefix}.blocks.{inner}'
                    if inner == 1:
                        branch = excitation(value, block)
                    else:
                        branch = functional.leaky_relu(conv(activated(value, block+'.m.0'), block+'.m.2') + value, .1)
                    inner_history.append(branch)
                    value = activated(torch.cat(inner_history, dim=1), f'{prefix}.singles.{inner}.m.0')
                outer_history.append(value)
                value = activated(torch.cat(outer_history, dim=1), f'1.singles.{outer}.m.0')
            value = activated(value + residual, '1.upsampler.m.0.0')
            return conv(functional.pixel_shuffle(value, 2), '1.exit_conv')
        return forward
    except EngineError:
        raise
    except Exception as error:
        raise EngineError(MODEL_LOAD_FAILED, f'Cannot load Waifu2x {model_id}: {error}') from error


def run_tiles(image, model, tile=64, check_interrupt=lambda: None, progress=None):
    """One model patch resident at a time; publish no partial output on cancellation."""
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from PIL import Image
    width, height = image.size
    validate_dimensions(width * 2, height * 2)
    plan = tile_plan(width, height, tile)
    check_interrupt()
    pixels = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
    source = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    source = functional.pad(source, (3, 3, 3, 3), mode='replicate')
    output = np.empty((height * 2, width * 2, 3), dtype=np.uint8)
    with torch.inference_mode():
        for index, (x, y, core_width, core_height) in enumerate(plan, 1):
            check_interrupt()
            patch = source[:, :, y:y+core_height+6, x:x+core_width+6]
            predicted = model(patch)
            check_interrupt()
            expected = (1, 3, (core_height+6)*2, (core_width+6)*2)
            if not isinstance(predicted, torch.Tensor) or tuple(predicted.shape) != expected or not torch.isfinite(predicted).all():
                raise EngineError(MODEL_LOAD_FAILED, 'Waifu2x model returned an invalid tile')
            cropped = predicted[0, :, 6:6+core_height*2, 6:6+core_width*2]
            # Legacy conversion clips then truncates, rather than rounding.
            converted = cropped.float().cpu().permute(1, 2, 0).numpy()
            output[y*2:(y+core_height)*2, x*2:(x+core_width)*2] = np.clip(converted * 255.0, 0, 255).astype(np.uint8)
            del predicted, cropped, converted
            if progress is not None:
                progress(index, _tile_count(width, height, tile))
    check_interrupt()
    return Image.fromarray(output, mode='RGB')


_model_cache = None


def unload():
    global _model_cache
    _model_cache = None


def model_for(model_id, device='cpu', cache_dir=None, allow_download=True, check_interrupt=lambda: None):
    global _model_cache
    asset = next((entry for entry in assets() if entry['id'] == model_id), None)
    if asset is None:
        raise EngineError(INVALID_PARAMS, f'Unknown Waifu2x model: {model_id}')
    directory = cache_dir or os.environ.get('SDFX_WAIFU2X_CACHE') or Path.home() / '.cache' / 'stablediffusion-nui' / 'waifu2x'
    path = materialize(asset, directory, allow_download, check_interrupt)
    key = model_id, str(device), str(path.resolve()), asset['sha256']
    if _model_cache is None or _model_cache[0] != key:
        check_interrupt()
        model = load_model(model_id, path, device)
        check_interrupt()
        _model_cache = key, model
    return _model_cache[1]


def upscale(image, width, height, model_id, device='cpu', tile=64, cache_dir=None,
            allow_download=True, check_interrupt=lambda: None, progress=None):
    """Repeated native 2x passes, then exact target resampling (including 1x denoise)."""
    from PIL import Image
    validate_dimensions(width, height)
    options(tile=tile)
    if not isinstance(allow_download, bool):
        raise EngineError(INVALID_PARAMS, 'Waifu2x allow_download must be boolean')
    ratio = max(width / image.width, height / image.height)
    passes = max(1, math.ceil(math.log2(ratio)))
    if passes > 4:
        raise EngineError(INVALID_PARAMS, 'Waifu2x target may not exceed 16x the input size')
    validate_dimensions(image.width * 2**passes, image.height * 2**passes)
    check_interrupt()
    model = model_for(model_id, device, cache_dir, allow_download, check_interrupt)
    current = image.convert('RGB')
    total = sum(_tile_count(image.width * 2**index, image.height * 2**index, tile) for index in range(passes))
    done = 0
    for _ in range(passes):
        check_interrupt()
        count = _tile_count(current.width, current.height, tile)
        if progress is None:
            current = run_tiles(current, model, tile, check_interrupt)
        else:
            offset = done
            current = run_tiles(current, model, tile, check_interrupt, lambda part, _: progress(offset + part, total))
        done += count
    if current.size != (width, height):
        current = current.resize((width, height), Image.Resampling.LANCZOS)
    check_interrupt()
    return current
