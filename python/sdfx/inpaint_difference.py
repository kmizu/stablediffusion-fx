"""Inpaint Difference mask and preview, based on the pinned MIT upstream behavior."""
import math
import re
from contextlib import contextmanager

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

MAX_PIXELS = 4096 * 4096
MAX_FILE_BYTES = 200_000_000


@contextmanager
def generation_request(params, mode, ctx):
    """Make an owned mask and canonical altered image for one generation."""
    import copy
    import tempfile
    from pathlib import Path
    from .hooks import hook_config
    from .protocol import EngineError, INVALID_PARAMS

    fields = params.get('params', {}) if isinstance(params, dict) else {}
    config = hook_config(fields.get('hooks', {}), 'inpaint_difference', {})
    if config is None:
        yield params
        return
    image_params = fields.get('img2img')
    if mode != 'img2img' or not isinstance(image_params, dict):
        raise EngineError(INVALID_PARAMS, 'Inpaint Difference requires an img2img input')
    base_path, altered_path = config.get('base_path'), image_params.get('init_image')
    try:
        for path in (base_path, altered_path):
            if not isinstance(path, str) or not path or len(path) > 32768:
                raise ValueError('Both base and altered image paths are required')
            item = Path(path)
            if not item.is_file() or item.stat().st_size > MAX_FILE_BYTES:
                raise ValueError('An image is missing or too large')
        options = {key: config[key] for key in ('threshold', 'contours', 'dilation', 'erosion') if key in config}
        ctx.check_interrupt()
        with Image.open(base_path) as base, Image.open(altered_path) as altered:
            mask, _ = compute(base, altered, **options)
            canonical = ImageOps.exif_transpose(altered).convert('RGB')
        ctx.check_interrupt()
    except (ValueError, OSError) as exc:
        raise EngineError(INVALID_PARAMS, f'Inpaint Difference: {exc}') from exc
    with tempfile.TemporaryDirectory(prefix='sdfx-inpaint-difference-') as directory:
        try:
            mask_path, image_path = Path(directory) / 'mask.png', Path(directory) / 'altered.png'
            mask.save(mask_path, format='PNG')
            canonical.save(image_path, format='PNG')
            ctx.check_interrupt()
            owned = copy.deepcopy(params)
            owned['params']['img2img']['init_image'] = str(image_path)
            owned['params']['img2img']['mask'] = str(mask_path)
        except (ValueError, OSError) as exc:
            raise EngineError(INVALID_PARAMS, f'Inpaint Difference: {exc}') from exc
        yield owned


def preview_request(params, ctx):
    """Create preview PNGs in a unique, caller-owned child directory."""
    import os
    from pathlib import Path
    import shutil
    import tempfile

    if not isinstance(params, dict):
        raise ValueError('Preview request must be an object')
    paths = []
    for key in ('base_path', 'altered_path'):
        value = params.get(key)
        if not isinstance(value, str) or not value or len(value) > 32768:
            raise ValueError(f'{key} must identify an image')
        path = Path(value)
        if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f'{key} is missing or too large')
        paths.append(path)
    output = params.get('output_dir')
    if not isinstance(output, str) or not output or len(output) > 32768:
        raise ValueError('output_dir is required')
    options = {key: params[key] for key in ('threshold', 'contours', 'dilation', 'erosion', 'blur', 'color', 'show_under', 'invert', 'preview_mode') if key in params}
    ctx.check_interrupt()
    with Image.open(paths[0]) as first, Image.open(paths[1]) as second:
        mask, preview = compute(first, second, **options)
    ctx.check_interrupt()
    os.makedirs(output, exist_ok=True)
    owned = tempfile.mkdtemp(prefix='inpaint-difference-', dir=output)
    try:
        mask_path, preview_path = Path(owned) / 'mask.png', Path(owned) / 'preview.png'
        mask.save(mask_path, format='PNG')
        ctx.check_interrupt()
        preview.save(preview_path, format='PNG')
        ctx.check_interrupt()
        return dict(artifact_dir=owned, mask_path=str(mask_path), preview_path=str(preview_path),
                    width=mask.width, height=mask.height)
    except BaseException:
        shutil.rmtree(owned)
        raise


def compute(base, altered, *, threshold=1., contours=False, dilation=0, erosion=0,
            blur=0, color='#ffffff', show_under=True, invert=False, preview_mode='source'):
    if (not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
            or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('Difference threshold must be between 0 and 1')
    for name, value, high in (('dilation', dilation, 100), ('erosion', erosion, 100), ('blur', blur, 64)):
        if type(value) is not int or not 0 <= value <= high:
            raise ValueError(f'{name} must be an integer between 0 and {high}')
    if type(contours) is not bool or type(show_under) is not bool or type(invert) is not bool:
        raise ValueError('Preview and contour flags must be boolean')
    if preview_mode not in ('source', 'effective'):
        raise ValueError('Preview mode must be source or effective')
    if not isinstance(color, str) or not re.fullmatch(r'#[0-9a-fA-F]{6}', color):
        raise ValueError('Mask color must be #RRGGBB')
    for image in (base, altered):
        if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0 or image.width * image.height > MAX_PIXELS:
            raise ValueError('Input image dimensions are invalid or too large')
    base = ImageOps.exif_transpose(base).convert('RGB')
    altered = ImageOps.exif_transpose(altered).convert('RGB')
    if base.size != altered.size:
        base = base.resize(altered.size)
    a = np.asarray(base, dtype=np.uint8)
    b = np.asarray(altered, dtype=np.uint8)
    mean = cv2.absdiff(a, b).sum(axis=2, dtype=np.uint16) // 3
    mask = np.where(mean / 255 > 1 - threshold, 255, 0).astype(np.uint8)
    if contours:
        found, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, found, -1, 255, 1)
        mask[0, :] = mask[-1, :] = 0
        mask[:, 0] = mask[:, -1] = 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    if dilation:
        mask = cv2.dilate(mask, kernel, iterations=dilation, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    if erosion:
        mask = cv2.erode(mask, kernel, iterations=erosion, borderType=cv2.BORDER_CONSTANT, borderValue=255)
    # Upstream keeps threshold masks as float64, contours as float32, and
    # morphology returns uint8; OpenCV's blur rounds differently by dtype.
    if preview_mode == 'effective':
        effective = Image.fromarray(mask, 'L')
        if invert:
            effective = ImageOps.invert(effective)
        if blur:
            effective = effective.filter(ImageFilter.GaussianBlur(blur))
        blurred = np.asarray(effective)
    else:
        preview_mask = mask if dilation or erosion else mask.astype(np.float32 if contours else np.float64)
        size = 2 * int(2.5 * blur + .5) + 1
        blurred = cv2.GaussianBlur(preview_mask, (size, size), blur)
    rgb = np.asarray([int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    overlay = blurred[:, :, None] * rgb
    if show_under:
        alpha = blurred[:, :, None] / 255
        overlay = b * (1 - alpha) + overlay * alpha
    return Image.fromarray(mask, 'L'), Image.fromarray(overlay.astype(np.uint8), 'RGB')
