"""Reference-guided frame processing for the native Temporal Lab.

Controls and algorithm details were informed by AbyszOne/Abysz-LAB-Ext source
at deff0d4b4a208f862299b14b93e088f613373153 (GPL-3.0); see THIRD_PARTY_NOTICES.md.
This implementation does not claim complete numerical equivalence: masks use
grayscale difference and optional expansion/blur with a dynamic Gaussian pass;
the exact-white percentage selects the default dissolve strength. Inter-blur is an intentional working enhancement
(the pinned upstream control does not affect its output). Frame extraction
uses frame-index timestamps rather than the upstream decoder's POS_MSEC value.
"""
from __future__ import annotations

import base64
from io import BytesIO
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .protocol import EngineError, INVALID_PARAMS, UNSUPPORTED
from .temporal_lab_io import (MAX_DIMENSION, MAX_FRAME_PIXELS, MAX_FRAMES, MAX_TOTAL_PIXELS,
                              OutputTransaction, _absolute, _reject_links, read_sequence)

MAX_OUTPUT_BYTES = 4 * 1024 * 1024 * 1024
MAX_PREVIEW_SIDE = 1024
# Conservative scratch estimate for RGB float64 overlay/NumPy temporaries,
# decoded frames and stage-local arrays; this is a per-operation guard, not a
# process-wide allocator limit. It may reject images below the IO pixel limit.
# A video codec allocates a frame before Python can inspect its actual shape.
MAX_WORKING_BYTES = 512 * 1024 * 1024
ESTIMATED_BYTES_PER_PIXEL = 256


def _number(params, name, default, minimum, maximum, *, integer=False, odd=False):
    value = params.get(name, default)
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite number")
    if not minimum <= value <= maximum or (integer and int(value) != value) or (odd and int(value) % 2 != 1):
        raise ValueError(f"{name} must be {'an odd integer' if odd else 'in range'} {minimum}..{maximum}")
    return int(value) if integer else float(value)


def _path(params, name):
    value = params.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path")
    return value


def _cv():
    try:
        import cv2
    except ImportError as exc:
        raise EngineError(UNSUPPORTED, "Temporal Lab requires OpenCV") from exc
    return cv2


def _read_rgb(path, ctx=None):
    with Image.open(path) as image:
        if 'Encrypt' in image.info:
            password = getattr(ctx, 'image_password', None)
            if not password:
                raise ValueError('image password required')
            from .encrypt_image import decode_png
            decoded, verified = decode_png(Path(path).read_bytes(), password)
            if verified is None and callable(getattr(ctx, 'warn', None)):
                ctx.warn('Legacy image password is unverified; the decoded pixels may be incorrect.')
            with Image.open(BytesIO(decoded)) as plain:
                return np.array(plain.convert('RGB'), dtype=np.uint8)
        return np.array(image.convert("RGB"), dtype=np.uint8)


def _dfi_options(params):
    return {
        "source_denoise": _number(params, "source_denoise", 0, 0, 30, integer=True),
        "tolerance": _number(params, "tolerance", 5, 0.5, 20),
        "expand": _number(params, "expand", 0, 0, 50, integer=True),
        "smooth": _number(params, "smooth", 1, 1, 99, integer=True, odd=True),
        "inter_denoise": _number(params, "inter_denoise", 9, 1, 25, integer=True),
        "inter_denoise_size": _number(params, "inter_denoise_size", 9, 1, 25, integer=True, odd=True),
        "inter_denoise_frequency": _number(params, "inter_denoise_frequency", 3, 1, 15, integer=True),
        "inter_blur": _number(params, "inter_blur", 3, 1, 5),
        "refresh_frequency": _number(params, "refresh_frequency", 0, 0, 30, integer=True),
        "preserve": _number(params, "preserve", 0, 0, 100),
        "frame_limit": _number(params, "frame_limit", 0, 0, MAX_FRAMES, integer=True),
    }


def _pairs(first, second, ctx, equal_dimensions=True):
    a, b = read_sequence(first, ctx.check_interrupt), read_sequence(second, ctx.check_interrupt)
    if a.count != b.count:
        raise ValueError("frame sequences must have equal counts")
    if equal_dimensions and a.dimensions != b.dimensions:
        raise ValueError("reference and generated frame dimensions must match")
    return a, b


def _check_working_memory(*dimensions):
    """Conservatively guard largest source/target frame before full decode."""
    pixels = max(width * height for width, height in dimensions)
    estimated = pixels * ESTIMATED_BYTES_PER_PIXEL
    if estimated > MAX_WORKING_BYTES:
        raise ValueError(f"estimated working memory {estimated} bytes exceeds {MAX_WORKING_BYTES} bytes; use smaller images")


def _align_reference(image, target_dimensions):
    """Center-crop reference to generated aspect ratio, then resize it."""
    width, height = target_dimensions
    source_height, source_width = image.shape[:2]
    if source_width * height > width * source_height:
        cropped_width = max(1, int(source_height * width / height))
        left = (source_width - cropped_width) // 2
        image = image[:, left:left + cropped_width]
    elif source_width * height < width * source_height:
        cropped_height = max(1, int(source_width * height / width))
        top = (source_height - cropped_height) // 2
        image = image[top:top + cropped_height, :]
    if image.shape[:2] != (height, width):
        image = _cv().resize(image, (width, height))
    return image


def _mask(previous, current, options):
    cv = _cv()
    if options["source_denoise"]:
        d = options["source_denoise"]
        previous = cv.bilateralFilter(previous, d, 31, 31)
        current = cv.bilateralFilter(current, d, 31, 31)
    p = cv.cvtColor(previous, cv.COLOR_RGB2GRAY)
    c = cv.cvtColor(current, cv.COLOR_RGB2GRAY)
    stable = np.where(cv.absdiff(p, c) <= options["tolerance"], 255, 0).astype(np.uint8)
    if options["expand"]:
        width = options["expand"]
        changing = cv.dilate(255 - stable, np.ones((width, width), dtype=np.uint8))
        stable = 255 - changing
    if options["smooth"] > 1:
        width = options["smooth"]
        stable = cv.GaussianBlur(stable, (width, width), 0)
    return stable


def _preview(params, ctx):
    options = _dfi_options(params)
    reference, generated = _pairs(_path(params, "reference_dir"), _path(params, "generated_dir"), ctx,
                                  equal_dimensions=False)
    _check_working_memory(reference.dimensions, generated.dimensions)
    idx = _number(params, "frame_index", 1, 1, reference.count - 1, integer=True)
    ctx.check_interrupt()
    mask = _mask(_align_reference(_read_rgb(reference.paths[idx - 1], ctx), generated.dimensions),
                 _align_reference(_read_rgb(reference.paths[idx], ctx), generated.dimensions), options)
    preview = Image.fromarray(mask, "L")
    preview.thumbnail((MAX_PREVIEW_SIDE, MAX_PREVIEW_SIDE))
    with BytesIO() as buffer:
        preview.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"operation": "preview", "mask_png_base64": encoded, "frame_index": idx,
            "dimensions": list(generated.dimensions), "preview_dimensions": list(preview.size),
            "generated_previous": str(generated.paths[idx - 1]), "generated_current": str(generated.paths[idx])}


def _writer(tx, image, index, bytes_used, ctx=None):
    assert tx.staging is not None
    output = tx.staging / f"frame{index:06d}.png"
    pixels = Image.fromarray(image, "RGB")
    password = getattr(ctx, 'image_password', None)
    if password:
        from .encrypt_image import encode_png
        buffer = BytesIO()
        pixels.save(buffer, format='PNG')
        ctx.check_interrupt()
        output.write_bytes(encode_png(buffer.getvalue(), password))
    else:
        pixels.save(output, format="PNG")
    size = output.stat().st_size
    total = bytes_used + size
    if total > MAX_OUTPUT_BYTES:
        raise ValueError(f"output exceeds {MAX_OUTPUT_BYTES} bytes")
    return total


def _transaction(params, inputs):
    return OutputTransaction(_path(params, "output_parent"), _path(params, "output_name"), inputs)


def _dfi(params, ctx):
    options = _dfi_options(params)
    reference_dir, generated_dir = _path(params, "reference_dir"), _path(params, "generated_dir")
    reference, generated = _pairs(reference_dir, generated_dir, ctx, equal_dimensions=False)
    _check_working_memory(reference.dimensions, generated.dimensions)
    count = min(reference.count, options["frame_limit"] or reference.count)
    ctx.check_interrupt()
    with _transaction(params, [reference_dir, generated_dir]) as tx:
        prior_ref = _align_reference(_read_rgb(reference.paths[0], ctx), generated.dimensions)
        previous = _read_rgb(generated.paths[0], ctx)
        used = _writer(tx, previous, 1, 0, ctx)
        ctx.progress(1, count, pass_="dfi")
        for idx in range(1, count):
            ctx.check_interrupt()
            current_ref = _align_reference(_read_rgb(reference.paths[idx], ctx), generated.dimensions)
            current = _read_rgb(generated.paths[idx], ctx)
            pre_mask = _mask(prior_ref, current_ref, options)
            percentage = round(float(np.count_nonzero(pre_mask == 255)) / pre_mask.size * 100, 1)
            kernel = math.ceil((100 - percentage) / 3)
            if kernel % 2 == 0:
                kernel += 1
            dynamic = _cv().GaussianBlur(pre_mask, (kernel, kernel), 0) if kernel > 1 else pre_mask
            mask = dynamic.astype(np.float32) / 255.0
            frequency = options["refresh_frequency"]
            # CopyOpacity transfers the blurred mask into previous-frame alpha;
            # dissolve scales that alpha before compositing over the new frame.
            retention = (options["preserve"] / 100.0 if frequency and (idx - 1) % frequency == 0
                         else 1.0 if frequency else percentage / 100.0)
            weight = (mask * retention)[:, :, None]
            previous = np.clip(np.rint(previous.astype(np.float32) * weight + current * (1 - weight)), 0, 255).astype(np.uint8)
            if (idx - 1) % options["inter_denoise_frequency"] == 0:
                cv = _cv()
                if options["inter_denoise"] > 1 and options["inter_denoise_size"] > 1:
                    previous = cv.bilateralFilter(previous, options["inter_denoise"],
                                                  options["inter_denoise_size"], options["inter_denoise_size"])
                blur = int(round(options["inter_blur"]))
                if blur > 1:
                    width = blur if blur % 2 else blur + 1
                    previous = cv.GaussianBlur(previous, (width, width), 0)
            used = _writer(tx, previous, idx + 1, used, ctx)
            prior_ref = current_ref
            ctx.progress(idx + 1, count, pass_="dfi")
        ctx.check_interrupt()
        result_dir = ctx.publish(tx.commit)
    return {"operation": "dfi", "output_dir": str(result_dir), "frame_count": count,
            "dimensions": list(generated.dimensions), "output_bytes": used,
            "reference_aligned": reference.dimensions != generated.dimensions}


def _overlay(target, blend, strength):
    a, b = target.astype(np.float64) / 255.0, blend.astype(np.float64) / 255.0
    # Branch on target (the prior/style frame), then mix with the current frame.
    composite = np.where(a <= 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
    return np.clip((b * (1 - strength) + composite * strength) * 255, 0, 255).astype(np.uint8)


def _blend_stage(frames, strength):
    cv = _cv()
    previous = next(frames)
    yield previous
    for current in frames:
        yield cv.addWeighted(current, strength, previous, 1 - strength, 0)
        previous = current


def _overlay_stage(frames, strength):
    previous = next(frames)
    current = next(frames, None)
    if current is None:
        yield previous
        return

    def combine(left, right):
        result = _overlay(left, right, strength)
        if strength >= 0.4:
            result = np.minimum(result.astype(np.uint16) + 10, 255).astype(np.uint8)
        return result

    first = combine(previous, current)
    yield first
    yield first.copy()
    previous = current
    for current in frames:
        yield combine(previous, current)
        previous = current


def _normalize_stage(frames):
    cv = _cv()
    previous = next(frames)
    yield previous
    for current in frames:
        a = float(np.mean(cv.cvtColor(previous, cv.COLOR_RGB2GRAY)))
        b = float(np.mean(cv.cvtColor(current, cv.COLOR_RGB2GRAY)))
        # Both black inputs have no defined upstream ratio: preserve black.
        yield current if a + b == 0 else cv.addWeighted(previous, a / (a + b), current, b / (a + b), 0)
        previous = current


def _deflicker(params, ctx):
    blend = _number(params, "blend", 0, 0, 1)
    overlay = _number(params, "overlay", 0, 0, 1)
    normalize = params.get("normalize", False)
    if not isinstance(normalize, bool):
        raise ValueError("normalize must be boolean")
    frames_dir = _path(params, "frames_dir")
    sequence = read_sequence(frames_dir, ctx.check_interrupt)
    _check_working_memory(sequence.dimensions)
    with _transaction(params, [frames_dir]) as tx:
        def read_frames():
            for path in sequence.paths:
                ctx.check_interrupt()
                yield _read_rgb(path, ctx)

        frames = read_frames()
        if blend:
            frames = _blend_stage(frames, blend)
        if overlay:
            frames = _overlay_stage(frames, overlay)
        if normalize:
            frames = _normalize_stage(frames)
        used = 0
        for idx, current in enumerate(frames):
            ctx.check_interrupt()
            used = _writer(tx, current, idx + 1, used, ctx)
            ctx.progress(idx + 1, sequence.count, pass_="deflicker")
        ctx.check_interrupt()
        output = ctx.publish(tx.commit)
    return {"operation": "deflicker", "output_dir": str(output), "frame_count": sequence.count,
            "dimensions": list(sequence.dimensions), "output_bytes": used}


def _fuse(params, ctx):
    strength = _number(params, "strength", 0.5, 0.1, 1)
    style_dir, video_dir = _path(params, "style_dir"), _path(params, "video_dir")
    style, video = _pairs(style_dir, video_dir, ctx, equal_dimensions=False)
    _check_working_memory(style.dimensions, video.dimensions)
    with _transaction(params, [style_dir, video_dir]) as tx:
        used = 0
        for idx, (a, b) in enumerate(zip(style.paths, video.paths)):
            ctx.check_interrupt()
            style_frame = np.array(Image.fromarray(_read_rgb(a, ctx), 'RGB').resize(video.dimensions, Image.Resampling.LANCZOS))
            fused = _overlay(style_frame, _read_rgb(b, ctx), strength)
            used = _writer(tx, fused, idx + 1, used, ctx)
            ctx.progress(idx + 1, video.count, pass_="fuse")
        ctx.check_interrupt()
        output = ctx.publish(tx.commit)
    return {"operation": "fuse", "output_dir": str(output), "frame_count": video.count,
            "dimensions": list(video.dimensions), "resized_style": style.dimensions != video.dimensions,
            "output_bytes": used}


def _extract(params, ctx):
    cv = _cv()
    path = _path(params, "video_path")
    _reject_links(_absolute(path))
    if not Path(path).is_file():
        raise ValueError("video file does not exist")
    requested = _number(params, "fps", 0, 0, 1000)
    capture = cv.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise ValueError("could not open video")
        source_fps = float(capture.get(cv.CAP_PROP_FPS))
        raw_metadata = [capture.get(cv.CAP_PROP_FRAME_WIDTH), capture.get(cv.CAP_PROP_FRAME_HEIGHT),
                        capture.get(cv.CAP_PROP_FRAME_COUNT)]
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in raw_metadata):
            raise ValueError("video dimensions or frame count are invalid")
        width, height, expected = map(int, raw_metadata)
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("video FPS is invalid or missing")
        if width <= 0 or height <= 0 or max(width, height) > MAX_DIMENSION or width * height > MAX_FRAME_PIXELS:
            raise ValueError("video dimensions invalid or exceed limits")
        _check_working_memory((width, height))
        if expected <= 0 or expected > MAX_FRAMES or expected * width * height > MAX_TOTAL_PIXELS:
            raise ValueError("video frame count is unknown or exceeds limits")
        fps = requested or source_fps
        if fps > source_fps:
            raise ValueError("requested FPS exceeds source FPS")
        used = 0
        written = 0
        next_sample = 0.0
        with _transaction(params, [path]) as tx:
            for frame_index in range(expected):
                ctx.check_interrupt()
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"video decode stopped at frame {frame_index} of {expected}")
                if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
                    raise ValueError("video decoder returned an invalid frame")
                actual_height, actual_width = frame.shape[:2]
                if (actual_width <= 0 or actual_height <= 0 or max(actual_width, actual_height) > MAX_DIMENSION
                        or actual_width * actual_height > MAX_FRAME_PIXELS):
                    raise ValueError("decoded video frame dimensions invalid or exceed limits")
                _check_working_memory((actual_width, actual_height))
                if frame.shape[:2] != (height, width):
                    raise ValueError("video frame dimensions changed")
                if frame_index / source_fps + 1e-9 >= next_sample:
                    used = _writer(tx, cv.cvtColor(frame, cv.COLOR_BGR2RGB), written + 1, used, ctx)
                    written += 1
                    next_sample += 1.0 / fps
                ctx.progress(frame_index + 1, expected, pass_="extract")
            if not written:
                raise ValueError("no frames extracted")
            ctx.check_interrupt()
            output = ctx.publish(tx.commit)
        return {"operation": "extract", "output_dir": str(output), "frame_count": written,
                "dimensions": [width, height], "fps": fps, "source_fps": source_fps,
                "duration": expected / source_fps, "output_bytes": used}
    finally:
        capture.release()


def run(params: dict, ctx) -> dict:
    if not isinstance(params, dict):
        raise EngineError(INVALID_PARAMS, "Temporal Lab parameters must be an object")
    operations = {"preview": _preview, "dfi": _dfi, "deflicker": _deflicker, "fuse": _fuse, "extract": _extract}
    operation = params.get("operation")
    if not isinstance(operation, str) or operation not in operations:
        raise EngineError(INVALID_PARAMS, "unknown Temporal Lab operation")
    try:
        result = operations[operation](params, ctx)
        if getattr(ctx, 'warnings', None):
            result['warnings'] = list(ctx.warnings)
        return result
    except EngineError:
        raise
    except (ValueError, OSError) as exc:
        raise EngineError(INVALID_PARAMS, str(exc)) from exc
