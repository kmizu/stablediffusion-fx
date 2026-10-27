"""A1111 parameters for owned WD batch PNGs (#141).

GenerationParams uses the existing snake_case worker contract. PNG text uses
the same labels/JSON quoting as the native Infotext reader, not a second
serialized generation schema. WD provenance is informational extra text.
"""
import json
import os
import struct
import tempfile
import zlib
from pathlib import Path

from .params import GenerationParams, Img2ImgParams, from_dict

MAX_METADATA_BYTES = 1024 * 1024
MAX_PNG_BYTES = 128 * 1024 * 1024


def _value(value):
    if isinstance(value, float):
        return format(value, '.15g')
    return str(value)


def parameters(generation, row, image, result, provenance):
    p = from_dict(GenerationParams, generation['params'])
    ii = p.img2img or Img2ImgParams()
    seed = image.get('seed')
    if type(seed) is not int or seed < 0:
        raise ValueError('generated image has no actual seed')
    fields = dict(p.extra)
    fields.update({'Steps': p.steps, 'Sampler': p.sampler_name,
        'Schedule type': result.get('scheduler') or p.scheduler,
        'CFG scale': p.cfg_scale, 'Seed': seed,
        'Size': f"{image.get('width') or p.width}x{image.get('height') or p.height}",
        'Batch size': p.batch_size, 'Batch count': p.n_iter,
        'Denoising strength': ii.denoising_strength, 'Resize mode': ii.resize_mode,
        'Clip skip': p.clip_skip, 'Eta': p.eta if p.eta is not None else 'Automatic',
        'VAE': result.get('vae_name') or 'Automatic'})
    for key, label in (('model_name', 'Model'), ('model_hash', 'Model hash'),
                       ('vae_name', 'VAE'), ('vae_hash', 'VAE hash')):
        value = result.get(key)
        if value is not None:
            fields[label] = value[:10] if key.endswith('_hash') else value
    if p.subseed_strength > 0:
        subseed = image.get('subseed')
        if type(subseed) is not int or subseed < 0:
            raise ValueError('generated image has no actual variation seed')
        fields.update({'Variation seed': subseed, 'Variation seed strength': p.subseed_strength})
    if ii.mask:
        fields.update({'Mask mode': 'Inpaint not masked' if ii.inpainting_mask_invert else 'Inpaint masked',
            'Mask blur': ii.mask_blur, 'Inpaint area': 'Only masked' if ii.inpaint_full_res else 'Whole picture',
            'Masked area padding': ii.inpaint_full_res_padding,
            'Masked content': {0: 'fill', 1: 'original', 2: 'latent noise', 3: 'latent nothing'}[ii.inpainting_fill]})
    hardware = image.get('hardware_info') or {}
    for key, label in (('hardware_info', 'Hardware Info'), ('time_taken', 'Time taken'), ('remark', 'Remark')):
        if hardware.get(key) is not None:
            fields[label] = hardware[key]
    fields['WD interrogators'] = json.dumps(provenance, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    fields['WD input'] = row['input']
    # Original image identity is descriptive only; Restore must retain the
    # currently selected init image/mask instead of opening metadata paths.
    if row.get('input_sha256'):
        fields['WD input SHA256'] = row['input_sha256']
    def quote(value):
        text = _value(value)
        return json.dumps(text, ensure_ascii=False) if any(c in text for c in ',\n:') else text
    text = row['positive']
    if row['negative']:
        text += '\nNegative prompt: ' + row['negative']
    text += '\n' + ', '.join(f'{key}: {quote(value)}' for key, value in fields.items())
    if len(text.encode('utf-8')) > MAX_METADATA_BYTES:
        raise ValueError('PNG generation metadata exceeds limit')
    return text


def write_parameters(path, text, check_cancel):
    """Replace parameters without decoding/re-encoding IDAT or other chunks.

    Caller must validate path ownership first. All writes stay in the private
    transaction directory; cancellation/failure never replaces the PNG.
    """
    payload = text.encode('utf-8')
    if len(payload) > MAX_METADATA_BYTES:
        raise ValueError('PNG generation metadata exceeds limit')
    path = Path(path)
    if path.stat().st_size > MAX_PNG_BYTES:
        raise ValueError('generated PNG exceeds limit')
    check_cancel()
    fd, temp = tempfile.mkstemp(prefix='.metadata-', suffix='.part', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as dst, path.open('rb') as src:
            signature = src.read(8)
            if signature != b'\x89PNG\r\n\x1a\n':
                raise ValueError('generated output is not PNG')
            dst.write(signature)
            total = 8
            inserted = False
            while True:
                check_cancel()
                header = src.read(8)
                if len(header) != 8:
                    raise ValueError('truncated generated PNG')
                length, kind = struct.unpack('>I4s', header)
                total += length + 12
                if total > MAX_PNG_BYTES:
                    raise ValueError('generated PNG exceeds limit')
                data, crc = src.read(length), src.read(4)
                if len(data) != length or len(crc) != 4 or struct.unpack('>I', crc)[0] != zlib.crc32(kind + data):
                    raise ValueError('invalid generated PNG chunk')
                # Metadata precedes IDAT so both ImageIO metadata-only reads
                # and Pillow Image.open see it without decoding image pixels.
                if kind == b'IDAT' and not inserted:
                    encoded = b'parameters\x00\x00\x00\x00\x00' + payload
                    dst.write(struct.pack('>I', len(encoded)) + b'iTXt' + encoded
                              + struct.pack('>I', zlib.crc32(b'iTXt' + encoded)))
                    inserted = True
                if kind == b'IEND':
                    if length or src.read(1) or not inserted:
                        raise ValueError('invalid generated PNG end')
                    dst.write(header + data + crc)
                    break
                if kind not in (b'tEXt', b'iTXt', b'zTXt') or data.split(b'\x00', 1)[0] != b'parameters':
                    dst.write(header + data + crc)
        check_cancel()
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
