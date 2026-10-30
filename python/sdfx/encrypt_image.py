"""Compatibility with sd-encrypt-image pixel scrambling, not cryptographic encryption.

The transformation preserves the image's pixel values and leaks metadata. It is
only suitable for interoperating with the upstream formats, not confidentiality.
16-bit RGB(A) PNGs use OpenCV to avoid Pillow silently reducing precision.
"""
from __future__ import annotations

import hashlib
import io
import hmac
import struct
import zlib

import numpy as np
from PIL import Image, PngImagePlugin, UnidentifiedImageError


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def password_marker(password: str) -> str:
    return _digest(_digest(password) + 'Encrypt')


def verify_password(password: str, marker: str | None) -> bool | None:
    """Return None when the upstream file has no password verifier."""
    if marker is None:
        return None
    if not isinstance(marker, str) or len(marker) != 64 or any(c not in '0123456789abcdef' for c in marker):
        return False
    return hmac.compare_digest(password_marker(password), marker)


def _indices(length: int, key: str, version: int) -> list[int]:
    seed = _digest(key)
    sequence = list(range(length))
    doubled = seed * 2
    for i in range(length):
        offset = i % len(seed)
        target = int(doubled[offset:offset + 8], 16) % (length - i)
        source = length - i - 1 if version == 3 else i
        sequence[source], sequence[target] = sequence[target], sequence[source]
    return sequence


def _transform(image: Image.Image, password: str, version: int, reverse: bool) -> Image.Image:
    if version not in (1, 2, 3):
        raise ValueError('unsupported pixel scramble version')
    if not isinstance(password, str) or not password:
        raise ValueError('password must not be empty')
    allowed = ('RGB', 'RGBA', 'LA', 'L', 'P', 'I;16') if version == 1 and reverse else ('RGB', 'RGBA', 'LA')
    if image.mode not in allowed:
        raise ValueError('pixel scrambling requires RGB, RGBA or LA, except legacy v1 decoding also supports L, P and I;16')
    key = _digest(password)
    xs = _indices(image.width, key, version)
    ys = _indices(image.height, _digest(key), version)
    data = np.array(image, copy=True)
    if version == 1:
        x_iter = range(image.width - 1, -1, -1) if reverse else range(image.width)
        y_iter = range(image.height - 1, -1, -1) if reverse else range(image.height)
        for x in x_iter:
            for y in y_iter:
                other_x, other_y = xs[x], ys[y]
                value = data[y, x].copy()
                data[y, x] = data[other_y, other_x]
                data[other_y, other_x] = value
    elif version == 2:
        def rows():
            for y in (range(image.height - 1, -1, -1) if reverse else range(image.height)):
                value = data[y, :].copy()
                data[y, :] = data[ys[y], :]
                data[ys[y], :] = value
        def columns():
            for x in (range(image.width - 1, -1, -1) if reverse else range(image.width)):
                value = data[:, x].copy()
                data[:, x] = data[:, xs[x]]
                data[:, xs[x]] = value
        if reverse:
            columns(); rows()
        else:
            rows(); columns()
    elif reverse:
        restored = np.empty_like(data)
        restored[:, xs] = data
        data = restored
        restored = np.empty_like(data)
        restored[ys, :] = data
        data = restored
    else:
        data = data[ys, :, :][:, xs, :]
    output = Image.fromarray(data, image.mode)
    if image.mode == 'P' and image.palette is not None:
        output.putpalette(image.palette.copy())
    output.info = image.info.copy()
    return output


def scramble(image: Image.Image, password: str) -> Image.Image:
    return _transform(image, password, 3, False)


def unscramble(image: Image.Image, password: str, version: int) -> Image.Image:
    return _transform(image, password, version, True)


def _open_png(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data), formats=['PNG'])
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError('invalid PNG image') from exc
    return image


def _serialize_png(image: Image.Image, text: dict[str, str], extra: dict) -> bytes:
    info = PngImagePlugin.PngInfo()
    for key, value in text.items():
        info.add_text(key, value)
    output = io.BytesIO()
    image.save(output, format='PNG', pnginfo=info, **extra)
    return output.getvalue()


def _metadata(image: Image.Image) -> tuple[dict[str, str], dict]:
    text = dict(image.text)
    extra = {k: image.info[k] for k in ('icc_profile', 'exif', 'dpi') if k in image.info}
    return text, extra


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    """Validate framing and CRC before rewriting a 16-bit PNG."""
    if not data.startswith(_PNG_SIGNATURE):
        raise ValueError('invalid PNG image')
    chunks = []
    pos = len(_PNG_SIGNATURE)
    while pos < len(data):
        if len(data) - pos < 12:
            raise ValueError('invalid PNG chunk')
        size = struct.unpack_from('>I', data, pos)[0]
        end = pos + size + 12
        if end > len(data):
            raise ValueError('invalid PNG chunk length')
        kind = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + size]
        crc = struct.unpack_from('>I', data, pos + 8 + size)[0]
        if zlib.crc32(kind + payload) != crc:
            raise ValueError('invalid PNG chunk CRC')
        chunks.append((kind, payload))
        pos = end
        if kind == b'IEND':
            break
    kinds = [kind for kind, _ in chunks]
    if (pos != len(data) or not chunks or kinds[0] != b'IHDR' or len(chunks[0][1]) != 13
            or kinds[-1] != b'IEND' or chunks[-1][1] or kinds.count(b'IHDR') != 1
            or kinds.count(b'IEND') != 1 or b'IDAT' not in kinds):
        raise ValueError('invalid PNG structure')
    first, last = kinds.index(b'IDAT'), len(kinds) - 1 - kinds[::-1].index(b'IDAT')
    if any(kind != b'IDAT' for kind in kinds[first:last + 1]):
        raise ValueError('non-consecutive PNG image data')
    if any(kind[0] & 32 == 0 and kind not in (b'IHDR', b'PLTE', b'IDAT', b'IEND') for kind in kinds):
        raise ValueError('unsupported critical PNG chunk')
    return chunks


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload))


def _is_16bit_png(data: bytes) -> bool:
    return (len(data) >= 29 and data.startswith(_PNG_SIGNATURE)
            and data[8:16] == b'\x00\x00\x00\x0dIHDR' and data[24] == 16)


def _transform_16bit_png(data: bytes, password: str, reverse: bool) -> tuple[bytes, bool | None]:
    import cv2

    chunks = _png_chunks(data)
    width, height, depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', chunks[0][1])
    if depth != 16 or color not in (0, 2, 6) or compression or filtering or interlace not in (0, 1):
        raise ValueError('unsupported 16-bit PNG sample layout')
    if not isinstance(password, str) or not password:
        raise ValueError('password must not be empty')
    # Pillow is used for text parsing only; OpenCV retains all 16 sample bits.
    with _open_png(data) as image:
        text = dict(image.text)
    if reverse:
        versions = {'pixel_shuffle': 1, 'pixel_shuffle_2': 2, 'pixel_shuffle_3': 3}
        version = versions.get(text.get('Encrypt'))
        if version is None:
            raise ValueError('unsupported or missing pixel scramble marker')
        checked = verify_password(password, text.get('EncryptPwdSha'))
        if checked is False:
            raise ValueError('incorrect password or invalid password marker')
    else:
        if 'Encrypt' in text:
            raise ValueError('image already has a scramble marker')
        version, checked = 3, None
    if color == 0 and not (reverse and version == 1):
        raise ValueError('16-bit grayscale supports legacy v1 decoding only')
    pixels = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    channels = 3 if color == 2 else 4
    expected = (height, width) if color == 0 else (height, width, channels)
    if pixels is None or pixels.dtype != np.uint16 or pixels.shape != expected:
        raise ValueError('unsupported or invalid 16-bit PNG sample layout')
    # OpenCV uses BGR(A), but the same permutation of whole pixels commutes
    # with channel order, so there is no need to convert and risk precision.
    key = _digest(password)
    xs, ys = _indices(width, key, version), _indices(height, _digest(key), version)
    if version == 1:
        for x in range(width - 1, -1, -1):
            for y in range(height - 1, -1, -1):
                other_x, other_y = xs[x], ys[y]
                value = pixels[y, x].copy()
                pixels[y, x] = pixels[other_y, other_x]
                pixels[other_y, other_x] = value
    elif version == 2:
        for x in range(width - 1, -1, -1):
            value = pixels[:, x].copy()
            pixels[:, x] = pixels[:, xs[x]]
            pixels[:, xs[x]] = value
        for y in range(height - 1, -1, -1):
            value = pixels[y].copy()
            pixels[y] = pixels[ys[y]]
            pixels[ys[y]] = value
    elif reverse:
        restored = np.empty_like(pixels)
        restored[:, xs] = pixels
        pixels = restored
        restored = np.empty_like(pixels)
        restored[ys, :] = pixels
        pixels = restored
    else:
        pixels = pixels[ys, :, :][:, xs, :]
    ok, encoded = cv2.imencode('.png', pixels)
    if not ok:
        raise ValueError('could not encode 16-bit PNG')
    generated = _png_chunks(encoded.tobytes())
    image_data = [(kind, payload) for kind, payload in generated if kind == b'IDAT']
    result = bytearray(_PNG_SIGNATURE)
    wrote_data = False
    for kind, payload in chunks:
        if kind == b'IHDR':
            result.extend(_chunk(kind, generated[0][1]))
        elif kind == b'IDAT':
            if not wrote_data:
                if not reverse:
                    result.extend(_chunk(b'tEXt', b'Encrypt\0pixel_shuffle_3'))
                    result.extend(_chunk(b'tEXt', b'EncryptPwdSha\0' + password_marker(password).encode('ascii')))
                for new_kind, new_payload in image_data:
                    result.extend(_chunk(new_kind, new_payload))
                wrote_data = True
        elif kind in (b'tEXt', b'zTXt', b'iTXt') and payload.split(b'\0', 1)[0] in (b'Encrypt', b'EncryptPwdSha'):
            continue
        else:
            result.extend(_chunk(kind, payload))
    return bytes(result), checked


def encode_png(data: bytes, password: str) -> bytes:
    if _is_16bit_png(data):
        return _transform_16bit_png(data, password, False)[0]
    with _open_png(data) as image:
        text, extra = _metadata(image)
        if 'Encrypt' in text:
            raise ValueError('image already has a scramble marker')
        output = scramble(image, password)
        text['Encrypt'] = 'pixel_shuffle_3'
        text['EncryptPwdSha'] = password_marker(password)
        return _serialize_png(output, text, extra)


def decode_png(data: bytes, password: str) -> tuple[bytes, bool | None]:
    if _is_16bit_png(data):
        return _transform_16bit_png(data, password, True)
    with _open_png(data) as image:
        text, extra = _metadata(image)
        tag = text.pop('Encrypt', None)
        versions = {'pixel_shuffle': 1, 'pixel_shuffle_2': 2, 'pixel_shuffle_3': 3}
        if tag not in versions:
            raise ValueError('unsupported or missing pixel scramble marker')
        checked = verify_password(password, text.pop('EncryptPwdSha', None))
        if checked is False:
            raise ValueError('incorrect password or invalid password marker')
        output = unscramble(image, password, versions[tag])
        return _serialize_png(output, text, extra), checked
