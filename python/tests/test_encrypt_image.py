import json
from pathlib import Path

import pytest
from PIL import Image

from sdfx import encrypt_image

CASES = json.loads((Path(__file__).parent / 'fixtures/encrypt_image_reference.json').read_text(encoding='utf-8'))['cases']

@pytest.mark.parametrize('case', CASES, ids=lambda c: f"v{c['version']}-{c['width']}x{c['height']}-{c['mode']}")
def test_upstream_pixel_compatibility(case):
    size = (case['width'], case['height'])
    plain = Image.frombytes(case['mode'], size, bytes.fromhex(case['plain']))
    scrambled = Image.frombytes(case['mode'], size, bytes.fromhex(case['scrambled']))
    result = encrypt_image.unscramble(scrambled, case['password'], case['version'])
    assert result.tobytes() == plain.tobytes()
    assert scrambled.tobytes() == bytes.fromhex(case['scrambled'])
    if case['version'] == 3:
        assert encrypt_image.scramble(plain, case['password']).tobytes() == scrambled.tobytes()
        assert plain.tobytes() == bytes.fromhex(case['plain'])


def test_password_verifier_and_unsupported_inputs():
    import hashlib
    password = '秘密🔑'
    h = hashlib.sha256(password.encode()).hexdigest()
    marker = hashlib.sha256((h + 'Encrypt').encode()).hexdigest()
    assert encrypt_image.password_marker(password) == marker
    assert encrypt_image.verify_password(password, marker) is True
    assert encrypt_image.verify_password('wrong', marker) is False
    assert encrypt_image.verify_password('anything', None) is None
    with pytest.raises(ValueError):
        encrypt_image.unscramble(Image.new('RGB', (2, 3)), 'x', 4)
    with pytest.raises(ValueError):
        encrypt_image.scramble(Image.new('L', (2, 3)), 'x')
    with pytest.raises(ValueError):
        encrypt_image.scramble(Image.new('RGB', (2, 3)), '')


def test_png_bytes_roundtrip_preserves_text_and_verifies_password():
    import io
    from PIL import PngImagePlugin
    image = Image.new('RGBA', (3, 2), (32, 44, 128, 55))
    info = PngImagePlugin.PngInfo()
    info.add_text('parameters', 'prompt: 猫')
    buffer = io.BytesIO()
    image.save(buffer, format='PNG', pnginfo=info)
    original = buffer.getvalue()
    protected = encrypt_image.encode_png(original, '秘密')
    with Image.open(io.BytesIO(protected)) as encoded:
        assert encoded.info['Encrypt'] == 'pixel_shuffle_3'
        assert encoded.info['parameters'] == 'prompt: 猫'
        assert encoded.info['EncryptPwdSha'] == encrypt_image.password_marker('秘密')
    restored, checked = encrypt_image.decode_png(protected, '秘密')
    assert checked is True
    with Image.open(io.BytesIO(restored)) as result:
        assert result.tobytes() == image.tobytes()
        assert result.info['parameters'] == 'prompt: 猫'
        assert 'Encrypt' not in result.info
        assert 'EncryptPwdSha' not in result.info
    with pytest.raises(ValueError, match='password'):
        encrypt_image.decode_png(protected, 'wrong')


def test_unverified_legacy_marker_and_malformed_marker_are_explicit():
    import io
    from PIL import PngImagePlugin
    case = next(c for c in CASES if c['version'] == 2 and c['width'] == 2)
    image = Image.frombytes(case['mode'], (case['width'], case['height']), bytes.fromhex(case['scrambled']))
    info = PngImagePlugin.PngInfo()
    info.add_text('Encrypt', 'pixel_shuffle_2')
    out = io.BytesIO()
    image.save(out, format='PNG', pnginfo=info)
    restored, checked = encrypt_image.decode_png(out.getvalue(), case['password'])
    assert checked is None
    with Image.open(io.BytesIO(restored)) as result:
        assert result.tobytes() == bytes.fromhex(case['plain'])
    assert encrypt_image.verify_password('x', '💥') is False
    with pytest.raises(ValueError):
        encrypt_image.decode_png(b'not png', 'x')


@pytest.mark.parametrize("mode", ["L", "P"])
def test_upstream_v1_single_channel_png_decoding(mode):
    # Generated with pinned scripts/core/core.py encrypt_image and sha256("secret").
    import io
    from PIL import PngImagePlugin
    scrambled = Image.frombytes(mode, (4, 3), bytes.fromhex("090a0003050604070102080b"))
    if mode == "P":
        scrambled.putpalette([v for i in range(256) for v in (i, 255 - i, (i * 7) % 256)])
        scrambled.info["transparency"] = bytes(range(256))
    else:
        scrambled.info["transparency"] = 9
    info = PngImagePlugin.PngInfo()
    info.add_text("Encrypt", "pixel_shuffle")
    buffer = io.BytesIO()
    scrambled.save(buffer, format="PNG", pnginfo=info)
    restored_bytes, verified = encrypt_image.decode_png(buffer.getvalue(), "secret")
    assert verified is None
    with Image.open(io.BytesIO(restored_bytes)) as restored:
        assert restored.mode == mode
        assert restored.tobytes() == bytes(range(12))
        assert restored.info["transparency"] == scrambled.info["transparency"]
        if mode == "P":
            assert restored.getpalette() == scrambled.getpalette()
    for version in (2, 3):
        with pytest.raises(ValueError):
            encrypt_image.unscramble(scrambled, "secret", version)


@pytest.mark.parametrize("color_type,channels", [(2, 3), (6, 4)])
def test_16_bit_rgb_roundtrip_and_crc(color_type, channels):
    import cv2
    import numpy as np
    import struct
    import zlib

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    ihdr = struct.pack(">IIBBBBB", 2, 2, 16, color_type, 0, 0, 0)
    rows = []
    for y in range(2):
        rows.append(b"\x00" + b"".join(struct.pack(">H", 1000 + (y * 2 * channels + i) * 301)
                                         for i in range(2 * channels)))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) +
           chunk(b"tEXt", b"parameters\0test") + chunk(b"IDAT", zlib.compress(b"".join(rows))) +
           chunk(b"IEND", b""))
    def samples(value):
        return cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    original = samples(png)
    assert original.dtype == np.uint16
    protected = encrypt_image.encode_png(png, "secret")
    assert not np.array_equal(samples(protected), original)
    with pytest.raises(ValueError, match="password"):
        encrypt_image.decode_png(protected, "wrong")
    restored, verified = encrypt_image.decode_png(protected, "secret")
    assert verified is True
    assert np.array_equal(samples(restored), original)
    with Image.open(__import__('io').BytesIO(restored)) as image:
        assert image.text["parameters"] == "test"
        assert "Encrypt" not in image.text
    corrupt = bytearray(protected)
    corrupt[29] ^= 1
    with pytest.raises(ValueError, match="CRC"):
        encrypt_image.decode_png(bytes(corrupt), "secret")


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"16bit-v{c['version']}-{c['width']}x{c['height']}-{c['mode']}")
def test_16_bit_matches_upstream_permutation_fixtures(case):
    import cv2
    import numpy as np
    import struct
    import zlib

    channels = 4 if case["mode"] == "RGBA" else 3
    width, height = case["width"], case["height"]
    def to_array(hex_data):
        eight = np.frombuffer(bytes.fromhex(hex_data), dtype=np.uint8).reshape(height, width, channels)
        return eight.astype(np.uint16) * 257 + 1
    def png(array, marker=None):
        def chunk(kind, value):
            return struct.pack(">I", len(value)) + kind + value + struct.pack(">I", zlib.crc32(kind + value))
        raw = b"".join(b"\0" + row.astype(">u2").tobytes() for row in array)
        header = struct.pack(">IIBBBBB", width, height, 16, 6 if channels == 4 else 2, 0, 0, 0)
        text = chunk(b"tEXt", b"Encrypt\0" + marker.encode()) if marker else b""
        return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + text + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    def read(value):
        bgr = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        return bgr[:, :, [2, 1, 0, 3] if channels == 4 else [2, 1, 0]]
    plain, scrambled = to_array(case["plain"]), to_array(case["scrambled"])
    version = case["version"]
    marker = "pixel_shuffle" if version == 1 else f"pixel_shuffle_{version}"
    restored, verified = encrypt_image.decode_png(png(scrambled, marker), case["password"])
    assert verified is None
    assert np.array_equal(read(restored), plain)
    if version == 3:
        assert np.array_equal(read(encrypt_image.encode_png(png(plain), case["password"])), scrambled)


@pytest.mark.parametrize("version,encoded", [
    (1, "ab66be5500ff39cc5faa72994cbb858813ee26dd9877d144"),
    (2, "13ee26dd00ff39cc5faa72994cbb8588ab66be559877d144"),
    (3, "be559877d144ab6626dd00ff39cc13ee72994cbb85885faa"),
])
def test_upstream_la_fixtures(version, encoded):
    import io
    from PIL import PngImagePlugin
    expected = bytes(x for i in range(12) for x in (i * 19, 255 - i * 17))
    scrambled = Image.frombytes("LA", (4, 3), bytes.fromhex(encoded))
    info = PngImagePlugin.PngInfo()
    info.add_text("Encrypt", "pixel_shuffle" if version == 1 else f"pixel_shuffle_{version}")
    info.add_text("parameters", "alpha fixture")
    stream = io.BytesIO()
    scrambled.save(stream, format="PNG", pnginfo=info)
    restored, verified = encrypt_image.decode_png(stream.getvalue(), "secret")
    assert verified is None
    with Image.open(io.BytesIO(restored)) as image:
        assert image.mode == "LA"
        assert image.tobytes() == expected
        assert image.text["parameters"] == "alpha fixture"
    if version == 3:
        assert encrypt_image.scramble(Image.frombytes("LA", (4, 3), expected), "secret").tobytes() == bytes.fromhex(encoded)


def test_upstream_v1_16bit_grayscale_png_and_metadata():
    import io
    from PIL import PngImagePlugin
    image = Image.new("I;16", (4, 3))
    image.putdata([45000, 50000, 0, 15000, 25000, 30000, 20000, 35000, 5000, 10000, 40000, 55000])
    info = PngImagePlugin.PngInfo()
    info.add_text("Encrypt", "pixel_shuffle")
    info.add_text("parameters", "gray fixture")
    stream = io.BytesIO()
    image.save(stream, format="PNG", pnginfo=info, dpi=(144, 72))
    restored, verified = encrypt_image.decode_png(stream.getvalue(), "secret")
    assert verified is None
    with Image.open(io.BytesIO(restored)) as decoded:
        assert list(decoded.getdata()) == [i * 5000 for i in range(12)]
        assert decoded.text["parameters"] == "gray fixture"
        assert decoded.info["dpi"] == pytest.approx((144, 72), abs=.02)
    with pytest.raises(ValueError, match="legacy v1"):
        encrypt_image.encode_png(restored, "secret")
