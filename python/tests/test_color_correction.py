import numpy as np
import pytest
from PIL import Image
from sdfx.color_correction import adain


@pytest.mark.parametrize('method', ['a1111', 'adain', 'wavelet'])
def test_all_algorithms_single_pixel_and_validation(method):
    from sdfx import color_correction
    transform = getattr(color_correction, method)
    source = Image.new('RGB', (1, 1), (30, 80, 120))
    reference = Image.new('RGB', (1, 1), (100, 150, 200))
    result = transform(source, reference)
    assert result.size == (1, 1)
    assert np.isfinite(np.asarray(result)).all()
    with pytest.raises(ValueError, match='strength'):
        transform(source, reference, -0.1)
    with pytest.raises(ValueError, match='strength'):
        transform(source, reference, float('inf'))


def test_a1111_zero_strength_and_invalid_strength():
    from sdfx.color_correction import a1111
    source = Image.new('RGB', (5, 4), (20, 70, 90))
    np.testing.assert_array_equal(a1111(source, source, 0), source)
    with pytest.raises(ValueError, match='strength'):
        a1111(source, source, float('nan'))


def test_all_methods_comparison_uses_same_target(tmp_path):
    from sdfx.image_tools import run
    source = Image.fromarray(np.random.default_rng(51).integers(20, 220, (8, 9, 3), dtype=np.uint8))
    path = tmp_path / 'reference.png'
    Image.new('RGB', (4, 3), (110, 120, 130)).save(path)
    params = {'method': 'A1111', 'reference_path': str(path), 'all_methods': True}
    result, extra = run('color_correction', source, params)
    for method in ('AdaIN', 'Wavelet'):
        separate, _ = run('color_correction', source, {**params, 'method': method, 'all_methods': False})
        np.testing.assert_array_equal(extra[method], separate)
    assert result.size == source.size
    assert extra['method'] == 'A1111'


def test_wavelet_constant_color_transfer_and_identity():
    from sdfx.color_correction import wavelet
    source = Image.new('RGB', (7, 9), (30, 40, 50))
    reference = Image.new('RGB', (3, 2), (100, 110, 120))
    assert wavelet(source, reference).getpixel((0, 0)) == (100, 110, 120)
    np.testing.assert_array_equal(wavelet(source, source), source)
    np.testing.assert_array_equal(wavelet(source, reference, 0), source)


def test_wavelet_matches_direct_dilated_kernel_and_dispatch(tmp_path):
    from sdfx.image_tools import run
    rng = np.random.default_rng(78)
    source = rng.integers(50, 190, (9, 11, 3), dtype=np.uint8)
    reference = rng.integers(50, 190, (9, 11, 3), dtype=np.uint8)

    def direct_low(array):
        array = array.astype(float)
        for radius in (1, 2, 4, 8, 16):
            padded = np.pad(array, ((radius, radius), (radius, radius), (0, 0)), mode='edge')
            result = np.zeros_like(array)
            for y, wy in enumerate((1, 2, 1)):
                for x, wx in enumerate((1, 2, 1)):
                    result += padded[y * radius:y * radius + 9, x * radius:x * radius + 11] * (wy * wx / 16)
            array = result
        return array

    path = tmp_path / 'reference.png'
    Image.fromarray(reference).save(path)
    result, info = run('color_correction', Image.fromarray(source),
                       {'method': 'Wavelet', 'reference_path': str(path)})
    expected = np.rint(np.clip(source.astype(float) - direct_low(source) + direct_low(reference), 0, 255)).astype(np.uint8)
    np.testing.assert_array_equal(result, expected)
    assert info['method'] == 'Wavelet'


def test_image_tool_adain_dispatch_and_reference_lifetime(tmp_path):
    from sdfx.image_tools import run
    path = tmp_path / 'reference.png'
    Image.new('RGB', (3, 4), (90, 100, 110)).save(path)
    result, info = run('color_correction', Image.new('RGB', (7, 8)),
                       {'method': 'AdaIN', 'reference_path': str(path)})
    assert result.getpixel((0, 0)) == (90, 100, 110)
    assert info['method'] == 'AdaIN'
    path.unlink()  # no open reference handle after processing


def test_image_tool_rejects_unknown_correction_method(tmp_path):
    from sdfx.image_tools import run
    with pytest.raises(ValueError, match='method'):
        run('color_correction', Image.new('RGB', (2, 2)), {'method': 'bogus'})


def test_image_tool_requires_explicit_original_fallback():
    from sdfx.image_tools import run
    with pytest.raises(ValueError, match='reference_path'):
        run('color_correction', Image.new('RGB', (2, 2)), {})


def test_image_tool_reports_unreadable_reference(tmp_path):
    from sdfx.image_tools import run
    path = tmp_path / 'bad.png'
    path.write_text('not an image')
    with pytest.raises(ValueError, match='cannot read color reference'):
        run('color_correction', Image.new('RGB', (2, 2)), {'reference_path': str(path)})
    path.unlink()


def test_engine_request_persists_all_methods(tmp_path):
    from sdfx.base import BaseEngine
    source = tmp_path / 'source.png'
    reference = tmp_path / 'reference.png'
    Image.new('RGB', (9, 7), (50, 80, 110)).save(source)
    Image.new('RGB', (4, 3), (100, 130, 160)).save(reference)
    result = BaseEngine().image_tool({'tool': 'color_correction', 'method': 'A1111',
        'image_path': str(source), 'reference_path': str(reference),
        'all_methods': True, 'output_dir': str(tmp_path / 'out')}, None)
    assert set(result['outputs']) == {'AdaIN', 'Wavelet'}
    for path in [result['path'], *result['outputs'].values()]:
        with Image.open(path) as output:
            assert output.size == (9, 7)
    assert result['info']['method'] == 'A1111'


def test_engine_request_classifies_corrupt_reference_as_invalid_params(tmp_path):
    from sdfx.base import BaseEngine
    from sdfx.protocol import EngineError, INVALID_PARAMS
    source = tmp_path / 'source.png'
    reference = tmp_path / 'bad.png'
    Image.new('RGB', (3, 4)).save(source)
    reference.write_bytes(b'broken')
    with pytest.raises(EngineError) as caught:
        BaseEngine().image_tool({'tool': 'color_correction', 'method': 'AdaIN',
            'image_path': str(source), 'reference_path': str(reference)}, None)
    assert caught.value.code == INVALID_PARAMS


def test_adain_constant_reference_and_different_sizes():
    source = Image.fromarray(np.tile(np.arange(32, dtype=np.uint8), (24, 1))[..., None].repeat(3, 2))
    reference = Image.new('RGB', (9, 7), (40, 90, 160))
    result = adain(source, reference)
    assert result.size == source.size
    # Variance epsilon retains a small amount of source detail even for a flat reference.
    values = np.asarray(source).astype(float) / 255
    expected = (values - values.mean((0, 1))) / np.sqrt(values.var((0, 1), ddof=1) + 1e-5)
    expected = expected * np.sqrt(1e-5) + np.array([40, 90, 160]) / 255
    np.testing.assert_array_equal(result, np.clip(expected * 255 + 1e-10, 0, 255).astype(np.uint8))


def test_adain_zero_strength_preserves_pixels():
    source = Image.fromarray(np.random.default_rng(3).integers(0, 256, (13, 17, 3), dtype=np.uint8))
    np.testing.assert_array_equal(adain(source, Image.new('RGB', (2, 2)), 0), source)


def test_adain_transfers_nonconstant_channel_statistics():
    ramp = np.arange(64, dtype=np.uint8).reshape(8, 8)
    source = Image.fromarray(np.stack([ramp + 30, ramp + 50, ramp + 70], axis=2))
    reference = Image.fromarray(np.stack([ramp * 2 + 40, ramp + 90, ramp + 110], axis=2))
    result = np.asarray(adain(source, reference)).astype(float)
    expected = np.asarray(reference).astype(float)
    np.testing.assert_allclose(result.mean((0, 1)), expected.mean((0, 1)), atol=0.5)
    np.testing.assert_allclose(result.std((0, 1)), expected.std((0, 1)), atol=0.5)
    halfway = np.asarray(adain(source, reference, 0.5)).astype(float)
    np.testing.assert_allclose(halfway, (np.asarray(source).astype(float) + result) / 2, atol=0.5)


def test_adain_uses_normalized_sample_variance_with_epsilon():
    source = np.array([[[100, 100, 100], [101, 101, 101]]], dtype=np.uint8)
    reference = np.array([[[80, 80, 80], [160, 160, 160]]], dtype=np.uint8)
    a, b = source.astype(float) / 255, reference.astype(float) / 255
    expected = (a - a.mean((0, 1))) / np.sqrt(a.var((0, 1), ddof=1) + 1e-5)
    expected = expected * np.sqrt(b.var((0, 1), ddof=1) + 1e-5) + b.mean((0, 1))
    expected = np.clip(expected * 255, 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(adain(Image.fromarray(source), Image.fromarray(reference)), expected)


@pytest.mark.parametrize('strength', [-1, 1.1, float('nan'), float('inf')])
def test_adain_rejects_invalid_strength(strength):
    with pytest.raises(ValueError, match='strength'):
        adain(Image.new('RGB', (2, 2)), Image.new('RGB', (2, 2)), strength)


def test_adain_identity_and_constant_source_are_finite():
    source = Image.new('RGB', (5, 6), (20, 30, 40))
    np.testing.assert_array_equal(adain(source, source), source)
    result = adain(source, Image.new('RGB', (3, 2), (100, 120, 140)))
    assert np.isfinite(np.asarray(result)).all()
    assert result.getpixel((0, 0)) == (100, 120, 140)
