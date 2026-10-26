"""CPU checks for the acceptance harness's scope and pixel comparison contract."""
import importlib.util
from pathlib import Path
import pytest


def harness():
    path = Path(__file__).resolve().parents[2] / 'scripts/dev/prompt_fusion_check.py'
    spec = importlib.util.spec_from_file_location('prompt_fusion_check', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_acceptance_covers_source_language_and_native_sampling_paths():
    cases = harness().build_cases(False)
    names = {c['name'] for c in cases}
    assert {'baseline', 'ordinary-enabled', 'ordinary-disabled', 'variables-functions',
            'linear', 'bezier', 'catmull', 'weighted-mean', 'alternation', 'contextual-attention',
            'slerp-negative-origin', 'slerp-empty-origin', 'negative-curve', 'long-conditioning',
            'batch', 'heun', 'hires', 'img2img', 'restored'} <= names
    for family in (False, True):
        cases = harness().build_cases(family)
        baseline = next(c for c in cases if c['name'] == 'baseline')
        assert baseline['params']['width'] == (768 if family else 512)
        for name in ('ordinary-enabled', 'ordinary-disabled', 'variables-functions', 'constant-curve', 'restored'):
            assert next(c for c in cases if c['name'] == name)['equal_to'] == 'baseline'


def test_guard_cases_cover_main_and_override_inverse_routes_and_numeric_errors():
    guards = harness().build_guards('input.png')
    assert {'photomaker', 'negpip', 'alternative-main', 'alternative-override',
            'nonfinite-position', 'nonfinite-attention'} <= {c['name'] for c in guards}
    assert all(c['expected_code'] in ('unsupported', 'invalid_params') for c in guards)


def test_pixel_identity_ignores_png_metadata_and_reports_real_pixel_difference(tmp_path):
    image = pytest.importorskip('PIL.Image')
    png = pytest.importorskip('PIL.PngImagePlugin')
    original, same, different = [tmp_path / f'{name}.png' for name in ('original', 'same', 'different')]
    image.new('RGB', (8, 8), (10, 20, 30)).save(original)
    metadata = png.PngInfo()
    metadata.add_text('parameters', 'different metadata')
    image.new('RGB', (8, 8), (10, 20, 30)).save(same, pnginfo=metadata)
    image.new('RGB', (8, 8), (11, 21, 31)).save(different)
    module = harness()
    assert module.pixel_record(original)['pixel_sha256'] == module.pixel_record(same)['pixel_sha256']
    assert module.image_difference(original, same) == {'exact': True, 'mean_absolute_difference': 0.0}
    assert module.image_difference(original, different) == {'exact': False, 'mean_absolute_difference': 1.0}
