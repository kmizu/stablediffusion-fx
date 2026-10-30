"""Actual generation control flow with only expensive model calls substituted."""
import copy
from types import SimpleNamespace
import pytest
from PIL import Image
from test_hires_repeat_engine import harness, context
from sdfx.protocol import EngineError, INTERRUPTED


def request_for(harness):
    engine, request, calls = harness
    request['params'].update(hires={'enabled': False}, hooks={'img2img_hires_fix': {
        'upscaler': 'Lanczos', 'ratio': 2, 'steps': 5, 'denoise': .6}})
    return engine, request, calls


def test_post_image_refinement_all_items_once_with_independent_seeds(harness):
    engine, request, calls = request_for(harness)
    result = engine.txt2img(request, context())
    runs = [c for c in calls if isinstance(c, dict)]
    assert [r['pass_'] for r in runs] == ['base', 'img2img_hires', 'img2img_hires'] * 2
    assert [r['seeds'] for r in runs if r['pass_'] == 'img2img_hires'] == [[10], [11], [12], [13]]
    assert [(im['width'], im['height']) for im in result['images']] == [(128, 128)] * 4
    assert all(r['steps'] == 5 and r['expected_steps'] == 3 for r in runs if r['pass_'] == 'img2img_hires')


def test_zero_denoise_upscales_without_refinement(harness):
    engine, request, calls = request_for(harness)
    request['params']['hooks']['img2img_hires_fix']['denoise'] = 0
    result = engine.txt2img(request, context())
    assert [r['pass_'] for r in calls if isinstance(r, dict)] == ['base', 'base']
    assert [(im['width'], im['height']) for im in result['images']] == [(128, 128)] * 4


def test_second_item_cancel_preserves_first_completed_output(harness, monkeypatch):
    engine, request, calls = request_for(harness)
    original = engine._run
    def run(*args, **kw):
        if kw['pass_'] == 'img2img_hires' and kw['image_idx'] == 1:
            raise EngineError(INTERRUPTED, 'cancel second item')
        return original(*args, **kw)
    monkeypatch.setattr(engine, '_run', run)
    result = engine.txt2img(request, context())
    assert result['interrupted']
    assert [im['seed'] for im in result['images']] == [10]
    with Image.open(result['images'][0]['path']) as image:
        assert image.size == (128, 128)


@pytest.mark.parametrize('masked', [False, True])
def test_img2img_and_inpaint_refine_completed_image(harness, tmp_path, masked):
    engine, request, calls = request_for(harness)
    source = tmp_path / 'source.png'
    Image.new('RGB', (64, 64), 'red').save(source)
    request['params']['img2img'] = {'init_image': str(source), 'denoising_strength': .5}
    if masked:
        mask = tmp_path / 'mask.png'
        Image.new('L', (64, 64), 255).save(mask)
        request['params']['img2img']['mask'] = str(mask)
    result = engine.img2img(request, context())
    assert len(result['images']) == 4
    assert all(im['width'] == 128 for im in result['images'])
    assert len([r for r in calls if isinstance(r, dict) and r['pass_'] == 'img2img_hires']) == 4


def test_override_adapters_are_active_only_during_refinement_and_restore_on_error(harness, monkeypatch, tmp_path):
    engine, request, calls = request_for(harness)
    path = tmp_path / 'ref.safetensors'
    path.write_bytes(b'fixture')
    request['img2img_hires_loras'] = [{'name': 'ref', 'weight': .4}]
    request['lora_paths'] = {'ref': str(path)}
    active = []
    def apply(gp, req, ctx):
        active.append([l.name for l in gp.loras])
        return {}
    monkeypatch.setattr(engine, '_apply_loras', apply)
    original = engine._run
    def run(*args, **kw):
        if kw['pass_'] == 'img2img_hires':
            assert active[-1] == ['ref']
            raise EngineError('internal', 'sampling failure')
        return original(*args, **kw)
    monkeypatch.setattr(engine, '_run', run)
    with pytest.raises(EngineError, match='sampling failure'):
        engine.txt2img(request, context())
    assert active[-1] == []


def test_native_latent_repeats_stay_latent_before_independent_pixel_refinement(harness, monkeypatch):
    import torch
    engine, request, calls = harness
    request['params']['hires']['upscaler'] = 'Latent'
    request['params']['hooks']['img2img_hires_fix'] = {'upscaler': 'Lanczos'}
    original = engine._run
    def run(*args, **kw):
        output = original(*args, **kw)
        if kw['pass_'] == 'base':
            assert kw['output_type'] == 'latent'
            return torch.ones(2, 4, 8, 8), output[1]
        if kw['pass_'] == 'hires':
            assert isinstance(kw['image'], torch.Tensor)
            assert kw['image'].shape == (2, 4, 16, 16)
        else:
            assert isinstance(kw['image'][0], Image.Image)
            assert kw['image'][0].size == (256, 256)
        return output
    monkeypatch.setattr(engine, '_run', run)
    result = engine.txt2img(request, context())
    assert len(result['images']) == 12
    assert all(im['width'] == 256 for im in result['images'])


def test_disabled_feature_preserves_native_output(harness):
    engine, request, calls = harness
    ordinary = engine.txt2img(request, context())
    before = [Image.open(im['path']).tobytes() for im in ordinary['images']]
    request['params']['hooks']['img2img_hires_fix'] = {'enabled': False, 'ratio': 0}
    disabled = engine.txt2img(request, context())
    assert before == [Image.open(im['path']).tobytes() for im in disabled['images']]
    assert not any(r['pass_'] == 'img2img_hires' for r in calls if isinstance(r, dict))


def test_refinement_has_independent_clock_even_after_native_hires(harness, monkeypatch):
    engine, request, calls = harness
    request['params']['hooks']['img2img_hires_fix'] = {'upscaler': 'Lanczos', 'steps': 5}
    original = engine._run
    def run(task, gp, *args, **kw):
        if kw['pass_'] == 'img2img_hires':
            from sdfx.clora.runtime import clock_total
            assert clock_total(gp, {'hires_global_step': True}, kw['expected_steps']) == kw['expected_steps']
        return original(task, gp, *args, **kw)
    monkeypatch.setattr(engine, '_run', run)
    engine.txt2img(request, context())
