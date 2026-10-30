import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REFERENCE = json.loads((Path(__file__).parent / 'fixtures/inpaint_difference_reference.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('case', REFERENCE['cases'])
def test_pinned_source_mask_and_colored_preview(case):
    from sdfx.inpaint_difference import compute
    base = Image.fromarray(np.asarray(REFERENCE['base'], dtype=np.uint8))
    altered = Image.fromarray(np.asarray(REFERENCE['altered'], dtype=np.uint8))
    mask, preview = compute(base, altered, threshold=case['threshold'], contours=case['contours'],
        dilation=case['dilation'], erosion=case['erosion'], blur=1, color='#8030ff', show_under=True)
    np.testing.assert_array_equal(np.asarray(mask), np.asarray(case['mask'], dtype=np.uint8))
    np.testing.assert_array_equal(np.asarray(preview), np.asarray(case['preview'], dtype=np.uint8))


@pytest.mark.parametrize('case', REFERENCE['edge_cases'], ids=lambda c: c['name'])
def test_pinned_source_edges_and_preview_options(case):
    from sdfx.inpaint_difference import compute
    base = Image.fromarray(np.asarray(case['base'], dtype=np.uint8))
    altered = Image.fromarray(np.asarray(case['altered'], dtype=np.uint8))
    mask, preview = compute(base, altered, threshold=case['threshold'], contours=case['contours'],
        dilation=case['dilation'], erosion=case['erosion'], blur=case['blur'], color='#8030ff',
        show_under=case['show_under'])
    np.testing.assert_array_equal(np.asarray(mask), np.asarray(case['mask'], dtype=np.uint8))
    np.testing.assert_array_equal(np.asarray(preview), np.asarray(case['preview'], dtype=np.uint8))


def test_invalid_settings_fail_and_inputs_are_unchanged():
    from sdfx.inpaint_difference import compute
    base = Image.new('RGB', (4, 4), 'black')
    altered = Image.new('RGB', (4, 4), 'white')
    before = (base.tobytes(), altered.tobytes())
    for kwargs in ({'threshold': -0.01}, {'threshold': float('nan')}, {'dilation': 101},
                   {'erosion': -1}, {'blur': -1}, {'color': '#invalid'}):
        with pytest.raises(ValueError):
            compute(base, altered, **kwargs)
    assert (base.tobytes(), altered.tobytes()) == before


def test_different_dimensions_and_grayscale_are_normalized():
    from sdfx.inpaint_difference import compute
    mask, preview = compute(Image.new('L', (2, 2), 0), Image.new('RGBA', (4, 3), (255, 255, 255, 255)))
    assert mask.mode == 'L' and mask.size == (4, 3)
    assert preview.mode == 'RGB' and preview.size == mask.size
    assert np.all(np.asarray(mask) == 255)


def test_exif_orientation_aligns_generation_image_and_mask(tmp_path):
    from sdfx.inpaint_difference import generation_request
    class Context:
        def check_interrupt(self): pass
    base = Image.new('RGB', (3, 2), 'black')
    altered = Image.new('RGB', (2, 3), 'black')
    altered.putpixel((0, 0), (255, 255, 255))
    exif = Image.Exif(); exif[274] = 6
    base_path, altered_path = tmp_path / 'base.png', tmp_path / 'rotated.png'
    base.save(base_path); altered.save(altered_path, exif=exif)
    request = {'params': {'img2img': {'init_image': str(altered_path)},
                          'hooks': {'inpaint_difference': {'base_path': str(base_path)}}}}
    with generation_request(request, 'img2img', Context()) as prepared:
        ip = prepared['params']['img2img']
        image = Image.open(ip['init_image']); mask = Image.open(ip['mask'])
        assert image.size == mask.size == (3, 2)
        assert image.getpixel((2, 0)) == (255, 255, 255)
        assert mask.getpixel((2, 0)) == 255 and mask.getpixel((0, 0)) == 0


def test_oversized_image_is_rejected_before_conversion():
    from sdfx.inpaint_difference import compute
    oversized = Image.new('L', (4097, 4097))
    with pytest.raises(ValueError, match='too large'):
        compute(oversized, Image.new('RGB', (1, 1)))


@pytest.mark.parametrize('invert', [False, True])
def test_effective_preview_matches_inpaint_mask_before_resize(invert):
    from sdfx.inpaint_difference import compute
    from sdfx.inpaint import prepare
    base = Image.new('RGB', (32, 32), 'black')
    altered = base.copy()
    from PIL import ImageDraw
    ImageDraw.Draw(altered).rectangle((8, 9, 17, 19), fill='white')
    mask, preview = compute(base, altered, blur=3, color='#ffffff', show_under=False,
                            invert=invert, preview_mode='effective')
    job = prepare(altered, mask, 32, 32, 0, 3, 1, False, 0, invert, 42)
    expected = np.asarray(job.overlay_mask)
    np.testing.assert_array_equal(np.asarray(preview), np.repeat(expected[:, :, None], 3, axis=2))


def test_dryrun_stdio_preview_returns_owned_mask_and_preserves_inputs(tmp_path):
    from test_server_dryrun import Engine
    base, altered = tmp_path / 'base.png', tmp_path / 'altered.png'
    Image.new('RGB', (8, 8), 'black').save(base)
    Image.new('RGB', (8, 8), 'white').save(altered)
    originals = (base.read_bytes(), altered.read_bytes())
    engine = Engine()
    try:
        hello = engine.next()
        assert 'inpaint_difference' in hello['features']
        response, _ = engine.call('inpaint_difference_preview', dict(base_path=str(base), altered_path=str(altered),
            output_dir=str(tmp_path / 'previews'), threshold=1., blur=0))
        assert response['type'] == 'result', response
        result = response['result']
        assert Path(result['mask_path']).parent == Path(result['artifact_dir'])
        assert Path(result['preview_path']).parent == Path(result['artifact_dir'])
        assert np.all(np.asarray(Image.open(result['mask_path'])) == 255)
        assert (base.read_bytes(), altered.read_bytes()) == originals
        invalid, _ = engine.call('inpaint_difference_preview', dict(base_path=str(base), altered_path=str(altered),
            output_dir=str(tmp_path / 'bad'), threshold=-1))
        assert invalid['type'] == 'error' and invalid['code'] == 'invalid_params'
        assert not list((tmp_path / 'bad').glob('*'))
    finally:
        engine.close()


def test_interrupted_preview_removes_only_its_own_partial_artifacts(tmp_path):
    from sdfx.inpaint_difference import preview_request
    base, altered = tmp_path / 'base.png', tmp_path / 'altered.png'
    Image.new('RGB', (8, 8), 'black').save(base)
    Image.new('RGB', (8, 8), 'white').save(altered)
    out = tmp_path / 'previews'
    out.mkdir()
    existing = out / 'keep.txt'
    existing.write_text('untouched')
    class Interrupted:
        calls = 0
        def check_interrupt(self):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError, match='interrupted'):
        preview_request(dict(base_path=str(base), altered_path=str(altered), output_dir=str(out)), Interrupted())
    assert list(out.iterdir()) == [existing]
    assert existing.read_text() == 'untouched'
    assert base.is_file() and altered.is_file()


def test_dryrun_generation_uses_altered_image_and_owned_computed_mask(tmp_path, monkeypatch):
    import io
    from sdfx.dryrun import FakeEngine
    from sdfx.protocol import Emitter, JobContext
    base, altered = tmp_path / 'base.png', tmp_path / 'altered.png'
    Image.new('RGB', (8, 8), 'black').save(base)
    Image.new('RGB', (8, 8), 'white').save(altered)
    originals = (base.read_bytes(), altered.read_bytes())
    engine = FakeEngine()
    parse = engine.parse_generation
    observed = {}
    def capture(params, mode, ctx):
        img = params['params']['img2img']
        observed.update(image=img['init_image'], mask=img['mask'])
        assert np.all(np.asarray(Image.open(img['mask'])) == 255)
        assert np.all(np.asarray(Image.open(img['init_image'])) == 255)
        return parse(params, mode, ctx)
    monkeypatch.setattr(engine, 'parse_generation', capture)
    monkeypatch.setattr(engine, '_run_pass', lambda *args: None)
    request = dict(output_dir=str(tmp_path / 'output'), params=dict(width=64, height=64, steps=1,
        img2img=dict(init_image=str(altered)), hooks=dict(inpaint_difference=dict(base_path=str(base)))))
    result = engine.img2img(request, JobContext(Emitter(io.BytesIO()), 'difference'))
    assert len(result['images']) == 1
    assert not Path(observed['mask']).exists() and not Path(observed['image']).exists()
    assert request['params']['img2img'] == {'init_image': str(altered)}
    assert (base.read_bytes(), altered.read_bytes()) == originals


def test_generation_body_errors_are_not_reclassified_and_owned_images_are_removed(tmp_path):
    from sdfx.inpaint_difference import generation_request
    class Context:
        def check_interrupt(self): pass
    base, altered = tmp_path / 'base.png', tmp_path / 'altered.png'
    Image.new('RGB', (8, 8), 'black').save(base)
    Image.new('RGB', (8, 8), 'white').save(altered)
    request = {'params': {'img2img': {'init_image': str(altered)},
                          'hooks': {'inpaint_difference': {'base_path': str(base)}}}}
    owned = None
    with pytest.raises(ValueError, match='downstream error'):
        with generation_request(request, 'img2img', Context()) as prepared:
            owned = Path(prepared['params']['img2img']['mask'])
            raise ValueError('downstream error')
    assert owned is not None and not owned.exists()


def test_disabled_generation_never_imports_optional_difference_module(tmp_path, monkeypatch):
    import builtins
    import io
    from sdfx.dryrun import FakeEngine
    from sdfx.protocol import Emitter, JobContext
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.endswith('inpaint_difference'):
            raise AssertionError('disabled generation imported optional dependency')
        return original_import(name, *args, **kwargs)
    engine = FakeEngine()
    monkeypatch.setattr(engine, '_run_pass', lambda *args: None)
    monkeypatch.setattr(builtins, '__import__', guarded)
    result = engine.txt2img(dict(output_dir=str(tmp_path), params=dict(width=64, height=64, steps=1)),
                            JobContext(Emitter(io.BytesIO()), 'plain'))
    assert len(result['images']) == 1


def test_missing_opencv_preview_and_generation_report_unsupported(tmp_path, monkeypatch):
    import builtins
    import io
    import sys
    from sdfx.dryrun import FakeEngine
    from sdfx.protocol import Emitter, EngineError, JobContext
    module = sys.modules.pop('sdfx.inpaint_difference', None)
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == 'cv2':
            raise ImportError('opencv missing')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', blocked)
    ctx = JobContext(Emitter(io.BytesIO()), 'missing')
    try:
        with pytest.raises(EngineError) as preview:
            FakeEngine().inpaint_difference_preview({}, ctx)
        assert preview.value.code == 'unsupported'
        with pytest.raises(EngineError) as generation:
            FakeEngine().img2img({'params': {'hooks': {'inpaint_difference': {}}, 'img2img': {}}}, ctx)
        assert generation.value.code == 'unsupported'
    finally:
        if module is not None:
            sys.modules['sdfx.inpaint_difference'] = module
