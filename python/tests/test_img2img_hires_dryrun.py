import io
import pytest
from sdfx.dryrun import FakeEngine
from sdfx.protocol import Emitter, JobContext, EngineError


def test_dryrun_declares_feature_and_refines_each_item(tmp_path, monkeypatch):
    engine = FakeEngine()
    assert 'img2img_hires_fix' in engine.features()
    passes = []
    monkeypatch.setattr(engine, '_run_pass', lambda ctx, steps, kind, idx, total, *args:
        passes.append((kind, idx, total)))
    result = engine.txt2img(dict(output_dir=str(tmp_path), params=dict(seed=7, width=64, height=64,
        batch_size=2, hooks={'img2img_hires_fix': {'upscaler': 'Lanczos', 'ratio': 2}})),
        JobContext(Emitter(io.BytesIO()), 'test'))
    assert [(i['width'], i['height']) for i in result['images']] == [(128, 128)] * 2
    assert passes == [('base', 0, 2), ('img2img_hires', 0, 2), ('img2img_hires', 1, 2)]


def test_invalid_refinement_rejected_before_loading(tmp_path, monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(engine, 'ensure_model', lambda *a: pytest.fail('model touched'))
    with pytest.raises(EngineError):
        engine.txt2img(dict(output_dir=str(tmp_path), params=dict(hooks={
            'img2img_hires_fix': {'ratio': 0, 'width': 512}})), JobContext(Emitter(io.BytesIO()), 'test'))


def test_composed_native_and_refinement_size_rejected_before_loading(tmp_path, monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(engine, 'ensure_model', lambda *a: pytest.fail('model touched'))
    with pytest.raises(EngineError, match='16384'):
        engine.txt2img(dict(output_dir=str(tmp_path), params=dict(width=4096, height=64,
            hires={'enabled': True, 'scale': 2}, hooks={
            'img2img_hires_fix': {'ratio': 4, 'upscaler': 'Lanczos'}})), JobContext(Emitter(io.BytesIO()), 'test'))


def test_cancel_second_refinement_returns_first_completed_image(tmp_path, monkeypatch):
    engine = FakeEngine()
    def run(ctx, steps, kind, idx, total, *args):
        if kind == 'img2img_hires' and idx == 1:
            raise EngineError('interrupted', 'cancel')
    monkeypatch.setattr(engine, '_run_pass', run)
    result = engine.txt2img(dict(output_dir=str(tmp_path), params=dict(seed=7, width=64, height=64,
        batch_size=2, hooks={'img2img_hires_fix': {'upscaler': 'Lanczos'}})),
        JobContext(Emitter(io.BytesIO()), 'test'))
    assert result['interrupted'] and len(result['images']) == 1
