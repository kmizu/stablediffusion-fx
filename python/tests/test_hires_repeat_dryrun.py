import io
import pytest
from sdfx.dryrun import FakeEngine
from sdfx.protocol import Emitter, JobContext, EngineError


def test_dryrun_repeat_seed_order_and_progress(tmp_path, monkeypatch):
    engine = FakeEngine()
    passes = []
    monkeypatch.setattr(engine, '_run_pass', lambda ctx, steps, kind, idx, total, *args:
        passes.append((kind, idx, total)))
    result = engine.txt2img(dict(output_dir=str(tmp_path), params=dict(seed=10, subseed=20,
        width=64, height=64, batch_size=2, n_iter=2, hires=dict(enabled=True),
        hooks=dict(hires_fix_tweaks=dict(batch_count=2, seed_enabled=True, seed=100)))),
        JobContext(Emitter(io.BytesIO()), 'test'))
    assert passes == [('base', 0, 8), ('hires', 0, 8), ('hires', 2, 8),
                      ('base', 4, 8), ('hires', 4, 8), ('hires', 6, 8)]
    assert [x['seed'] for x in result['images']] == [10, 11, 10, 11, 12, 13, 12, 13]
    assert [x['hires_seed'] for x in result['images']] == [100, 101, 101, 102, 102, 103, 103, 104]


def test_large_seed_output_is_lossless_for_json_double_clients(tmp_path, monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(engine, '_run_pass', lambda *a: None)
    result = engine.txt2img(dict(output_dir=str(tmp_path), params=dict(seed=10, subseed=20, width=64, height=64,
        hires=dict(enabled=True), hooks=dict(hires_fix_tweaks=dict(seed_enabled=True, seed='9007199254740993')))),
        JobContext(Emitter(io.BytesIO()), 'test'))
    assert result['images'][0]['hires_seed'] == '9007199254740993'


def test_dryrun_rejects_excess_before_model_or_generation(tmp_path, monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(engine, 'ensure_model', lambda *a: pytest.fail('model touched'))
    with pytest.raises(EngineError, match='1600-image'):
        engine.txt2img(dict(output_dir=str(tmp_path), params=dict(batch_size=16, n_iter=100,
            hires=dict(enabled=True), hooks=dict(hires_fix_tweaks=dict(batch_count=2)))),
            JobContext(Emitter(io.BytesIO()), 'test'))


def test_dryrun_returns_completed_images_with_interrupted_flag(tmp_path, monkeypatch):
    engine = FakeEngine()
    def run(ctx, steps, kind, idx, total, *args):
        if kind == 'hires' and idx == 1:
            raise EngineError('interrupted', 'fixture')
    monkeypatch.setattr(engine, '_run_pass', run)
    result = engine.txt2img(dict(output_dir=str(tmp_path), params=dict(seed=10, width=64, height=64,
        hires=dict(enabled=True), hooks=dict(hires_fix_tweaks=dict(batch_count=2)))),
        JobContext(Emitter(io.BytesIO()), 'test'))
    assert result['interrupted'] is True
    assert len(result['images']) == 1
