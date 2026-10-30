import pytest
from sdfx.vram_estimator import normalize, cases
from sdfx.protocol import EngineError


def test_four_mode_sweep_includes_cap_and_actual_batch_sizes():
    plan = cases(normalize({'max_size': 320, 'max_batch': 2}))
    assert len(plan) == 16
    assert {item['mode'] for item in plan} == {'txt2img', 'txt2img_hr', 'txt2img_hr_latent', 'img2img'}
    assert [(item['target'], item['batch']) for item in plan[:4]] == [(256, 1), (320, 1), (256, 2), (320, 2)]
    hires = next(item for item in plan if item['mode'] == 'txt2img_hr' and item['target'] == 320)
    assert hires['width'] == hires['height'] == 160
    assert hires['upscaler'] == 'Lanczos'
    latent = next(item for item in plan if item['mode'] == 'txt2img_hr_latent')
    assert latent['upscaler'] == 'Latent'


@pytest.mark.parametrize('config', [{'max_size': 255}, {'max_size': 300}, {'max_size': 4096},
    {'max_size': float('nan')}, {'max_batch': 0}, {'max_batch': 17}, {'max_batch': 1.5},
    {'max_batch': True}, {'steps': 0}, {'steps': float('inf')}])
def test_invalid_sweep_rejected_before_engine_work(config):
    with pytest.raises(EngineError):
        normalize(config)


def test_default_workload_is_bounded_and_no_oom_is_claimed_by_a_cap():
    config = normalize({})
    assert config['steps'] == 1
    plan = cases(config)
    assert len(plan) <= 2048
    assert plan[-1]['target'] == config['max_size']
    assert 'capacity' not in config



def rows_for(function, sizes=(256, 384, 512, 640), batches=(1,)):
    return [dict(mode='txt2img', pixels=size * size, batch_size=batch,
                 reserved_delta_mb=function(size, batch), status='complete')
            for batch in batches for size in sizes]


def test_known_batch_cubic_matches_independent_polynomial_and_system_buffer():
    from sdfx.vram_estimator import estimate
    function = lambda x, b: 100 + .5*x + .001*x*x + .000001*x*x*x
    result = estimate(rows_for(function), 'txt2img', 448, 448, 1, 2000, 24000)
    assert result['available'] and result['method'] == 'per_batch_polynomial'
    assert result['degree'] == 3 and not result['extrapolated']
    assert result['incremental_mb'] == pytest.approx(function(448, 1))
    assert result['estimated_total_mb'] == pytest.approx(2000 + function(448, 1)*1.1)
    assert result['training_rmse_mb'] < 1e-6


def test_two_samples_use_explicit_linear_fit_and_outside_range_is_labeled():
    from sdfx.vram_estimator import estimate
    result = estimate(rows_for(lambda x, b: 100+x, sizes=(256, 512)), 'txt2img', 768, 768, 1, 1000, 24000)
    assert result['available'] and result['degree'] == 1
    assert result['incremental_mb'] == pytest.approx(868)
    assert result['extrapolated']


def test_single_sample_does_not_invent_a_curve():
    from sdfx.vram_estimator import estimate
    rows = rows_for(lambda x, b: 123, sizes=(512,))
    exact = estimate(rows, 'txt2img', 512, 512, 1, 1000, 24000)
    assert exact['available'] and exact['method'] == 'measured_point'
    assert not estimate(rows, 'txt2img', 768, 768, 1, 1000, 24000)['available']


def test_unmeasured_batch_requires_full_rank_surface_and_recovers_holdout():
    from sdfx.vram_estimator import estimate
    function = lambda x, b: 100+.2*x+20*b+.001*x*x+.01*x*b+2*b*b+.000001*x*x*x+.0001*x*x*b+.002*x*b*b+.5*b*b*b
    rows = rows_for(function, batches=(1, 2, 4, 5))
    result = estimate(rows, 'txt2img', 448, 448, 3, 1000, 24000)
    assert result['available'] and result['method'] == 'cubic_surface'
    assert not result['extrapolated']
    assert result['incremental_mb'] == pytest.approx(function(448, 3), abs=1e-6)
    assert not estimate(rows_for(function, batches=(1, 2)), 'txt2img', 448, 448, 3, 1000, 24000)['available']


def test_failed_rows_and_other_modes_do_not_pollute_fits():
    from sdfx.vram_estimator import estimate
    rows = rows_for(lambda x, b: 200+x)
    rows += [dict(mode='txt2img', pixels=512*512, batch_size=1, status='oom'),
             dict(mode='img2img', pixels=512*512, batch_size=1, reserved_delta_mb=99999, status='complete')]
    result = estimate(rows, 'txt2img', 512, 512, 1, 1000, 24000)
    assert result['incremental_mb'] == pytest.approx(712)
    assert result['samples'] == 4


@pytest.mark.parametrize('field,value', [('pixels', 0), ('pixels', float('nan')), ('batch_size', True),
                                         ('reserved_delta_mb', -1), ('reserved_delta_mb', float('inf'))])
def test_malformed_successful_rows_are_rejected(field, value):
    from sdfx.vram_estimator import estimate
    rows = rows_for(lambda x, b: 200+x)
    rows[0][field] = value
    with pytest.raises(EngineError):
        estimate(rows, 'txt2img', 512, 512, 1, 1000, 24000)


def test_nonphysical_extrapolation_is_unavailable_and_capacity_warning_is_explicit():
    from sdfx.vram_estimator import estimate
    assert not estimate(rows_for(lambda x, b: 1000-x, sizes=(256,512)), 'txt2img', 2048, 2048, 1, 0, 24000)['available']
    result = estimate(rows_for(lambda x,b: 5000), 'txt2img', 512, 512, 1, 23000, 24000)
    assert result['over_capacity'] and result['estimated_total_mb'] > 24000


def test_rectangular_shape_and_missing_mode_are_explicit():
    from sdfx.vram_estimator import estimate
    rows = rows_for(lambda x,b: x+100)
    result = estimate(rows, 'txt2img', 256, 1024, 1, 1000, 24000)
    assert result['shape_assumption'] == 'square_calibration_by_pixel_area'
    assert not estimate(rows, 'img2img', 512, 512, 1, 1000, 24000)['available']



class SweepContext:
    cancelled = False
    def check_interrupt(self):
        from sdfx.protocol import INTERRUPTED
        if self.cancelled:
            raise EngineError(INTERRUPTED, 'cancelled')
    def progress(self, *args): pass


def test_sweep_oom_stops_only_larger_sizes_of_that_mode_batch_and_keeps_rows():
    from sdfx.vram_estimator import run_sweep
    from sdfx.protocol import OOM
    attempted, cleanups, updates = [], [], []
    def measure(case):
        attempted.append((case['mode'], case['batch'], case['target']))
        if attempted[-1] == ('txt2img', 1, 320):
            raise EngineError(OOM, 'out of memory')
        return dict(reserved_delta_mb=case['target'])
    result = run_sweep(normalize({'max_size': 384, 'max_batch': 2}), measure, SweepContext(),
                       lambda: cleanups.append(True), updates.append)
    assert ('txt2img', 1, 384) not in attempted
    assert ('txt2img', 2, 384) in attempted
    assert len(attempted) == 23 and len(result['rows']) == 23
    assert sum(row['status'] == 'oom' for row in result['rows']) == 1
    assert result['status'] == 'stopped_after_error' and cleanups == [True]
    assert updates[0]['rows'] == [] and len(updates[-1]['rows']) == 23


def test_sweep_interruption_publishes_partial_state_and_propagates_cancel():
    from sdfx.vram_estimator import run_sweep
    from sdfx.protocol import INTERRUPTED
    context, updates = SweepContext(), []
    def publish(state):
        updates.append(state)
        if len(state['rows']) == 1:
            context.cancelled = True
    with pytest.raises(EngineError) as failure:
        run_sweep(normalize({'max_size': 320, 'max_batch': 1}), lambda _: dict(reserved_delta_mb=123),
                  context, lambda: None, publish)
    assert failure.value.code == INTERRUPTED
    assert updates[-1]['status'] == 'interrupted' and len(updates[-1]['rows']) == 1
    assert updates[0]['rows'] == [], 'published snapshots must not mutate afterward'


def test_atomic_profile_write_failure_keeps_old_bytes_and_removes_only_own_temp(tmp_path, monkeypatch):
    import os
    from sdfx.vram_estimator import write_profile
    target = tmp_path / 'profile.json'
    write_profile(target, {'status': 'complete', 'rows': []})
    before = target.read_bytes()
    def fail(*args): raise PermissionError('injected atomic replace failure')
    monkeypatch.setattr(os, 'replace', fail)
    with pytest.raises(PermissionError):
        write_profile(target, {'status': 'running', 'rows': []})
    assert target.read_bytes() == before
    assert list(tmp_path.iterdir()) == [target]


def test_profile_refuses_nonfinite_data_without_touching_previous_record(tmp_path):
    from sdfx.vram_estimator import write_profile
    target = tmp_path / 'profile.json'
    target.write_bytes(b'previous')
    with pytest.raises(ValueError):
        write_profile(target, {'rows': [{'reserved_delta_mb': float('nan')}]})
    assert target.read_bytes() == b'previous'



def test_generation_requests_cover_real_hires_and_img2img_paths():
    from sdfx.vram_estimator import generation_request
    plan = cases(normalize({'max_size': 320, 'max_batch': 2}))
    for mode in ('txt2img', 'txt2img_hr', 'txt2img_hr_latent', 'img2img'):
        case = next(item for item in plan if item['mode'] == mode and item['target'] == 320 and item['batch'] == 2)
        operation, request = generation_request(case, 1, '/output', '/synthetic.png')
        params = request['params']
        assert params['batch_size'] == 2 and params['n_iter'] == 1
        assert params['hooks'] == {} and request['preview_method'] == 'none'
        if mode == 'img2img':
            assert operation == 'img2img' and params['img2img']['init_image'] == '/synthetic.png'
        elif mode.startswith('txt2img_hr'):
            assert operation == 'txt2img' and params['width'] == 160
            assert params['hires']['resize_x'] == params['hires']['resize_y'] == 320
            assert params['hires']['second_pass_steps'] == 1
            assert params['hires']['upscaler'] == case['upscaler']
        else:
            assert operation == 'txt2img' and params['width'] == 320



def test_benchmark_restores_engine_flags_and_persists_actual_case_rows(tmp_path):
    import json
    from sdfx.vram_estimator import benchmark
    class Engine:
        cuda, dry_run, model = True, False, {'hash': 'fixture'}
        _benchmark_no_save = 'original'
        calls = []
        def txt2img(self, request, ctx):
            assert self._benchmark_no_save is True
            self.calls.append(request)
            params = request['params']
            target = params.get('hires', {}).get('resize_x', params['width'])
            return {'images': [dict(width=target, height=target, path='') for _ in range(params['batch_size'])]}
        img2img = txt2img
        def on_oom(self): pass
    class Meter:
        def measure(self, generate): return generate(), dict(reserved_delta_mb=123, reserved_peak_mb=456)
    engine = Engine()
    result = benchmark(engine, {'max_size': 256, 'max_batch': 1, 'output_dir': str(tmp_path)},
                       SweepContext(), meter=Meter(), identity={'model': 'fixture'})
    assert len(result['rows']) == 4 and result['status'] == 'complete'
    assert engine._benchmark_no_save == 'original'
    assert not hasattr(engine, '_vram_capture_sample') and not hasattr(engine, '_vram_sample')
    assert json.loads(__import__('pathlib').Path(result['profile_path']).read_text())['status'] == 'complete'
    assert not list(tmp_path.rglob('synthetic-input.png'))


def test_cpu_benchmark_is_not_mislabeled_as_vram_measurement(tmp_path):
    from types import SimpleNamespace
    from sdfx.vram_estimator import benchmark
    with pytest.raises(EngineError) as error:
        benchmark(SimpleNamespace(cuda=False, dry_run=False, model={}), {'output_dir': str(tmp_path)}, SweepContext())
    assert error.value.code == 'unsupported'
    assert not list(tmp_path.iterdir())


def test_profile_fingerprint_distinguishes_model_runtime_and_is_key_order_stable():
    from sdfx.vram_estimator import fingerprint
    assert fingerprint({'model':'a', 'dtype':'fp16'}) == fingerprint({'dtype':'fp16', 'model':'a'})
    assert fingerprint({'model':'a', 'dtype':'fp16'}) != fingerprint({'model':'b', 'dtype':'fp16'})



def test_interrupted_benchmark_restores_flags_and_keeps_partial_profile(tmp_path):
    from pathlib import Path
    from sdfx.vram_estimator import benchmark
    from sdfx.protocol import INTERRUPTED
    class Engine:
        cuda, dry_run, model = True, False, {'hash': 'fixture'}
        count = 0
        def txt2img(self, request, ctx):
            self.count += 1
            if self.count == 2:
                raise EngineError(INTERRUPTED, 'cancelled')
            params = request['params']
            target = params.get('hires', {}).get('resize_x', params['width'])
            return {'images': [dict(width=target, height=target, path='')]}
        img2img = txt2img
        def on_oom(self): pass
    class Meter:
        def measure(self, generate): return generate(), {'reserved_delta_mb': 123}
    engine = Engine()
    with pytest.raises(EngineError) as error:
        benchmark(engine, {'max_size': 256, 'max_batch': 1, 'output_dir': str(tmp_path)}, SweepContext(),
                  meter=Meter(), identity={'model':'fixture'})
    assert error.value.code == INTERRUPTED
    assert engine._vram_last_profile['status'] == 'interrupted'
    assert len(engine._vram_last_profile['rows']) == 1
    assert Path(engine._vram_last_profile['profile_path']).is_file()
    assert all(not hasattr(engine, key) for key in ('_benchmark_no_save', '_vram_capture_sample', '_vram_sample'))
    assert not list(tmp_path.rglob('synthetic-input.png'))



def profile_for(identity=None):
    from sdfx.vram_estimator import fingerprint
    identity = identity or {'model': 'a', 'dtype': 'fp16'}
    return dict(version=1, status='complete', identity=identity, fingerprint=fingerprint(identity),
                rows=rows_for(lambda x,b: 100+x))


def test_estimate_dispatch_uses_current_memory_only_after_identity_match():
    from types import SimpleNamespace
    from sdfx.vram_estimator import dispatch
    profile = profile_for()
    engine = SimpleNamespace(cuda=True, dry_run=False, model={'hash':'a'}, _vram_last_profile=profile)
    called = []
    def memory():
        called.append(True)
        return {'system_used_mb': 1000, 'total_mb': 24000}
    result = dispatch(engine, {'operation':'estimate', 'mode':'txt2img', 'width':512, 'height':512, 'batch':1},
                      SweepContext(), identity=profile['identity'], memory=memory)
    assert result['available'] and result['incremental_mb'] == pytest.approx(612)
    assert result['estimated_total_mb'] == pytest.approx(1673.2) and called == [True]
    result = dispatch(engine, {'operation':'estimate'}, SweepContext(), identity={'model':'different'}, memory=memory)
    assert not result['available'] and result['reason'] == 'profile_identity_mismatch'
    assert called == [True], 'mismatched profile must not query or reset GPU counters'


def test_loaded_profile_fingerprint_is_checked_against_its_contents():
    from sdfx.vram_estimator import validate_profile
    profile = profile_for()
    profile['identity']['model'] = 'changed'
    with pytest.raises(EngineError):
        validate_profile(profile)


def test_profile_dispatch_keeps_interrupted_data_and_marks_compatibility():
    from types import SimpleNamespace
    from sdfx.vram_estimator import dispatch
    profile = profile_for(); profile['status'] = 'interrupted'
    engine = SimpleNamespace(cuda=True, dry_run=False, model={'hash':'a'}, _vram_last_profile=profile)
    result = dispatch(engine, {'operation':'profile'}, SweepContext(), identity=profile['identity'])
    assert result['compatible'] and result['profile']['status'] == 'interrupted'
    result = dispatch(engine, {'operation':'estimate', 'profile':profile}, SweepContext(), identity=profile['identity'],
                      memory=lambda: {'system_used_mb': 1000, 'total_mb':24000})
    assert result['available'] and result['profile_status'] == 'interrupted'


def test_no_profile_and_unknown_operation_are_explicit():
    from types import SimpleNamespace
    from sdfx.vram_estimator import dispatch
    engine = SimpleNamespace(cuda=True, dry_run=False, model={'hash':'a'})
    assert dispatch(engine, {'operation':'estimate'}, SweepContext(), identity={'model':'a'})['reason'] == 'no_profile'
    with pytest.raises(EngineError):
        dispatch(engine, {'operation':'unrecognized'}, SweepContext())



def test_fingerprint_survives_equivalent_json_integer_float_roundtrip():
    from sdfx.vram_estimator import fingerprint
    original = {'gpu': {'total_mb': 24576.0, 'capability': [8, 6]}, 'runtime': {'vae_tiling': False}}
    roundtrip = {'runtime': {'vae_tiling': False}, 'gpu': {'capability': [8.0, 6.0], 'total_mb': 24576}}
    assert fingerprint(original) == fingerprint(roundtrip)
    assert fingerprint({'x': True}) != fingerprint({'x': 1})
