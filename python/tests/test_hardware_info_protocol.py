"""Actual dry-run stdio attachment contract; this never loads a diffusion model or Torch."""
from pathlib import Path
from test_server_dryrun import Engine, gen_params


def test_hardware_records_are_per_image_opt_in_literal_and_do_not_change_image_bytes(tmp_path):
    engine = Engine()
    try:
        assert engine.next()['dry_run'] is True
        params = gen_params(tmp_path, steps=1, batch_size=2, n_iter=2, seed=71)
        plain, _ = engine.call('txt2img', params)
        assert plain['type'] == 'result'
        baseline = plain['result']['images']
        assert len(baseline) == 4
        assert all(image.get('hardware_info') is None for image in baseline)
        params['hardware_info'] = {'enabled': True, 'remark': '机, 日本語\nquiet'}
        enabled, _ = engine.call('txt2img', params)
        assert enabled['type'] == 'result'
        images = enabled['result']['images']
        assert [image['seed'] for image in images] == [71, 72, 73, 74]
        for original, annotated in zip(baseline, images):
            record = annotated['hardware_info']
            assert record['hardware_info'] and record['time_taken'].endswith('sec.')
            assert record['remark'] == '机, 日本語\nquiet'
            assert Path(original['path']).read_bytes() == Path(annotated['path']).read_bytes()
        params['hardware_info'] = {'enabled': False, 'remark': 'must not leak'}
        disabled, _ = engine.call('txt2img', params)
        assert all(image.get('hardware_info') is None for image in disabled['result']['images'])
    finally:
        engine.close()


def test_extras_records_whole_operation_and_preserves_source_bytes_without_another_generation(tmp_path):
    engine = Engine()
    try:
        engine.next()
        initial, _ = engine.call('txt2img', gen_params(tmp_path, steps=1))
        source = Path(initial['result']['images'][0]['path'])
        before = source.read_bytes()
        request = {'image_path': str(source), 'upscaler': 'Lanczos', 'scale': 2.0,
            'output_dir': str(tmp_path / 'extras'), 'upscaler_paths': {}}
        plain, _ = engine.call('upscale', request)
        assert plain['type'] == 'result'
        assert plain['result'].get('hardware_info') is None
        request['hardware_info'] = {'enabled': True, 'remark': 'Extras, 日本語'}
        annotated, _ = engine.call('upscale', request)
        assert annotated['type'] == 'result'
        record = annotated['result']['hardware_info']
        assert record['hardware_info'] and record['time_taken'].endswith('sec.')
        assert record['remark'] == 'Extras, 日本語'
        assert source.read_bytes() == before
        assert Path(plain['result']['path']).read_bytes() == Path(annotated['result']['path']).read_bytes()
    finally:
        engine.close()
