import io
import json
from types import SimpleNamespace
import pytest
from sdfx.base import BaseEngine
from sdfx.protocol import Emitter, JobContext, EngineError


def test_cpu_snapshot_is_real_and_omits_private_identity_paths_and_environment(monkeypatch):
    from sdfx import system_info as si
    monkeypatch.setenv('PRIVATE_TOKEN', 'should-never-appear')
    e = BaseEngine()
    e.model = {'name': 'test', 'kind': 'sd15', 'hash': 'abc', 'path': 'C:/private/user/model.safetensors'}
    r = si.snapshot(e, full=True)
    encoded = json.dumps(r)
    assert 'should-never-appear' not in encoded and 'C:/private/user' not in encoded
    assert r['model']['name'] == 'test'
    assert r['runtime']['pipeline'] == 'diffusers'
    assert r['memory']['process_rss_mb'] >= 0
    assert r['platform']['python']
    assert 'hostname' not in r and 'username' not in r


def test_snapshot_reports_actual_component_precisions_without_loading_them():
    from sdfx import system_info as si
    engine = BaseEngine()
    engine.device, engine.dtype = 'cpu', 'float16'
    engine.pipe = SimpleNamespace(vae=SimpleNamespace(dtype='float32'), unet=SimpleNamespace(dtype='float16', attn_processors={}))
    result = si.snapshot(engine)
    assert result['device'] == {'active': 'cpu', 'dtype': 'float16', 'vae_dtype': 'float32', 'unet_dtype': 'float16'}


def test_benchmark_levels_and_steps_are_exact_and_invalid_inputs_rejected():
    from sdfx import system_info as si
    assert si.benchmark_config({'level': 'normal', 'steps': 'turbo'})['batches'] == [1, 2, 4]
    assert si.benchmark_config({'level': 'ludicrous', 'steps': 'long'})['batches'][-1] == 64
    assert si.benchmark_config({'steps': 'default'})['inference_steps'] == 20
    for params in ({'width': 0}, {'height': 65}, {'width': 64, 'height': 64}, {'level': 'other'}, {'warmup': 'bogus'}, {'steps': 'bogus'}):
        with pytest.raises(EngineError) as error:
            si.benchmark_config(params)
        assert error.value.code == 'invalid_params'


def test_metadata_whitelist_is_opt_in_and_does_not_include_paths():
    from sdfx import system_info as si
    data = {'platform': {'python': '3.12', 'system': 'Windows'}, 'model': {'path': 'private'}}
    assert si.metadata(data, []) == {}
    assert si.metadata(data, ['platform.python', 'model.path']) == {'platform.python': '3.12'}
    native = {'runtime': {'attention': 'SDPA', 'pipeline': 'diffusers', 'flags': ['gpu']}, 'libraries': {'torch': '2.9'}}
    assert si.metadata(native, ['torch.version', 'crossattention', 'pipeline', 'flags']) == {
        'torch.version': '2.9', 'crossattention': 'SDPA', 'pipeline': 'diffusers', 'flags': 'gpu'}


def test_submission_requires_preview_and_uses_exact_mocked_transport(monkeypatch):
    from sdfx import system_info as si
    record = {'timestamp': 'now', 'fingerprint': '123456', 'results': [{'batch': 1, 'iterations_per_second': 2}],
              'system': {}, 'config': {}, 'note': 'test'}
    e = BaseEngine()
    preview = si.submission_preview(e, {'record': record, 'username': 'tester', 'include_hostname': False})
    assert preview['destination'] == 'logs3.papertrailapp.com:32554/udp'
    assert all('SDBENCHMARK: tester ' in packet for packet in preview['packets'])
    assert all('anonymous' in packet for packet in preview['packets'])
    payload = preview['packets'][0].split('SDBENCHMARK: tester ', 1)[1].removesuffix('\x00')
    assert len(payload.split('|')) == 12  # eleven source columns plus submission digest
    sent = []
    class Socket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def settimeout(self, value): pass
        def sendto(self, data, endpoint): sent.append((data, endpoint)); return len(data)
    monkeypatch.setattr(si.socket, 'socket', lambda *a, **kw: Socket())
    with pytest.raises(EngineError):
        si.submit(e, {'token': 'wrong'})
    result = si.submit(e, {'token': preview['token']})
    assert result['status'] == 'sent_unconfirmed'
    assert sent == [(packet.encode('utf-8'), ('logs3.papertrailapp.com', 32554)) for packet in preview['packets']]
    with pytest.raises(EngineError):
        si.submit(e, {'token': preview['token']})


@pytest.mark.parametrize('record', [{'results': [1]}, {'results': [], 'system': []}, {'results': [], 'fingerprint': None}])
def test_malformed_submission_record_is_invalid_params(record):
    from sdfx import system_info as si
    with pytest.raises(EngineError) as error:
        si.submission_preview(BaseEngine(), {'record': record, 'username': 'test'})
    assert error.value.code == 'invalid_params'


def test_real_server_status_remains_responsive_while_worker_is_busy():
    import threading
    from sdfx.server import Server
    started, release = threading.Event(), threading.Event()
    class BusyEngine(BaseEngine):
        def txt2img(self, params, ctx):
            ctx.progress(3, 10)
            started.set()
            assert release.wait(5)
            return {}
    stream = io.BytesIO()
    server = Server(BusyEngine(), Emitter(stream))
    server.worker.start()
    try:
        server.handle_line(json.dumps({'id': 'busy', 'type': 'txt2img'}))
        assert started.wait(3)
        server.handle_line(json.dumps({'id': 'status', 'type': 'system_status'}))
        messages = [json.loads(line) for line in stream.getvalue().splitlines()]
        result = next(m['result'] for m in messages if m.get('id') == 'status')
        assert result['state']['request_id'] == 'busy'
        assert result['state']['progress']['step'] == 3
        assert not any(m.get('id') == 'busy' and m['type'] == 'result' for m in messages)
    finally:
        release.set()
        server.q.put(None)
        server.worker.join(5)


def test_benchmark_computes_wall_time_scores_cleans_images_and_stops_on_oom(monkeypatch):
    from pathlib import Path
    from sdfx import system_info as si
    requests = []
    class BenchEngine(BaseEngine):
        def __init__(self):
            super().__init__()
            self.model = {'name': 'test', 'hash': 'abc'}
            self.fail = False
            self.oom_cleanup = False
        def txt2img(self, params, ctx):
            batch = params['params']['batch_size']
            requests.append(params)
            if self.fail and batch == 2:
                raise RuntimeError('out of memory')
            images = []
            for index in range(batch):
                path = Path(params['output_dir']) / f'{index}.png'
                path.write_bytes(b'test fixture image')
                images.append({'path': str(path)})
            return {'images': images}
        def on_oom(self): self.oom_cleanup = True
    engine = BenchEngine()
    ticks = iter([1., 3., 10., 14., 20., 28.])
    monkeypatch.setattr(si.time, 'perf_counter', lambda: next(ticks))
    record = si.benchmark(engine, {'level': 'normal', 'steps': 'turbo', 'warmup': False}, JobContext(Emitter(io.BytesIO()), 'test'))
    assert [r['iterations_per_second'] for r in record['results']] == [5., 5., 5.]
    assert [p['params']['batch_size'] for p in requests] == [1, 2, 4]
    assert all(not Path(p['output_dir']).exists() for p in requests)
    engine.fail = True
    requests.clear()
    ticks = iter([1., 3., 4.])
    record = si.benchmark(engine, {'level': 'normal', 'steps': 'turbo', 'warmup': False}, JobContext(Emitter(io.BytesIO()), 'test'))
    assert [p['params']['batch_size'] for p in requests] == [1, 2]
    assert record['status'] == 'stopped_after_error' and record['results'][-1]['status'] == 'oom'
    assert engine.oom_cleanup


def test_inventory_embedding_vectors_use_only_safe_headers_and_paths_are_omitted(tmp_path):
    import struct
    from sdfx import system_info as si
    header = json.dumps({'emb_params': {'shape': [4, 768], 'dtype': 'F32', 'data_offsets': [0, 0]}}).encode()
    (tmp_path / 'embedding.safetensors').write_bytes(struct.pack('<Q', len(header)) + header)
    (tmp_path / 'unsafe.pt').write_bytes(b'not loaded as pickle')
    result = si.inventory(BaseEngine(), {'embeddings': [str(tmp_path)]}, False, JobContext(Emitter(io.BytesIO()), 'scan'))
    assert {e['name']: e.get('vectors') for e in result['embeddings']} == {'embedding': 4, 'unsafe': None}
    assert str(tmp_path) not in json.dumps(result)


def test_dryrun_benchmark_is_not_reported_as_performance_and_metadata_is_explicit():
    from test_server_dryrun import Engine
    engine = Engine()
    try:
        engine.next()
        result, _ = engine.call('system_benchmark', {'level': 'quick'})
        assert result['type'] == 'error' and result['code'] == 'unsupported'
        status, _ = engine.call('system_status')
        assert status['result']['runtime']['dry_run'] is True
    finally:
        engine.close()


def test_benchmark_no_save_flag_is_scoped_and_restored_on_failure(monkeypatch):
    from sdfx import system_info as si
    engine = BaseEngine()
    def fail(e, params, ctx):
        assert e._benchmark_no_save is True
        raise RuntimeError('forced benchmark failure')
    monkeypatch.setattr(si, 'benchmark', fail)
    with pytest.raises(RuntimeError, match='forced benchmark failure'):
        engine.system_benchmark({}, None)
    assert engine._benchmark_no_save is False


def test_interrupted_benchmark_propagates_interrupt_and_restores_no_save():
    import threading
    class InterruptEngine(BaseEngine):
        def __init__(self):
            super().__init__()
            self.model = {'name': 'test', 'hash': 'abc'}
        def txt2img(self, params, ctx):
            assert self._benchmark_no_save
            ctx.interrupt_event.set()
            ctx.check_interrupt()
    engine = InterruptEngine()
    with pytest.raises(EngineError) as error:
        engine.system_benchmark({'level': 'quick', 'warmup': False}, JobContext(Emitter(io.BytesIO()), 'cancel'))
    assert error.value.code == 'interrupted'
    assert not engine._benchmark_no_save


def test_explicit_metadata_survives_real_dryrun_protocol_without_a_model(tmp_path):
    from test_server_dryrun import Engine, gen_params
    engine = Engine()
    try:
        engine.next()
        params = gen_params(str(tmp_path), steps=1)
        plain, _ = engine.call('txt2img', params)
        assert plain['result']['system_info'] is None
        params['system_info_fields'] = ['version.app', 'platform.python', 'model.path']
        opted_in, _ = engine.call('txt2img', params)
        assert set(opted_in['result']['system_info']) == {'version.app', 'platform.python'}
        assert opted_in['result']['system_info']['version.app'] == 'stablediffusion-fx'
    finally:
        engine.close()
