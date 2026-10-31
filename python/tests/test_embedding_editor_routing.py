import sys
from types import SimpleNamespace
import pytest


def test_editor_worker_is_serialized_and_requires_real_loaded_model():
    from sdfx.base import BaseEngine
    from sdfx.protocol import EngineError
    from sdfx.server import QUEUED_TYPES
    assert 'embedding_editor' in QUEUED_TYPES
    engine = BaseEngine(); engine.dry_run = True
    with pytest.raises(EngineError, match='real loaded model'):
        engine.embedding_editor({'operation': 'open'}, None)
    engine.dry_run = False
    with pytest.raises(EngineError, match='load a checkpoint'):
        engine.embedding_editor({'operation': 'open'}, None)


def test_editor_worker_forwards_owned_context_to_service(monkeypatch):
    from sdfx.base import BaseEngine
    calls = []
    monkeypatch.setitem(sys.modules, 'sdfx.embedding_editor_service',
        SimpleNamespace(run_editor_service=lambda engine, params, ctx: calls.append((engine, params, ctx)) or {'ok': True}))
    engine = BaseEngine(); engine.model = {'model_token': 'm'}; engine.pipe = object()
    context = object(); params = {'operation': 'open', 'name': 'concept'}
    assert engine.embedding_editor(params, context) == {'ok': True}
    assert calls == [(engine, params, context)]


def test_dry_run_route_returns_protocol_unsupported_without_torch():
    import io, json, time
    from sdfx.dryrun import FakeEngine
    from sdfx.protocol import Emitter
    from sdfx.server import Server
    output = io.BytesIO(); server = Server(FakeEngine(), Emitter(output)); server.worker.start()
    try:
        server.handle_line(json.dumps({'id': 'editor', 'type': 'embedding_editor', 'params': {'operation': 'open', 'name': 'cat'}}))
        deadline = time.monotonic() + 3
        while b'"id":"editor"' not in output.getvalue() and time.monotonic() < deadline: time.sleep(.01)
        reply = next(json.loads(line) for line in output.getvalue().splitlines())
        assert reply['type'] == 'error' and reply['code'] == 'unsupported'
    finally:
        server.q.put(None); server.worker.join(timeout=5)


def test_live_route_passes_worker_context_and_result_through_protocol(monkeypatch):
    import io, json, time
    from sdfx.base import BaseEngine
    from sdfx.protocol import Emitter
    from sdfx.server import Server
    seen = []
    def run(engine, params, ctx):
        ctx.check_interrupt(); seen.append((params['name'], ctx)); return {'opened': params['name']}
    monkeypatch.setitem(sys.modules, 'sdfx.embedding_editor_service', SimpleNamespace(run_editor_service=run))
    engine = BaseEngine(); engine.model = {'model_token': 'm'}; engine.pipe = object()
    output = io.BytesIO(); server = Server(engine, Emitter(output)); server.worker.start()
    try:
        server.handle_line(json.dumps({'id': 'editor', 'type': 'embedding_editor', 'params': {'operation': 'open', 'name': 'cat'}}))
        deadline = time.monotonic() + 3
        while b'"id":"editor"' not in output.getvalue() and time.monotonic() < deadline: time.sleep(.01)
        reply = next(json.loads(line) for line in output.getvalue().splitlines())
        assert reply['type'] == 'result' and reply['result'] == {'opened': 'cat'}
        assert len(seen) == 1 and seen[0][0] == 'cat'
    finally:
        server.q.put(None); server.worker.join(timeout=5)
