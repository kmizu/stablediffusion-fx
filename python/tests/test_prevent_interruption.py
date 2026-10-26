"""Reader-thread and policy contract tests; no model, GPU or network activity."""
import io
import json
import threading

import pytest
from sdfx.base import BaseEngine
from sdfx.protocol import Emitter
from sdfx.server import Server


def request(server, stream, kind, params=None, rid='control'):
    server.handle_line(json.dumps(dict(id=rid, type=kind, params=params or {})))
    messages = [json.loads(line) for line in stream.getvalue().splitlines()]
    reply = next(message for message in reversed(messages) if message.get('id') == rid and message['type'] in ('result', 'error'))
    assert reply['type'] == 'result', reply
    return reply['result']


@pytest.fixture
def idle():
    stream = io.BytesIO()
    return Server(BaseEngine(), Emitter(stream)), stream


def test_default_off_shared_get_set_and_new_server_resets_runtime_state(idle):
    server, stream = idle
    first = request(server, stream, 'interruption_protection')
    assert first == {'enabled': False, 'revision': 0}
    changed = request(server, stream, 'interruption_protection', {'enabled': True}, 'frontend-one')
    assert changed == {'enabled': True, 'revision': 1}
    assert request(server, stream, 'interruption_protection', rid='frontend-two') == changed
    assert request(server, stream, 'interruption_protection', {'enabled': True}) == changed  # idempotent
    other = io.BytesIO()
    assert request(Server(BaseEngine(), Emitter(other)), other, 'interruption_protection') == first


@pytest.mark.parametrize('value', ['false', 1, None, [], {}])
def test_set_requires_real_boolean_and_invalid_input_cannot_unlock(idle, value):
    server, stream = idle
    request(server, stream, 'interruption_protection', {'enabled': True})
    server.handle_line(json.dumps(dict(id='invalid', type='interruption_protection', params={'enabled': value})))
    result = next(json.loads(line) for line in stream.getvalue().splitlines() if json.loads(line).get('id') == 'invalid')
    assert result['type'] == 'error' and result['code'] == 'invalid_params'
    assert request(server, stream, 'interruption_protection')['enabled'] is True


def test_protected_interrupt_changes_neither_event_nor_pending_cancel_set(idle):
    server, stream = idle
    server.current_id = 'running'
    server.queued.append('later')
    request(server, stream, 'interruption_protection', {'enabled': True})
    for target in ('running', 'later', None):
        outcome = request(server, stream, 'interrupt', {} if target is None else {'target_id': target})
        assert outcome['blocked'] is True and outcome['accepted'] is False
        assert outcome['protection']['enabled'] is True
        assert not server.interrupt.is_set()
        assert not server.cancelled
    request(server, stream, 'interruption_protection', {'enabled': False})
    outcome = request(server, stream, 'interrupt', {'target_id': 'running'})
    assert outcome['accepted'] is True and outcome['blocked'] is False
    assert server.interrupt.is_set()


def test_query_toggle_and_interrupt_are_responsive_while_worker_stays_busy():
    entered, release = threading.Event(), threading.Event()
    class Busy(BaseEngine):
        def txt2img(self, params, ctx):
            entered.set()
            assert release.wait(3)
            ctx.check_interrupt()
            return {'completed': True}
    stream = io.BytesIO()
    server = Server(Busy(), Emitter(stream))
    server.worker.start()
    try:
        server.handle_line(json.dumps(dict(id='job', type='txt2img', params={})))
        assert entered.wait(2)
        assert request(server, stream, 'interruption_protection', {'enabled': True})['enabled'] is True
        assert request(server, stream, 'interruption_protection', rid='other-frontend')['enabled'] is True
        assert request(server, stream, 'interrupt')['blocked'] is True
        assert not server.interrupt.is_set()
        assert not any(json.loads(line).get('id') == 'job' for line in stream.getvalue().splitlines())
    finally:
        release.set()
        server.q.put(None)
        server.worker.join(4)
    result = next(json.loads(line) for line in stream.getvalue().splitlines() if json.loads(line).get('id') == 'job')
    assert result['type'] == 'result' and result['result']['completed'] is True


def test_explicit_restart_authorization_is_blocked_but_shutdown_is_not(idle):
    server, stream = idle
    request(server, stream, 'interruption_protection', {'enabled': True})
    result = request(server, stream, 'restart_authorization')
    assert result['blocked'] is True and result['accepted'] is False
    assert not server.interrupt.is_set()
    request(server, stream, 'interruption_protection', {'enabled': False})
    result = request(server, stream, 'restart_authorization')
    assert result['accepted'] is True and result['blocked'] is False
    request(server, stream, 'interruption_protection', {'enabled': True})
    assert server.handle_line(json.dumps(dict(id='close', type='shutdown', params={}))) is False
    assert server.interrupt.is_set()  # internal teardown retains its existing semantics


def test_capability_is_exposed_without_loading_models():
    from sdfx.dryrun import FakeEngine
    assert 'prevent_interrupt' in BaseEngine().features()
    assert 'prevent_interrupt' in FakeEngine().features()


def test_actual_engine_advertises_control_capability_without_a_pipeline():
    pytest.importorskip('torch')
    from sdfx.engine import Engine
    engine = Engine()
    assert engine.pipe is None
    assert 'prevent_interrupt' in engine.features()


def test_real_json_lines_process_accepts_shared_query_toggle_and_resets_on_restart():
    from test_server_dryrun import Engine
    for expected_initial in (False, False):
        engine = Engine()
        try:
            hello = engine.next()
            assert 'prevent_interrupt' in hello['features']
            result, _ = engine.call('interruption_protection')
            assert result['result']['enabled'] is expected_initial
            result, _ = engine.call('interruption_protection', {'enabled': True})
            assert result['result']['enabled'] is True
            result, _ = engine.call('interruption_protection')
            assert result['result']['enabled'] is True
            result, _ = engine.call('restart_authorization')
            assert result['result']['blocked'] is True
        finally:
            assert engine.close() == 0
