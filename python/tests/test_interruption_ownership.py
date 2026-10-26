"""Reader admission binds native job ownership without changing generation parameters."""
import io
import json
import threading
import pytest
from sdfx.base import BaseEngine
from sdfx.protocol import Emitter
from sdfx.server import Server


def send(server, stream, typ, params=None, owner=None, rid='control'):
    message = dict(id=rid, type=typ, params=params or {})
    if owner is not None:
        message['owner_id'] = owner
    server.handle_line(json.dumps(message))
    replies = [json.loads(line) for line in stream.getvalue().splitlines()]
    return next((r for r in reversed(replies) if r.get('id') == rid and r['type'] in ('result', 'error')), None)


def test_stale_owner_cannot_interrupt_current_or_queued_next_job():
    stream = io.BytesIO()
    server = Server(BaseEngine(), Emitter(stream))
    server.current_id, server.current_owner = 'request-b', 'job-b'
    send(server, stream, 'txt2img', owner='job-b', rid='queued-b')
    for params in ({'owner_id': 'job-a'}, {'owner_id': 'job-a', 'target_id': 'request-b'}, {'owner_id': 'job-a', 'target_id': 'queued-b'}):
        reply = send(server, stream, 'interrupt', params)['result']
        assert reply['accepted'] and not reply['affected']
        assert not server.interrupt.is_set() and not server.cancelled
    # Existing unowned raw clients still target the current request.
    assert send(server, stream, 'interrupt', {'target_id': 'request-b'})['result']['affected']


def test_owner_bound_interrupt_can_cancel_its_queued_request_behind_other_work():
    stream = io.BytesIO()
    server = Server(BaseEngine(), Emitter(stream))
    server.current_id, server.current_owner = 'request-b', 'job-b'
    send(server, stream, 'txt2img', owner='job-b', rid='queued-b')
    send(server, stream, 'txt2img', owner='job-a', rid='queued-a')
    assert send(server, stream, 'interrupt', {'owner_id': 'job-a'})['result']['affected']
    assert server.cancelled == {'queued-a'}
    assert not server.interrupt.is_set()


@pytest.mark.parametrize('value', [True, 1, '', 'x' * 257, {}, []])
def test_invalid_owner_cannot_turn_targeted_control_into_global_interrupt(value):
    stream = io.BytesIO()
    server = Server(BaseEngine(), Emitter(stream))
    server.current_id, server.current_owner = 'request-b', 'job-b'
    reply = send(server, stream, 'interrupt', {'owner_id': value})
    assert reply['type'] == 'error' and reply['code'] == 'invalid_params'
    assert not server.interrupt.is_set()
    reply = send(server, stream, 'txt2img', owner=value, rid='bad-generation')
    assert reply['type'] == 'error' and reply['code'] == 'invalid_params'
    assert not server.queued


def test_actual_worker_owner_is_transport_only_and_clears_on_completion():
    entered, release = threading.Event(), threading.Event()
    seen = []
    class Busy(BaseEngine):
        def txt2img(self, params, ctx):
            seen.append(params)
            entered.set()
            assert release.wait(3)
            ctx.check_interrupt()
            return {'complete': True}
    stream = io.BytesIO()
    server = Server(Busy(), Emitter(stream))
    server.worker.start()
    try:
        send(server, stream, 'txt2img', {'seed': 71}, owner='job-b', rid='actual-b')
        assert entered.wait(2)
        assert server.current_owner == 'job-b'
        reply = send(server, stream, 'interrupt', {'owner_id': 'old-job-a', 'target_id': 'actual-b'})['result']
        assert reply['accepted'] and not reply['affected']
        assert not server.interrupt.is_set()
    finally:
        release.set()
        server.q.put(None)
        server.worker.join(4)
    assert seen == [{'seed': 71}]
    assert server.current_owner is None
    assert not server.queued_owners
    replies = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert next(r for r in replies if r.get('id') == 'actual-b')['result']['complete']


@pytest.mark.parametrize('queued', [False, True])
def test_interrupt_reports_only_the_actually_active_admitted_request(queued):
    stream = io.BytesIO()
    server = Server(BaseEngine(), Emitter(stream))
    server.current_id, server.current_owner = 'request-b', 'job-b'
    if queued:
        send(server, stream, 'txt2img', owner='job-a', rid='queued-a')
    result = send(server, stream, 'interrupt', {'owner_id': 'job-a' if queued else 'job-b'})['result']
    assert result['affected']
    # Queued cancellation must not arm the native active-work restart grace timer.
    assert result['active_id'] == (None if queued else 'request-b')
