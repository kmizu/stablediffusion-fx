import io
import json

from sdfx.protocol import Emitter
from sdfx.server import Server


def test_generation_secret_is_scoped_to_context_and_not_params_or_response():
    class Engine:
        def txt2img(self, params, ctx):
            assert params == {'prompt': 'cat'}
            assert ctx.image_password == 'sensitive-secret'
            return {'ok': True}
    stream = io.BytesIO()
    server = Server(Engine(), Emitter(stream))
    server.handle_line(json.dumps({'id': 'a', 'type': 'txt2img', 'params': {'prompt': 'cat'},
                                   'image_password': 'sensitive-secret'}))
    server._handle(server.q.get_nowait())
    assert b'sensitive-secret' not in stream.getvalue()
    assert server.current_ctx is None


def test_invalid_or_non_generation_secret_is_rejected_without_echo():
    stream = io.BytesIO()
    server = Server(object(), Emitter(stream))
    server.handle_line(json.dumps({'id': 'a', 'type': 'img2img', 'params': {}, 'image_password': ''}))
    server.handle_line(json.dumps({'id': 'b', 'type': 'ping', 'params': {}, 'image_password': 'sensitive-secret'}))
    messages = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(messages) == 2 and all(m['type'] == 'error' for m in messages)
    assert b'sensitive-secret' not in stream.getvalue()
    assert server.q.empty()


def test_plaintext_video_permission_is_request_scoped_and_not_in_params():
    seen = []
    class Engine:
        def txt2img(self, params, ctx):
            seen.append((params, ctx.allow_plaintext_video))
            return {}
    stream = io.BytesIO()
    server = Server(Engine(), Emitter(stream))
    for rid, permit in [('a', True), ('b', False)]:
        server.handle_line(json.dumps({'id': rid, 'type': 'txt2img', 'params': {},
                                       'image_password': 'secret', 'allow_plaintext_video': permit}))
        server._handle(server.q.get_nowait())
    assert seen == [({}, True), ({}, False)]
    assert b'secret' not in stream.getvalue()


def test_plaintext_video_permission_requires_boolean_and_password():
    stream = io.BytesIO()
    server = Server(object(), Emitter(stream))
    for index, extra in enumerate([{'allow_plaintext_video': True},
                                    {'image_password': 'secret', 'allow_plaintext_video': 'true'}]):
        server.handle_line(json.dumps({'id': str(index), 'type': 'txt2img', 'params': {}, **extra}))
    assert len(stream.getvalue().splitlines()) == 2
    assert all(json.loads(line)['type'] == 'error' for line in stream.getvalue().splitlines())
    assert server.q.empty()


def test_wd_execute_receives_secret_outside_prepared_params():
    class Engine:
        def batch_interrogate_execute(self, params, ctx):
            assert params == {'token': 'token'}
            assert ctx.image_password == 'sensitive-secret'
            return {'ok': True}
    stream = io.BytesIO()
    server = Server(Engine(), Emitter(stream))
    server.handle_line(json.dumps({'id': 'wd', 'type': 'batch_interrogate_execute',
                                   'params': {'token': 'token'}, 'image_password': 'sensitive-secret'}))
    server._handle(server.q.get_nowait())
    assert b'sensitive-secret' not in stream.getvalue()


def test_temporal_job_receives_secret_without_persisting_it_in_params():
    class Engine:
        def temporal_lab(self, params, ctx):
            assert params == {'operation': 'preview'}
            assert ctx.image_password == 'sensitive-secret'
            return {'operation': 'preview'}
    stream = io.BytesIO(); server = Server(Engine(), Emitter(stream))
    server.handle_line(json.dumps({'id': 't', 'type': 'temporal_lab',
                                  'params': {'operation': 'preview'}, 'image_password': 'sensitive-secret'}))
    server._handle(server.q.get_nowait())
    assert b'sensitive-secret' not in stream.getvalue()
