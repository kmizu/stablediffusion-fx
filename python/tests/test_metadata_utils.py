"""No torch: exercise real safetensors bytes and filesystem publication."""
import hashlib
import json
import pathlib
import struct
import tempfile
import unittest
import io
import threading
import time
from unittest.mock import patch


def fixture(path, metadata=None, payload=b'\x00\x00\x80?\x00\x00\x00@'):
    header = {'v': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]}}
    if metadata is not None:
        header['__metadata__'] = metadata
    raw = json.dumps(header).encode()
    raw += b' ' * (-len(raw) % 8)
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + payload)


def unpack(path):
    raw = path.read_bytes()
    size, = struct.unpack('<Q', raw[:8])
    return json.loads(raw[8:8+size]), raw[8+size:]


class MetadataUtilsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name).resolve()
        self.src = self.root / 'model.safetensors'
        fixture(self.src, {'plain': '日本語', 'nested': '{"x":2}'})

    def api(self):
        from sdfx import metadata_utils
        return metadata_utils

    def write(self, metadata_json, new_name='', **options):
        token = self.api().read_metadata(self.src)['source_token']
        return self.api().write_metadata(self.src, metadata_json, new_name, source_token=token, **options)

    def test_read_exposes_raw_strings_and_source_compatible_object_display(self):
        result = self.api().read_metadata(self.src)
        self.assertEqual(result['metadata'], {'plain': '日本語', 'nested': '{"x":2}'})
        self.assertEqual(json.loads(result['editable']), {'plain': '日本語', 'nested': {'x': 2}})
        self.assertEqual(result['tensor_bytes'], 8)
        self.assertEqual(json.loads(result['source_token'])[1], hashlib.sha256(self.src.read_bytes()).hexdigest())

    def test_read_fingerprint_cancellation_does_not_return_a_token(self):
        from sdfx.protocol import Emitter, JobContext, EngineError
        ctx = JobContext(Emitter(io.BytesIO()), 'read')
        ctx.progress = lambda *args: ctx.interrupt_event.set()
        with self.assertRaises(EngineError) as caught:
            self.api().run({'operation': 'read', 'path': str(self.src)}, ctx)
        self.assertEqual(caught.exception.code, 'interrupted')
        self.assertEqual(list(self.root.iterdir()), [self.src])

    def test_read_detects_source_mutation_during_fingerprinting(self):
        def mutate(done, total):
            with self.src.open('ab') as stream:
                stream.write(b'changed')
        with self.assertRaises(ValueError):
            self.api().read_metadata(self.src, progress=mutate)

    def test_worker_write_requires_read_token(self):
        from sdfx.protocol import Emitter, JobContext, EngineError
        with self.assertRaises(EngineError) as caught:
            self.api().run({'operation': 'write', 'path': str(self.src), 'metadata_json': '{}'},
                JobContext(Emitter(io.BytesIO()), 'write', publication_lock=threading.Lock()))
        self.assertIn('source token', caught.exception.message)
        self.assertEqual(list(self.root.iterdir()), [self.src])

    def test_save_default_copy_preserves_source_and_payload_exactly(self):
        original = self.src.read_bytes()
        result = self.write('{"title":"日本語","nested":{"n":3}}')
        target = self.root / 'model_md.safetensors'
        self.assertEqual(pathlib.Path(result['path']), target)
        self.assertEqual(self.src.read_bytes(), original)
        header, data = unpack(target)
        self.assertEqual(data, b'\x00\x00\x80?\x00\x00\x00@')
        self.assertEqual(header['v'], {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]})
        self.assertEqual(header['__metadata__'], {'title': '日本語', 'nested': '{"n":3}'})
        self.assertEqual(result['tensor_sha256'], hashlib.sha256(data).hexdigest())

    def test_empty_metadata_and_explicit_unicode_stem(self):
        fixture(self.src)
        self.assertEqual(self.api().read_metadata(self.src)['editable'], '{}')
        result = self.write('{}', 'コピー')
        self.assertEqual(pathlib.Path(result['path']).name, 'コピー.safetensors')
        self.assertEqual(unpack(pathlib.Path(result['path']))[0]['__metadata__'], {})

    def test_collision_never_overwrites_existing_file(self):
        target = self.root / 'model_md.safetensors'
        target.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            self.write('{}')
        self.assertEqual(target.read_bytes(), b'keep')

    def test_existing_hardlink_to_source_is_never_overwritten(self):
        import os
        target = self.root / 'model_md.safetensors'
        original = self.src.read_bytes()
        os.link(self.src, target)
        with self.assertRaises(FileExistsError):
            self.write('{}')
        self.assertEqual(self.src.read_bytes(), original)
        self.assertEqual(target.read_bytes(), original)

    def test_failed_atomic_publication_removes_owned_temp_only(self):
        original = self.src.read_bytes()
        with patch('sdfx.metadata_utils.os.link', side_effect=OSError('filesystem does not support link')):
            with self.assertRaises(OSError):
                self.write('{}')
        self.assertEqual(list(self.root.iterdir()), [self.src])
        self.assertEqual(self.src.read_bytes(), original)

    def test_invalid_metadata_fails_without_any_output(self):
        for raw in ('[]', '{"x":1,"x":2}', '{"x":NaN}', 'broken'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.write(raw)
        self.assertEqual(list(self.root.iterdir()), [self.src])

    def test_bad_names_never_escape_or_replace_source(self):
        for name in ('../escape', 'nested/x', 'NUL', 'a.', 'a:b', 'model'):
            with self.subTest(name=name), self.assertRaises((ValueError, FileExistsError)):
                self.write('{}', name)
        self.assertEqual(list(self.root.iterdir()), [self.src])

    def test_truncated_payload_rejected_before_copy(self):
        fixture(self.src, payload=b'short')
        with self.assertRaises(ValueError):
            self.write('{}')
        self.assertEqual(list(self.root.iterdir()), [self.src])

    def test_truncated_oversized_and_nonobject_headers_rejected(self):
        for raw in (b'abc', struct.pack('<Q', 101 * 1024 * 1024),
                    struct.pack('<Q', 20) + b'{}', struct.pack('<Q', 2) + b'[]'):
            self.src.write_bytes(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.write('{}')
        self.assertEqual(list(self.root.iterdir()), [self.src])

    def test_scalar_tensor_and_literal_json_string_keep_exact_data(self):
        raw = b'{"v":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}'
        self.src.write_bytes(struct.pack('<Q', len(raw)) + raw + b'1234')
        result = self.write('{"literal":"  { \\"x\\": 2 } "}')
        header, data = unpack(pathlib.Path(result['path']))
        self.assertEqual(data, b'1234')
        self.assertEqual(header['__metadata__']['literal'], '  { "x": 2 } ')

    def test_cancel_during_copy_leaves_no_temp_or_output(self):
        def cancel(done, total):
            raise InterruptedError('cancel')
        with self.assertRaises(InterruptedError):
            self.write('{}', progress=cancel)
        self.assertEqual(list(self.root.iterdir()), [self.src])

    def test_racing_destination_creation_is_not_overwritten(self):
        target = self.root / 'model_md.safetensors'
        def race(operation):
            target.write_bytes(b'racer')
            return operation()
        with self.assertRaises(FileExistsError):
            self.write('{}', publish=race)
        self.assertEqual(target.read_bytes(), b'racer')
        self.assertEqual(set(self.root.iterdir()), {self.src, target})

    def test_source_mutation_during_copy_rejects_output(self):
        def mutate(done, total):
            with self.src.open('ab') as stream:
                stream.write(b'changed')
        with self.assertRaises(ValueError):
            self.write('{}', progress=mutate)
        self.assertEqual(list(self.root.iterdir()), [self.src])

    def test_mutation_at_publication_boundary_rejects_output(self):
        def mutate(operation):
            self.src.write_bytes(b'replaced')
            return operation()
        with self.assertRaises(ValueError):
            self.write('{}', publish=mutate)
        self.assertEqual(list(self.root.iterdir()), [self.src])

    def test_metadata_input_is_bounded(self):
        with self.assertRaises(ValueError):
            self.write('{"x":"' + 'a' * (4 * 1024 * 1024) + '"}')
        self.assertEqual(list(self.root.iterdir()), [self.src])

    def test_invalid_tensor_layout_is_rejected(self):
        for desc in ({'dtype': 'F32', 'shape': [2], 'data_offsets': [1, 9]},
                     {'dtype': 'F32', 'shape': [-1], 'data_offsets': [0, 8]},
                     {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 8]}):
            raw = json.dumps({'v': desc}).encode()
            raw += b' ' * (-len(raw) % 8)
            self.src.write_bytes(struct.pack('<Q', len(raw)) + raw + b'12345678')
            with self.subTest(desc=desc), self.assertRaises(ValueError):
                self.api().read_metadata(self.src)

    def test_empty_tensor_file_copies_with_zero_payload(self):
        raw = b'{"v":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}'
        raw += b' ' * (-len(raw) % 8)
        self.src.write_bytes(struct.pack('<Q', len(raw)) + raw)
        result = self.write('{}')
        self.assertEqual(unpack(pathlib.Path(result['path']))[1], b'')
        self.assertEqual(result['tensor_sha256'], hashlib.sha256(b'').hexdigest())

    def test_valid_unpadded_header_and_complex_fp8_payloads_are_preserved(self):
        # Official safetensors reader accepts arbitrary header byte lengths.
        for dtype, payload in (('C64', b'12345678'), ('F8_E4M3FNUZ', b'a'), ('F8_E5M2FNUZ', b'b')):
            with self.subTest(dtype=dtype):
                raw = json.dumps({'v': {'dtype': dtype, 'shape': [1], 'data_offsets': [0, len(payload)]}}).encode()
                self.src.write_bytes(struct.pack('<Q', len(raw)) + raw + payload)
                result = self.write('{}', dtype)
                self.assertEqual(unpack(pathlib.Path(result['path']))[1], payload)

    def test_worker_publication_and_cancel_are_owned(self):
        from sdfx.protocol import Emitter, JobContext, EngineError
        ctx = JobContext(Emitter(io.BytesIO()), 'copy', publication_lock=threading.Lock())
        result = self.api().run({'operation': 'write', 'path': str(self.src), 'metadata_json': '{}', 'source_token': self.api().read_metadata(self.src)['source_token']}, ctx)
        self.assertTrue(pathlib.Path(result['path']).is_file())
        self.assertEqual(ctx.publication_state, 'committed')
        cancelled = JobContext(Emitter(io.BytesIO()), 'cancel', publication_lock=threading.Lock())
        cancelled.interrupt_event.set()
        with self.assertRaises(EngineError) as caught:
            self.api().run({'operation': 'write', 'path': str(self.src), 'new_name': 'cancelled', 'metadata_json': '{}'}, cancelled)
        self.assertEqual(caught.exception.code, 'interrupted')
        self.assertFalse((self.root / 'cancelled.safetensors').exists())

    def test_actual_context_cancel_after_copy_chunk_never_publishes(self):
        from sdfx.protocol import Emitter, JobContext, EngineError
        ctx = JobContext(Emitter(io.BytesIO()), 'copy', publication_lock=threading.Lock())
        with self.assertRaises(EngineError) as caught:
            self.write('{}', check=ctx.check_interrupt,
                progress=lambda done, total: ctx.interrupt_event.set(), publish=ctx.publish)
        self.assertEqual(caught.exception.code, 'interrupted')
        self.assertEqual(ctx.publication_state, 'open')
        self.assertEqual(list(self.root.iterdir()), [self.src])

    def test_worker_invalid_operation_does_not_write(self):
        from sdfx.protocol import Emitter, JobContext, EngineError
        ctx = JobContext(Emitter(io.BytesIO()), 'bad')
        with self.assertRaises(EngineError) as caught:
            self.api().run({'operation': 'delete', 'path': str(self.src)}, ctx)
        self.assertEqual(caught.exception.code, 'invalid_params')
        self.assertEqual(list(self.root.iterdir()), [self.src])

    def test_worker_rejects_replacement_with_restored_timestamp(self):
        self.stale_source(True)

    def test_worker_rejects_inplace_same_size_edit_with_restored_timestamp(self):
        self.stale_source(False)

    def stale_source(self, replacement):
        import os
        from sdfx.protocol import Emitter, JobContext, EngineError
        fixture(self.src, {'version': 'A'})
        read = self.api().run({'operation': 'read', 'path': str(self.src)},
            JobContext(Emitter(io.BytesIO()), 'read'))
        before = self.src.stat()
        changed = self.src.read_bytes()[:-8] + b'87654321'
        if replacement:
            other = self.root / 'replacement.safetensors'
            other.write_bytes(changed)
            os.replace(other, self.src)
        else:
            self.src.write_bytes(changed)
        os.utime(self.src, ns=(before.st_atime_ns, before.st_mtime_ns))
        ctx = JobContext(Emitter(io.BytesIO()), 'write', publication_lock=threading.Lock())
        with self.assertRaises(EngineError):
            self.api().run({'operation': 'write', 'path': str(self.src), 'metadata_json': read['editable'],
                'source_token': read.get('source_token', '')}, ctx)
        self.assertFalse((self.root / 'model_md.safetensors').exists())
        self.assertEqual(self.src.read_bytes(), changed)

    def test_real_queued_protocol_copies_and_returns_readable_metadata(self):
        from sdfx.dryrun import FakeEngine
        from sdfx.protocol import Emitter
        from sdfx.server import Server
        stream = io.BytesIO()
        server = Server(FakeEngine(), Emitter(stream))
        server.worker.start()
        try:
            server.handle_line(json.dumps({'id': 'copy', 'type': 'metadata_utils', 'params': {
                'operation': 'write', 'path': str(self.src), 'metadata_json': '{"name":"new"}', 'source_token': self.api().read_metadata(self.src)['source_token']}}))
            deadline = time.monotonic() + 5
            final = []
            while time.monotonic() < deadline:
                final = [json.loads(line) for line in stream.getvalue().splitlines()
                         if json.loads(line).get('id') == 'copy' and json.loads(line)['type'] in ('result', 'error')]
                if final:
                    break
                time.sleep(.01)
            self.assertEqual(len(final), 1)
            self.assertEqual(final[0]['type'], 'result', final)
            self.assertEqual(unpack(pathlib.Path(final[0]['result']['path']))[0]['__metadata__'], {'name': 'new'})
        finally:
            server.q.put(None)
            server.worker.join(timeout=5)


if __name__ == '__main__':
    unittest.main()
