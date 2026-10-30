import io
import json
import time

from sdfx.dryrun import FakeEngine
from sdfx.protocol import Emitter
from sdfx.server import Server


def test_dry_run_inspector_reports_unsupported_through_queue_without_torch():
    output = io.BytesIO()
    server = Server(FakeEngine(), Emitter(output))
    server.worker.start()
    try:
        server.handle_line(json.dumps({"id": "inspect", "type": "embedding_inspector",
                                       "params": {"operation": "inspect", "text": "cat"}}))
        deadline = time.monotonic() + 3
        while b'"id":"inspect"' not in output.getvalue() and time.monotonic() < deadline:
            time.sleep(0.01)
        reply = next(json.loads(line) for line in output.getvalue().splitlines())
        assert reply["type"] == "error"
        assert reply["code"] == "unsupported"
    finally:
        server.q.put(None)
        server.worker.join(timeout=5)
