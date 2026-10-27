"""Actual Temporal Lab server publication/cancellation ownership races."""
import io
import json
import threading

from PIL import Image
import pytest

from sdfx.base import BaseEngine
from sdfx.protocol import Emitter, JobContext
from sdfx.server import Server
from sdfx.temporal_lab_io import OutputTransaction


@pytest.mark.parametrize("winner", ["cancel", "commit"])
def test_owned_temporal_publication_race(tmp_path, monkeypatch, winner):
    output = tmp_path / "output"
    frames = tmp_path / "frames"
    output.mkdir()
    frames.mkdir()
    Image.new("RGB", (16, 16), (20, 90, 130)).save(frames / "frame0001.png")
    params = {"operation": "deflicker", "frames_dir": str(frames),
              "output_parent": str(output), "output_name": "result"}
    stream = io.BytesIO()
    server = Server(BaseEngine(), Emitter(stream))
    entered, release = threading.Event(), threading.Event()
    if winner == "cancel":
        original = JobContext.publish
        def held(ctx, operation):
            entered.set()
            assert release.wait(5)
            return original(ctx, operation)
        monkeypatch.setattr(JobContext, "publish", held)
    else:
        original = OutputTransaction.commit
        def held(tx):
            entered.set()
            assert release.wait(5)
            return original(tx)
        monkeypatch.setattr(OutputTransaction, "commit", held)
    job = threading.Thread(target=server._handle, args=(("job", "temporal_lab", params, "owner"),))
    job.start()
    try:
        assert entered.wait(3), "job never reached publication boundary"
        server.handle_line(json.dumps({"id": "cancel", "type": "interrupt",
                                      "params": {"target_id": "job", "owner_id": "owner"}}))
        server.handle_line(json.dumps({"id": "ping", "type": "ping"}))
    finally:
        release.set()
        job.join(5)
    assert not job.is_alive()
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    cancel = next(e["result"] for e in events if e.get("id") == "cancel")
    terminal = next(e for e in reversed(events) if e.get("id") == "job" and e["type"] in ("result", "error"))
    assert any(e.get("id") == "ping" and e["type"] == "result" for e in events)
    if winner == "cancel":
        assert cancel["accepted"] and cancel["affected"]
        assert terminal.get("code") == "interrupted"
        assert not (output / "result").exists()
    else:
        assert cancel.get("too_late") and not cancel["accepted"] and not cancel["affected"]
        assert terminal["type"] == "result"
        assert (output / "result").is_dir()
    assert not list(output.glob(".sdfx-*"))
