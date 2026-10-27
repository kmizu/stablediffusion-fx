"""Internal immutable request boundary; no worker protocol registration."""
import pytest
from pathlib import Path
from sdfx.batch_face_manifest import FileSnapshot
from sdfx.batch_interrogate import BatchInterrogateOptions
from sdfx.batch_interrogate_runner import BatchInput, TransactionFailure
from sdfx.batch_interrogate_request import prepare_request, execute_prepared, internal_support


def snapshot(tmp_path):
    path = tmp_path / "input.png"
    return BatchInput(str(path), FileSnapshot(path, (1, 1), "a" * 64, 1, 2, 5, 7))


def test_request_freezes_mutable_generation_and_selection(tmp_path):
    generation = {"params": {"prompt": "original", "img2img": {"denoising_strength": .5}}}
    models = ["wd_ext"]
    req = prepare_request("token", [snapshot(tmp_path)], generation, models, BatchInterrogateOptions(), tmp_path, "result")
    generation["params"]["prompt"] = "changed"; models.append("clip")
    assert req.models == ("wd_ext",) and req.generation()["params"]["prompt"] == "original"
    copy = req.generation(); copy["params"]["prompt"] = "also changed"
    assert req.generation()["params"]["prompt"] == "original"


def test_support_reports_partial_and_missing_backend_not_public_capability():
    support = internal_support()
    assert support.public_capability is False and support.full_parity is False
    assert support.candidate == "wd-vit-tagger-v3" and set(support.unavailable) >= {"clip", "deepbooru", "clip_ext"}


@pytest.mark.parametrize("models", [["clip"], ["wd_ext", "deepbooru"], [], ["wd_ext", "wd_ext"]])
def test_unsupported_or_duplicate_selection_fails_at_admission(tmp_path, models):
    with pytest.raises(ValueError):
        prepare_request("token", [snapshot(tmp_path)], {"params": {}}, models, BatchInterrogateOptions(), tmp_path, "result")


def test_checkpoint_change_aborts_before_worker_or_generation(tmp_path, monkeypatch):
    import sdfx.batch_interrogate_request as boundary
    req = prepare_request("token", [snapshot(tmp_path)], {"params": {}}, ["wd_ext"], BatchInterrogateOptions(), tmp_path, "result")
    calls = []
    class Engine:
        model = {"model_token": "other"}
        def img2img(self, request, ctx): calls.append("generated")
    monkeypatch.setattr(boundary, "run_registered_batch", lambda *args: calls.append("worker"))
    with pytest.raises(TransactionFailure): execute_prepared(Engine(), req, object(), object())
    assert not calls


def test_checkpoint_reload_between_interrogation_and_generation_aborts(tmp_path, monkeypatch):
    import sdfx.batch_interrogate_request as boundary
    req = prepare_request("token", [snapshot(tmp_path)], {"params": {}}, ["wd_ext"], BatchInterrogateOptions(), tmp_path, "result")
    calls = []
    class Engine:
        model = {"model_token": "token"}
        def img2img(self, request, ctx): calls.append("generated")
    engine = Engine()
    def dispatch(proxy, *args, **kwargs):
        engine.model = {"model_token": "reloaded"}
        return proxy.img2img({}, object())
    monkeypatch.setattr(boundary, "run_registered_batch", dispatch)
    with pytest.raises(TransactionFailure): execute_prepared(engine, req, object(), object())
    assert not calls


@pytest.mark.parametrize("winner", ["cancel", "commit"])
def test_empty_batch_result_has_server_owned_terminal_race(tmp_path, monkeypatch, winner):
    import hashlib
    import io
    import json
    import threading
    from PIL import Image
    from sdfx.base import BaseEngine
    from sdfx.batch_interrogate_backends import BackendRegistry
    from sdfx.protocol import Emitter, JobContext
    from sdfx.server import Server
    image = tmp_path / "source.png"; Image.new("RGB", (2, 2)).save(image)
    info = image.stat()
    item = BatchInput(str(image), FileSnapshot(image, (2, 2), hashlib.sha256(image.read_bytes()).hexdigest(),
                      info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns))
    output = tmp_path / "out"; output.mkdir()
    req = prepare_request("token", [item], {"params": {}}, ["wd_ext"], BatchInterrogateOptions(), output, "result")
    class Backend:
        def interrogate(self, *args): raise ValueError("cannot interrogate")
        def unload(self): pass
    registry = BackendRegistry({"wd_ext": Backend()})
    class Engine(BaseEngine):
        def __init__(self):
            super().__init__()
            self.model = {"model_token": "token"}
        def batch_local(self, params, ctx): return {"rows": execute_prepared(self, req, registry, ctx)}
    stream = io.BytesIO(); server = Server(Engine(), Emitter(stream))
    entered, release = threading.Event(), threading.Event()
    original = JobContext.publish
    def held_publish(ctx, operation):
        if winner == "cancel":
            entered.set(); assert release.wait(5)
            return original(ctx, operation)
        def held_operation():
            entered.set(); assert release.wait(5)
            return operation()
        return original(ctx, held_operation)
    monkeypatch.setattr(JobContext, "publish", held_publish)
    job = threading.Thread(target=server._handle, args=(("job", "batch_local", {}, "owner"),))
    job.start()
    try:
        assert entered.wait(3), "job did not enter terminal ownership boundary"
        server.handle_line(json.dumps({"id": "cancel", "type": "interrupt", "params": {"target_id": "job", "owner_id": "owner"}}))
    finally:
        release.set(); job.join(5)
    assert not job.is_alive()
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    cancel = next(e["result"] for e in events if e.get("id") == "cancel")
    terminal = next(e for e in reversed(events) if e.get("id") == "job" and e["type"] in ("result", "error"))
    if winner == "cancel":
        assert cancel["accepted"] and cancel["affected"] and terminal.get("code") == "interrupted"
    else:
        assert cancel.get("too_late") and not cancel["accepted"] and terminal["type"] == "result"
        assert terminal["result"]["rows"][0]["status"] == "error"
    assert list(output.iterdir()) == [] and image.exists()
