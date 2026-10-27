"""Real Server routing with a synthetic trusted provider and fake generator."""
import io
import json
from pathlib import Path
from PIL import Image
import pytest
from sdfx.base import BaseEngine
from sdfx.batch_interrogate_backends import BackendRegistry
from sdfx.batch_interrogate_worker import BatchInterrogateService
from sdfx.protocol import Emitter
from sdfx.server import Server, QUEUED_TYPES


class Provider:
    def interrogate(self, image, ctx): return "caption"
    def unload(self): pass


class Engine(BaseEngine):
    def __init__(self):
        from sdfx.wd_v3_assets import AssetPin
        from sdfx.wd_v3_provider import WdV3Provider
        from sdfx.wd_v3_runtime import _MODEL_IDENTITY, _TAGS_IDENTITY
        super().__init__(); self.model = {"model_token": "token"}
        model = AssetPin(Path("fake-model"), *_MODEL_IDENTITY)
        tags = AssetPin(Path("fake-tags"), *_TAGS_IDENTITY)
        self._batch_interrogate_service = BatchInterrogateService(
            BackendRegistry({"wd_ext": WdV3Provider(model, tags, Path("fake-temp"))}))
    def img2img(self, request, ctx):
        path = Path(request["output_dir"]) / "fake.png"; path.write_bytes(b"fake")
        return {"images": [{"path": str(path)}]}


@pytest.fixture(autouse=True)
def fake_inference(monkeypatch):
    import sdfx.wd_v3_provider as provider
    from sdfx.wd_v3_runtime import ScoredTag
    monkeypatch.setattr(provider, "run_internal", lambda *args: (ScoredTag("caption", 0, .8),))


def invoke(server, stream, request_id, typ, params, owner="owner"):
    server._handle((request_id, typ, params, owner))
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    return next(row for row in reversed(rows) if row.get("id") == request_id and row["type"] in ("error", "result"))


def fixture(tmp_path):
    images = tmp_path / "images"; images.mkdir()
    Image.new("RGB", (2, 2), "red").save(images / "one.png")
    output = tmp_path / "out"; output.mkdir()
    return {"images_root": str(images), "generation": {"params": {"prompt": "base"}},
            "output_parent": str(output), "output_name": "result", "checkpoint_token": "token"}, output


def test_real_server_prepares_worker_owned_snapshot_and_executes_once(tmp_path):
    params, output = fixture(tmp_path); stream = io.BytesIO(); server = Server(Engine(), Emitter(stream))
    assert "batch_interrogate_prepare" in QUEUED_TYPES and "batch_interrogate_execute" in QUEUED_TYPES
    prep = invoke(server, stream, "prep", "batch_interrogate_prepare", params)
    assert prep["type"] == "result" and prep["result"]["image_count"] == 1
    token = prep["result"]["token"]
    wrong_owner = invoke(server, stream, "other", "batch_interrogate_execute", {"token": token}, "other")
    assert wrong_owner["type"] == "error" and not (output / "result").exists()
    result = invoke(server, stream, "run", "batch_interrogate_execute", {"token": token})
    assert result["type"] == "result" and result["result"]["rows"][0]["positive"] == "caption, base"
    assert (output / "result" / "item-0000" / "fake.png").exists()
    replay = invoke(server, stream, "again", "batch_interrogate_execute", {"token": token})
    assert replay["type"] == "error"


def test_public_cannot_supply_registry_snapshot_or_trusted_pins(tmp_path):
    params, output = fixture(tmp_path); stream = io.BytesIO(); server = Server(Engine(), Emitter(stream))
    params["registry"] = {"wd_ext": "forged"}
    result = invoke(server, stream, "bad", "batch_interrogate_prepare", params)
    assert result["type"] == "error" and list(output.iterdir()) == []


def test_worker_without_trusted_service_is_explicitly_unsupported(tmp_path):
    params, output = fixture(tmp_path); stream = io.BytesIO(); server = Server(BaseEngine(), Emitter(stream))
    result = invoke(server, stream, "unsupported", "batch_interrogate_prepare", params)
    assert result["type"] == "error" and result["code"] == "unsupported"
    assert "batch_interrogate" not in server.engine.features()


def test_service_rejects_fake_provider_or_unapproved_pin(tmp_path):
    import pytest
    from sdfx.wd_v3_assets import AssetPin
    from sdfx.wd_v3_provider import WdV3Provider
    with pytest.raises(ValueError): BatchInterrogateService(BackendRegistry({"wd_ext": Provider()}))
    pin = AssetPin(tmp_path / "unapproved", 1, "a" * 64)
    with pytest.raises(ValueError):
        BatchInterrogateService(BackendRegistry({"wd_ext": WdV3Provider(pin, pin, tmp_path)}))


def test_prepared_token_capacity_expiry_and_owner(tmp_path, monkeypatch):
    import sdfx.batch_interrogate_worker as worker
    params, output = fixture(tmp_path); stream = io.BytesIO(); server = Server(Engine(), Emitter(stream))
    clock = [100.0]
    monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(worker, "MAX_PREPARED", 1)
    prep = invoke(server, stream, "prep", "batch_interrogate_prepare", params)
    token = prep["result"]["token"]
    full = invoke(server, stream, "full", "batch_interrogate_prepare", params)
    assert full["type"] == "error" and "pending" in full["message"]
    wrong = invoke(server, stream, "wrong", "batch_interrogate_execute", {"token": token}, "other")
    assert wrong["type"] == "error"
    clock[0] += worker.PREPARED_TTL_SECONDS + 1
    expired = invoke(server, stream, "expired", "batch_interrogate_execute", {"token": token})
    assert expired["type"] == "error" and list(output.iterdir()) == []
    fresh = invoke(server, stream, "fresh", "batch_interrogate_prepare", params)
    assert fresh["type"] == "result"


@pytest.mark.parametrize("point", ["before_consume", "after_consume"])
def test_server_cancel_around_single_use_consumption(tmp_path, monkeypatch, point):
    import threading
    import sdfx.wd_v3_provider as provider
    from sdfx.wd_v3_runtime import ScoredTag
    params, output = fixture(tmp_path); stream = io.BytesIO(); engine = Engine(); server = Server(engine, Emitter(stream))
    token = invoke(server, stream, "prep", "batch_interrogate_prepare", params)["result"]["token"]
    entered, release = threading.Event(), threading.Event()
    if point == "before_consume":
        original = engine._batch_interrogate_service._prune
        def held_prune(): original(); entered.set(); assert release.wait(5)
        monkeypatch.setattr(engine._batch_interrogate_service, "_prune", held_prune)
    else:
        def held_inference(model, tags, parent, image, ctx):
            entered.set(); assert release.wait(5); ctx.check_interrupt()
            return (ScoredTag("caption", 0, .8),)
        monkeypatch.setattr(provider, "run_internal", held_inference)
    job = threading.Thread(target=server._handle, args=(("run", "batch_interrogate_execute", {"token": token}, "owner"),))
    job.start()
    try:
        assert entered.wait(3)
        server.handle_line(json.dumps({"id": "cancel", "type": "interrupt", "params": {"target_id": "run", "owner_id": "owner"}}))
    finally:
        release.set(); job.join(5)
    assert not job.is_alive()
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert next(e["result"] for e in events if e.get("id") == "cancel")["affected"]
    assert next(e for e in reversed(events) if e.get("id") == "run" and e["type"] == "error")["code"] == "interrupted"
    monkeypatch.undo()
    # Restore fake inference after undo and check whether the token remains.
    monkeypatch.setattr(provider, "run_internal", lambda *args: (ScoredTag("caption", 0, .8),))
    retry = invoke(server, stream, "retry", "batch_interrogate_execute", {"token": token})
    assert retry["type"] == ("result" if point == "before_consume" else "error")


@pytest.mark.parametrize("winner", ["cancel", "commit"])
def test_prepare_token_publication_is_atomic_with_cancel(tmp_path, monkeypatch, winner):
    import threading
    from sdfx.protocol import JobContext
    params, output = fixture(tmp_path); stream = io.BytesIO(); engine = Engine(); server = Server(engine, Emitter(stream))
    entered, release = threading.Event(), threading.Event()
    original = JobContext.publish
    def held(ctx, action):
        if winner == "cancel":
            entered.set(); assert release.wait(5)
            return original(ctx, action)
        def action_held(): entered.set(); assert release.wait(5); return action()
        return original(ctx, action_held)
    monkeypatch.setattr(JobContext, "publish", held)
    job = threading.Thread(target=server._handle, args=(("prep", "batch_interrogate_prepare", params, "owner"),))
    job.start()
    try:
        assert entered.wait(3)
        server.handle_line(json.dumps({"id": "cancel", "type": "interrupt", "params": {"target_id": "prep", "owner_id": "owner"}}))
    finally:
        release.set(); job.join(5)
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    cancel = next(e["result"] for e in events if e.get("id") == "cancel")
    terminal = next(e for e in reversed(events) if e.get("id") == "prep" and e["type"] in ("result", "error"))
    if winner == "cancel":
        assert cancel["affected"] and terminal.get("code") == "interrupted"
        assert not engine._batch_interrogate_service._entries
    else:
        assert cancel.get("too_late") and terminal["type"] == "result"
        assert terminal["result"]["token"] in engine._batch_interrogate_service._entries
    assert list(output.iterdir()) == []
