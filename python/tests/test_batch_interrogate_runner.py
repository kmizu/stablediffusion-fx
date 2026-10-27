"""Per-item orchestration using synthetic backends and owned transactions."""
import copy
import pytest
from sdfx.batch_interrogate import BatchInterrogateOptions
from sdfx.batch_interrogate_runner import BatchInput, run_batch
from sdfx.protocol import EngineError


class Context:
    def __init__(self): self.cancelled = False
    def check_interrupt(self):
        if self.cancelled: raise EngineError("interrupted", "cancelled")


class Backend:
    def __init__(self): self.calls = []
    def interrogate(self, image, models, ctx):
        self.calls.append((image, tuple(models)))
        return [f"caption-{image}", "shared"]


class Transaction:
    def __init__(self, index, commits):
        self.directory = f"owned-stage-{index}"
        self.index, self.commits = index, commits
    def bind_input(self, item, ctx): return item.path
    def __enter__(self): return self
    def __exit__(self, kind, error, tb): return False
    def commit(self, result, ctx):
        ctx.check_interrupt()
        self.commits.append(self.index)
        return result


class Engine:
    def __init__(self): self.requests = []
    def img2img(self, params, ctx):
        self.requests.append(copy.deepcopy(params))
        return {"images": [{"path": params["output_dir"] + "/generated.png"}]}


def test_each_input_uses_frozen_original_prompt_and_ordered_models():
    engine, backend, ctx, commits = Engine(), Backend(), Context(), []
    request = {"params": {"prompt": "base", "negative_prompt": "bad", "seed": 42,
                          "img2img": {"denoising_strength": 0.5}}, "output_dir": "user-output"}
    original = copy.deepcopy(request)
    inputs = (BatchInput("first.png", "first-id"), BatchInput("second.png", "second-id"))
    checks = []
    results = run_batch(engine, inputs, request, ("clip", "wd"), BatchInterrogateOptions(), backend,
                        lambda item: checks.append(item.identity), lambda index: Transaction(index, commits), ctx)
    assert backend.calls == [("first.png", ("clip", "wd")), ("second.png", ("clip", "wd"))]
    assert [r["params"]["prompt"] for r in engine.requests] == ["caption-first.png, shared, base", "caption-second.png, shared, base"]
    assert [r["params"]["negative_prompt"] for r in engine.requests] == ["bad", "bad"]
    assert [r["params"]["img2img"]["init_image"] for r in engine.requests] == ["first.png", "second.png"]
    assert [r["output_dir"] for r in engine.requests] == ["owned-stage-0", "owned-stage-1"]
    assert checks == ["first-id", "first-id", "second-id", "second-id"]
    assert commits == [0, 1] and len(results) == 2 and request == original


def test_cancellation_stops_before_generation_and_later_items():
    engine, ctx, commits = Engine(), Context(), []
    class CancelBackend(Backend):
        def interrogate(self, image, models, ctx):
            ctx.cancelled = True
            return ["caption"]
    with pytest.raises(EngineError) as error:
        run_batch(engine, (BatchInput("a", "id"), BatchInput("b", "id2")),
                  {"params": {"prompt": "base", "negative_prompt": ""}}, ("clip",),
                  BatchInterrogateOptions(), CancelBackend(), lambda _: None,
                  lambda i: Transaction(i, commits), ctx)
    assert error.value.code == "interrupted" and not engine.requests and not commits


def test_failure_never_reuses_prior_caption_and_continues_per_item():
    class Sometimes(Backend):
        def interrogate(self, image, models, ctx):
            if image == "bad": raise ValueError("unreadable image")
            return [image]
    engine, ctx, commits = Engine(), Context(), []
    results = run_batch(engine, (BatchInput("good", "1"), BatchInput("bad", "2"), BatchInput("last", "3")),
        {"params": {"prompt": "base", "negative_prompt": ""}}, ("clip",), BatchInterrogateOptions(),
        Sometimes(), lambda _: None, lambda i: Transaction(i, commits), ctx)
    assert [r["status"] for r in results] == ["complete", "error", "complete"]
    assert [r["params"]["prompt"] for r in engine.requests] == ["good, base", "last, base"]
    assert commits == [0, 2]


def test_unavailable_backend_and_unsafe_missing_transaction_fail_before_generation():
    engine = Engine()
    with pytest.raises(ValueError):
        run_batch(engine, (BatchInput("a", "id"),), {"params": {}}, ("clip",),
                  BatchInterrogateOptions(), None, lambda _: None, None, Context())
    assert not engine.requests


def test_backend_runtime_error_is_per_item_and_does_not_stop_later_work():
    class Broken(Backend):
        def interrogate(self, image, models, ctx):
            if image == "bad": raise RuntimeError("decoder failed")
            return [image]
    engine, ctx, commits = Engine(), Context(), []
    result = run_batch(engine, (BatchInput("bad", "1"), BatchInput("good", "2")),
        {"params": {"prompt": "base", "negative_prompt": ""}}, ("clip",), BatchInterrogateOptions(),
        Broken(), lambda _: None, lambda i: Transaction(i, commits), ctx)
    assert [row["status"] for row in result] == ["error", "complete"]
    assert "decoder failed" in result[0]["error"] and len(engine.requests) == 1


@pytest.mark.parametrize("extra", [
    {"vae_recovery_dir": "/outside"},
    {"params": {"prompt": "base", "hooks": {"save_intermediate": {"output_dir": "/outside"}}}},
    {"params": {"prompt": "base", "hooks": {"composable_lora": {"plot": True}}}},
])
def test_side_effecting_generation_options_rejected_before_any_backend(extra):
    engine, backend, ctx = Engine(), Backend(), Context()
    request = {"params": {"prompt": "base", "negative_prompt": ""}, **extra}
    with pytest.raises(ValueError, match="(?i)(unsupported|output|hooks)"):
        run_batch(engine, (BatchInput("a", "id"),), request, ("clip",), BatchInterrogateOptions(),
                  backend, lambda _: None, lambda i: Transaction(i, []), ctx)
    assert not backend.calls and not engine.requests


def test_real_owned_transaction_binds_source_and_publishes_only_complete_group(tmp_path):
    import hashlib
    from sdfx.batch_face_manifest import FileSnapshot
    from sdfx.batch_interrogate_runner import OwnedBatchTransaction
    source = tmp_path / "input.png"
    source.write_bytes(b"original")
    stat = source.stat()
    snap = FileSnapshot(source, (1, 1), hashlib.sha256(b"original").hexdigest(),
                        stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    output = tmp_path / "out"; output.mkdir()
    item = BatchInput(str(source), snap)
    ctx = Context()
    ctx.publish = lambda action: action()
    with OwnedBatchTransaction(output, "result", (item,), ctx) as transaction:
        bound = transaction.bind_input(item, 0)
        source.write_bytes(b"modified")  # same length; bound bytes cannot change
        assert bound.read_bytes() == b"original"
        stage = transaction.item_directory(0)
        (stage / "output.png").write_bytes(b"generated")
        result = transaction.publish([{"status": "complete", "input": str(source),
             "result": {"images": [{"path": str(stage / "output.png")}]}}])
    assert result[0]["result"]["images"][0]["path"] == str(output / "result" / "item-0000" / "output.png")
    assert (output / "result" / "item-0000" / "output.png").read_bytes() == b"generated"
    assert source.read_bytes() == b"modified"


def test_owned_runner_uses_same_bound_bytes_for_backend_and_engine(tmp_path):
    import hashlib
    from sdfx.batch_face_manifest import FileSnapshot
    from sdfx.batch_interrogate_runner import run_owned_batch
    source = tmp_path / "source.png"
    source.write_bytes(b"original")
    info = source.stat()
    item = BatchInput(str(source), FileSnapshot(source, (1, 1), hashlib.sha256(b"original").hexdigest(),
                      info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns))
    output = tmp_path / "out"; output.mkdir()
    class ReadBackend:
        def interrogate(self, image, models, ctx):
            from pathlib import Path
            assert Path(image).read_bytes() == b"original"
            source.write_bytes(b"modified")
            return ["caption"]
    class WriteEngine:
        def img2img(self, request, ctx):
            from pathlib import Path
            assert Path(request["params"]["img2img"]["init_image"]).read_bytes() == b"original"
            path = Path(request["output_dir"]) / "result.png"
            path.write_bytes(b"output")
            return {"images": [{"path": str(path)}]}
    ctx = Context(); ctx.publish = lambda action: action()
    result = run_owned_batch(WriteEngine(), (item,), {"params": {}}, ("clip",),
                             BatchInterrogateOptions(), ReadBackend(), output, "result", ctx)
    assert result[0]["status"] == "complete"
    assert (output / "result" / "item-0000" / "result.png").read_bytes() == b"output"


def test_owned_batch_rejects_aggregate_input_budget_before_staging(tmp_path, monkeypatch):
    import hashlib
    import sdfx.batch_interrogate_runner as runner
    from sdfx.batch_face_manifest import FileSnapshot
    monkeypatch.setattr(runner, "MAX_TOTAL_INPUT_BYTES", 5)
    items = []
    for i in range(2):
        path = tmp_path / f"{i}.png"; path.write_bytes(b"abcd"); info = path.stat()
        items.append(BatchInput(str(path), FileSnapshot(path, (1, 1), hashlib.sha256(b"abcd").hexdigest(),
                     info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)))
    out = tmp_path / "out"; out.mkdir()
    with pytest.raises(ValueError, match="(?i)(total|aggregate)"):
        with runner.OwnedBatchTransaction(out, "result", items, Context()): pass
    assert list(out.iterdir()) == []


def test_owned_batch_rejects_excess_generated_count_before_running(tmp_path, monkeypatch):
    import hashlib
    import sdfx.batch_interrogate_runner as runner
    from sdfx.batch_face_manifest import FileSnapshot
    path = tmp_path / "input.png"; path.write_bytes(b"abcd"); info = path.stat()
    item = BatchInput(str(path), FileSnapshot(path, (1, 1), hashlib.sha256(b"abcd").hexdigest(),
                     info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns))
    out = tmp_path / "out"; out.mkdir(); engine = Engine(); backend = Backend()
    monkeypatch.setattr(runner, "MAX_OUTPUT_FILES", 1)
    with pytest.raises(ValueError, match="(?i)(output|limit)"):
        runner.run_owned_batch(engine, (item,), {"params": {"batch_size": 2}}, ("clip",),
                               BatchInterrogateOptions(), backend, out, "result", Context())
    assert not engine.requests and not backend.calls and list(out.iterdir()) == []


def test_item_cleanup_failure_preserves_interruption(tmp_path, monkeypatch):
    import hashlib
    import sdfx.batch_interrogate_runner as runner
    from sdfx.batch_face_manifest import FileSnapshot
    path = tmp_path / "input.png"; path.write_bytes(b"abcd"); info = path.stat()
    item = BatchInput(str(path), FileSnapshot(path, (1, 1), hashlib.sha256(b"abcd").hexdigest(),
                     info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns))
    out = tmp_path / "out"; out.mkdir()
    def fail(*args): raise OSError("cleanup denied")
    monkeypatch.setattr(runner.OwnedBatchTransaction, "discard_item", fail)
    class Cancel:
        def interrogate(self, *args): raise EngineError("interrupted", "cancelled")
    with pytest.raises(EngineError) as error:
        runner.run_owned_batch(Engine(), (item,), {"params": {}}, ("clip",),
                               BatchInterrogateOptions(), Cancel(), out, "result", Context())
    assert error.value.code == "interrupted" and "cleanup denied" in error.value.message


def test_cleanup_failure_aborts_remaining_items_preserving_original_engine_error(tmp_path, monkeypatch):
    import hashlib
    import sdfx.batch_interrogate_runner as runner
    from sdfx.batch_face_manifest import FileSnapshot
    items = []
    for index in range(2):
        path = tmp_path / f"{index}.png"; path.write_bytes(b"abcd"); info = path.stat()
        items.append(BatchInput(str(path), FileSnapshot(path, (1, 1), hashlib.sha256(b"abcd").hexdigest(),
                     info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)))
    out = tmp_path / "out"; out.mkdir()
    monkeypatch.setattr(runner.OwnedBatchTransaction, "discard_item",
                        lambda *args: (_ for _ in ()).throw(OSError("cleanup denied")))
    class Failure:
        calls = 0
        def interrogate(self, *args):
            self.calls += 1
            raise EngineError("invalid_params", "original problem")
    backend = Failure(); engine = Engine()
    with pytest.raises(Exception) as error:
        runner.run_owned_batch(engine, items, {"params": {}}, ("clip",),
                               BatchInterrogateOptions(), backend, out, "result", Context())
    assert backend.calls == 1 and not engine.requests
    assert "invalid_params" in str(error.value) and "original problem" in str(error.value)
    assert "cleanup denied" in str(error.value)


def test_owner_cleanup_failure_retains_plain_primary_reason_and_stage_path(tmp_path, monkeypatch):
    import hashlib
    import sdfx.batch_interrogate_runner as runner
    from sdfx.batch_face_manifest import FileSnapshot
    path = tmp_path / "input.png"; path.write_bytes(b"abcd"); info = path.stat()
    item = BatchInput(str(path), FileSnapshot(path, (1, 1), hashlib.sha256(b"abcd").hexdigest(),
                     info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns))
    out = tmp_path / "out"; out.mkdir()
    transaction = runner.OwnedBatchTransaction(out, "result", [item], Context())
    with pytest.raises(runner.TransactionFailure) as error:
        with transaction:
            monkeypatch.setattr(transaction.transaction, "_check_identity",
                                lambda: (_ for _ in ()).throw(OSError("cleanup denied")))
            raise ValueError("original problem")
    assert "original problem" in str(error.value)
    assert "cleanup denied" in str(error.value)
    assert str(transaction.transaction.staging) in str(error.value)


def test_registered_runner_unloads_before_publication_and_on_preflight_failure(tmp_path):
    import hashlib
    from PIL import Image
    from sdfx.batch_face_manifest import FileSnapshot
    from sdfx.batch_interrogate_backends import BackendRegistry
    from sdfx.batch_interrogate_runner import run_registered_batch
    image = tmp_path / "source.png"; Image.new("RGB", (2, 2)).save(image)
    info = image.stat()
    item = BatchInput(str(image), FileSnapshot(image, (2, 2), hashlib.sha256(image.read_bytes()).hexdigest(),
                      info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns))
    output = tmp_path / "output"; output.mkdir()
    class Provider:
        unloaded = 0
        def interrogate(self, image, ctx): return "caption"
        def unload(self): self.unloaded += 1
    provider = Provider(); registry = BackendRegistry({"clip": provider})
    class WriteEngine:
        def img2img(self, request, ctx):
            from pathlib import Path
            path = Path(request["output_dir"]) / "result.png"
            path.write_bytes(b"output")
            return {"images": [{"path": str(path)}]}
    class Publish(Context):
        def publish(self, action):
            assert provider.unloaded == 1
            return action()
    result = run_registered_batch(WriteEngine(), [item], {"params": {}}, ["clip"],
                                  BatchInterrogateOptions(), registry, output, "result", Publish())
    assert result[0]["status"] == "complete" and provider.unloaded == 1
    with pytest.raises(ValueError):
        run_registered_batch(WriteEngine(), [], {"params": {}}, ["clip"],
                             BatchInterrogateOptions(), registry, output, "second", Publish())
    assert provider.unloaded == 2

