"""Fake CPU session tests; no real model, network or inference."""
import hashlib
import weakref
import numpy as np
import pytest
from PIL import Image
onnx = pytest.importorskip("onnx")
from onnx import helper, TensorProto
from sdfx.protocol import EngineError
from sdfx.wd_v3_assets import AssetPin
from sdfx.wd_v3_onnx_audit import AuditPolicy
import sdfx.wd_v3_runtime as runtime


def setup(tmp_path, monkeypatch):
    graph = helper.make_graph([helper.make_node("Identity", ["image"], ["scores"])], "fake",
          [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 2, 2, 3])],
          [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 2])])
    data = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]).SerializeToString()
    model = tmp_path / "source.onnx"; model.write_bytes(data)
    tags = tmp_path / "source.csv"; tags.write_bytes(b"tag_id,name,category,count\n1,a,0,1\n2,b,9,1\n")
    mp = AssetPin(model, len(data), hashlib.sha256(data).hexdigest())
    tp = AssetPin(tags, tags.stat().st_size, hashlib.sha256(tags.read_bytes()).hexdigest())
    monkeypatch.setattr(runtime, "_MODEL_IDENTITY", (mp.size, mp.sha256))
    monkeypatch.setattr(runtime, "_TAGS_IDENTITY", (tp.size, tp.sha256))
    monkeypatch.setattr(runtime, "_POLICY", AuditPolicy(17, ("Identity",), (1, 2, 2, 3), (1, 2)))
    return mp, tp, data


class Context:
    def check_interrupt(self): pass


class FakeOrt:
    class GraphOptimizationLevel:
        ORT_DISABLE_ALL = 0
    class SessionOptions: pass
    last_ref = None
    captured = None
    result = np.array([[0.2, 0.8]], dtype=np.float32)
    class Info:
        def __init__(self, name, shape): self.name, self.shape, self.type = name, shape, "tensor(float)"
    class Session:
        def get_inputs(self): return [FakeOrt.Info("image", [1, 2, 2, 3])]
        def get_outputs(self): return [FakeOrt.Info("scores", [1, 2])]
        def get_providers(self): return ["CPUExecutionProvider"]
        def run(self, names, inputs):
            assert list(inputs["image"][0, 0, 0]) == [255, 255, 255]
            return [FakeOrt.result]
    @staticmethod
    def InferenceSession(model, sess_options, providers):
        assert providers == ["CPUExecutionProvider"] and isinstance(model, bytes)
        FakeOrt.captured = model
        result = FakeOrt.Session(); FakeOrt.last_ref = weakref.ref(result)
        return result


def test_same_audited_bytes_feed_cpu_session_preserving_alpha_and_release(tmp_path, monkeypatch):
    mp, tp, data = setup(tmp_path, monkeypatch)
    original = runtime.inspect_model_bytes
    def audit_then_replace(blob, *args, **kwargs):
        result = original(blob, *args, **kwargs)
        next(tmp_path.glob("sdfx-wd-v3-*/model.onnx")).write_bytes(b"substituted")
        return result
    monkeypatch.setattr(runtime, "inspect_model_bytes", audit_then_replace)
    image = Image.new("RGBA", (2, 2), (255, 0, 0, 0))
    result = runtime.run_internal(mp, tp, tmp_path, image, Context(), _ort=FakeOrt)
    assert FakeOrt.captured == data and FakeOrt.last_ref() is None
    assert len(result) == 2 and result[0].name == "a"
    assert not list(tmp_path.glob("sdfx-wd-v3-*"))


@pytest.mark.parametrize("result", [np.array([[float("nan"), 0.3]], dtype=np.float32),
                                      np.array([[0.3]], dtype=np.float32),
                                      np.array([[1.1, 0.0]], dtype=np.float32)])
def test_invalid_scores_fail_and_session_releases(tmp_path, monkeypatch, result):
    mp, tp, _ = setup(tmp_path, monkeypatch); monkeypatch.setattr(FakeOrt, "result", result)
    with pytest.raises(ValueError):
        runtime.run_internal(mp, tp, tmp_path, Image.new("RGBA", (2, 2)), Context(), _ort=FakeOrt)
    assert FakeOrt.last_ref() is None and not list(tmp_path.glob("sdfx-wd-v3-*"))


def test_cancellation_before_session_never_calls_runtime(tmp_path, monkeypatch):
    mp, tp, _ = setup(tmp_path, monkeypatch)
    class Cancel:
        def check_interrupt(self): raise EngineError("interrupted", "cancelled")
    monkeypatch.setattr(FakeOrt, "InferenceSession", lambda *a, **k: pytest.fail("session must not start"))
    with pytest.raises(EngineError): runtime.run_internal(mp, tp, tmp_path, Image.new("RGB", (2, 2)), Cancel(), _ort=FakeOrt)
    assert not list(tmp_path.glob("sdfx-wd-v3-*"))


def test_constructor_failure_drops_model_bytes_from_traceback(tmp_path, monkeypatch):
    mp, tp, _ = setup(tmp_path, monkeypatch)
    def fail(model, **kwargs):
        raise RuntimeError("session construction failed")
    monkeypatch.setattr(FakeOrt, "InferenceSession", fail)
    with pytest.raises(RuntimeError, match="construction failed") as caught:
        runtime.run_internal(mp, tp, tmp_path, Image.new("RGB", (2, 2)), Context(), _ort=FakeOrt)
    trace = caught.value.__traceback__
    runtime_frames = []
    while trace is not None:
        assert trace.tb_frame.f_code is not fail.__code__, "constructor traceback retains its model argument"
        if trace.tb_frame.f_code is runtime.run_internal.__code__:
            runtime_frames.append(trace.tb_frame)
        trace = trace.tb_next
    assert len(runtime_frames) == 1
    assert runtime_frames[0].f_locals.get("model_bytes") is None
    assert not list(tmp_path.glob("sdfx-wd-v3-*"))


def test_audit_failure_clears_bytes_and_parser_frames(tmp_path, monkeypatch):
    mp, tp, _ = setup(tmp_path, monkeypatch)
    def failed_audit(data, *args, **kwargs):
        raise ValueError("audit rejected model")
    monkeypatch.setattr(runtime, "inspect_model_bytes", failed_audit)
    with pytest.raises(ValueError, match="audit rejected model") as caught:
        runtime.run_internal(mp, tp, tmp_path, Image.new("RGB", (2, 2)), Context(), _ort=FakeOrt)
    trace = caught.value.__traceback__
    while trace is not None:
        assert trace.tb_frame.f_code is not failed_audit.__code__
        if trace.tb_frame.f_code is runtime.run_internal.__code__:
            assert trace.tb_frame.f_locals.get("model_bytes") is None
        trace = trace.tb_next
    assert not list(tmp_path.glob("sdfx-wd-v3-*"))


def test_interrupt_after_audit_clears_model_bytes(tmp_path, monkeypatch):
    mp, tp, _ = setup(tmp_path, monkeypatch)
    class CancelAfterAudit:
        audited = False
        def check_interrupt(self):
            if self.audited: raise EngineError("interrupted", "after audit")
    ctx = CancelAfterAudit()
    original = runtime.inspect_model_bytes
    def set_cancel(*args, **kwargs):
        result = original(*args, **kwargs)
        ctx.audited = True
        return result
    monkeypatch.setattr(runtime, "inspect_model_bytes", set_cancel)
    with pytest.raises(EngineError) as caught:
        runtime.run_internal(mp, tp, tmp_path, Image.new("RGB", (2, 2)), ctx, _ort=FakeOrt)
    assert caught.value.code == "interrupted"
    trace = caught.value.__traceback__
    while trace is not None:
        if trace.tb_frame.f_code is runtime.run_internal.__code__:
            assert trace.tb_frame.f_locals.get("model_bytes") is None
        trace = trace.tb_next
    assert not list(tmp_path.glob("sdfx-wd-v3-*"))
