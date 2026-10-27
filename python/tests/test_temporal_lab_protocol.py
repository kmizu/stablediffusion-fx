"""Temporal Lab queued worker routing, ownership and dry-run protocol."""
import io
import json
import os
from pathlib import Path
import threading
import subprocess
import sys

import pytest

from sdfx.base import BaseEngine
from sdfx.dryrun import FakeEngine
from sdfx.protocol import Emitter, EngineError, INVALID_PARAMS, INTERRUPTED, JobContext
from sdfx.server import Server


def emit(server, kind, rid, params=None, owner=None):
    message = {"id": rid, "type": kind, "params": params or {}}
    if owner is not None:
        message["owner_id"] = owner
    server.handle_line(json.dumps(message))


def responses(stream, rid):
    return [x for line in stream.getvalue().splitlines() if (x := json.loads(line)).get("id") == rid]


def make_context():
    return JobContext(Emitter(io.BytesIO()), "direct")


def test_feature_is_advertised_by_fake_engine():
    assert "temporal_lab" in FakeEngine().features()


@pytest.mark.parametrize("operation,fields", [
    ("dfi", {"reference_dir": "r", "generated_dir": "g", "output_parent": "o", "output_name": "result"}),
    ("preview", {"reference_dir": "r", "generated_dir": "g", "frame_index": 1}),
    ("deflicker", {"frames_dir": "f", "output_parent": "o", "output_name": "result"}),
    ("fuse", {"style_dir": "s", "video_dir": "v", "output_parent": "o", "output_name": "result"}),
    ("extract", {"video_path": "video.avi", "output_parent": "o", "output_name": "result"}),
])
def test_real_engine_dispatch_preserves_operation_parameters(monkeypatch, operation, fields):
    import sdfx.temporal_lab as lab
    seen = []
    ctx = make_context()
    def dispatch(params, context):
        seen.append((params, context))
        return {"operation": operation, "marker": 3}
    monkeypatch.setattr(lab, "run", dispatch)
    params = {"operation": operation, **fields}
    result = BaseEngine().temporal_lab(params, ctx)
    assert result == {"operation": operation, "marker": 3}
    assert seen == [(params, ctx)]


@pytest.mark.parametrize("operation,fields", [
    ("dfi", {"reference_dir": "r", "generated_dir": "g", "output_parent": "o", "output_name": "result"}),
    ("preview", {"reference_dir": "r", "generated_dir": "g", "frame_index": 1}),
    ("deflicker", {"frames_dir": "f", "output_parent": "o", "output_name": "result"}),
    ("fuse", {"style_dir": "s", "video_dir": "v", "output_parent": "o", "output_name": "result"}),
    ("extract", {"video_path": "video.avi", "output_parent": "o", "output_name": "result"}),
])
def test_fake_engine_validates_and_identifies_synthetic_result_without_files(tmp_path, monkeypatch, operation, fields):
    monkeypatch.chdir(tmp_path)
    result = FakeEngine().temporal_lab({"operation": operation, **fields}, make_context())
    assert result["operation"] == operation and result["dry_run"] is True
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("params", [
    {}, {"operation": []}, {"operation": "bad"},
    {"operation": "dfi", "reference_dir": "", "generated_dir": "x", "output_parent": "y", "output_name": "z"},
    {"operation": "dfi", "reference_dir": "r", "generated_dir": "g", "output_parent": "o", "output_name": "z", "tolerance": -1},
    {"operation": "extract", "video_path": "v", "output_parent": "o", "output_name": "z", "fps": -1},
])
def test_fake_engine_rejects_invalid_requests_without_output(params):
    with pytest.raises(EngineError) as e:
        FakeEngine().temporal_lab(params, make_context())
    assert e.value.code == INVALID_PARAMS


def test_temporal_lab_is_queued_serially_and_owned_interrupt_is_scoped(monkeypatch):
    import sdfx.temporal_lab as lab
    entered = threading.Event()
    release = threading.Event()
    seen = []
    def dispatch(params, ctx):
        seen.append(params)
        entered.set()
        assert release.wait(4)
        ctx.check_interrupt()
        ctx.progress(1, 1, pass_="dfi")
        return {"operation": params["operation"]}
    monkeypatch.setattr(lab, "run", dispatch)
    stream = io.BytesIO()
    server = Server(BaseEngine(), Emitter(stream))
    server.worker.start()
    try:
        emit(server, "temporal_lab", "a", {"operation": "dfi"}, "owner-a")
        assert entered.wait(3)
        emit(server, "temporal_lab", "b", {"operation": "fuse"}, "owner-b")
        assert server.current_id == "a" and list(server.queued) == ["b"]
        emit(server, "interrupt", "cancel-b", {"owner_id": "owner-b"})
        response = responses(stream, "cancel-b")[-1]
        assert response["result"]["affected"] and response["result"]["active_id"] is None
    finally:
        release.set()
        server.q.put(None)
        server.worker.join(5)
    assert not server.worker.is_alive()
    assert seen == [{"operation": "dfi"}]
    assert responses(stream, "a")[-1]["result"]["operation"] == "dfi"
    assert any(x.get("event") == "progress" for x in responses(stream, "a"))
    assert responses(stream, "b")[-1]["code"] == INTERRUPTED


def test_worker_survives_invalid_temporal_request_and_reports_context(monkeypatch):
    import sdfx.temporal_lab as lab
    def dispatch(params, ctx):
        if params.get("operation") == "bad":
            raise EngineError(INVALID_PARAMS, "invalid operation")
        return {"operation": params["operation"]}
    monkeypatch.setattr(lab, "run", dispatch)
    stream = io.BytesIO()
    server = Server(BaseEngine(), Emitter(stream))
    server.worker.start()
    try:
        emit(server, "temporal_lab", "bad", {"operation": "bad"}, "owner")
        emit(server, "temporal_lab", "next", {"operation": "fuse"}, "owner")
        server.q.put(None)
        server.worker.join(5)
    finally:
        if server.worker.is_alive():
            server.q.put(None)
            server.worker.join(5)
    assert responses(stream, "bad")[-1]["code"] == INVALID_PARAMS
    assert responses(stream, "next")[-1]["result"]["operation"] == "fuse"


def test_wrong_owner_cannot_interrupt_active_temporal_request(monkeypatch):
    import sdfx.temporal_lab as lab
    entered, release = threading.Event(), threading.Event()
    def dispatch(params, ctx):
        entered.set()
        assert release.wait(4)
        ctx.check_interrupt()
        return {"operation": "dfi"}
    monkeypatch.setattr(lab, "run", dispatch)
    stream = io.BytesIO()
    server = Server(BaseEngine(), Emitter(stream))
    server.worker.start()
    try:
        emit(server, "temporal_lab", "a", {"operation": "dfi"}, "owner-a")
        assert entered.wait(3)
        emit(server, "interrupt", "wrong", {"owner_id": "owner-b", "target_id": "a"})
        assert not responses(stream, "wrong")[-1]["result"]["affected"]
        assert not server.interrupt.is_set()
        emit(server, "interrupt", "right", {"owner_id": "owner-a", "target_id": "a"})
        assert responses(stream, "right")[-1]["result"]["affected"]
    finally:
        release.set()
        server.q.put(None)
        server.worker.join(5)
    assert responses(stream, "a")[-1]["code"] == INTERRUPTED


def test_fake_temporal_validation_without_opencv_or_ml_packages(tmp_path):
    program = r'''
import builtins, io
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in ('cv2', 'torch', 'transformers', 'diffusers'):
        raise ImportError('optional module blocked: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
from sdfx.dryrun import FakeEngine
from sdfx.protocol import Emitter, JobContext
engine = FakeEngine()
assert 'temporal_lab' in engine.features()
result = engine.temporal_lab({'operation':'dfi', 'reference_dir':'r', 'generated_dir':'g',
                              'output_parent':'o', 'output_name':'result'},
                             JobContext(Emitter(io.BytesIO()), 'fake'))
assert result == {'operation': 'dfi', 'dry_run': True}
'''
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    result = subprocess.run([sys.executable, "-c", program], cwd=tmp_path, env=env,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("missing", [("numpy",), ("PIL",), ("numpy", "PIL")])
def test_fake_temporal_without_optional_image_libraries_is_not_advertised_and_is_actionable(tmp_path, missing):
    program = r"""
import builtins, io, json, sys
missing = set(json.loads(sys.argv[1]))
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in missing:
        raise ImportError('optional module blocked: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
from sdfx.dryrun import FakeEngine
from sdfx.protocol import Emitter, EngineError, JobContext, UNSUPPORTED
engine = FakeEngine()
assert 'temporal_lab' not in engine.features()
try:
    engine.temporal_lab({'operation':'dfi', 'reference_dir':'r', 'generated_dir':'g',
                         'output_parent':'o', 'output_name':'result'},
                        JobContext(Emitter(io.BytesIO()), 'fake'))
except EngineError as exc:
    assert exc.code == UNSUPPORTED
    assert 'numpy' in exc.message.lower() and 'pillow' in exc.message.lower()
else:
    raise AssertionError('missing dependencies were accepted')
"""
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    result = subprocess.run([sys.executable, "-c", program, json.dumps(missing)], cwd=tmp_path, env=env,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("opencv_available", [True, False])
def test_real_engine_temporal_capability_tracks_opencv(monkeypatch, opencv_available):
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from sdfx.engine import Engine, cn_mod
    real_has = cn_mod._has
    monkeypatch.setattr(cn_mod, "_has", lambda name: opencv_available if name == "cv2" else real_has(name))
    assert ("temporal_lab" in Engine.__new__(Engine).features()) is opencv_available
