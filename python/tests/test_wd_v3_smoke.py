"""Smoke helper contract with fake session; no actual ONNX inference."""
import importlib.util
import json
from pathlib import Path
import types
import weakref
from PIL import Image
import pytest
from sdfx import wd_v3_runtime


spec = importlib.util.spec_from_file_location("wd_v3_cpu_smoke", Path(__file__).parents[2] / "scripts" / "wd_v3_cpu_smoke.py")
smoke = importlib.util.module_from_spec(spec); spec.loader.exec_module(smoke)


def test_smoke_reports_hashes_scores_cpu_and_session_release(tmp_path):
    model = tmp_path / "model"; model.write_bytes(b"model")
    tags = tmp_path / "tags"; tags.write_bytes(b"tags")
    image = tmp_path / "source.png"; Image.new("RGBA", (2, 3), (10, 20, 30, 0)).save(image)
    class Session: pass
    def run(model_pin, tags_pin, parent, source, ctx, *, _ort):
        assert source.mode == "RGBA" and source.size == (2, 3)
        session = _ort.InferenceSession(b"fake", providers=["CPUExecutionProvider"])
        assert session is not None
        return (wd_v3_runtime.ScoredTag("blue_hair", 0, .8), wd_v3_runtime.ScoredTag("general", 9, .9))
    module = types.SimpleNamespace(InferenceSession=lambda *a, **k: Session())
    report = smoke.run_smoke(model, tags, image, tmp_path, _runner=run, _ort_module=module)
    assert report["session_released"] is True and report["provider"] == "CPUExecutionProvider"
    assert report["input"]["size"] == image.stat().st_size and report["input"]["dimensions"] == [2, 3]
    assert len(report["input"]["sha256"]) == 64 and len(report["scores_sha256"]) == 64
    assert report["top_tags"][0]["name"] == "blue_hair"
    json.dumps(report, allow_nan=False)


def test_smoke_detects_source_change(tmp_path):
    model = tmp_path / "model"; model.write_bytes(b"model")
    tags = tmp_path / "tags"; tags.write_bytes(b"tags")
    image = tmp_path / "source.png"; Image.new("RGB", (2, 2)).save(image)
    def mutate(*args, **kwargs):
        image.write_bytes(b"changed")
        return ()
    with pytest.raises(ValueError, match="(?i)(changed|input)"):
        smoke.run_smoke(model, tags, image, tmp_path, _runner=mutate,
                        _ort_module=types.SimpleNamespace(InferenceSession=lambda *a, **k: None))
