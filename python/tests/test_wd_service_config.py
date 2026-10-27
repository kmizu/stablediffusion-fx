"""Explicit local configuration with tiny synthetic assets, no network/inference."""
import hashlib
from sdfx.base import BaseEngine
from sdfx.wd_service_config import WdLocalConfig, configure_wd_service
from sdfx.__main__ import parse_args


def test_missing_incomplete_or_dryrun_never_advertises(tmp_path, monkeypatch):
    import sdfx.wd_service_config as setup
    monkeypatch.setattr(setup, "_dependencies_available", lambda: True)
    engine = BaseEngine()
    assert not configure_wd_service(engine, None).available
    assert "batch_interrogate_wd_v3_partial" not in engine.features()
    assert not configure_wd_service(engine, WdLocalConfig(str(tmp_path / "missing"), "", str(tmp_path))).available
    class Dry(BaseEngine): dry_run = True
    assert not configure_wd_service(Dry(), WdLocalConfig("a", "b", "c")).available


def test_verified_local_files_enable_partial_only_and_missing_revokes(tmp_path, monkeypatch):
    import sdfx.wd_service_config as setup
    import sdfx.wd_v3_runtime as runtime
    monkeypatch.setattr(setup, "_dependencies_available", lambda: True)
    monkeypatch.setattr(setup, "_initialize_dependencies", lambda: None)
    model = tmp_path / "model.onnx"; model.write_bytes(b"model")
    tags = tmp_path / "tags.csv"; tags.write_bytes(b"tags")
    monkeypatch.setattr(runtime, "_MODEL_IDENTITY", (5, hashlib.sha256(b"model").hexdigest()))
    monkeypatch.setattr(runtime, "_TAGS_IDENTITY", (4, hashlib.sha256(b"tags").hexdigest()))
    engine = BaseEngine()
    result = configure_wd_service(engine, WdLocalConfig(str(model), str(tags), str(tmp_path)))
    assert result.available and "batch_interrogate_wd_v3_partial" in engine.features()
    assert "batch_interrogate" not in engine.features()
    tags.unlink()
    assert "batch_interrogate_wd_v3_partial" not in engine.features()


def test_wrong_digest_and_missing_dependency_never_enable(tmp_path, monkeypatch):
    import sdfx.wd_service_config as setup
    model = tmp_path / "model"; model.write_bytes(b"wrong")
    tags = tmp_path / "tags"; tags.write_bytes(b"wrong")
    cfg = WdLocalConfig(str(model), str(tags), str(tmp_path)); engine = BaseEngine()
    monkeypatch.setattr(setup, "_dependencies_available", lambda: False)
    assert not configure_wd_service(engine, cfg).available
    monkeypatch.setattr(setup, "_dependencies_available", lambda: True)
    assert not configure_wd_service(engine, cfg).available
    assert "batch_interrogate_wd_v3_partial" not in engine.features()


def test_explicit_cli_paths_are_optional_and_not_guessed():
    default = parse_args([])
    assert default.wd_v3_model is None and default.wd_v3_tags is None and default.wd_v3_temp is None
    selected = parse_args(["--wd-v3-model", "model", "--wd-v3-tags", "tags", "--wd-v3-temp", "temp"])
    assert (selected.wd_v3_model, selected.wd_v3_tags, selected.wd_v3_temp) == ("model", "tags", "temp")


def test_unexpected_optional_configuration_error_keeps_base_engine(monkeypatch):
    import sys
    import types
    import sdfx.server as server
    import sdfx.wd_service_config as setup
    engine = BaseEngine()
    fake = types.ModuleType("sdfx.engine"); fake.Engine = lambda: engine
    monkeypatch.setitem(sys.modules, "sdfx.engine", fake)
    def fail(*args): raise RuntimeError("optional config exploded")
    monkeypatch.setattr(setup, "configure_wd_service", fail)
    built = server.build_engine(False, wd_paths=("model", "tags", "temp"))
    assert built is engine and "batch_interrogate_wd_v3_partial" not in built.features()


def test_configured_wd_dependencies_initialize_on_calling_thread_before_readiness(tmp_path, monkeypatch):
    import importlib
    import threading
    import sdfx.wd_service_config as setup
    import sdfx.wd_v3_runtime as runtime
    monkeypatch.setattr(setup, "_dependencies_available", lambda: True)
    model = tmp_path / "model"; model.write_bytes(b"model")
    tags = tmp_path / "tags"; tags.write_bytes(b"tags")
    monkeypatch.setattr(runtime, "_MODEL_IDENTITY", (5, hashlib.sha256(b"model").hexdigest()))
    monkeypatch.setattr(runtime, "_TAGS_IDENTITY", (4, hashlib.sha256(b"tags").hexdigest()))
    imported = []
    def track(name, *args, **kwargs):
        imported.append((name, threading.current_thread()))
        return object()
    monkeypatch.setattr(importlib, "import_module", track)
    engine = BaseEngine()
    assert configure_wd_service(engine, WdLocalConfig(str(model), str(tags), str(tmp_path))).available
    assert {"numpy", "onnx", "onnxruntime", "PIL.Image"} <= {name for name, _ in imported}
    assert all(thread is threading.current_thread() for _, thread in imported)


def test_dependency_initialization_failure_keeps_service_disabled(tmp_path, monkeypatch):
    import importlib
    import sdfx.wd_service_config as setup
    import sdfx.wd_v3_runtime as runtime
    monkeypatch.setattr(setup, "_dependencies_available", lambda: True)
    model = tmp_path / "model"; model.write_bytes(b"model")
    tags = tmp_path / "tags"; tags.write_bytes(b"tags")
    monkeypatch.setattr(runtime, "_MODEL_IDENTITY", (5, hashlib.sha256(b"model").hexdigest()))
    monkeypatch.setattr(runtime, "_TAGS_IDENTITY", (4, hashlib.sha256(b"tags").hexdigest()))
    def fail(name, *args, **kwargs): raise ImportError("broken native dependency")
    monkeypatch.setattr(importlib, "import_module", fail)
    engine = BaseEngine()
    result = configure_wd_service(engine, WdLocalConfig(str(model), str(tags), str(tmp_path)))
    assert not result.available and "batch_interrogate_wd_v3_partial" not in engine.features()
