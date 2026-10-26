"""The notice collector must not turn a transient GitHub failure into a missing licence."""
import base64
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("third_party", Path(__file__).resolve().parents[2] / "scripts/third_party.py")
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)


def response(code, stderr="", data=None):
    return SimpleNamespace(returncode=code, stderr=stderr, stdout=json.dumps(data or {}))


def test_transient_failure_is_retried(monkeypatch):
    calls = iter([response(1, "gh: Bad Gateway (HTTP 502)"), response(0, data={
        "license": {"spdx_id": "MIT"}, "content": base64.b64encode(b"original text").decode()})])
    monkeypatch.setattr(tp.subprocess, "run", lambda *a, **kw: next(calls))
    assert tp.gh_license("example/project") == ("MIT", "original text")


def test_persistent_fetch_error_does_not_mean_no_licence(monkeypatch):
    monkeypatch.setattr(tp.subprocess, "run", lambda *a, **kw: response(1, "gh: Forbidden (HTTP 403)"))
    with pytest.raises(RuntimeError, match="example/project"):
        tp.gh_license("example/project")


def test_actual_404_can_report_missing_licence(monkeypatch):
    monkeypatch.setattr(tp.subprocess, "run", lambda *a, **kw: response(1, "gh: Not Found (HTTP 404)"))
    assert tp.gh_license("example/project") == (None, None)


def test_missing_upstream_file_preserves_archived_notice(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/extensions-status.json").write_text("[]", encoding="utf-8")
    (tmp_path / "licenses").mkdir()
    archived = tmp_path / "licenses/example__project.txt"
    archived.write_text("previously retrieved licence", encoding="utf-8")
    monkeypatch.setattr(tp, "ROOT", str(tmp_path))
    monkeypatch.setattr(tp, "BASE", [("example/project", "reimplemented", "test")])
    monkeypatch.setattr(tp, "gh_license", lambda repo: (None, None))
    tp.main()
    row = next(line for line in (tmp_path / "THIRD_PARTY_NOTICES.md").read_text().splitlines()
               if line.startswith("| [example/project]"))
    assert "[text](licenses/example__project.txt)" in row
    assert archived.read_text() == "previously retrieved licence"
