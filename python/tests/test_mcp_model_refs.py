import sys
from types import ModuleType

import pytest

from sdfx.mcp_model_refs import encoder_source


@pytest.fixture
def hub(monkeypatch):
    module = ModuleType("huggingface_hub")
    module.snapshot_download = lambda **kw: pytest.fail("unexpected download")
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return module


def test_explicit_remote_reference_cannot_be_shadowed_by_a_local_directory(tmp_path, monkeypatch, hub):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "owner" / "model").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(hub, "snapshot_download", lambda **kw: calls.append(kw) or "/cache/snapshot")
    assert encoder_source("hf://owner/model") == "/cache/snapshot"
    assert calls == [{"repo_id": "owner/model"}]


@pytest.mark.parametrize("source", ["hf://../private", "hf://owner/../private", "hf://owner/model?x=1", "hf://C:/file"])
def test_invalid_remote_references_fail_before_download(source, monkeypatch, hub):
    monkeypatch.setattr(hub, "snapshot_download", lambda **kw: pytest.fail("must not download"))
    with pytest.raises(ValueError):
        encoder_source(source)


def test_normal_desktop_reference_is_unchanged():
    assert encoder_source("h94/IP-Adapter") == "h94/IP-Adapter"
    assert encoder_source("C:/models/encoder") == "C:/models/encoder"
