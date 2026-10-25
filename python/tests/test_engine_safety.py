"""Pickle safety of the real engine loaders (runs only where torch + diffusers are installed)."""
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from sdfx.engine import safe_torch_load  # noqa: E402
from sdfx.protocol import EngineError  # noqa: E402


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned > pwned.txt",))


def test_malicious_pickle_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "evil.ckpt"
    torch.save({"state_dict": {"w": torch.zeros(1)}, "x": _Evil()}, p)
    with pytest.raises(EngineError) as ei:
        safe_torch_load(str(p), "checkpoint")
    assert ei.value.code == "unsupported" and "refused for safety" in ei.value.message
    assert not (tmp_path / "pwned.txt").exists()


def test_plain_tensor_file_loads(tmp_path):
    p = tmp_path / "ok.pt"
    torch.save({"string_to_param": {"*": torch.ones(2, 768)}, "name": "tok", "step": 10}, p)
    sd = safe_torch_load(str(p), "embedding")
    assert sd["string_to_param"]["*"].shape == (2, 768)


def test_lightning_metadata_is_inert(tmp_path):
    """Old A1111 VAEs pickle a pytorch_lightning ModelCheckpoint; it is mapped to an inert stub."""
    import sys
    import types
    mod = types.ModuleType("pytorch_lightning.callbacks.model_checkpoint")

    class ModelCheckpoint:
        def __init__(self):
            self.monitor = "val/loss"

    ModelCheckpoint.__module__ = mod.__name__
    ModelCheckpoint.__qualname__ = "ModelCheckpoint"
    mod.ModelCheckpoint = ModelCheckpoint
    for name in ("pytorch_lightning", "pytorch_lightning.callbacks", mod.__name__):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules[mod.__name__] = mod
    p = tmp_path / "vae.pt"
    try:
        torch.save({"state_dict": {"w": torch.zeros(1)}, "callbacks": {"cb": ModelCheckpoint()}}, p)
    finally:
        for name in ("pytorch_lightning.callbacks.model_checkpoint", "pytorch_lightning.callbacks", "pytorch_lightning"):
            sys.modules.pop(name, None)
    sd = safe_torch_load(str(p), "VAE")
    assert "w" in sd["state_dict"]
    cb = sd["callbacks"]["cb"]
    assert type(cb).__name__ == "ModelCheckpoint" and type(cb).__mro__[1].__name__ == "_InertStub"
