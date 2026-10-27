"""Synthetic local asset and pixel fixtures; no ONNX session or network."""
import hashlib
from pathlib import Path
import pytest
from PIL import Image
from sdfx.wd_v3_assets import AssetPin, owned_assets, parse_tags, prepare_image


def pin(path):
    data = path.read_bytes()
    return AssetPin(path, len(data), hashlib.sha256(data).hexdigest())


class Context:
    def check_interrupt(self): pass


def test_owned_assets_verify_both_copies_and_cleanup(tmp_path):
    model = tmp_path / "model.onnx"; model.write_bytes(b"synthetic onnx")
    tags = tmp_path / "tags.csv"; tags.write_bytes(b"tag_id,name,category,count\n1,general,9,1\n")
    with owned_assets(pin(model), pin(tags), tmp_path, Context()) as assets:
        assert assets.model.read_bytes() == model.read_bytes()
        assert assets.tags.read_bytes() == tags.read_bytes()
        model.write_bytes(b"changed source")
        assert assets.model.read_bytes() == b"synthetic onnx"
        staged = assets.model.parent
    assert not staged.exists() and model.exists() and tags.exists()


def test_wrong_digest_and_link_source_are_rejected_without_staging(tmp_path):
    model = tmp_path / "model.onnx"; model.write_bytes(b"model")
    tags = tmp_path / "tags.csv"; tags.write_bytes(b"tags")
    wrong = AssetPin(model, 5, "0" * 64)
    with pytest.raises(ValueError):
        with owned_assets(wrong, pin(tags), tmp_path, Context()): pass
    alias = tmp_path / "alias.onnx"
    try: alias.symlink_to(model)
    except OSError: pytest.skip("symlink unavailable")
    with pytest.raises(ValueError):
        with owned_assets(pin(alias), pin(tags), tmp_path, Context()): pass
    assert not list(tmp_path.glob("sdfx-wd-v3-*"))


def test_strict_csv_schema_and_bounded_values():
    rows = parse_tags(b"tag_id,name,category,count\n1,general,9,3\n2,blue_hair,0,4\n3,person,4,9\n")
    assert [(row.name, row.category) for row in rows] == [("general", 9), ("blue_hair", 0), ("person", 4)]
    for invalid in (b"tag_id,name,category\n1,x,0\n", b"tag_id,name,category,count\n1,x,7,3\n",
                    b"tag_id,name,category,count\n1,x,0,1\n1,y,0,1\n",
                    b"tag_id,name,category,count\n1,x,0,1,extra\n"):
        with pytest.raises(ValueError): parse_tags(invalid)


def test_preprocessing_alpha_padding_bgr_and_bounded_square():
    image = Image.new("RGBA", (2, 1), (255, 0, 0, 0)); image.putpixel((0, 0), (255, 0, 0, 255))
    array = prepare_image(image, 2)
    assert array.shape == (1, 2, 2, 3) and str(array.dtype) == "float32"
    assert list(array[0, 0, 0]) == [0.0, 0.0, 255.0]
    assert list(array[0, 0, 1]) == [255.0, 255.0, 255.0]
    assert list(array[0, 1, 0]) == [255.0, 255.0, 255.0]
    with pytest.raises(ValueError): prepare_image(Image.new("RGB", (1, 5000)), 448)


def test_cleanup_attempts_every_file_and_retains_interruption_details(tmp_path, monkeypatch):
    from sdfx.protocol import EngineError
    model = tmp_path / "model.onnx"; model.write_bytes(b"model")
    tags = tmp_path / "tags.csv"; tags.write_bytes(b"tags")
    original = Path.unlink
    attempted = []
    def unlink(path, *args, **kwargs):
        if path.parent.name.startswith("sdfx-wd-v3-"):
            attempted.append(path.name)
            if path.name == "model.onnx": raise OSError("model unlink denied")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", unlink)
    stage = None
    with pytest.raises(EngineError) as caught:
        with owned_assets(pin(model), pin(tags), tmp_path, Context()) as bound:
            stage = bound.model.parent
            raise EngineError("interrupted", "original interruption")
    assert attempted == ["model.onnx", "tags.csv"]
    assert (stage / "model.onnx").exists() and not (stage / "tags.csv").exists()
    assert caught.value.code == "interrupted" and "original interruption" in caught.value.message
    assert "model unlink denied" in caught.value.message and str(stage) in caught.value.message


def test_failed_copy_reports_primary_and_cleanup_failure(tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"; model.write_bytes(b"model")
    tags = tmp_path / "tags.csv"; tags.write_bytes(b"tags")
    original = Path.unlink
    def unlink(path, *args, **kwargs):
        if path.parent.name.startswith("sdfx-wd-v3-"): raise OSError("cleanup denied")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(ValueError) as caught:
        with owned_assets(AssetPin(model, 5, "0" * 64), pin(tags), tmp_path, Context()): pass
    assert "approved identity" in str(caught.value) and "cleanup denied" in str(caught.value)
    assert "sdfx-wd-v3-" in str(caught.value)


def test_interrupt_during_copy_removes_partial_stage_and_keeps_sources(tmp_path, monkeypatch):
    import sdfx.wd_v3_assets as assets
    from sdfx.protocol import EngineError
    model = tmp_path / "model.onnx"; model.write_bytes(b"model bytes")
    tags = tmp_path / "tags.csv"; tags.write_bytes(b"tag bytes")
    monkeypatch.setattr(assets, "_CHUNK", 2)
    class CancelDuringCopy:
        calls = 0
        partial_seen = False
        def check_interrupt(self):
            self.calls += 1
            if self.calls == 3:
                staged = list(tmp_path.glob("sdfx-wd-v3-*/model.onnx"))
                assert len(staged) == 1
                self.partial_seen = True
                raise EngineError("interrupted", "copy cancelled")
    ctx = CancelDuringCopy()
    with pytest.raises(EngineError) as caught:
        with owned_assets(pin(model), pin(tags), tmp_path, ctx): pass
    assert ctx.partial_seen and caught.value.code == "interrupted"
    assert not list(tmp_path.glob("sdfx-wd-v3-*"))
    assert model.read_bytes() == b"model bytes" and tags.read_bytes() == b"tag bytes"


def test_cleanup_failure_after_success_is_marked_fatal(tmp_path, monkeypatch):
    from sdfx.wd_v3_assets import WdCleanupFailure
    model = tmp_path / "model.onnx"; model.write_bytes(b"model")
    tags = tmp_path / "tags.csv"; tags.write_bytes(b"tags")
    original = Path.unlink
    def fail(path, *args, **kwargs):
        if path.parent.name.startswith("sdfx-wd-v3-"): raise OSError("cleanup denied")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", fail)
    with pytest.raises(WdCleanupFailure) as error:
        with owned_assets(pin(model), pin(tags), tmp_path, Context()): pass
    assert error.value._sdfx_cleanup_failed and "cleanup denied" in str(error.value)
