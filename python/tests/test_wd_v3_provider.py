"""Internal WD provider contract using synthetic scores; no real inference."""
from pathlib import Path
import pytest
from PIL import Image
from sdfx.batch_interrogate_backends import BackendRegistry
from sdfx.wd_v3_assets import AssetPin
from sdfx.wd_v3_runtime import ScoredTag
from sdfx.wd_v3_provider import WdV3Options, WdV3Provider


class Context:
    def check_interrupt(self): pass


def test_wd_provider_preserves_alpha_and_thresholds_while_other_backend_remains_rgb(tmp_path, monkeypatch):
    import sdfx.wd_v3_provider as provider
    image = tmp_path / "image.png"; Image.new("RGBA", (2, 2), (255, 0, 0, 0)).save(image)
    seen = []
    def fake_run(model, tags, parent, pixels, ctx):
        seen.append(("wd", pixels.mode, pixels.getpixel((0, 0))))
        return (ScoredTag("blue_hair", 0, .7), ScoredTag("equal", 0, .35),
                ScoredTag(">_<", 0, .8), ScoredTag("rating", 9, .5), ScoredTag("other", 9, .49))
    monkeypatch.setattr(provider, "run_internal", fake_run)
    class Rgb:
        def interrogate(self, pixels, ctx): seen.append(("rgb", pixels.mode, pixels.getpixel((0, 0)))); return "rgb"
        def unload(self): seen.append(("rgb", "unload"))
    fakepin = AssetPin(tmp_path / "fake", 1, "0" * 64)
    wd = WdV3Provider(fakepin, fakepin, tmp_path, WdV3Options(append_ratings=True))
    registry = BackendRegistry({"clip": Rgb(), "wd_ext": wd})
    with registry.owned(("clip", "wd_ext")):
        result = registry.interrogate(image, ("clip", "wd_ext"), Context())
    assert result == ["rgb", "blue hair, >_<, rating"]
    assert seen == [("rgb", "RGB", (255, 0, 0)), ("wd", "RGBA", (255, 0, 0, 0)), ("rgb", "unload")]


def test_provider_rejects_bad_options_and_never_downloads(tmp_path, monkeypatch):
    pin = AssetPin(tmp_path / "model", 1, "0" * 64)
    for opts in (WdV3Options(threshold=float("nan")), WdV3Options(rating_threshold=1.1),
                 WdV3Options(remove_underscores=1)):
        with pytest.raises(ValueError): WdV3Provider(pin, pin, tmp_path, opts)


def test_registry_rejects_unknown_input_mode_before_backend_invocation(tmp_path):
    class Bad:
        input_mode = "CMYK"
        def interrogate(self, *args): pytest.fail("invalid mode must not run")
        def unload(self): pass
    with pytest.raises(ValueError): BackendRegistry({"clip": Bad()})


def test_wd_asset_cleanup_failure_aborts_owned_batch_before_next_input(tmp_path, monkeypatch):
    import hashlib
    import sdfx.wd_v3_provider as provider
    from sdfx.batch_face_manifest import FileSnapshot
    from sdfx.batch_interrogate import BatchInterrogateOptions
    from sdfx.batch_interrogate_runner import BatchInput, run_registered_batch
    from sdfx.wd_v3_assets import owned_assets
    model = tmp_path / "model.onnx"; model.write_bytes(b"model")
    tags = tmp_path / "tags.csv"; tags.write_bytes(b"tags")
    bad = AssetPin(model, 5, "0" * 64)
    tag_pin = AssetPin(tags, 4, hashlib.sha256(b"tags").hexdigest())
    inputs = []
    for index in range(2):
        path = tmp_path / f"image-{index}.png"; Image.new("RGB", (1, 1)).save(path)
        info = path.stat()
        inputs.append(BatchInput(str(path), FileSnapshot(path, (1, 1), hashlib.sha256(path.read_bytes()).hexdigest(),
                                 info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)))
    calls = []
    def fake_run(model_pin, tags_pin, parent, image, ctx):
        calls.append(1)
        with owned_assets(model_pin, tags_pin, parent, ctx): pass
    monkeypatch.setattr(provider, "run_internal", fake_run)
    original = Path.unlink
    def fail_unlink(path, *args, **kwargs):
        if path.parent.name.startswith("sdfx-wd-v3-"): raise OSError("cleanup denied")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    wd = WdV3Provider(bad, tag_pin, tmp_path)
    registry = BackendRegistry({"wd_ext": wd})
    output = tmp_path / "out"; output.mkdir()
    class Engine:
        def img2img(self, *args): pytest.fail("generation must not start")
    class Publish(Context):
        def publish(self, action): pytest.fail("failed batch must not publish")
    with pytest.raises(ValueError) as caught:
        run_registered_batch(Engine(), inputs, {"params": {}}, ("wd_ext",),
                             BatchInterrogateOptions(), registry, output, "result", Publish())
    assert len(calls) == 1 and "approved identity" in str(caught.value)
    assert "cleanup denied" in str(caught.value) and "sdfx-wd-v3-" in str(caught.value)


@pytest.mark.parametrize("case", ["later_failure", "success_cleanup", "engine_error_cleanup"])
def test_wd_cleanup_failure_keeps_batch_unpublished_for_all_exit_paths(tmp_path, monkeypatch, case):
    import hashlib
    import sdfx.wd_v3_provider as provider
    from sdfx.batch_face_manifest import FileSnapshot
    from sdfx.batch_interrogate import BatchInterrogateOptions
    from sdfx.batch_interrogate_runner import BatchInput, run_registered_batch
    from sdfx.protocol import EngineError
    from sdfx.wd_v3_assets import owned_assets
    model = tmp_path / "model.onnx"; model.write_bytes(b"model")
    tags = tmp_path / "tags.csv"; tags.write_bytes(b"tags")
    good = AssetPin(model, 5, hashlib.sha256(b"model").hexdigest())
    bad = AssetPin(model, 5, "0" * 64)
    tag_pin = AssetPin(tags, 4, hashlib.sha256(b"tags").hexdigest())
    inputs = []
    for index in range(2):
        path = tmp_path / f"source-{index}.png"; Image.new("RGB", (1, 1)).save(path)
        info = path.stat()
        inputs.append(BatchInput(str(path), FileSnapshot(path, (1, 1), hashlib.sha256(path.read_bytes()).hexdigest(),
                                 info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)))
    calls, generated, publications = [], [], []
    def fake_run(model_pin, tags_pin, parent, image, ctx):
        calls.append(1)
        if case == "later_failure" and len(calls) == 1:
            return (ScoredTag("first", 0, .8),)
        with owned_assets(bad if case == "later_failure" else good, tags_pin, parent, ctx):
            if case == "engine_error_cleanup": raise EngineError("invalid_params", "model rejected")
        return (ScoredTag("unused", 0, .8),)
    monkeypatch.setattr(provider, "run_internal", fake_run)
    original = Path.unlink
    def fail_unlink(path, *args, **kwargs):
        if path.parent.name.startswith("sdfx-wd-v3-"): raise OSError("cleanup denied")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    output = tmp_path / "out"; output.mkdir()
    class Engine:
        def img2img(self, request, ctx):
            path = Path(request["output_dir"]) / "result.png"; Image.new("RGB", (64, 64)).save(path)
            generated.append(path)
            return {"images": [{"path": str(path), "seed": 123, "width": 64, "height": 64}]}
    class Publish(Context):
        def publish(self, action): publications.append(1); return action()
    registry = BackendRegistry({"wd_ext": WdV3Provider(good, tag_pin, tmp_path)})
    with pytest.raises(Exception) as caught:
        run_registered_batch(Engine(), inputs, {"params": {}}, ("wd_ext",),
                             BatchInterrogateOptions(), registry, output, "result", Publish())
    assert not publications and not (output / "result").exists() and list(output.iterdir()) == []
    assert len(generated) == (1 if case == "later_failure" else 0)
    assert "cleanup denied" in str(caught.value)
    if case == "engine_error_cleanup":
        assert isinstance(caught.value, EngineError) and caught.value.code == "invalid_params"
        assert "model rejected" in caught.value.message


def test_registered_batch_returns_wd_model_identity_with_actual_prompt(tmp_path, monkeypatch):
    import hashlib
    import sdfx.wd_v3_provider as provider
    from sdfx.batch_face_manifest import FileSnapshot
    from sdfx.batch_interrogate import BatchInterrogateOptions
    from sdfx.batch_interrogate_runner import BatchInput, run_registered_batch
    image = tmp_path / "source.png"; Image.new("RGB", (1, 1)).save(image)
    info = image.stat()
    item = BatchInput(str(image), FileSnapshot(image, (1, 1), hashlib.sha256(image.read_bytes()).hexdigest(),
                      info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns))
    model = AssetPin(tmp_path / "model", 123, "a" * 64)
    tags = AssetPin(tmp_path / "tags", 45, "b" * 64)
    monkeypatch.setattr(provider, "run_internal", lambda *args: (ScoredTag("blue_hair", 0, .8),))
    wd = WdV3Provider(model, tags, tmp_path)
    output = tmp_path / "out"; output.mkdir()
    class Engine:
        def img2img(self, request, ctx):
            path = Path(request["output_dir"]) / "image.png"; Image.new("RGB", (64, 64)).save(path)
            return {"images": [{"path": str(path), "seed": 123, "width": 64, "height": 64}]}
    class Publish(Context):
        def publish(self, action): return action()
    result = run_registered_batch(Engine(), [item], {"params": {"prompt": "base"}}, ["wd_ext"],
               BatchInterrogateOptions(), BackendRegistry({"wd_ext": wd}), output, "result", Publish())
    row = result[0]
    assert row["positive"] == "blue hair, base"
    assert row["interrogators"] == [{"backend": "wd_ext", "model_id": wd.model_id,
                                      "model_sha256": "a" * 64, "tags_sha256": "b" * 64,
                                      "threshold": .35, "remove_underscores": True,
                                      "append_ratings": False, "rating_threshold": .5,
                                      "preprocessing": "exif-white-alpha-square-pillow-bicubic-bgr-f32-v1"}]


def test_provider_provenance_records_nondefault_options(tmp_path):
    pin = AssetPin(tmp_path / "fake", 1, "a" * 64)
    wd = WdV3Provider(pin, pin, tmp_path, WdV3Options(.2, False, True, .7))
    metadata = BackendRegistry({"wd_ext": wd}).provenance(("wd_ext",))[0]
    assert (metadata["threshold"], metadata["remove_underscores"], metadata["append_ratings"],
            metadata["rating_threshold"]) == (.2, False, True, .7)


def test_huge_thresholds_are_structured_invalid_options_and_provenance(tmp_path):
    pin = AssetPin(tmp_path / "fake", 1, "a" * 64)
    with pytest.raises(ValueError): WdV3Provider(pin, pin, tmp_path, WdV3Options(threshold=10 ** 1000))
    class Provider:
        def interrogate(self, *args): return ""
        def unload(self): pass
        def provenance(self):
            return {"model_id": "fake", "model_sha256": "a" * 64, "tags_sha256": "b" * 64,
                    "threshold": 10 ** 1000, "rating_threshold": .5, "remove_underscores": True,
                    "append_ratings": False, "preprocessing": "fake"}
    with pytest.raises(ValueError): BackendRegistry({"wd_ext": Provider()}).provenance(("wd_ext",))
