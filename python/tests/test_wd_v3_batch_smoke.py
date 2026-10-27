"""Exercise smoke wiring with fake scores and a synthetic image generator."""
import importlib.util
from pathlib import Path
from PIL import Image
import pytest

spec = importlib.util.spec_from_file_location("wd_batch_smoke", Path(__file__).parents[2] / "scripts" / "wd_v3_batch_smoke.py")
smoke = importlib.util.module_from_spec(spec); spec.loader.exec_module(smoke)


def test_smoke_connects_wd_caption_to_synthetic_image_and_preserves_source(tmp_path, monkeypatch):
    import sdfx.wd_v3_provider as provider
    from sdfx.wd_v3_runtime import ScoredTag
    model = tmp_path / "model"; model.write_bytes(b"model")
    tags = tmp_path / "tags"; tags.write_bytes(b"tags")
    image = tmp_path / "source.png"; Image.new("RGBA", (2, 3), (1, 2, 3, 4)).save(image)
    original = image.read_bytes()
    output = tmp_path / "output"; output.mkdir()
    def fake_wd(model_pin, tags_pin, parent, pixels, ctx):
        assert pixels.mode == "RGBA" and pixels.getpixel((0, 0)) == (1, 2, 3, 4)
        return (ScoredTag("blue_hair", 0, .8),)
    monkeypatch.setattr(provider, "run_internal", fake_wd)
    result = smoke.run_batch_smoke(model, tags, image, tmp_path, output, "result", "original prompt")
    assert result["generator"] == "synthetic test image; no Stable Diffusion inference"
    assert result["caption"] == "blue hair" and result["generation_prompt"] == "blue hair, original prompt"
    assert result["png_prompt"] == result["generation_prompt"] and result["input_unchanged"] is True
    assert result["staging_clean"] is True and image.read_bytes() == original
    assert Path(result["output"]["path"]).is_file()


def test_smoke_failure_does_not_publish_output(tmp_path, monkeypatch):
    import sdfx.wd_v3_provider as provider
    model = tmp_path / "model"; model.write_bytes(b"model")
    tags = tmp_path / "tags"; tags.write_bytes(b"tags")
    image = tmp_path / "source.png"; Image.new("RGB", (1, 1)).save(image)
    output = tmp_path / "output"; output.mkdir()
    monkeypatch.setattr(provider, "run_internal", lambda *args: (_ for _ in ()).throw(ValueError("failed")))
    with pytest.raises(RuntimeError, match="failed"):
        smoke.run_batch_smoke(model, tags, image, tmp_path, output, "result", "base")
    assert list(output.iterdir()) == [] and image.is_file()
