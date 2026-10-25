"""Dry-run protocol tests for the non-generation built-ins (face restore, caption, img2vid, novel view, custom code)
and for the new hook names being accepted by txt2img/img2img."""
import os

import pytest

from test_server_dryrun import Engine, gen_params, write_ckpt

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


@pytest.fixture
def eng():
    e = Engine()
    yield e
    if e.p.poll() is None:
        e.p.kill()


def img(tmp_path, name="in.png", size=(64, 48)):
    p = tmp_path / name
    Image.new("RGB", size, (120, 60, 30)).save(p)
    return str(p)


def test_hello_lists_new_features(eng):
    hello = eng.next()
    for f in ("apg", "mahiro", "kohya_hrfix", "multidiffusion", "soft_inpainting", "ip_adapter", "img2vid", "caption"):
        assert f in hello["features"]
    assert "custom_code" not in hello["features"]


def test_restore_faces_caption_novel_view(eng, tmp_path):
    eng.next()
    src = img(tmp_path)
    model = tmp_path / "GFPGANv1.4.pth"
    model.write_bytes(b"x")
    r, _ = eng.call("restore_faces", {"image_path": src, "model": str(model), "output_dir": str(tmp_path)})
    assert r["type"] == "result" and os.path.isfile(r["result"]["path"])
    r, _ = eng.call("restore_faces", {"image_path": src, "model": str(tmp_path / "missing.pth"), "output_dir": str(tmp_path)})
    assert r["type"] == "error" and r["code"] == "invalid_params"
    r, _ = eng.call("caption", {"image_path": src, "method": "blip"})
    assert r["result"]["caption"]
    r, _ = eng.call("caption", {"image_path": src, "method": "nope"})
    assert r["type"] == "error"
    r, _ = eng.call("novel_view", {"image_path": src, "output_dir": str(tmp_path), "model": str(tmp_path)})
    assert r["type"] == "result" and Image.open(r["result"]["path"]).size == (64, 48)


def test_img2vid_progress_and_gif(eng, tmp_path):
    eng.next()
    src = img(tmp_path)
    model = tmp_path / "svd.safetensors"
    model.write_bytes(b"x")
    r, events = eng.call("img2vid", {"image_path": src, "model": str(model), "output_dir": str(tmp_path), "frames": 6,
                                     "steps": 3, "width": 256, "height": 128})
    assert r["type"] == "result"
    assert r["result"]["path"].endswith(".gif") and Image.open(r["result"]["path"]).n_frames == 6
    assert os.path.isfile(r["result"]["preview"])
    assert [e["step"] for e in events if e.get("event") == "progress"] == [1, 2, 3]


def test_custom_code_disabled_by_default(eng, tmp_path):
    eng.next()
    r, _ = eng.call("run_code", {"code": "result = 1", "output_dir": str(tmp_path)})
    assert r["type"] == "error" and r["code"] == "unsupported"


def test_custom_code_enabled_with_env(tmp_path):
    e = Engine(env={"SDFX_ALLOW_CODE": "1"})
    try:
        hello = e.next()
        assert "custom_code" in hello["features"]
        r, _ = e.call("run_code", {"code": "print('x')\nresult = 2 + 2", "output_dir": str(tmp_path)})
        assert r["result"]["result"] == "4" and "x" in r["result"]["stdout"]
    finally:
        e.close()


def test_generation_accepts_all_new_hooks(eng, tmp_path):
    eng.next()
    ck = write_ckpt(tmp_path / "m.safetensors")
    eng.call("load_model", {"path": ck})
    hooks = {k: {} for k in ("apg", "mahiro", "rescale_cfg", "model_sampling", "kohya_hrfix", "multidiffusion",
                             "latent_modifier", "sag", "neveroom", "reference", "revision", "ip_adapter")}
    r, _ = eng.call("txt2img", gen_params(str(tmp_path), hooks=hooks))
    assert r["type"] == "result", r


def test_new_preprocessors_dry_run(eng, tmp_path):
    hello = eng.next()
    assert hello["dry_run"]
    r, _ = eng.call("list_samplers", {})
    pres = r["result"]["preprocessors"]
    assert {"recolor_luminance", "tile_resample", "depth_marigold", "inpaint"} <= set(pres)
    assert "LDSR" in r["result"]["upscalers"]
    src = img(tmp_path)
    for name in ("recolor_luminance", "tile_colorfix", "depth_marigold"):
        r, _ = eng.call("preprocess", {"preprocessor": name, "image_path": src, "output_dir": str(tmp_path), "args": {}})
        assert r["type"] == "result" and Image.open(r["result"]["path"]).size == (64, 48)
