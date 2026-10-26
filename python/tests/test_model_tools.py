"""Checkpoint merge / convert on tiny synthetic state dicts."""

import pytest

torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")

from sdfx import model_tools as mt  # noqa: E402

KEYS = {
    "cond_stage_model.transformer.text_model.embeddings.position_ids": "ids",
    "cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1.weight": (4, 4),
    "model.diffusion_model.time_embed.0.weight": (4, 4),
    "model.diffusion_model.input_blocks.0.0.weight": (8, 4, 3, 3),
    "model.diffusion_model.input_blocks.4.1.proj_in.weight": (4, 4),
    "model.diffusion_model.middle_block.1.proj_in.weight": (4, 4),
    "model.diffusion_model.output_blocks.11.1.proj_out.weight": (4, 4),
    "first_stage_model.decoder.conv_in.weight": (4, 4),
}


def ckpt(path, value, extra=None, conv_in_ch=4):
    sd = {}
    for k, shape in KEYS.items():
        if shape == "ids":
            sd[k] = torch.arange(77, dtype=torch.float32).reshape(1, 77) + 0.3   # broken float position ids
        elif k.endswith("input_blocks.0.0.weight"):
            sd[k] = torch.full((8, conv_in_ch, 3, 3), float(value))
        else:
            sd[k] = torch.full(shape, float(value))
    sd.update(extra or {})
    st.save_file(sd, str(path))
    return str(path)


def load(p):
    return st.load_file(p)


def test_block_mapping():
    assert mt.block_of("model.diffusion_model.input_blocks.4.1.proj_in.weight") == "IN04"
    assert mt.block_of("model.diffusion_model.middle_block.1.x") == "M00"
    assert mt.block_of("model.diffusion_model.output_blocks.11.1.x") == "OUT11"
    assert mt.block_of("model.diffusion_model.time_embed.0.weight") == "BASE"
    assert mt.block_of("cond_stage_model.x") == "BASE" and mt.block_of("first_stage_model.x") is None
    assert len(mt.parse_block_weights(",".join(["0.5"] * 25), False, 0.1)) == 26
    assert mt.parse_block_weights(",".join(["0.2"] + ["1"] * 19), True, 0.9)["BASE"] == 0.2
    with pytest.raises(ValueError):
        mt.parse_block_weights("1,2,3", False, 0.5)


def test_weight_sum_add_difference_and_blocks(tmp_path):
    a, b, c = ckpt(tmp_path / "a.safetensors", 0), ckpt(tmp_path / "b.safetensors", 4), ckpt(tmp_path / "c.safetensors", 1)
    out = str(tmp_path / "m.safetensors")
    r = mt.merge(a, b, out, alpha=0.25, precision="fp32")
    m = load(out)
    assert torch.allclose(m["model.diffusion_model.middle_block.1.proj_in.weight"], torch.full((4, 4), 1.0))
    assert r["recipe"]["model_b"] == "b.safetensors" and r["sdxl"] is False
    mt.merge(a, b, out + "2.safetensors", c=c, mode="Add difference", alpha=0.5, precision="fp32")
    assert load(out + "2.safetensors")["model.diffusion_model.time_embed.0.weight"][0, 0] == pytest.approx(1.5)
    bw = ",".join(["0"] * 4 + ["1"] + ["0"] * 7 + ["0.5"] + ["0"] * 11 + ["1"])      # IN04 = 1, M00 = 0.5, OUT11 = 1
    mt.merge(a, b, out + "3.safetensors", alpha=0.0, block_weights=bw, vae_source="B", precision="fp16")
    m3 = load(out + "3.safetensors")
    assert m3["model.diffusion_model.input_blocks.4.1.proj_in.weight"][0, 0] == 4
    assert m3["model.diffusion_model.middle_block.1.proj_in.weight"][0, 0] == 2
    assert m3["model.diffusion_model.time_embed.0.weight"][0, 0] == 0                 # BASE = alpha = 0
    assert m3["first_stage_model.decoder.conv_in.weight"][0, 0] == 4                  # VAE from B
    assert m3["model.diffusion_model.output_blocks.11.1.proj_out.weight"].dtype == torch.float16


def test_inpainting_conv_in_and_architecture_check(tmp_path):
    a = ckpt(tmp_path / "inp.safetensors", 0, conv_in_ch=9)
    b = ckpt(tmp_path / "b.safetensors", 2)
    out = str(tmp_path / "m.safetensors")
    mt.merge(a, b, out, alpha=0.5, precision="fp32")
    w = load(out)["model.diffusion_model.input_blocks.0.0.weight"]
    assert w.shape[1] == 9 and w[0, 0, 0, 0] == 1 and w[0, 8, 0, 0] == 0
    xl = str(tmp_path / "xl.safetensors")
    st.save_file({"conditioner.embedders.0.x": torch.zeros(2)}, xl)
    with pytest.raises(ValueError):
        mt.merge(b, xl, out + "x.safetensors")


def test_convert_precision_ema_junk_clip_and_vae(tmp_path):
    ema = {"model_ema.diffusion_modelmiddle_block1proj_inweight": torch.full((4, 4), 9.0),
           "optimizer.state": torch.zeros(1)}
    src = ckpt(tmp_path / "a.safetensors", 1, extra=ema)
    vae = tmp_path / "vae.safetensors"
    st.save_file({"decoder.conv_in.weight": torch.full((4, 4), 7.0)}, str(vae))
    out = str(tmp_path / "o.safetensors")
    r = mt.convert(src, out, precision="fp16", ema="ema-only", bake_vae=str(vae))
    o = load(out)
    assert r["ema_weights_used"] == 1 and r["clip_fixed"] == 1 and r["vae_baked"]
    assert "optimizer.state" not in o and not any(k.startswith("model_ema.") for k in o)
    assert o["model.diffusion_model.middle_block.1.proj_in.weight"][0, 0] == 9
    ids = o["cond_stage_model.transformer.text_model.embeddings.position_ids"]
    assert ids.dtype == torch.int64 and ids[0, 76] == 76
    assert o["first_stage_model.decoder.conv_in.weight"][0, 0] == 7
    assert o["model.diffusion_model.time_embed.0.weight"].dtype == torch.float16
    with pytest.raises(ValueError):
        mt.convert(ckpt(tmp_path / "noema.safetensors", 1), out + "2.safetensors", ema="ema-only")


def test_engine_requests_validate_paths(tmp_path):
    from sdfx.base import BaseEngine
    from sdfx.protocol import EngineError
    a = ckpt(tmp_path / "a.safetensors", 0)

    class Ctx:
        def progress(self, *a, **k): pass
        def check_interrupt(self): pass
    eng = BaseEngine.__new__(BaseEngine)
    with pytest.raises(EngineError):
        eng.merge_models({"model_a": a, "model_b": str(tmp_path / "missing.safetensors"), "output": str(tmp_path / "o.safetensors")}, Ctx())
    with pytest.raises(EngineError):
        eng.convert_model({"model": a, "output": a}, Ctx())
    with pytest.raises(EngineError):
        eng.convert_model({"model": a, "output": str(tmp_path / "o.ckpt")}, Ctx())
    r = eng.convert_model({"model": a, "output": str(tmp_path / "o.safetensors")}, Ctx())
    assert r["path"].endswith("o.safetensors")
    with pytest.raises(EngineError):  # exists, no overwrite
        eng.convert_model({"model": a, "output": str(tmp_path / "o.safetensors")}, Ctx())


def test_block_presets_follow_the_layout():
    v = mt.parse_block_weights("preset:GRAD_V", False, 0.3)
    assert v["IN00"] == 1 and v["M00"] == 0 and v["OUT11"] == 1 and v["BASE"] == 0.3
    x = mt.parse_block_weights("preset:MID_ONLY", True, 0.0)
    assert len(x) == 20 and x["M00"] == 1 and x["IN08"] == 0
    assert mt.parse_block_weights("preset:flat_25", False, 0)["OUT05"] == 0.25
    with pytest.raises(ValueError):
        mt.parse_block_weights("preset:NOPE", False, 0)
