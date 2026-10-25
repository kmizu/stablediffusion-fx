import json
import struct

import pytest

from sdfx import models


def write_st(path, tensors: dict, extra: dict | None = None):
    hdr = {}
    off = 0
    for k, shape in tensors.items():
        n = 2
        for s in shape:
            n *= s
        n = min(n, 2)  # do not actually write the data; offsets only need to be well-formed for our reader
        hdr[k] = {"dtype": "F16", "shape": shape, "data_offsets": [off, off + n]}
        off += n
    if extra:
        hdr.update(extra)
    b = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(b)))
        f.write(b)
        f.write(b"\0" * off)
    return path


UNET_IN = "model.diffusion_model.input_blocks.0.0.weight"


def sd15(inch=4):
    return {UNET_IN: [320, inch, 3, 3],
            "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight": [49408, 768],
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": [320, 768]}


def sdxl(inch=4):
    return {UNET_IN: [320, inch, 3, 3],
            "model.diffusion_model.label_emb.0.0.weight": [1280, 2816],
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight": [49408, 768],
            "conditioner.embedders.1.model.token_embedding.weight": [49408, 1280]}


@pytest.mark.parametrize("tensors,kind", [
    (sd15(), "sd15"),
    (sd15(9), "sd15-inpaint"),
    (sdxl(), "sdxl"),
    (sdxl(9), "sdxl-inpaint"),
    ({UNET_IN: [320, 4, 3, 3], "cond_stage_model.model.token_embedding.weight": [49408, 1024]}, "sd2"),
    ({"double_blocks.0.img_attn.qkv.weight": [9216, 3072]}, "unsupported"),
    ({"model.diffusion_model.double_blocks.0.img_attn.qkv.weight": [9216, 3072]}, "unsupported"),
    ({"model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight": [4608, 1536]}, "unsupported"),
    ({"lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight": [4, 320]}, "unsupported"),
    ({UNET_IN: [320, 4, 3, 3], "model.diffusion_model.label_emb.0.0.weight": [1280, 2560],
      "conditioner.embedders.0.model.token_embedding.weight": [49408, 1280]}, "unsupported"),  # refiner
    ({UNET_IN: [320, 8, 3, 3], "cond_stage_model.transformer.x": [1]}, "unsupported"),
])
def test_kind_detection(tmp_path, tensors, kind):
    p = write_st(tmp_path / "m.safetensors", tensors)
    assert models.detect_kind(p) == kind


def test_flags_and_garbage(tmp_path):
    p = write_st(tmp_path / "v.safetensors", sdxl(), {"v_pred": {"dtype": "F16", "shape": [0], "data_offsets": [0, 0]}})
    assert models.checkpoint_flags(models.read_safetensors_header(p))["v_pred"]
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"\xff" * 4)
    assert models.detect_kind(bad) == "unknown"
    ck = tmp_path / "x.ckpt"
    ck.write_bytes(b"not read")
    assert models.detect_kind(ck) == "unknown"


def test_scan_models_recursive(tmp_path):
    (tmp_path / "sub" / "deep").mkdir(parents=True)
    write_st(tmp_path / "a.safetensors", sd15())
    write_st(tmp_path / "sub" / "deep" / "b.safetensors", sdxl())
    (tmp_path / "sub" / "c.ckpt").write_bytes(b"x" * 10)
    (tmp_path / "readme.txt").write_text("x")
    res = models.scan_models([str(tmp_path), str(tmp_path / "missing")])
    names = {m["name"]: m for m in res}
    assert set(names) == {"a", "sub/deep/b", "sub/c"}
    assert names["a"]["kind"] == "sd15" and names["sub/deep/b"]["kind"] == "sdxl" and names["sub/c"]["kind"] == "unknown"
    assert all(set(m) == {"name", "path", "kind", "size_mb"} for m in res)


def test_scan_files(tmp_path):
    (tmp_path / "cn_dir").mkdir()
    (tmp_path / "cn_dir" / "config.json").write_text("{}")
    (tmp_path / "cn_dir" / "diffusion_pytorch_model.safetensors").write_bytes(b"1234")
    (tmp_path / "x.pth").write_bytes(b"1")
    (tmp_path / "y.txt").write_bytes(b"1")
    res = models.scan_files("controlnet", [str(tmp_path)])
    assert [r["name"] for r in res] == ["cn_dir", "x"]
    res = models.scan_files("lora", [str(tmp_path)])
    assert [r["name"] for r in res] == ["cn_dir/diffusion_pytorch_model", "x"]
    with pytest.raises(ValueError):
        models.scan_files("bogus", [])


def test_short_hash_and_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SDFX_HASH_CACHE", str(tmp_path / "cache.json"))
    f = tmp_path / "f.bin"
    f.write_bytes(b"hello world")
    # sha256("hello world")
    assert models.short_hash(f) == "b94d27b993"
    cache = json.loads((tmp_path / "cache.json").read_text())
    assert any(v["sha256"].startswith("b94d27b993") for v in cache.values())
    assert models.short_hash(f) == "b94d27b993"
