"""Unit tests for the reForge built-in rewrites (pure helpers; torch parts skip when torch is missing)."""
import math
import os
import types

import pytest

from sdfx import adapters, extensions, extras, latent_ops, unet_hooks
from sdfx.protocol import EngineError


# ---------------------------------------------------------------------------------------------- deep shrink / tiles
def test_kohya_block_location_sd15_and_sdxl():
    sd15 = [2, 2, 2, 2]
    assert unet_hooks.kohya_block_location(1, sd15) == (0, 0)
    assert unet_hooks.kohya_block_location(3, sd15) == (0, 2)  # downsampler of level 0
    assert unet_hooks.kohya_block_location(4, sd15) == (1, 0)
    assert unet_hooks.kohya_block_location(11, sd15) == (3, 1)
    assert unet_hooks.kohya_block_location(12, sd15) is None
    sdxl = [2, 2, 2]
    assert unet_hooks.kohya_block_location(8, sdxl) == (2, 1)
    assert unet_hooks.kohya_block_location(9, sdxl) is None
    assert unet_hooks.kohya_block_location(0, sdxl) is None


def test_timestep_window():
    hi, lo = unet_hooks.timestep_window(0.0, 0.35)
    assert hi == pytest.approx(999.0) and lo == pytest.approx(999 * 0.65)


@pytest.mark.parametrize("size,tile,overlap", [(128, 96, 8), (96, 96, 8), (300, 64, 16), (65, 64, 0)])
def test_tile_starts_cover_and_overlap(size, tile, overlap):
    starts = unet_hooks.tile_starts(size, tile, overlap)
    assert starts[0] == 0 and starts[-1] + min(tile, size) == size
    covered = set()
    for s in starts:
        covered.update(range(s, s + min(tile, size)))
    assert covered == set(range(size))
    for a, b in zip(starts, starts[1:]):
        assert a + tile - b >= overlap


def test_tile_grid_and_gaussian_weights():
    tiles = unet_hooks.tile_grid(128, 256, 96, 96, 8)
    assert all(y1 - y0 == 96 and x1 - x0 == 96 for y0, y1, x0, x1 in tiles)
    w = unet_hooks.gaussian_weights(9, 9)
    assert w[4][4] == max(max(r) for r in w)
    assert min(min(r) for r in w) >= w[4][4] * 1e-3  # floor keeps corners usable


def test_normalizers_coerce_types():
    k = unet_hooks.normalize_kohya({"block_number": "4", "downscale_after_skip": "false", "upscale_method": "weird"})
    assert k["block_number"] == 4 and k["downscale_after_skip"] is False and k["upscale_method"] == "bicubic"
    m = unet_hooks.normalize_multidiffusion({"tile_width": "512.0"})
    assert m["tile_width"] == 512 and m["method"] == "MultiDiffusion"


# ------------------------------------------------------------------------------------------------------- latent ops
def test_soft_mask_threshold_schedule():
    assert latent_ops.soft_mask_threshold(0, 10, 1.0) == pytest.approx(0.9)
    assert latent_ops.soft_mask_threshold(9, 10, 1.0) == pytest.approx(0.0)
    assert latent_ops.soft_mask_threshold(4, 10, 2.0) == pytest.approx(0.25)


def test_composite_mask_numpy():
    np = pytest.importorskip("numpy")
    orig = np.zeros((4, 4, 3), np.float32)
    gen = np.ones((4, 4, 3), np.float32)
    m = np.zeros((4, 4), np.float32)
    m[:2] = 1.0
    out = latent_ops.composite_mask(orig, gen, m, latent_ops.normalize_soft_inpaint({}))
    assert out[:2].min() == 1.0 and out[2:].max() == 0.0


def test_normalize_latent_modifier_types():
    c = latent_ops.normalize_latent_modifier({"sharpness_multiplier": "0.5", "extra_noise_lowpass": "50", "tonemap_method": "arctan"})
    assert c["sharpness_multiplier"] == 0.5 and c["extra_noise_lowpass"] == 50 and c["tonemap_method"] == "arctan"


def test_modify_latents_identity_and_effects():
    torch = pytest.importorskip("torch")
    x = torch.randn(1, 4, 16, 16, generator=torch.Generator().manual_seed(0))
    base = latent_ops.normalize_latent_modifier({})
    assert torch.allclose(latent_ops.modify_latents(x, base, 0.5), x)
    drift = dict(base, combat_cfg_drift=1.0)
    y = latent_ops.modify_latents(x + 3.0, drift, 0.5)
    assert y.mean(dim=(-2, -1)).abs().max() < 1e-4
    for k in ("sharpness_multiplier", "tonemap_multiplier", "contrast_multiplier", "divisive_norm_multiplier",
              "spectral_mod_multiplier", "extra_noise_multiplier"):
        y = latent_ops.modify_latents(x, dict(base, **{k: 0.5}), 0.2, torch.Generator().manual_seed(1))
        assert y.shape == x.shape and torch.isfinite(y).all() and not torch.allclose(y, x), k


def test_soft_inpaint_blend():
    torch = pytest.importorskip("torch")
    gen = torch.ones(1, 4, 8, 8)
    orig = torch.zeros(1, 4, 8, 8)
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., :4] = 1.0
    out = latent_ops.soft_inpaint_blend(gen, orig, mask, 5, 10, latent_ops.normalize_soft_inpaint({}))
    assert out[..., :4].min() == 1.0 and out[..., 4:].max() == 0.0


# --------------------------------------------------------------------------------------------------------- guidance
def test_apg_momentum_schedule():
    from sdfx import guidance
    assert guidance.apg_momentum(0.5, 0.0, 500) == 0.5
    assert guidance.apg_momentum(0.5, 0.18, 999) == pytest.approx(0.5 - 0.5 * 0.18 ** 4 * 1)
    assert guidance.apg_momentum(0.5, 1.0, 0) == 0.0
    assert guidance.apg_momentum(-0.5, 1.0, 0) == 0.0


def test_apg_and_mahiro_math():
    torch = pytest.importorskip("torch")
    from sdfx import guidance
    g = torch.Generator().manual_seed(0)
    c, u = torch.randn(1, 4, 8, 8, generator=g), torch.randn(1, 4, 8, 8, generator=g)
    # eta = 1, no momentum, no threshold -> plain CFG
    cfg = u + 5 * (c - u)
    assert torch.allclose(guidance.normalized_guidance(c, u, 5.0, None, 1.0, 0.0), cfg, atol=1e-5)
    # eta = 0 drops the component parallel to cond
    r = guidance.normalized_guidance(c, u, 5.0, None, 0.0, 0.0) - c
    assert abs(float((r * c).sum())) < 1e-3
    m = guidance.mahiro(c, u, cfg, 5.0)
    assert m.shape == c.shape and torch.isfinite(m).all()
    a = guidance.normalize_apg({"momentum": "0.2", "guidance_limiter": "true"})
    assert a["momentum"] == 0.2 and a["guidance_limiter"] is True


# --------------------------------------------------------------------------------------------------------- adapters
def test_ldm_block_mapping_and_lllite_names():
    assert adapters.ldm_block_to_diffusers("input", 4) == ("down_blocks", 1, 0)
    assert adapters.ldm_block_to_diffusers("input", 8) == ("down_blocks", 2, 1)
    assert adapters.ldm_block_to_diffusers("output", 0) == ("up_blocks", 0, 0)
    assert adapters.ldm_block_to_diffusers("output", 5) == ("up_blocks", 1, 2)
    t = adapters.lllite_target("lllite_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q")
    assert t == ("down_blocks", 1, 0, 0, "attn1", "to_q")
    assert adapters.lllite_target("lllite_unet_middle_block_1_transformer_blocks_3_attn2_to_k")[:4] == ("mid_block", 0, 0, 3)
    assert adapters.lllite_target("something_else") is None


def test_adapters_normalize():
    c = adapters.normalize({"strength": "2", "ignore_prompt": "yes"}, adapters.REVISION_DEFAULTS)
    assert c["strength"] == 2.0 and c["ignore_prompt"] is True


def test_revision_apply():
    torch = pytest.importorskip("torch")
    emb = {"prompt_embeds": torch.ones(2, 77, 8), "pooled_prompt_embeds": torch.full((2, 1280), 3.0),
           "negative_pooled_prompt_embeds": torch.ones(2, 1280)}
    ie = torch.ones(1, 1280)
    out = adapters.apply_revision(emb, ie, 0.5, False)
    # forge: positive pooled replaced (not added), negative pooled zeroed, text tokens kept
    assert torch.allclose(out["pooled_prompt_embeds"], torch.full((2, 1280), 0.5))
    assert out["negative_pooled_prompt_embeds"].abs().sum() == 0
    assert torch.equal(out["prompt_embeds"], emb["prompt_embeds"])
    out2 = adapters.apply_revision(emb, ie, 1.0, True)
    assert out2["prompt_embeds"].abs().sum() == 0
    with pytest.raises(EngineError):
        adapters.apply_revision({"prompt_embeds": torch.ones(1, 77, 8)}, ie, 1.0, False)


def test_lllite_module_build_and_forward():
    torch = pytest.importorskip("torch")
    w = {"conditioning1.0.weight": torch.zeros(16, 3, 4, 4), "conditioning1.0.bias": torch.zeros(16),
         "conditioning1.2.weight": torch.zeros(32, 16, 2, 2), "conditioning1.2.bias": torch.zeros(32),
         "down.0.weight": torch.zeros(8, 64), "down.0.bias": torch.zeros(8),
         "mid.0.weight": torch.zeros(8, 40), "mid.0.bias": torch.zeros(8),
         "up.0.weight": torch.zeros(64, 8), "up.0.bias": torch.ones(64)}
    m = adapters.LLLiteModule.build(w, 2.0)
    m.cond_image = torch.zeros(1, 3, 64, 64)
    x = torch.zeros(2, 64, 64)  # 8x8 tokens after the /8 conditioning
    y = m(x)
    assert y.shape == x.shape and torch.allclose(y, torch.full_like(x, 2.0))
    m.active = False
    assert m(x).abs().sum() == 0


# ----------------------------------------------------------------------------------------------------------- extras
def test_expand_box_square_and_clamped():
    assert extras.expand_box((10, 10, 20, 20), 100, 100) == (4, 4, 36, 36)
    l, t, r, b = extras.expand_box((90, 90, 20, 20), 100, 100)
    assert r == 100 and b == 100 and r - l == b - t


def test_run_code_requires_opt_in(monkeypatch):
    monkeypatch.delenv("SDFX_ALLOW_CODE", raising=False)
    with pytest.raises(EngineError):
        extras.run_code("result = 1", {})
    monkeypatch.setenv("SDFX_ALLOW_CODE", "1")
    r = extras.run_code("print('hi')\nresult = 6 * 7", {})
    assert r["result"] == "42" and "hi" in r["stdout"] and r["images"] == []
    with pytest.raises(EngineError):
        extras.run_code("raise ValueError('x')", {})


def test_save_gif(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    frames = [Image.new("RGB", (8, 8), (i * 40, 0, 0)) for i in range(4)]
    p = extras.save_gif(frames, str(tmp_path / "a.gif"), 6)
    assert Image.open(p).n_frames == 4


# -------------------------------------------------------------------------------------------------------- extensions
def gp(**hooks):
    return types.SimpleNamespace(hooks=hooks)


def test_model_sampling_overrides():
    assert extensions.model_sampling_overrides(gp()) == {}
    assert extensions.model_sampling_overrides(gp(model_sampling={"sampling": "v_prediction"})) == {"prediction_type": "v_prediction"}
    z = extensions.model_sampling_overrides(gp(model_sampling={"sampling": "eps", "zsnr": True}))
    assert z == {"prediction_type": "epsilon", "rescale_betas_zero_snr": True, "timestep_spacing": "trailing"}
    assert extensions.model_sampling_overrides(gp(model_sampling={"enabled": False, "sampling": "eps"})) == {}
    with pytest.raises(EngineError):
        extensions.model_sampling_overrides(gp(model_sampling={"sampling": "edm"}))


def test_pre_run_plans():
    eng = types.SimpleNamespace(kind="sd15", sdxl=False)
    p = extensions.pre_run(eng, "inpaint", gp(soft_inpainting={}), mask_present=True, controlnet=False, pag=False)
    assert p.task == "img2img" and p.soft_cfg is not None
    p = extensions.pre_run(eng, "txt2img", gp(soft_inpainting={}), mask_present=False, controlnet=False, pag=False)
    assert p.task == "txt2img" and p.warnings
    p = extensions.pre_run(eng, "txt2img", gp(sag={"scale": 1}), mask_present=False, controlnet=False, pag=False)
    assert p.sag == {"scale": 1.0, "blur_sigma": 2.0}
    p = extensions.pre_run(eng, "img2img", gp(sag={}), mask_present=False, controlnet=False, pag=False)
    assert p.sag is None and p.warnings
    with pytest.raises(EngineError):
        extensions.pre_run(eng, "txt2img", gp(photomaker={"model": "x"}), mask_present=False, controlnet=False, pag=False)


def test_known_hooks_cover_extensions():
    from sdfx.hooks import KNOWN_HOOKS, unknown_hooks
    assert set(extensions.EXTENSION_HOOKS) <= set(KNOWN_HOOKS)
    assert unknown_hooks({"freeu_xyz": {}, "apg": {}}) == ["freeu_xyz"]


# ---------------------------------------------------------------------------------------------- models / preprocessors
def test_scan_new_file_kinds(tmp_path):
    from sdfx import models
    (tmp_path / "GFPGANv1.4.pth").write_bytes(b"x")
    svd = tmp_path / "svd_xt"
    svd.mkdir()
    (svd / "model_index.json").write_text("{}")
    (tmp_path / "inpaint_v26.fooocus.patch").write_bytes(b"x")
    assert [e["name"] for e in models.scan_files("face_restore", [str(tmp_path)])][0] == "GFPGANv1.4"
    assert "svd_xt" in [e["name"] for e in models.scan_files("svd", [str(tmp_path)])]
    assert any(e["name"].startswith("inpaint_v26") for e in models.scan_files("controlnet", [str(tmp_path)]))


@pytest.mark.parametrize("name", ["recolor_luminance", "recolor_intensity", "tile_resample", "tile_colorfix", "blur_gaussian"])
def test_pil_preprocessors(name):
    Image = pytest.importorskip("PIL.Image")
    pytest.importorskip("numpy")
    from sdfx import controlnet
    img = Image.new("RGB", (32, 24), (200, 30, 60))
    out = controlnet.pil_preprocess(name, img, {"down_sampling_rate": 2.0})
    assert out.size == img.size and out.mode == "RGB"
    if name.startswith("recolor"):
        r, g, b = out.getpixel((0, 0))
        assert r == g == b


def test_upscaler_list_has_ldsr():
    from sdfx.base import PIXEL_UPSCALERS
    assert "LDSR" in PIXEL_UPSCALERS
    assert math.isfinite(1.0) and os.sep


def test_marigold_depth_to_image_near_is_bright():
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from sdfx import controlnet
    pred = np.array([[[[0.0], [1.0]], [[0.5], [0.25]]]], dtype=np.float32)  # N,H,W,1; 0 = near
    im = controlnet.depth_to_image(pred)
    assert im.size == (2, 2) and im.mode == "RGB"
    assert im.getpixel((0, 0))[0] == 255 and im.getpixel((1, 0))[0] == 0
    flat = controlnet.depth_to_image(np.zeros((3, 4), np.float32))
    assert flat.size == (4, 3)


def test_zero123_camera_and_variant_detection(tmp_path):
    cam = extras.zero123_camera(30.0, 90.0)
    assert cam[0] == pytest.approx(math.radians(30)) and cam[1] == pytest.approx(1.0) and abs(cam[2]) < 1e-9
    assert cam[3] == 0.0
    m = tmp_path / "model"
    (m / "unet").mkdir(parents=True)
    (m / "vae").mkdir()
    (m / "clip_camera_projection").mkdir()
    (m / "unet" / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"x")
    (m / "vae" / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"x")
    (m / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    assert extras.fp16_variant(str(m), "unet") == "fp16"
    assert extras.fp16_variant(str(m), "vae") is None
    assert extras.fp16_variant(str(m)) == "fp16"
    assert extras._is_zero123_dir(str(m))
    assert not extras._is_zero123_dir(str(tmp_path))
