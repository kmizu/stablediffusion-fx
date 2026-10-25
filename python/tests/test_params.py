import pytest

from sdfx.params import (GenerationParams, GenerationRequest, HiresParams, ParamError, center_crop_box, from_dict, hires_steps,
                         img2img_steps, job_seeds, parse_generation_request)


def test_defaults_match_scala():
    p = GenerationParams()
    assert (p.sampler_name, p.scheduler, p.steps, p.cfg_scale, p.width, p.height) == ("DPM++ 2M", "Automatic", 20, 7.0, 512, 512)
    assert (p.seed, p.subseed, p.batch_size, p.n_iter, p.clip_skip) == (-1, -1, 1, 1, 1)
    assert p.hires == HiresParams() and p.hires.denoising_strength == 0.55 and p.hires.upscaler == "Latent"
    r = GenerationRequest()
    assert r.preview_every == 2 and r.preview_method == "approx"
    assert p.validate() == []


def test_parse_full_snake_case():
    data = {
        "params": {
            "prompt": "a", "negative_prompt": "b", "seed": 5, "subseed": 7.0, "subseed_strength": 0.5,
            "sampler_name": "Euler a", "scheduler": "Karras", "steps": 30, "cfg_scale": 6, "width": 768, "height": 512,
            "batch_size": 2, "n_iter": 3, "clip_skip": 2, "eta": None,
            "hires": {"enabled": True, "scale": 1.5, "upscaler": "Lanczos", "second_pass_steps": 10, "resize_x": 0,
                      "resize_y": 0, "denoising_strength": 0.4, "sampler_name": None, "scheduler": None, "prompt": "",
                      "negative_prompt": "", "cfg_scale": None},
            "img2img": {"init_image": "x.png", "mask": None, "denoising_strength": 0.6, "resize_mode": 1, "mask_blur": 4,
                        "inpainting_fill": 1, "inpaint_full_res": False, "inpaint_full_res_padding": 32,
                        "inpainting_mask_invert": False},
            "loras": [{"name": "foo", "weight": 0.8}],
            "controlnet": [{"enabled": True, "model": "cn", "image": None, "preprocessor": "canny", "weight": 1.0,
                            "guidance_start": 0.1, "guidance_end": 0.9, "preprocessor_args": {"low_threshold": 50}}],
            "hooks": {"freeu": {"b1": 1.1}, "freeu_xyz": 3},
            "styles": [], "checkpoint": None, "vae": None, "extra": {"k": "v"}, "unknown_key": 1,
        },
        "output_dir": "out", "lora_paths": {"foo": "foo.safetensors"}, "embedding_paths": [],
        "controlnet_paths": {"cn": "cn.safetensors"}, "upscaler_paths": {}, "preview_every": 3, "preview_method": "taesd",
    }
    r = parse_generation_request(data)
    p = r.params
    assert p.subseed == 7 and isinstance(p.cfg_scale, float) and p.n_iter == 3 and p.clip_skip == 2
    assert p.hires.enabled and p.hires.scale == 1.5 and p.hires.cfg_scale is None
    assert p.img2img.resize_mode == 1 and p.is_img2img and not p.is_inpaint
    assert p.loras[0].name == "foo" and p.loras[0].weight == 0.8
    assert p.controlnet[0].guidance_start == 0.1 and p.controlnet[0].preprocessor_args == {"low_threshold": 50.0}
    assert p.hooks["freeu"] == {"b1": 1.1}
    assert r.preview_every == 3 and r.lora_paths == {"foo": "foo.safetensors"}
    assert p.images_per_job == 6


def test_lenient_missing_and_null():
    r = parse_generation_request({"params": {"steps": None, "hires": None}, "output_dir": "o"})
    assert r.params.steps == 20 and r.params.hires == HiresParams()


def test_bad_types():
    with pytest.raises(ParamError):
        parse_generation_request({"params": {"steps": "abc"}})
    with pytest.raises(ParamError):
        parse_generation_request({"params": {"width": True}})
    with pytest.raises(ParamError):
        parse_generation_request({"output_dir": "x"})


def test_validate_messages():
    p = GenerationParams(steps=0, width=500, height=4104, batch_size=17, n_iter=0, cfg_scale=31, clip_skip=13,
                         subseed_strength=2)
    assert p.validate() == [
        "steps must be 1..150 (got 0)",
        "width must be a multiple of 8 in 64..4096 (got 500)",
        "height must be a multiple of 8 in 64..4096 (got 4104)",
        "batch size must be 1..16 (got 17)",
        "batch count must be 1..100 (got 0)",
        "CFG scale must be 0..30 (got 31.0)",
        "clip skip must be 1..12 (got 13)",
        "variation strength must be 0..1",
    ]
    q = from_dict(GenerationParams, {"img2img": {"init_image": "a", "denoising_strength": 1.5},
                                     "hires": {"enabled": True, "denoising_strength": -0.1}})
    assert q.validate() == ["denoising strength must be 0..1", "hires denoising must be 0..1"]


# Golden table shared with core/src/test/scala/sdfx/core/GenerationParamsSuite.scala (HiresParams.targetSize)
@pytest.mark.parametrize("scale,rx,ry,w,h,expected", [
    (2.0, 0, 0, 512, 512, (1024, 1024)),
    (1.5, 0, 0, 832, 1216, (1248, 1824)),
    (1.25, 0, 0, 512, 768, (640, 960)),
    (1.3, 0, 0, 520, 520, (672, 672)),
    (1.1, 0, 0, 512, 512, (560, 560)),
    (1.0, 0, 0, 512, 768, (512, 768)),
    (2.0, 1024, 0, 512, 768, (1024, 1536)),
    (2.0, 1000, 0, 512, 768, (1000, 1496)),
    (2.0, 0, 1536, 512, 768, (1024, 1536)),
    (2.0, 0, 1200, 832, 1216, (816, 1200)),
    (4.0, 1920, 1080, 512, 512, (1920, 1080)),
    (2.0, 1023, 777, 512, 512, (1016, 776)),
    (0.05, 0, 0, 512, 512, (64, 64)),
    # A1111 truncation (not rounding)
    (1.5, 0, 0, 600, 600, (896, 896)),
    (1.3, 0, 0, 512, 512, (664, 664)),
    (1.0, 0, 0, 516, 512, (512, 512)),
    (2.0, 0, 900, 768, 512, (1344, 896)),
])
def test_hires_target_size(scale, rx, ry, w, h, expected):
    assert HiresParams(scale=scale, resize_x=rx, resize_y=ry).target_size(w, h) == expected


def test_hires_target_size_degenerate():
    assert HiresParams(resize_x=1024).target_size(0, 0)[0] == 1024


@pytest.mark.parametrize("rx,ry,w,h,upscale", [
    (0, 0, 512, 512, (1024, 1024)),          # scale: no crop
    (1000, 0, 512, 768, (1000, 1496)),       # one side: no crop
    (1920, 1080, 512, 512, (1920, 1920)),    # wider target: cover width, crop height (A1111 hr_upscale_to)
    (1080, 1920, 512, 512, (1920, 1920)),    # taller target: cover height, crop width
    (1000, 600, 512, 768, (1000, 1496)),
    (1024, 1024, 1024, 512, (2048, 1024)),
])
def test_hires_upscale_size_cover(rx, ry, w, h, upscale):
    hp = HiresParams(scale=2.0, resize_x=rx, resize_y=ry)
    uw, uh = hp.upscale_size(w, h)
    tw, th = hp.target_size(w, h)
    assert (uw, uh) == upscale
    assert uw >= tw and uh >= th and uw % 8 == 0 and uh % 8 == 0


def test_center_crop_box():
    assert center_crop_box(1920, 1920, 1920, 1080) == (0, 420, 1920, 1500)
    l, t, r, b = center_crop_box(1000, 1496, 1000, 600, 8)
    assert (r - l, b - t) == (1000, 600) and t % 8 == 0
    assert center_crop_box(64, 64, 64, 64) == (0, 0, 64, 64)


def test_step_helpers():
    assert img2img_steps(20, 0.75) == (20, 0.75, 15)
    n, s, a = img2img_steps(20, 0.01)
    assert a == 1 and int(n * s) == 1
    n, s, a = hires_steps(10, 0.5)
    assert a == 10 and int(n * s) == a
    n, s, a = hires_steps(15, 0.55)
    assert a >= 15 and int(n * s) == a
    assert job_seeds(10, 100, 3) == [(10, 100), (11, 101), (12, 102)]
