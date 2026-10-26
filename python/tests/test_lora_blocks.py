"""LoRA Block Weight: presets, explicit weights, diffusers block scales, start/stop windows."""
import pytest

from sdfx import lora_blocks as lb


def test_presets_and_explicit_weights():
    w = lb.parse_weights("OUTALL", False)
    assert w["BASE"] == 1 and w["OUT03"] == 1 and w["IN04"] == 0 and w["MID"] == 0
    half = lb.parse_weights("all0.5", True)
    assert half["IN04"] == 0.5 and half["OUT05"] == 0.5
    e = lb.parse_weights("1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0", False)
    assert [e[k] for k in lb.SD15_17] == [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    short = lb.parse_weights("1,0.5", True)                    # missing blocks are 0
    assert short["IN04"] == 0.5 and short["OUT05"] == 0
    assert lb.parse_weights("mine", False, {"MINE": "0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"})["BASE"] == 0
    with pytest.raises(ValueError):
        lb.parse_weights("NOPE", False)


def test_sd15_scales_follow_the_17_block_layout():
    w = lb.parse_weights("0.5,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16", False)
    s = lb.diffusers_scales(w, False, unet_weight=2.0, te_weight=0.5)
    assert s["text_encoder"] == 0.25
    u = s["unet"]
    assert u["down"]["block_0"] == [2, 4] and u["down"]["block_2"] == [10, 12] and u["mid"] == 14
    assert u["up"]["block_1"] == [16, 18, 20] and u["up"]["block_3"] == [28, 30, 32]
    assert u["down"]["block_3"] == 0 and u["up"]["block_0"] == 0   # no attention there in a plain LoRA


def test_sdxl_scales_and_both_text_encoders():
    w = lb.parse_weights("1,2,3,4,5,6,7,8,9,10,11,12", True)
    s = lb.diffusers_scales(w, True, 1.0, 1.0)
    assert s["text_encoder"] == s["text_encoder_2"] == 1
    assert s["unet"]["down"]["block_1"] == [2, 3] and s["unet"]["mid"] == 6
    assert s["unet"]["up"]["block_0"] == [7, 8, 9] and s["unet"]["up"]["block_1"] == [10, 11, 12]


def test_step_windows():
    assert lb.step_of(0.4, 20) == 8 and lb.step_of(10, 20) == 10 and lb.step_of(None, 20) is None
    assert lb.active(1, 20, None, 10) and lb.active(9, 20, None, 10) and not lb.active(10, 20, None, 10)
    assert not lb.active(4, 20, 5, None) and lb.active(5, 20, 5, None)
    assert lb.zero_unet({"unet": {"mid": 1}, "text_encoder": 0.7}, False) == {"unet": 0.0, "text_encoder": 0.7}
    assert lb.zero_unet(0.8, True) == {"unet": 0.0, "text_encoder": 0.8, "text_encoder_2": 0.8}
