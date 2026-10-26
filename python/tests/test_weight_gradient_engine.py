"""Engine-level scheduling and callback contract; no checkpoints or GPU allocations required."""
import types
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from sdfx.engine import Engine  # noqa: E402
from sdfx.params import GenerationParams, GenerationRequest, HiresParams, Img2ImgParams  # noqa: E402
from sdfx.protocol import EngineError  # noqa: E402


def fake_engine():
    engine = Engine.__new__(Engine)
    calls = []

    def encode(pos, neg, gp, bs):
        calls.append((pos, neg))
        v = len(calls)
        return {"prompt_embeds": torch.full((bs, 77, 4), v, dtype=torch.float32),
                "negative_prompt_embeds": torch.full((bs, 77, 4), -v, dtype=torch.float32),
                "pooled_prompt_embeds": torch.full((bs, 8), v, dtype=torch.float32),
                "negative_pooled_prompt_embeds": torch.full((bs, 8), -v, dtype=torch.float32)}

    engine._encode = encode
    return engine, calls


def test_engine_schedules_both_prompts_and_hires_with_opt_in():
    engine, calls = fake_engine()
    gp = GenerationParams(hooks={"weight_gradient": {}})
    out = engine._encode_scheduled("{cat::0-1}", "{dog::1-0}", gp, 2, 3)
    assert calls == [("(cat:0)", "(dog:1)"), ("(cat:0.5)", "(dog:0.5)"), ("(cat:1)", "(dog:0)")]
    assert len(out.schedule) == 3 and out["prompt_embeds"].shape == (2, 77, 4)
    calls.clear()
    hr = engine._encode_scheduled("{cat::0-1}", "", gp, 2, 20, 2)
    assert calls == [("(cat:0)", ""), ("(cat:1)", "")]
    assert len(hr.schedule) == 2
    assert "weight_gradient" in engine.features()


def test_photomaker_rejects_dynamic_gradient_instead_of_silently_using_raw_tags():
    engine, _ = fake_engine()
    gp = GenerationParams(prompt="a woman img {red scarf::0-1}", hooks={"weight_gradient": {}, "photomaker": {}})
    engine.parse_generation = lambda *args: GenerationRequest(params=gp)
    engine.ensure_model = lambda *args: None
    engine.kind = "sdxl"
    engine._apply_loras = lambda *args: {}
    engine._load_embeddings = lambda *args: None
    with pytest.raises(EngineError, match="PhotoMaker"):
        engine._generate({}, None, "txt2img")


@pytest.mark.parametrize("hooks", [{}, {"weight_gradient": {"enabled": False}}])
def test_disabled_gradient_stays_literal(hooks):
    engine, calls = fake_engine()
    engine._encode_scheduled("{cat::0-1}", "", GenerationParams(hooks=hooks), 1, 3)
    assert calls == [("{cat::0-1}", "")]


def test_generation_reencodes_inherited_hires_prompt_for_new_step_count():
    engine, calls = fake_engine()
    gp = GenerationParams(prompt="{cat::0-1}", steps=3, hooks={"weight_gradient": {}},
                          hires=HiresParams(enabled=True, second_pass_steps=2))
    engine.parse_generation = lambda *args: GenerationRequest(params=gp)
    engine.ensure_model = lambda *args: None
    engine.kind = "sd15"
    engine._apply_loras = lambda *args: {}
    engine._load_embeddings = lambda *args: None
    engine._pag_config = lambda *args: None
    engine._adjust_embeds = lambda gp, embeds, ctx: embeds

    class StopBeforeSampling(Exception):
        pass

    def stop(*args):
        raise StopBeforeSampling

    engine._controlnets = stop
    with pytest.raises(StopBeforeSampling):
        engine._generate({}, None, "txt2img")
    assert calls == [("(cat:0)", ""), ("(cat:0.5)", ""), ("(cat:1)", ""), ("(cat:0)", ""), ("(cat:1)", "")]


@pytest.mark.parametrize("alternative,masked,actual", [(False, False, 4), (True, False, 10), (True, True, 4)])
def test_img2img_gradient_reaches_end_at_actual_last_step(tmp_path, alternative, masked, actual):
    from PIL import Image
    src = tmp_path / "init.png"
    Image.new("RGB", (64, 64)).save(src)
    engine, calls = fake_engine()
    hooks = {"weight_gradient": {}}
    if alternative:
        hooks["img2imgalt"] = {}
    gp = GenerationParams(prompt="{cat::0-1}", steps=10, hooks=hooks,
                          img2img=Img2ImgParams(init_image=str(src), mask=str(src) if masked else None,
                                              denoising_strength=0.4))
    engine.parse_generation = lambda *args: GenerationRequest(params=gp)
    engine.ensure_model = lambda *args: None
    engine.kind = "sd15"
    engine._apply_loras = lambda *args: {}
    engine._load_embeddings = lambda *args: None
    engine._pag_config = lambda *args: None
    engine._adjust_embeds = lambda gp, embeds, ctx: embeds

    class StopBeforeSampling(Exception):
        pass

    def stop(*args):
        raise StopBeforeSampling

    engine._controlnets = stop
    with pytest.raises(StopBeforeSampling):
        engine._generate({}, None, "img2img")
    assert len(calls) == actual
    assert calls[0] == ("(cat:0)", "") and calls[-1] == ("(cat:1)", "")
    if actual == 4:
        assert calls == [("(cat:0)", ""), ("(cat:0.3333)", ""), ("(cat:0.6667)", ""), ("(cat:1)", "")]


@pytest.mark.parametrize("cfg,pag,count", [(True, False, 2), (True, True, 3), (False, False, 1), (False, True, 2)])
def test_gradient_swap_callback_uses_next_step_and_updates_pooled_cfg_pag(cfg, pag, count):
    engine, _ = fake_engine()
    out = engine._encode_scheduled("{cat::0-1}", "{dog::1-0}", GenerationParams(hooks={"weight_gradient": {}}), 2, 3)
    callback = engine._schedule_swapper(out.schedule)
    pipe = types.SimpleNamespace(do_classifier_free_guidance=cfg, do_perturbed_attention_guidance=pag)
    data = {"prompt_embeds": torch.zeros(count * 2, 77, 4), "add_text_embeds": torch.zeros(count * 2, 8)}
    data = callback(pipe, 0, None, data)
    expected = [-2, 2] if cfg and not pag else [-2, 2, 2] if cfg else [2, 2] if pag else [2]
    assert data["prompt_embeds"][:, 0, 0].tolist() == [v for v in expected for _ in range(2)]
    assert data["add_text_embeds"][:, 0].tolist() == [v for v in expected for _ in range(2)]


@pytest.mark.parametrize("scheduler_name", ["EulerDiscreteScheduler", "HeunDiscreteScheduler", "KDPM2DiscreteScheduler",
                                           "KDPM2AncestralDiscreteScheduler", "DPMSolverSDEScheduler"])
def test_real_scheduler_order_keeps_gradient_active_for_the_entire_pass(scheduler_name):
    import diffusers
    scheduler = getattr(diffusers, scheduler_name)()
    scheduler.set_timesteps(5)
    engine, _ = fake_engine()
    out = engine._encode_scheduled("{cat::0-1}", "", GenerationParams(hooks={"weight_gradient": {}}), 1, 5)
    callback = engine._schedule_swapper(out.schedule)
    pipe = types.SimpleNamespace(do_classifier_free_guidance=False, do_perturbed_attention_guidance=False,
                                 scheduler=scheduler)
    data = {"prompt_embeds": out["prompt_embeds"]}
    used = []
    for i, t in enumerate(scheduler.timesteps):
        used.append(int(data["prompt_embeds"][0, 0, 0]))
        data = callback(pipe, i, t, data)
    assert used == ([1, 2, 3, 4, 5] if scheduler.order == 1 else [1, 1, 2, 2, 3, 3, 4, 4, 5])
