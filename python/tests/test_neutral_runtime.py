import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from sdfx.neutral_runtime import coefficients, to_x0, from_x0, RuntimeHook, Plan
from sdfx.neutral_prompt import parse, compose
from types import SimpleNamespace


@pytest.mark.parametrize("name", ["DDIMScheduler", "EulerDiscreteScheduler", "EulerAncestralDiscreteScheduler", "LMSDiscreteScheduler", "HeunDiscreteScheduler",
    "KDPM2DiscreteScheduler", "KDPM2AncestralDiscreteScheduler", "DPMSolverSDEScheduler"])
@pytest.mark.parametrize("prediction", ["epsilon", "v_prediction"])
def test_conversion_matches_native_pred_original_sample_each_stage(name, prediction):
    torch.manual_seed(71)
    cls = getattr(diffusers, name)
    options = dict(prediction_type=prediction)
    if name in ("DDIMScheduler", "HeunDiscreteScheduler"):
        options["clip_sample"] = False
    scheduler = cls(**options)
    scheduler.set_timesteps(4)
    sample = torch.randn(1, 4, 8, 8)
    for t in scheduler.timesteps:
        model_input = scheduler.scale_model_input(sample, t)
        model_output = torch.randn_like(sample)
        expected_x0 = to_x0(model_input, model_output, coefficients(scheduler, t))
        result = scheduler.step(model_output, t, sample)
        torch.testing.assert_close(expected_x0, result.pred_original_sample, rtol=3e-5, atol=3e-5)
        sample = result.prev_sample


@pytest.mark.parametrize("name", ["DPMSolverMultistepScheduler", "DPMSolverSinglestepScheduler", "UniPCMultistepScheduler"])
@pytest.mark.parametrize("prediction", ["epsilon", "v_prediction", "sample"])
def test_conversion_matches_native_data_conversion(name, prediction):
    torch.manual_seed(71)
    scheduler = getattr(diffusers, name)(prediction_type=prediction)
    scheduler.set_timesteps(4)
    sample = torch.randn(1, 4, 8, 8)
    for t in scheduler.timesteps:
        model_input = scheduler.scale_model_input(sample, t)
        if scheduler.step_index is None:
            scheduler._init_step_index(t)
        model_output = torch.randn_like(sample)
        ours = to_x0(model_input, model_output, coefficients(scheduler, t))
        native = scheduler.convert_model_output(model_output, sample=sample)
        torch.testing.assert_close(ours, native, rtol=2e-5, atol=2e-5)
        sample = scheduler.step(model_output, t, sample).prev_sample


def test_runtime_composes_branches_and_restores_forward_on_error():
    class UNet(torch.nn.Module):
        def forward(self, sample, timestep, encoder_hidden_states=None, **kwargs):
            if encoder_hidden_states.isnan().any():
                raise RuntimeError("bad branch")
            return (encoder_hidden_states[:, :4, :].reshape_as(sample),)
    u = UNet()
    scheduler = diffusers.DDIMScheduler(clip_sample=False)
    scheduler.set_timesteps(4)
    negative = torch.tensor([[[1.], [2], [3], [4]]])
    pos = [negative + 1, negative + torch.tensor([[[2.], [0], [-1], [3]]])]
    branches = [{"prompt_embeds": p, "negative_prompt_embeds": negative} for p in pos]
    tree = parse("one :2 AND_PERP two :-1")
    hook = RuntimeHook(u, scheduler, Plan(tree, branches), cfg=3, phi=0)
    hook.install()
    x = torch.randn(2, 4, 1, 1)
    x[1] = x[0]
    t = scheduler.timesteps[0]
    result = u(x, t, encoder_hidden_states=torch.cat([negative, pos[0]]))[0]
    c = coefficients(scheduler, t)
    u0 = to_x0(x[:1], negative.reshape(1, 4, 1, 1), c)
    refs = [to_x0(x[:1], p.reshape(1, 4, 1, 1), c) for p in pos]
    target = from_x0(x[:1], compose(tree, refs, u0, 3), c)
    uncond, cond = result.chunk(2)
    torch.testing.assert_close(uncond + 3 * (cond - uncond), target)
    branches[1]["prompt_embeds"].fill_(torch.nan)
    with pytest.raises(RuntimeError, match="bad branch"):
        u(x, t, encoder_hidden_states=torch.cat([negative, pos[0]]))
    hook.remove()
    hook.remove()
    assert "forward" not in u.__dict__


@pytest.mark.parametrize("phase", ["before", "between"])
def test_protocol_interrupt_cleans_reference_bank_and_restores_forward(phase):
    from diffusers.models.attention_processor import Attention
    from sdfx.adapters import ReferenceOnlyHook
    from sdfx.protocol import EngineError, INTERRUPTED
    class UNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1 = Attention(query_dim=4, heads=2, dim_head=2)
            self.calls = 0
        def forward(self, sample, timestep, encoder_hidden_states=None, **kw):
            self.calls += 1
            hs = sample.flatten(2).transpose(1, 2) + encoder_hidden_states.mean(1)[:, None]
            return (self.attn1(hs).transpose(1, 2).reshape_as(sample),)
    unet = UNet()
    s = diffusers.DDIMScheduler()
    s.set_timesteps(4)
    ref = ReferenceOnlyHook(unet, torch.randn(1, 4, 3, 3), s, .5, .8).install(managed=True)
    checks = 0
    def interrupt():
        nonlocal checks
        checks += 1
        if checks == (1 if phase == "before" else 3):
            raise EngineError(INTERRUPTED, "cancelled by user")
    branch = dict(prompt_embeds=torch.ones(1, 5, 4), negative_prompt_embeds=torch.zeros(1, 5, 4))
    runtime = RuntimeHook(unet, s, Plan(parse("one AND two"), [branch, branch]), 6., 0.,
                          branch_hooks=[ref], check_interrupt=interrupt).install()
    try:
        with pytest.raises(EngineError) as error:
            unet(torch.randn(2, 4, 3, 3), s.timesteps[0])
        assert error.value.code == INTERRUPTED
        assert unet.calls == (0 if phase == "before" else 2)  # first branch's reference write/read completed
        assert ref.mode == "off" and not ref.bank
    finally:
        runtime.remove()
        ref.remove()
    assert "forward" not in unet.__dict__
    assert not unet.attn1._forward_pre_hooks and not unet.attn1._forward_hooks


def test_engine_encodes_scheduled_leaves_and_preserves_the_noop_path():
    from sdfx.engine import Engine
    from sdfx.params import GenerationParams
    engine = Engine.__new__(Engine)
    calls = []
    def encode(pos, neg, gp, bs):
        calls.append((pos, neg))
        return {"prompt_embeds": torch.ones(bs, 77, 4), "negative_prompt_embeds": torch.zeros(bs, 77, 4)}
    engine._encode = encode
    gp = GenerationParams(hooks={"neutral_prompt": {}})
    ordinary = engine._encode_scheduled("ordinary (weighted:1.3)", "lowres", gp, 2, 4)
    assert "_neutral" not in ordinary
    assert calls == [("ordinary (weighted:1.3)", "lowres")]
    calls.clear()
    out = engine._encode_scheduled("one AND_TOPK [two:three:0.5] :0.7", "lowres", gp, 2, 4)
    plan = out["_neutral"]
    assert len(plan.branches) == 2
    assert [s for s, _ in plan.branches[1].schedule] == [2, 4]
    assert calls == [("one ", "lowres"), ("two", "lowres"), ("three", "lowres")]
    assert plan.branches[0]["prompt_embeds"].shape[0] == 2
    calls.clear()
    out = engine._encode_scheduled("one :1", "", gp, 1, 4)
    assert "_neutral" in out
    assert calls == [("one ", "")]


@pytest.mark.parametrize("cfg", [0, 0.5, 1])
def test_runtime_without_pipeline_cfg_repeats_xl_time_ids_and_uses_negative_branch(cfg):
    class UNet(torch.nn.Module):
        def forward(self, sample, timestep, encoder_hidden_states, added_cond_kwargs=None, **kw):
            assert added_cond_kwargs["time_ids"].shape == (4, 6)
            assert added_cond_kwargs["text_embeds"].shape == (4, 8)
            return (encoder_hidden_states[:, :4].reshape_as(sample),)
    unet = UNet()
    scheduler = diffusers.DDIMScheduler(clip_sample=False)
    scheduler.set_timesteps(4)
    negative = torch.arange(8.).reshape(2, 4, 1)
    positive = negative + 2
    branch = dict(prompt_embeds=positive, negative_prompt_embeds=negative,
                  pooled_prompt_embeds=torch.ones(2, 8), negative_pooled_prompt_embeds=torch.zeros(2, 8))
    hook = RuntimeHook(unet, scheduler, Plan(parse("one"), [branch]), cfg, 0).install()
    try:
        result = unet(torch.zeros(2, 4, 1, 1), scheduler.timesteps[0], encoder_hidden_states=positive,
                      added_cond_kwargs={"time_ids": torch.ones(2, 6)})[0]
        torch.testing.assert_close(result.flatten(1), negative.flatten(1) + cfg * 2)
    finally:
        hook.remove()


def test_callback_counts_logical_steps():
    from sdfx.engine import ScheduledEmbeds
    from sdfx.neutral_runtime import at
    a, b = {}, {}
    branch = ScheduledEmbeds(a)
    branch.schedule = [(2, a), (4, b)]
    hook = RuntimeHook(None, SimpleNamespace(order=2), Plan(parse("one"), [branch]), 6, 0)
    for i, expected in enumerate([1, 2, 2, 3, 3]):
        hook.callback(None, i, None, {})
        assert hook.step == expected
        assert at(branch, hook.step) is (a if expected <= 2 else b)


def test_feature_advertisement_matches_new_hook():
    from sdfx.dryrun import FakeEngine
    from sdfx.hooks import KNOWN_HOOKS
    assert "neutral_prompt" in FakeEngine().features()
    assert "neutral_prompt" in KNOWN_HOOKS


def test_pending_branch_dependent_combinations_fail_explicitly():
    from sdfx.neutral_runtime import validate
    from sdfx.protocol import EngineError
    for name in ("detail_daemon", "photomaker", "sag", "apg", "mahiro", "latent_modifier", "dynamic_thresholding"):
        with pytest.raises(EngineError, match=name):
            validate({name: {}})
        validate({name: {"enabled": False}})
    validate({}, controlnet=True)
    validate({}, pag=True)
    validate({"cads": {}, "cfg_schedule": {}})


@pytest.mark.parametrize("name", ["photomaker", "sag", "apg", "mahiro", "latent_modifier", "dynamic_thresholding", "detail_daemon"])
def test_unsupported_combinations_reject_before_pipeline_or_adapter_setup(monkeypatch, name):
    from sdfx.engine import Engine
    from sdfx.params import GenerationParams
    from sdfx.protocol import EngineError, UNSUPPORTED
    from sdfx import extensions
    monkeypatch.setattr(extensions, "pre_run", lambda *a, **kw: pytest.fail("unsupported request reached pipeline setup"))
    engine = Engine.__new__(Engine)
    gp = GenerationParams(hooks={"neutral_prompt": {}, name: {}})
    with pytest.raises(EngineError) as error:
        engine._run("txt2img", gp, None, None, sampler="DPM++ 2M", schedule="Automatic", cfg_scale=6.,
                    embeds={"_neutral": Plan(parse("a AND b"), [{}, {}])}, steps=4, w=64, h=64, seeds=[1],
                    pass_="base", image_idx=0, images_total=1)
    assert error.value.code == UNSUPPORTED and name in error.value.message


def test_zero_snr_v_prediction_ddim_has_finite_native_coefficients():
    scheduler = diffusers.DDIMScheduler(prediction_type="v_prediction", rescale_betas_zero_snr=True,
                                       timestep_spacing="trailing", clip_sample=False)
    scheduler.set_timesteps(4)
    t = scheduler.timesteps[0]
    x, v = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
    ab = coefficients(scheduler, t)
    assert ab == (0., -1.)
    torch.testing.assert_close(to_x0(x, v, ab), scheduler.step(v, t, x).pred_original_sample)


def test_infinite_sigma_v_limit_and_undefined_epsilon_fail_clearly():
    from sdfx.protocol import EngineError
    scheduler = diffusers.EulerDiscreteScheduler(prediction_type="v_prediction")
    scheduler.sigmas = torch.tensor([torch.inf])
    scheduler._step_index = 0
    assert coefficients(scheduler, torch.tensor(999)) == (0., -1.)
    scheduler.register_to_config(prediction_type="epsilon")
    with pytest.raises(EngineError, match="zero SNR"):
        coefficients(scheduler, torch.tensor(999))


def test_low_noise_composition_does_not_cancel_against_large_shared_baseline():
    class UNet(torch.nn.Module):
        def forward(self, sample, timestep, encoder_hidden_states=None, **kwargs):
            return (encoder_hidden_states[:, :4, :].reshape_as(sample),)
    scheduler = diffusers.DDIMScheduler(clip_sample=False)
    scheduler.set_timesteps(4)
    negative = torch.tensor([[[0.1], [0.2], [0.3], [0.4]]])
    positive = negative + 0.37
    branch = dict(prompt_embeds=positive, negative_prompt_embeds=negative)
    unet = UNet()
    hook = RuntimeHook(unet, scheduler, Plan(parse("one :0.5 AND two :0.5"), [branch, branch]), 6, 0).install()
    try:
        out = unet(torch.full((2, 4, 1, 1), 10000.), scheduler.timesteps[-1],
                   encoder_hidden_states=torch.cat([negative, positive]))[0]
        torch.testing.assert_close(out[1:].flatten(), positive.flatten(), atol=1e-6, rtol=1e-6)
    finally:
        hook.remove()


@pytest.mark.parametrize("prediction", ["epsilon", "v_prediction"])
def test_inverse_ddim_conversion_uses_input_noise_level(prediction):
    scheduler = diffusers.DDIMInverseScheduler(prediction_type=prediction, clip_sample=False)
    scheduler.set_timesteps(4)
    x = torch.randn(1, 4, 8, 8)
    for t in scheduler.timesteps:
        v = torch.randn_like(x)
        ours = to_x0(x, v, coefficients(scheduler, t))
        result = scheduler.step(v, t, x)
        torch.testing.assert_close(ours, result.pred_original_sample, rtol=3e-5, atol=3e-5)
        x = result.prev_sample


@pytest.mark.parametrize("prediction", ["epsilon", "v_prediction", "sample"])
def test_lcm_coefficients_match_native_boundary_condition_output(prediction):
    scheduler = diffusers.LCMScheduler(prediction_type=prediction, clip_sample=False)
    scheduler.set_timesteps(4)
    x = torch.randn(1, 4, 8, 8)
    for t in scheduler.timesteps:
        v = torch.randn_like(x)
        ours = to_x0(x, v, coefficients(scheduler, t))
        skip, out = scheduler.get_scalings_for_boundary_condition_discrete(t)
        result = scheduler.step(v, t, x)
        torch.testing.assert_close(out * ours + skip * x, result.denoised, rtol=3e-5, atol=3e-5)
        x = result.prev_sample


@pytest.mark.parametrize("phi", [0, 0.7, 1])
def test_zero_noise_guidance_is_continuous_limit_of_x0_rescale(phi):
    from sdfx.neutral_runtime import zero_noise_guidance
    from sdfx.neutral_prompt import combine_deltas
    z = torch.tensor([[1., 3, -2, 0]])
    u = torch.tensor([[0.2, 0.3, -0.1, 0.4]])
    d = torch.tensor([[1., -0.5, 0.7, 2]])
    cfg = 4
    got = zero_noise_guidance(z, u, d, cfg, phi)
    b = -1e-6
    zz, uu, dd = z.double(), u.double(), d.double()
    x0u = zz + b * uu
    g, c = x0u + b * cfg * dd, x0u + b * dd
    gm, cm = g.mean(1, keepdim=True), c.mean(1, keepdim=True)
    reference = (1-phi)*g + phi * (cm + (g-gm)/g.std(1, keepdim=True)*c.std(1, keepdim=True))
    torch.testing.assert_close(got.double(), (reference-zz)/b, rtol=2e-5, atol=2e-5)


def test_zero_noise_saliency_limit_uses_centered_absolute_activations():
    from sdfx.neutral_prompt import combine_deltas
    root = parse("base AND_SALT extra :0.7")
    normal = torch.tensor([[4., 1, 1, 1]])
    other = torch.tensor([[0., 2, 0, 0]])
    got = combine_deltas(root, [normal, other], zero_noise_limit=True)
    # Maps at infinitesimal noise are uniform + |B|*(abs(v)-mean(abs(v)))/N.
    maps = torch.stack([normal.abs()-normal.abs().mean(1, keepdim=True), other.abs()-other.abs().mean(1, keepdim=True)])
    win = maps.argmax(0)
    expected = normal + 0.7*(win==1)*(other-normal)
    torch.testing.assert_close(got, expected)


def test_zero_noise_saliency_ties_use_higher_order_normalization_terms():
    from sdfx.neutral_prompt import combine_deltas
    root = parse("base AND_SALT extra")
    normal = torch.tensor([[3., 1, 0, 0]])
    other = torch.tensor([[-3., 0.5, 0.5, 0]])
    got = combine_deltas(root, [normal, other], zero_noise_limit=True)
    torch.testing.assert_close(got, torch.tensor([[-3., 1, 0.5, 0]]))


@pytest.mark.parametrize("cfg", [0, 0.5, 1, 2])
@pytest.mark.parametrize("phi", [0, 0.7])
def test_actual_inverse_loop_composes_and_rescales_from_zero_noise(cfg, phi):
    from sdfx.extras import invert_latents
    class UNet(torch.nn.Module):
        def forward(self, sample, timestep, encoder_hidden_states=None, **kw):
            v = encoder_hidden_states[:, 0, 0]
            return SimpleNamespace(sample=torch.ones_like(sample) * v[:, None, None, None])
    unet = UNet()
    scheduler = diffusers.DDIMScheduler(clip_sample=False, set_alpha_to_one=True)
    pipe = SimpleNamespace(unet=unet, scheduler=scheduler)
    def embeds(value):
        return dict(prompt_embeds=torch.full((1, 77, 1), value), negative_prompt_embeds=torch.ones(1, 77, 1))
    branches = [embeds(3.), embeds(5.)]
    composed = dict(branches[0], _neutral=Plan(parse("one :0.5 AND two :0.5"), branches, phi=phi))
    torch.manual_seed(71)
    z = torch.randn(1, 4, 2, 2)
    ours = invert_latents(SimpleNamespace(sdxl=False), pipe, z.clone(), composed, 3, cfg, 16, 16, lambda: None)
    target = 1 + cfg * 3 + phi * (1 - cfg) * 3
    native = invert_latents(SimpleNamespace(sdxl=False), pipe, z.clone(), embeds(target), 3, 1, 16, 16, lambda: None)
    torch.testing.assert_close(ours, native, atol=3e-5, rtol=3e-5)
    assert "forward" not in unet.__dict__


def test_ordinary_decode_override_keeps_legacy_adjustment_path(monkeypatch):
    from sdfx.engine import Engine
    from sdfx.params import GenerationParams
    from sdfx import extras, extensions
    engine = Engine.__new__(Engine)
    engine.pipe = SimpleNamespace(scheduler=None)
    engine._activate = lambda *a: None
    engine._scheduler = lambda *a: (SimpleNamespace(config={}), None)
    def encoded(*a, **kw):
        return dict(prompt_embeds=torch.full((1, 77, 1), 3.), negative_prompt_embeds=torch.zeros(1, 77, 1))
    engine._encode = encoded
    engine._encode_scheduled = encoded
    def adjust(gp, embeds, ctx):
        return {k: torch.zeros_like(v) for k, v in embeds.items()}
    engine._adjust_embeds = adjust  # Revision(ignore_prompt=True) can zero conditioning
    engine._run = lambda *a, **kw: ([], None)
    latent = torch.zeros(1, 4, 2, 2)
    monkeypatch.setattr(extensions, "encode_image", lambda *a: latent)
    seen = []
    def invert(engine, pipe, x, embeds, *a):
        seen.append(embeds["prompt_embeds"].clone())
        return latent
    monkeypatch.setattr(extras, "invert_latents", invert)
    for enabled in (False, True):
        hooks = {"revision": {"ignore_prompt": True}}
        if enabled:
            hooks["neutral_prompt"] = {}
        engine._img2imgalt(GenerationParams(hooks=hooks), None, SimpleNamespace(check_interrupt=lambda: None),
                          {"override_prompt": "a cat", "randomness": 0}, None, encoded(), [1], 0, 1, 16, 16, None, None)
    torch.testing.assert_close(seen[0], seen[1], rtol=0, atol=0)
    assert seen[1].mean() == 3


@pytest.mark.parametrize("cfg", [1, 6])
@pytest.mark.parametrize("phi", [0, 0.7])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("keyword", ["AND_PERP", "AND_SALT", "AND_TOPK"])
def test_pag_branch_composition_preserves_native_guidance_layout(cfg, phi, batch, keyword):
    class UNet(torch.nn.Module):
        def forward(self, sample, timestep, encoder_hidden_states=None, **kw):
            groups = 3 if cfg > 1 else 4
            assert sample.shape[0] == batch * groups
            result = encoder_hidden_states[:, :4].reshape_as(sample).clone()
            # Real PAG processors use the final third (CFG) or half (non-CFG) for the perturbed path.
            result[-batch:] += torch.tensor([0.7, -0.4, 0.5, -0.3]).view(1, 4, 1, 1)
            return (result,)
    torch.manual_seed(61)
    negative = torch.randn(batch, 4, 1)
    positives = [torch.randn(batch, 4, 1), torch.randn(batch, 4, 1)]
    branches = [dict(prompt_embeds=v, negative_prompt_embeds=negative) for v in positives]
    tree = parse("base " + keyword + " extra :0.7")
    unet = UNet()
    scheduler = diffusers.DDIMScheduler(clip_sample=False)
    scheduler.set_timesteps(4)
    t = scheduler.timesteps[1]
    z = torch.randn(batch, 4, 1, 1)
    ab = coefficients(scheduler, t)
    u = negative.reshape_as(z)
    xs = [v.reshape_as(z) for v in positives]
    perturb = torch.tensor([0.7, -0.4, 0.5, -0.3]).view(1, 4, 1, 1)
    u0 = to_x0(z, u, ab)
    ordinary = [to_x0(z, v, ab) for v in xs]
    perturbed = [to_x0(z, v + perturb, ab) for v in xs]
    target = from_x0(z, compose(tree, ordinary, u0, cfg, phi=phi), ab)
    raw_c = from_x0(z, compose(tree, ordinary, u0, 1), ab)
    raw_p = from_x0(z, compose(tree, perturbed, u0, 1), ab)
    hook = RuntimeHook(unet, scheduler, Plan(tree, branches), cfg, phi, pag=True).install()
    try:
        out = unet(torch.cat([z] * (3 if cfg > 1 else 2)), t, encoder_hidden_states=torch.empty(0))[0]
        if cfg > 1:
            uncond, cond, pert = out.chunk(3)
            actual = uncond + cfg * (cond - uncond) + 2.3 * (cond - pert)
        else:
            cond, pert = out.chunk(2)
            actual = cond + 2.3 * (cond - pert)
        torch.testing.assert_close(actual, target + 2.3 * (raw_c - raw_p), rtol=3e-5, atol=3e-5)
    finally:
        hook.remove()


@pytest.mark.parametrize("cfg", [1, 6])
def test_pag_config_with_native_pag_inactive_uses_ordinary_batch(cfg):
    # Native PAG scale=0 or layers=[] leaves ordinary CFG/no-CFG groups despite a PAG pipeline/config.
    class UNet(torch.nn.Module):
        def forward(self, sample, timestep, encoder_hidden_states=None, **kwargs):
            assert sample.shape[0] == 2
            return (encoder_hidden_states[:, :4].reshape_as(sample),)
    positive, negative = torch.ones(1, 4, 1) * 3, torch.ones(1, 4, 1)
    branch = dict(prompt_embeds=positive, negative_prompt_embeds=negative)
    scheduler = diffusers.DDIMScheduler(clip_sample=False)
    scheduler.set_timesteps(4)
    unet = UNet()
    hook = RuntimeHook(unet, scheduler, Plan(parse("one :0.5 AND two :0.5"), [branch, branch]), cfg, 0, pag=True).install()
    try:
        result = unet(torch.zeros(2 if cfg > 1 else 1, 4, 1, 1), scheduler.timesteps[0], encoder_hidden_states=positive)[0]
        assert result.shape[0] == (2 if cfg > 1 else 1)
    finally:
        hook.remove()


@pytest.mark.parametrize("guided", [False, True])
def test_real_pag_processors_keep_positive_results_when_negative_group_is_added(guided):
    from diffusers.models.attention_processor import Attention
    from diffusers.models.attention_processor import PAGIdentitySelfAttnProcessor2_0, PAGCFGIdentitySelfAttnProcessor2_0
    torch.manual_seed(66)
    attention = Attention(query_dim=4, heads=1, dim_head=4)
    attention.set_processor(PAGCFGIdentitySelfAttnProcessor2_0() if guided else PAGIdentitySelfAttnProcessor2_0())
    negative, positive = torch.randn(2, 9, 4), torch.randn(2, 9, 4)
    if guided:
        result = attention(torch.cat([negative, positive, positive]))
        assert result.shape[0] == 6
    else:
        native = attention(torch.cat([positive, positive]))
        expanded = attention(torch.cat([negative, positive, negative, positive]))
        torch.testing.assert_close(expanded[2:4], native[:2])
        torch.testing.assert_close(expanded[6:], native[2:])
