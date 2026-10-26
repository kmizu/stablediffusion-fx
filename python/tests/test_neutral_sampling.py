import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from sdfx.neutral_sampling import Controller
from sdfx import sampling_ext as native


def branch(offset=0., marked=False):
    x = torch.arange(24., dtype=torch.float32).reshape(2, 3, 4) / 10
    p, n = x + offset, x - 2
    if marked:
        flags = torch.tensor([1., -1., 0.]).view(1, 3, 1).repeat(2, 1, 1)
        p = torch.cat([p.masked_fill(flags == 0, 0), flags], -1)
        n = torch.cat([n.masked_fill(flags == 0, 0), flags], -1)
    return dict(prompt_embeds=p, negative_prompt_embeds=n, _negpip=marked)


def scheduler():
    s = diffusers.DDIMScheduler()
    s.set_timesteps(4)
    return s


def test_disabled_controller_and_late_cads_are_exact_passthroughs():
    s = scheduler()
    branches = [branch(), branch(1)]
    for hooks, t in [({}, s.timesteps[0]), ({"cads": {}}, s.timesteps[-1])]:
        c = Controller(s, hooks, 71)
        assert c.branches(branches, t) is branches
        assert c.cfg(6, t, guided=True) == 6


@pytest.mark.parametrize("apply_negative", [False, True])
@pytest.mark.parametrize("marked", [False, True])
def test_cads_is_reproducible_shares_negatives_and_preserves_source_and_signs(apply_negative, marked):
    s = scheduler()
    original = [branch(marked=marked), branch(1., marked)]
    saved = [{k: v.clone() for k, v in b.items() if torch.is_tensor(v)} for b in original]
    hooks = {"cads": dict(apply_to_uncond=apply_negative)}
    first = Controller(s, hooks, 71).branches(original, s.timesteps[0])
    second = Controller(s, hooks, 71).branches(original, s.timesteps[0])
    for old, snapshot, out, repeat in zip(original, saved, first, second):
        for key in snapshot:
            torch.testing.assert_close(old[key], snapshot[key], rtol=0, atol=0)
            torch.testing.assert_close(out[key], repeat[key], rtol=0, atol=0)
        assert not torch.equal(old["prompt_embeds"], out["prompt_embeds"])
        if not apply_negative:
            assert out["negative_prompt_embeds"] is old["negative_prompt_embeds"]
        if marked:
            for key in snapshot:
                torch.testing.assert_close(out[key][..., -1:], old[key][..., -1:], rtol=0, atol=0)
                assert torch.count_nonzero(out[key][:, -1, :-1]) == 0
    torch.testing.assert_close(first[0]["negative_prompt_embeds"], first[1]["negative_prompt_embeds"], rtol=0, atol=0)


def test_cads_first_plain_leaf_matches_existing_native_noise_function():
    s = scheduler()
    original = branch()
    c = Controller(s, {"cads": {}}, 71)
    actual = c.branches([original], s.timesteps[0])[0]["prompt_embeds"]
    expected = native.cads_noise(original["prompt_embeds"], 0., .25, 1., True,
                                 torch.Generator("cpu").manual_seed(71))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_shared_negative_randomness_never_replaces_each_leafs_payload_or_metadata():
    s = scheduler()
    original = [branch(marked=True), branch(1., True)]
    # Equal tensor sizes can still carry different valid-token/sign layouts after alignment.
    original[1]["negative_prompt_embeds"] = original[1]["negative_prompt_embeds"].clone()
    original[1]["negative_prompt_embeds"][:, 0, -1] = -1
    original[1]["negative_prompt_embeds"][:, 1, :-1] += 4
    c = Controller(s, {"cads": dict(apply_to_uncond=True)}, 71)
    got = c.branches(original, s.timesteps[0])
    for source, out in zip(original, got):
        torch.testing.assert_close(source["negative_prompt_embeds"][..., -1:],
                                   out["negative_prompt_embeds"][..., -1:], rtol=0, atol=0)
    assert not torch.equal(got[0]["negative_prompt_embeds"][..., :-1], got[1]["negative_prompt_embeds"][..., :-1])


@pytest.mark.parametrize("guided,base", [(True, 6.), (False, 1.), (False, 6.), (True, .5)])
def test_cfg_curve_uses_native_progress_and_preserves_ignore_rule(guided, base):
    s = scheduler()
    config = dict(native.CFG_SCHEDULE_DEFAULTS, start_cfg=9., end_cfg=0., curve="cosine")
    c = Controller(s, {"cfg_schedule": config}, 71)
    for t in s.timesteps:
        expected = native.cfg_at(config, native.Progress(s)(t)[2]) if guided and base > 1 else base
        assert c.cfg(base, t, guided=guided) == pytest.approx(expected)


@pytest.mark.parametrize("cfg", [1., 6.])
@pytest.mark.parametrize("target", [0., 3.5])
@pytest.mark.parametrize("phi", [0., .7])
@pytest.mark.parametrize("pag", [False, True])
def test_runtime_effective_cfg_rescale_and_native_pag_difference(cfg, target, phi, pag):
    from sdfx.neutral_runtime import RuntimeHook, Plan, coefficients, to_x0, from_x0
    from sdfx.neutral_prompt import parse, compose
    class UNet(torch.nn.Module):
        def forward(self, sample, timestep, encoder_hidden_states=None, **kw):
            out = encoder_hidden_states.reshape_as(sample).clone()
            if pag:
                out[-1:] += perturbation
            return (out,)
    torch.manual_seed(71)
    s, unet = scheduler(), UNet()
    u = torch.randn(1, 4, 2, 2)
    positives = [torch.randn_like(u), torch.randn_like(u)]
    perturbation = torch.randn_like(u) * .3
    branches = [dict(prompt_embeds=p.reshape(1, 4, 4), negative_prompt_embeds=u.reshape(1, 4, 4)) for p in positives]
    tree = parse("one AND_PERP two :0.7")
    c = Controller(s, {"cfg_schedule": dict(start_cfg=target, end_cfg=target)}, 71)
    runtime = RuntimeHook(unet, s, Plan(tree, branches), cfg, phi, pag=pag, sampling=c).install()
    z, t = torch.randn_like(u), s.timesteps[1]
    groups = 3 if pag and cfg > 1 else 2 if pag or cfg > 1 else 1
    effective = target if cfg > 1 else cfg
    ab = coefficients(s, t)
    x0u = to_x0(z, u, ab)
    x0s = [to_x0(z, p, ab) for p in positives]
    expected = from_x0(z, compose(tree, x0s, x0u, effective, phi=phi), ab)
    try:
        result = unet(torch.cat([z] * groups), t)[0]
    finally:
        runtime.remove()
    if pag:
        if cfg > 1:
            uncond, cond, perturbed = result.chunk(3)
            actual = uncond + cfg * (cond - uncond)
        else:
            cond, perturbed = result.chunk(2)
            actual = cond
        raw = from_x0(z, compose(tree, x0s, x0u, 1.), ab)
        raw_perturbed = from_x0(z, compose(tree, [to_x0(z, p + perturbation, ab) for p in positives], x0u, 1.), ab)
        torch.testing.assert_close(cond - perturbed, raw - raw_perturbed, rtol=3e-5, atol=3e-5)
    elif cfg > 1:
        uncond, cond = result.chunk(2)
        actual = uncond + cfg * (cond - uncond)
    else:
        actual = result
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("pag", [False, True])
@pytest.mark.parametrize("guided", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
def test_cads_context_is_shared_by_pag_copies_and_all_branch_preparation(pag, guided, batch):
    from sdfx.neutral_runtime import RuntimeHook, Plan
    from sdfx.neutral_prompt import parse
    captures = []
    class UNet(torch.nn.Module):
        def forward(self, sample, timestep, encoder_hidden_states=None, **kw):
            captures.append(encoder_hidden_states.clone())
            return (encoder_hidden_states[..., :4].reshape_as(sample),)
    class ImageHook:
        priority = 0
        def before_branch(self, forward, sample, timestep, kw, layout):
            captures.append(kw["encoder_hidden_states"].clone())
        def after_branch(self):
            pass
    s, unet = scheduler(), UNet()
    branches = [branch(marked=True), branch(1., marked=True)]
    for b in branches:
        for key in ("prompt_embeds", "negative_prompt_embeds"):
            b[key] = b[key][:batch]
    hooks = {"cads": dict(apply_to_uncond=True)}
    expected = Controller(s, hooks, 71).branches(branches, s.timesteps[0])
    c = Controller(s, hooks, 71)
    runtime = RuntimeHook(unet, s, Plan(parse("one AND two"), branches), 6 if guided else 1, 0,
                          sampling=c, pag=pag, branch_hooks=[ImageHook()]).install()
    groups = 3 if pag and guided else 2 if pag or guided else 1
    try:
        unet(torch.zeros(batch * groups, 3, 2, 2), s.timesteps[0])
    finally:
        runtime.remove()
    layout = [False, True, True] if guided and pag else [False, True, False, True] if pag else [False, True]
    for i, b in enumerate(expected):
        ctx = torch.cat([b["prompt_embeds" if positive else "negative_prompt_embeds"] for positive in layout])
        torch.testing.assert_close(captures[2 * i], ctx, rtol=0, atol=0)
        torch.testing.assert_close(captures[2 * i + 1], ctx, rtol=0, atol=0)


def test_actual_inverse_cfg_schedule_including_zero_noise_start():
    from sdfx.extras import invert_latents
    from sdfx.neutral_runtime import Plan
    from sdfx.neutral_prompt import parse
    from types import SimpleNamespace
    class UNet(torch.nn.Module):
        def forward(self, sample, timestep, encoder_hidden_states=None, **kw):
            return SimpleNamespace(sample=torch.ones_like(sample) * encoder_hidden_states[:, 0, 0, None, None, None])
    def b(value):
        return dict(prompt_embeds=torch.full((1, 5, 4), value), negative_prompt_embeds=torch.ones(1, 5, 4))
    s, unet = scheduler(), UNet()
    encoded = dict(b(3.), _neutral=Plan(parse("one :0.5 AND two :0.5"), [b(3.), b(5.)]))
    config = dict(native.CFG_SCHEDULE_DEFAULTS, start_cfg=7., end_cfg=0.)
    torch.manual_seed(71)
    z = torch.randn(1, 4, 2, 2)
    got = invert_latents(SimpleNamespace(sdxl=False), SimpleNamespace(unet=unet, scheduler=s), z.clone(),
                         encoded, 4, 6., 16, 16, lambda: None,
                         sampling_setup=lambda inv: Controller(inv, {"cfg_schedule": config}, 71))
    inv = diffusers.DDIMInverseScheduler.from_config(s.config, clip_sample=False)
    inv.set_timesteps(4)
    expected = z.clone()
    for t in inv.timesteps:
        scale = native.cfg_at(config, native.Progress(inv)(t)[2])
        expected = inv.step(torch.ones_like(z) * (1 + scale * 3), t, expected).prev_sample
    torch.testing.assert_close(got, expected, rtol=3e-5, atol=3e-5)
    assert "forward" not in unet.__dict__


@pytest.mark.parametrize("composed", [False, True])
def test_root_sampling_hooks_are_skipped_only_for_composed_prompts(composed):
    from sdfx.params import GenerationParams
    from types import SimpleNamespace
    unet = torch.nn.Module()
    cleanups = []
    try:
        native.install(SimpleNamespace(sdxl=False), SimpleNamespace(unet=unet),
                       GenerationParams(hooks={"cads": {}, "cfg_schedule": {}}), {},
                       SimpleNamespace(warn=lambda msg: pytest.fail(msg)), cleanups,
                       sched=scheduler(), cfg_scale=6., steps=4, seed=71, chain=lambda *a: None, composed=composed)
        assert len(unet._forward_pre_hooks) == (0 if composed else 1)
        assert len(unet._forward_hooks) == (0 if composed else 1)
    finally:
        for cleanup in reversed(cleanups):
            cleanup()
    assert not unet._forward_hooks and not unet._forward_pre_hooks


@pytest.mark.parametrize("guided", [False, True])
@pytest.mark.parametrize("apply_negative", [False, True])
def test_ordinary_cads_also_preserves_private_negpip_signs(guided, apply_negative):
    from types import SimpleNamespace
    unet = torch.nn.Module()
    unet.config = SimpleNamespace(cross_attention_dim=4)
    s = scheduler()
    b = branch(marked=True)
    context = torch.cat([b["negative_prompt_embeds"], b["prompt_embeds"]]) if guided else b["prompt_embeds"]
    hook = native.CadsHook(unet, s, dict(native.CADS_DEFAULTS, apply_to_uncond=apply_negative), 71,
                           lambda: guided).install()
    try:
        _, kw = hook.pre(unet, (torch.zeros(context.shape[0], 4, 2, 2), s.timesteps[0]),
                         {"encoder_hidden_states": context})
        got = kw["encoder_hidden_states"]
        torch.testing.assert_close(got[..., -1:], context[..., -1:], rtol=0, atol=0)
        assert torch.count_nonzero(got[:, -1, :-1]) == 0
        if guided and not apply_negative:
            torch.testing.assert_close(got[:2], context[:2], rtol=0, atol=0)
    finally:
        hook.remove()
