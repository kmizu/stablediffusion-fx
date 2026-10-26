import pytest

torch = pytest.importorskip("torch")
from sdfx import sampling_ext as se
from types import SimpleNamespace


def scheduler():
    from diffusers import DDIMScheduler
    s = DDIMScheduler()
    s.set_timesteps(4)
    return s


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("pag", [False, True])
@pytest.mark.parametrize("target", [0., 3., 6.])
@pytest.mark.parametrize("object_output", [False, True])
def test_cfg_schedule_preserves_native_pag_difference(batch, pag, target, object_output):
    torch.manual_seed(71)
    u, c, p = [torch.randn(batch, 4, 2, 2) for _ in range(3)]
    prediction = torch.cat([u, c, p] if pag else [u, c])
    class UNet(torch.nn.Module):
        def forward(self, sample, timestep):
            return SimpleNamespace(sample=prediction) if object_output else (prediction, "metadata")
    unet, s = UNet(), scheduler()
    hook = se.CfgScheduleHook(unet, s, 6., dict(se.CFG_SCHEDULE_DEFAULTS, start_cfg=target, end_cfg=target),
                              lambda: True, pag_active=lambda: pag).install()
    try:
        result = unet(torch.zeros_like(prediction), s.timesteps[0])
        out = result.sample if object_output else result[0]
        if not object_output:
            assert result[1] == "metadata"
        if target == 6.:
            torch.testing.assert_close(out, prediction, rtol=0, atol=0)
        parts = out.chunk(3 if pag else 2)
        actual = parts[0] + 6 * (parts[1] - parts[0])
        if pag:
            torch.testing.assert_close(parts[1] - parts[2], c - p, rtol=2e-5, atol=2e-5)
            actual += 2.3 * (parts[1] - parts[2])
        expected = u + target * (c - u) + (2.3 * (c - p) if pag else 0.)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    finally:
        hook.remove()
    assert not unet._forward_hooks


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("guided", [False, True])
@pytest.mark.parametrize("apply_negative", [False, True])
@pytest.mark.parametrize("marked", [False, True])
def test_pag_cads_noises_positive_once_and_preserves_metadata(batch, guided, apply_negative, marked):
    torch.manual_seed(71)
    u, c = torch.randn(batch, 3, 4), torch.randn(batch, 3, 4)
    if marked:
        signs = torch.tensor([1., -1., 0.]).view(1, 3, 1).repeat(batch, 1, 1)
        u, c = [torch.cat([x.masked_fill(signs == 0, 0.), signs], -1) for x in (u, c)]
    context = torch.cat([u, c, c] if guided else [c, c])
    snapshot = context.clone()
    unet = torch.nn.Module()
    unet.config = SimpleNamespace(cross_attention_dim=4)
    s = scheduler()
    config = dict(se.CADS_DEFAULTS, apply_to_uncond=apply_negative)
    hook = se.CadsHook(unet, s, config, 71, lambda: guided, pag_active=lambda: True)
    _, kw = hook.pre(unet, (torch.zeros(context.shape[0], 4, 2, 2), s.timesteps[0]),
                     dict(encoder_hidden_states=context))
    parts = kw["encoder_hidden_states"].chunk(3 if guided else 2)
    gen = torch.Generator("cpu").manual_seed(71)
    expected_u = se.cads_context(u, 0., config, gen, marked=marked) if guided and apply_negative else u
    expected_c = se.cads_context(c, 0., config, gen, marked=marked)
    if guided:
        torch.testing.assert_close(parts[0], expected_u, rtol=0, atol=0)
    torch.testing.assert_close(parts[-2], expected_c, rtol=0, atol=0)
    torch.testing.assert_close(parts[-1], expected_c, rtol=0, atol=0)
    torch.testing.assert_close(context, snapshot, rtol=0, atol=0)
    if marked:
        torch.testing.assert_close(kw["encoder_hidden_states"][..., -1:], context[..., -1:], rtol=0, atol=0)


@pytest.mark.parametrize("scale", [0., 2.])
@pytest.mark.parametrize("layers", [[], ["mid"]])
def test_install_reads_native_pag_activity_lazily(scale, layers):
    from diffusers.pipelines.pag.pag_utils import PAGMixin
    from sdfx.params import GenerationParams
    class UNet(torch.nn.Module):
        config = SimpleNamespace(cross_attention_dim=4)
        def forward(self, sample, timestep, encoder_hidden_states=None):
            self.context = encoder_hidden_states
            return (self.prediction,)
    class Pipe(PAGMixin):
        pass
    pipe, unet, s = Pipe(), UNet(), scheduler()
    pipe.unet, pipe.do_classifier_free_guidance = unet, True
    pipe._pag_scale, pipe.pag_applied_layers = 0., []
    cleanups = []
    try:
        se.install(SimpleNamespace(sdxl=False), pipe, GenerationParams(hooks={"cads": {}, "cfg_schedule": {
                       "start_cfg": 3., "end_cfg": 3.}}), {}, SimpleNamespace(warn=lambda msg: pytest.fail(msg)),
                   cleanups, sched=s, cfg_scale=6., steps=4, seed=71, chain=lambda *a: None)
        # Native pipelines set these flags inside __call__, after extensions are installed.
        pipe._pag_scale, pipe.pag_applied_layers = scale, layers
        active = scale > 0 and bool(layers)
        negative, positive = torch.randn(1, 3, 4), torch.randn(1, 3, 4)
        context = torch.cat([negative, positive, positive] if active else [negative, positive])
        unet.prediction = torch.tensor([1., 3., 4.] if active else [1., 3.]).reshape(-1, 1, 1, 1)
        result = unet(torch.zeros_like(unet.prediction), s.timesteps[0], encoder_hidden_states=context)[0]
        parts = result.flatten()
        actual = parts[0] + 6 * (parts[1] - parts[0])
        if active:
            actual += 2 * (parts[1] - parts[2])
            torch.testing.assert_close(unet.context[1], unet.context[2], rtol=0, atol=0)
        torch.testing.assert_close(unet.context[:1], negative, rtol=0, atol=0)
        assert float(actual) == pytest.approx(5. if active else 7.)
    finally:
        for cleanup in reversed(cleanups):
            cleanup()
    assert not unet._forward_hooks and not unet._forward_pre_hooks


@pytest.mark.parametrize("guided", [False, True])
def test_late_pag_cads_is_exact_passthrough_without_random_draws(guided):
    u, s = torch.nn.Module(), scheduler()
    hook = se.CadsHook(u, s, dict(se.CADS_DEFAULTS), 71, lambda: guided, pag_active=lambda: True)
    context = torch.randn(3 if guided else 2, 3, 4)
    state = hook.gen.get_state().clone()
    assert hook.pre(u, (torch.zeros(context.shape[0], 4, 2, 2), s.timesteps[-1]),
                    dict(encoder_hidden_states=context)) is None
    torch.testing.assert_close(hook.gen.get_state(), state, rtol=0, atol=0)


@pytest.mark.parametrize("guided", [False, True])
@pytest.mark.parametrize("apply_negative", [False, True])
@pytest.mark.parametrize("marked", [False, True])
def test_inactive_pag_cads_keeps_existing_random_draw_layout(guided, apply_negative, marked):
    torch.manual_seed(71)
    u, s = torch.nn.Module(), scheduler()
    u.config = SimpleNamespace(cross_attention_dim=4)
    context = torch.randn(4 if guided else 2, 3, 4)
    if marked:
        context = torch.cat([context, torch.tensor([1., -1., 0.]).view(1, 3, 1).repeat(context.shape[0], 1, 1)], -1)
        context[:, -1, :-1] = 0
    config = dict(se.CADS_DEFAULTS, apply_to_uncond=apply_negative)
    hook = se.CadsHook(u, s, config, 71, lambda: guided, pag_active=lambda: False)
    gen = torch.Generator("cpu").manual_seed(71)
    if guided and not apply_negative:
        negative, positive = context.chunk(2)
        expected = torch.cat([negative, se.cads_context(positive, 0., config, gen, marked=marked)])
    else:
        expected = se.cads_context(context, 0., config, gen, marked=marked)
    _, kw = hook.pre(u, (torch.zeros(context.shape[0], 4, 2, 2), s.timesteps[0]), dict(encoder_hidden_states=context))
    torch.testing.assert_close(kw["encoder_hidden_states"], expected, rtol=0, atol=0)


@pytest.mark.parametrize("pag", [False, True])
def test_constant_cfg_is_bit_exact_for_half_precision(pag):
    s = scheduler()
    torch.manual_seed(71)
    prediction = torch.randn(6 if pag else 4, 4, 2, 2).half()
    hook = se.CfgScheduleHook(None, s, 6., dict(se.CFG_SCHEDULE_DEFAULTS, start_cfg=6., end_cfg=6.),
                              lambda: True, pag_active=lambda: pag)
    assert hook.hook(None, (torch.zeros_like(prediction), s.timesteps[0]), {}, (prediction,)) is None
