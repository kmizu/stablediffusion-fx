import pytest
torch = pytest.importorskip("torch")

from sdfx.neutral_controlnet import ControlNetFrame


@pytest.mark.parametrize("guess", [False, True])
@pytest.mark.parametrize("layout", [[False, True], [False, True, True], [False, True, False, True]])
@pytest.mark.parametrize("batch", [1, 2])
def test_branch_residuals_follow_text_pooled_and_guess_mode(guess, layout, batch):
    class CN(torch.nn.Module):
        def forward(self, sample, timestep, encoder_hidden_states=None, controlnet_cond=None,
                    conditioning_scale=1., guess_mode=False, added_cond_kwargs=None, return_dict=False):
            value = encoder_hidden_states[:, 0, 0] + added_cond_kwargs["text_embeds"][:, 0]
            assert controlnet_cond.shape[0] == sample.shape[0]
            assert added_cond_kwargs["time_ids"].shape[0] == sample.shape[0]
            down = sample * 0 + value[:, None, None, None] * conditioning_scale
            return [down, down * 2], down * 3
    model = CN()
    frame = ControlNetFrame(model).install()
    native_batch = batch if guess else batch * 2
    model(torch.zeros(native_batch, 4, 2, 2), torch.tensor(750),
          encoder_hidden_states=torch.ones(native_batch, 77, 1), controlnet_cond=torch.ones(native_batch, 3, 16, 16),
          conditioning_scale=0.7, guess_mode=guess,
          added_cond_kwargs={"text_embeds": torch.ones(native_batch, 8), "time_ids": torch.ones(native_batch, 6)})
    snapshot = frame.latest
    kw = dict(encoder_hidden_states=torch.cat([torch.full((batch, 77, 1), 5. if p else 2.) for p in layout]),
              added_cond_kwargs={"text_embeds": torch.cat([torch.full((batch, 8), 11. if p else 3.) for p in layout]),
                                 "time_ids": torch.ones(batch * len(layout), 6)})
    try:
        down, mid = frame.residuals(kw, batch, layout)
        expected = torch.cat([torch.full((batch,), 16*0.7 if p else 0 if guess else 5*0.7) for p in layout])
        torch.testing.assert_close(down[0][:, 0, 0, 0], expected)
        torch.testing.assert_close(mid[:, 0, 0, 0], expected * 3)
        assert frame.latest is snapshot
    finally:
        frame.remove()
        frame.remove()
    assert not model._forward_pre_hooks


@pytest.mark.parametrize("guess", [False, True])
def test_runtime_uses_recomputed_residuals_instead_of_native_first_branch(guess):
    from types import SimpleNamespace
    from diffusers import DDIMScheduler
    from sdfx.neutral_runtime import RuntimeHook, Plan
    from sdfx.neutral_prompt import parse
    class CN(torch.nn.Module):
        def forward(self, sample, timestep, encoder_hidden_states, controlnet_cond, **kw):
            out = torch.ones_like(sample) * encoder_hidden_states[:, 0, 0, None, None, None]
            return [out], out
    class UNet(torch.nn.Module):
        def forward(self, sample, timestep, encoder_hidden_states, down_block_additional_residuals=None,
                    mid_block_additional_residual=None, **kwargs):
            return (down_block_additional_residuals[0],)
    cn, unet = CN(), UNet()
    frame = ControlNetFrame(cn).install()
    scheduler = DDIMScheduler(clip_sample=False)
    scheduler.set_timesteps(4)
    branches = [dict(prompt_embeds=torch.full((1, 77, 1), v), negative_prompt_embeds=torch.ones(1, 77, 1)) for v in (3., 7.)]
    hook = RuntimeHook(unet, scheduler, Plan(parse("a :0.5 AND b :0.5"), branches), 6, 0, controlnet=frame).install()
    n = 1 if guess else 2
    try:
        stale, mid = cn(torch.zeros(n, 4, 2, 2), scheduler.timesteps[0],
                        encoder_hidden_states=torch.full((n, 77, 1), 100.),
                        controlnet_cond=torch.zeros(n, 3, 16, 16), guess_mode=guess)
        out = unet(torch.zeros(2, 4, 2, 2), scheduler.timesteps[0], encoder_hidden_states=torch.zeros(2, 77, 1),
                   down_block_additional_residuals=stale, mid_block_additional_residual=mid)[0]
        torch.testing.assert_close(out[:1], torch.full_like(out[:1], 0 if guess else 1))
        torch.testing.assert_close(out[1:], torch.full_like(out[1:], 5))
    finally:
        hook.remove()
        frame.remove()
        frame.remove()
    assert not cn._forward_pre_hooks


def test_multi_control_images_and_window_scale_are_retained():
    seen = []
    class Multi(torch.nn.Module):
        def forward(self, sample, timestep, encoder_hidden_states=None, controlnet_cond=None,
                    conditioning_scale=None, **kw):
            assert len(controlnet_cond) == 2
            assert all(x.shape[0] == sample.shape[0] for x in controlnet_cond)
            seen.append(conditioning_scale)
            out = sample * sum(conditioning_scale)
            return [out], out
    cn = Multi()
    frame = ControlNetFrame(cn).install()
    try:
        for scales in ([0., 0.], [0.2, 0.5]):
            cn(torch.ones(1, 4, 2, 2), torch.tensor(2), encoder_hidden_states=torch.ones(1, 77, 1),
               controlnet_cond=[torch.zeros(1, 3, 16, 16), torch.ones(1, 3, 16, 16)], conditioning_scale=scales)
            down, _ = frame.residuals({"encoder_hidden_states": torch.ones(2, 77, 1)}, 1, [False, True])
            torch.testing.assert_close(down[0], torch.full_like(down[0], sum(scales)))
        assert seen == [[0., 0.], [0., 0.], [0.2, 0.5], [0.2, 0.5]]
    finally:
        frame.remove()


def test_failed_replay_keeps_native_frame_and_removes_hooks():
    class CN(torch.nn.Module):
        def forward(self, sample, timestep, encoder_hidden_states, **kw):
            if encoder_hidden_states.max() > 1:
                raise RuntimeError("bad control branch")
            return [sample], sample
    cn = CN()
    frame = ControlNetFrame(cn).install()
    cn(torch.zeros(2, 4, 2, 2), 5, encoder_hidden_states=torch.ones(2, 77, 1))
    snapshot = frame.latest
    try:
        with pytest.raises(RuntimeError, match="bad control branch"):
            frame.residuals({"encoder_hidden_states": torch.full((2, 77, 1), 2.)}, 1, [False, True])
        assert not frame.replaying
        assert frame.latest is snapshot
    finally:
        frame.remove()
    assert frame.latest is None and not cn._forward_pre_hooks
