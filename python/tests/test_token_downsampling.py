import pytest
from sdfx.protocol import EngineError


@pytest.mark.parametrize('config', [{'factor': 0}, {'factor2': 6}, {'factor': True},
    {'factor': float('nan')}, {'disable_after': float('inf')}, {'disable_after': -0.1}])
def test_invalid_configuration(config):
    from sdfx.token_downsampling import normalize
    with pytest.raises(EngineError):
        normalize(config)


def test_nearest_exact_resampling_keeps_batch_channels_and_odd_shape():
    torch = pytest.importorskip('torch')
    from sdfx.token_downsampling import resample
    x = torch.arange(2 * 5 * 7 * 3, dtype=torch.float32).reshape(2, 35, 3)
    expected = torch.nn.functional.interpolate(x.reshape(2, 5, 7, 3).permute(0, 3, 1, 2),
        size=(2, 3), mode='nearest-exact').permute(0, 2, 3, 1).reshape(2, 6, 3)
    assert torch.equal(resample(x, 5, 7, 2), expected)
    assert resample(x, 5, 7, 1) is x
    assert resample(x[:, :1], 1, 1, 5).shape == (2, 1, 3)
    with pytest.raises(EngineError):
        resample(x, 7, 7, 2)


def tiny_unet():
    torch = pytest.importorskip('torch')
    pytest.importorskip('diffusers')
    from diffusers import UNet2DConditionModel
    torch.manual_seed(41)
    torch.set_num_threads(2)
    return UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
        down_block_types=('CrossAttnDownBlock2D', 'CrossAttnDownBlock2D'),
        up_block_types=('CrossAttnUpBlock2D', 'CrossAttnUpBlock2D'),
        block_out_channels=(8, 16), layers_per_block=1, cross_attention_dim=8,
        attention_head_dim=2, norm_num_groups=4).eval()


@pytest.mark.parametrize('sdxl,depth', [(False, 1), (True, 2)])
def test_real_attention_keeps_queries_reduces_keys_and_restores_after_cutoff(sdxl, depth):
    torch = pytest.importorskip('torch')
    from sdfx.token_downsampling import TokenDownsampling
    unet = tiny_unet()
    x, context = torch.randn(2, 4, 9, 11), torch.randn(2, 5, 8)
    processors = dict(unet.attn_processors)
    observed = []
    target = unet.down_blocks[depth - 1].attentions[0].transformer_blocks[0].attn1
    handles = [target.to_q.register_forward_pre_hook(lambda m, a: observed.append(('q', a[0].shape[1]))),
               target.to_k.register_forward_pre_hook(lambda m, a: observed.append(('k', a[0].shape[1])))]
    with torch.no_grad():
        baseline = unet(x, 1, context).sample
        observed.clear()
        hook = TokenDownsampling(unet, dict(factor=2, factor2=1, disable_after=.5), sdxl, 2).install()
        try:
            active = unet(x, 1, context).sample
            first = list(observed)
            observed.clear()
            inactive = unet(x, 1, context).sample
            assert torch.equal(inactive, baseline)
            assert not torch.equal(active, baseline)
            assert first[1][1] < first[0][1]
            assert observed[0][1] == observed[1][1]
        finally:
            hook.remove()
        assert torch.equal(unet(x, 1, context).sample, baseline)
    assert all(unet.attn_processors[key] is value for key, value in processors.items())
    for handle in handles:
        handle.remove()
    assert all(not module._forward_pre_hooks for module in unet.modules())


@pytest.mark.parametrize('config', [dict(factor=1, factor2=1), dict(factor=3, disable_after=0)])
def test_noop_installs_no_hooks(config):
    from sdfx.token_downsampling import TokenDownsampling
    unet = tiny_unet()
    hook = TokenDownsampling(unet, config, False, 4).install()
    assert not hook.handles
    hook.remove()


def test_mask_and_keyword_inputs_sample_identical_keys_and_reinstall_cleans_up():
    torch = pytest.importorskip('torch')
    from sdfx.token_downsampling import TokenDownsampling, resample
    unet = tiny_unet()
    hook = TokenDownsampling(unet, dict(factor=1, factor2=2), False, 2).install()
    attention = unet.down_blocks[1].attentions[0].transformer_blocks[0].attn1
    states = torch.randn(2, 35, 16)
    mask = torch.arange(35.).reshape(1, 1, 35).repeat(2, 1, 1)
    try:
        args, kwargs = hook._attention([(5, 7)], 2, attention, (), dict(hidden_states=states, attention_mask=mask))
        assert not args and kwargs['hidden_states'] is states
        assert torch.equal(kwargs['encoder_hidden_states'], resample(states, 5, 7, 2))
        assert kwargs['attention_mask'].shape == (2, 1, 6)
        assert torch.equal(kwargs['attention_mask'][0, 0], torch.tensor([8., 10., 12., 22., 24., 26.]))
        count = len(hook.handles)
        hook.install()
        assert len(hook.handles) == count
    finally:
        hook.remove()
    hook.install().remove()
    assert all(not module._forward_pre_hooks for module in unet.modules())


def test_processor_error_does_not_change_identity_and_cleanup_is_safe():
    torch = pytest.importorskip('torch')
    from sdfx.token_downsampling import TokenDownsampling
    unet = tiny_unet()
    attention = unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1
    class UnknownProcessor:
        pass
    processor = UnknownProcessor()
    attention.set_processor(processor)
    hook = TokenDownsampling(unet, dict(factor=2), False, 2).install()
    try:
        with pytest.raises(EngineError):
            unet(torch.randn(1, 4, 8, 8), 1, torch.randn(1, 5, 8))
    finally:
        hook.remove()
    assert attention.processor is processor
    assert all(not module._forward_pre_hooks for module in unet.modules())


def test_hook_is_advertised_without_torch_import():
    from sdfx import hooks
    from sdfx.dryrun import FakeEngine
    assert 'token_downsampling' in hooks.KNOWN_HOOKS
    assert 'token_downsampling' in FakeEngine().features()


def test_cutoff_budget_uses_requested_steps_not_img2img_strength():
    from sdfx.token_downsampling import step_budget, TokenDownsampling
    assert step_budget(20, 4, 'base') == 20
    assert step_budget(20, 4, 'hires') == 4
    assert step_budget(20, 0, 'hires') == 20
    hook = TokenDownsampling(None, dict(factor=2, disable_after=.5), False, step_budget(20, 0, 'base'))
    assert hook.limit == 10  # A 10-evaluation strength=.5 pass stays active throughout.


def test_branch_forward_bypass_ticks_each_evaluation_and_not_outer_call():
    torch = pytest.importorskip('torch')
    from sdfx.token_downsampling import TokenDownsampling
    unet = tiny_unet()
    x, context = torch.randn(2, 4, 8, 8), torch.randn(2, 5, 8)
    with torch.no_grad():
        baseline = unet(x, 1, context).sample
        hook = TokenDownsampling(unet, dict(factor=2, disable_after=.5), False, 2, branch_mode=True).install()
        try:
            assert not unet._forward_pre_hooks
            hook.before_branch(unet.forward, x, 1, {}, [False, True])
            active = unet.forward(x, 1, context).sample
            hook.after_branch()
            hook.before_branch(unet.forward, x, 1, {}, [False, True])
            inactive = unet.forward(x, 1, context).sample
            hook.after_branch()
            assert hook.evaluations == 2
            assert not torch.equal(active, baseline)
            assert torch.equal(inactive, baseline)
        finally:
            hook.remove()
