import pytest
torch = pytest.importorskip("torch")
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from sdfx.reference_attention import attend


@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("mask_kind", [None, "bool", "additive"])
@pytest.mark.parametrize("normalization", [None, "layer_norm", "group_norm"])
def test_separate_projections_match_native_context_attention(heads, mask_kind, normalization):
    torch.manual_seed(77)
    a=Attention(query_dim=4,heads=heads,dim_head=4//heads,bias=True,residual_connection=True,
                cross_attention_norm=normalization,cross_attention_norm_num_groups=2)
    x,ref=torch.randn(2,9,4),torch.randn(2,9,4)
    mask=None
    if mask_kind:
        mask=torch.ones(2,1,9,dtype=torch.bool); mask[...,2]=False
        if mask_kind=="additive": mask=torch.zeros_like(mask,dtype=torch.float32).masked_fill(~mask,-torch.inf)
    extended=None if mask is None else torch.nn.functional.pad(mask,(0,9),value=True if mask_kind=="bool" else 0.)
    expected=AttnProcessor2_0()(a,x,encoder_hidden_states=torch.cat([x,ref*0.7],1),attention_mask=extended)
    lengths=[]
    h=a.to_k.register_forward_pre_hook(lambda m,args:lengths.append(args[0].shape[1]))
    try: got=attend(a,x,ref,0.7,attention_mask=mask)
    finally: h.remove()
    torch.testing.assert_close(got,expected)
    assert lengths==[9,9]


@pytest.mark.parametrize("fail", [False, True])
def test_branch_hooks_prepare_in_priority_order_and_always_unwind(fail):
    from diffusers import DDIMScheduler
    from sdfx.neutral_runtime import RuntimeHook, Plan
    from sdfx.neutral_prompt import parse
    events = []
    class UNet(torch.nn.Module):
        def forward(self, sample, timestep, **kw):
            events.append("forward")
            return (sample,)
    class Hook:
        def __init__(self, name, priority):
            self.name, self.priority = name, priority
        def before_branch(self, forward, sample, timestep, kw, layout):
            events.append(self.name + "+")
            assert layout == [False, True]
            if fail and self.name == "reference":
                raise RuntimeError("prepare failed")
        def after_branch(self):
            events.append(self.name + "-")
    scheduler = DDIMScheduler()
    scheduler.set_timesteps(4)
    unet = UNet()
    branch = dict(prompt_embeds=torch.ones(1, 3, 4), negative_prompt_embeds=torch.zeros(1, 3, 4))
    runtime = RuntimeHook(unet, scheduler, Plan(parse("one AND two"), [branch, branch]), 7, 0,
                          branch_hooks=[Hook("reference", 10), Hook("lllite", 0)]).install()
    try:
        if fail:
            with pytest.raises(RuntimeError, match="prepare failed"):
                unet(torch.randn(2, 4, 2, 2), scheduler.timesteps[0])
        else:
            unet(torch.randn(2, 4, 2, 2), scheduler.timesteps[0])
    finally:
        runtime.remove()
    expected = ["lllite+", "reference+"] + ([] if fail else ["forward"]) + ["reference-", "lllite-"]
    assert events == expected * (1 if fail else 2)


@pytest.mark.parametrize("layout", [[True], [False, True], [False, True, True], [False, True, False, True]])
@pytest.mark.parametrize("fidelity", [0., 0.7, 1.])
def test_managed_reference_positive_mask_and_branch_bank(layout, fidelity):
    from sdfx.adapters import ReferenceOnlyHook
    from diffusers import DDIMScheduler
    class UNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1 = Attention(query_dim=4, heads=2, dim_head=2)
            self.calls = []
        def forward(self, sample, timestep, encoder_hidden_states=None, **kw):
            self.calls.append(encoder_hidden_states.clone())
            x = sample.flatten(2).transpose(1, 2) + encoder_hidden_states.mean(1)[:, None]
            return (self.attn1(x).transpose(1, 2).reshape_as(sample),)
    torch.manual_seed(41)
    unet = UNet()
    scheduler = DDIMScheduler()
    scheduler.set_timesteps(4)
    sample = torch.randn(len(layout) * 2, 4, 3, 3)
    context = torch.randn(len(layout) * 2, 5, 4)
    hook = ReferenceOnlyHook(unet, torch.randn(1, 4, 3, 3), scheduler, fidelity, 0.8).install(managed=True)
    try:
        hook.before_branch(unet.forward, sample, scheduler.timesteps[0], {"encoder_hidden_states": context}, layout)
        assert len(unet.calls) == 1
        torch.testing.assert_close(unet.calls[0], context)
        bank = hook.bank[id(unet.attn1)]
        hs = sample.flatten(2).transpose(1, 2) + context.mean(1)[:, None]
        plain = AttnProcessor2_0()(unet.attn1, hs)
        referenced = attend(unet.attn1, hs, bank, 0.8)
        positive = torch.tensor(layout).repeat_interleave(2).view(-1, 1, 1)
        expected = torch.where(positive, referenced * (1 - fidelity) + plain * fidelity, referenced)
        got = unet.forward(sample, scheduler.timesteps[0], encoder_hidden_states=context)[0]
        torch.testing.assert_close(got.flatten(2).transpose(1, 2), expected)
    finally:
        hook.after_branch()
        assert hook.mode == "off" and not hook.bank
        hook.remove()


@pytest.mark.parametrize("guided", [False, True])
def test_reference_changes_pag_normal_attention_and_preserves_identity_branch(guided):
    from sdfx.adapters import ReferenceOnlyHook
    from diffusers import DDIMScheduler
    from diffusers.models.attention_processor import PAGIdentitySelfAttnProcessor2_0, PAGCFGIdentitySelfAttnProcessor2_0
    torch.manual_seed(71)
    unet = torch.nn.Module()
    unet.attn1 = Attention(query_dim=4, heads=2, dim_head=2)
    unet.attn1.set_processor(PAGCFGIdentitySelfAttnProcessor2_0() if guided else PAGIdentitySelfAttnProcessor2_0())
    layout = [False, True, True] if guided else [False, True, False, True]
    hs = torch.randn(len(layout) * 2, 9, 4)
    baseline = unet.attn1(hs)
    ref = torch.randn_like(hs)
    hook = ReferenceOnlyHook(unet, torch.randn(1, 4, 3, 3), DDIMScheduler(), 0.5, 0.8).install(managed=True)
    try:
        hook.mode, hook.layout = "read", layout
        hook.bank[id(unet.attn1)] = ref
        got = unet.attn1(hs)
        n = hs.shape[0] * (2 if guided else 1) // (3 if guided else 2)
        torch.testing.assert_close(got[n:], baseline[n:], rtol=0, atol=0)
        assert not torch.allclose(got[:n], baseline[:n])
        expected = attend(unet.attn1, hs[:n], ref[:n], 0.8)
        positive = torch.tensor(layout[:2]).repeat_interleave(2).view(-1, 1, 1)
        expected = torch.where(positive, expected * 0.5 + baseline[:n] * 0.5, expected)
        torch.testing.assert_close(got[:n], expected)
    finally:
        hook.remove()


def test_separate_reference_projections_keep_spatial_conditioning_active():
    torch.manual_seed(89)
    attn = Attention(query_dim=4, heads=2, dim_head=2)
    x, ref = torch.randn(2, 9, 4), torch.randn(2, 9, 4)
    spatial = torch.randn(2, 9, 4)
    def pre(module, args):
        # LLLite refuses conditioning when the token count is not its image grid size.
        return (args[0] + spatial,) if args[0].shape[1] == 9 else None
    handles = [linear.register_forward_pre_hook(pre) for linear in (attn.to_k, attn.to_v)]
    try:
        got = attend(attn, x, ref, 0.7)
        native_skipping_condition = AttnProcessor2_0()(attn, x, encoder_hidden_states=torch.cat([x, ref * 0.7], 1))
    finally:
        for handle in handles:
            handle.remove()
    expected = AttnProcessor2_0()(attn, x, encoder_hidden_states=torch.cat([x + spatial, ref * 0.7 + spatial], 1))
    torch.testing.assert_close(got, expected)
    assert not torch.allclose(got, native_skipping_condition)


def test_managed_lllite_window_activation_is_reset_between_branches():
    from sdfx.adapters import ControlLLLiteHook
    from types import SimpleNamespace
    hook = ControlLLLiteHook(torch.nn.Module(), "unused", None, 1., 0.25, 0.75)
    module = SimpleNamespace(active=True)
    hook.modules["test"] = module
    for timestep, expected in [(900, False), (500, True), (100, False)]:
        hook.before_branch(None, None, timestep, {}, [False, True])
        assert module.active == expected
        hook.after_branch()
        assert module.active is False


def test_reference_zero_weight_never_writes_or_changes_attention():
    from sdfx.adapters import ReferenceOnlyHook
    from diffusers import DDIMScheduler
    unet = torch.nn.Module()
    unet.attn1 = Attention(query_dim=4, heads=2, dim_head=2)
    scheduler = DDIMScheduler()
    scheduler.set_timesteps(4)
    hook = ReferenceOnlyHook(unet, torch.randn(1, 4, 3, 3), scheduler, 0.5, 0).install(managed=True)
    hs = torch.randn(2, 9, 4)
    expected = AttnProcessor2_0()(unet.attn1, hs)
    try:
        hook.before_branch(lambda *a, **kw: pytest.fail("zero weight wrote reference"),
                           torch.randn(2, 4, 3, 3), scheduler.timesteps[0], {}, [False, True])
        torch.testing.assert_close(unet.attn1(hs), expected, rtol=0, atol=0)
    finally:
        hook.after_branch()
        hook.remove()
