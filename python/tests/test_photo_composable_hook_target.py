"""Real PEFT injection and reload must not change PhotoMaker hook ordering."""
import pytest

torch = pytest.importorskip('torch')
peft = pytest.importorskip('peft')


def test_photo_hook_target_survives_injection_disable_and_adapter_reload():
    from sdfx.clora.runtime import _photo_hook_target

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(1, 1, bias=False).to(torch.bfloat16)
            self.proj.weight.data.fill_(1)
        def forward(self, x):
            return self.proj(x)

    net = Net()
    base = net.proj
    x = torch.ones(1, 1, dtype=torch.bfloat16)
    def addition(module, args, result):
        return result + torch.full_like(result, .00390625)
    first_handle = _photo_hook_target(base, True).register_forward_hook(addition)
    cfg = peft.LoraConfig(r=1, lora_alpha=1, target_modules=['proj'], init_lora_weights=False)
    peft.inject_adapter_in_model(cfg, net, adapter_name='photomaker')
    with torch.no_grad():
        net.proj.lora_A['photomaker'].weight.fill_(1)
        net.proj.lora_B['photomaker'].weight.fill_(.0078125)
    first = net(x)
    first_handle.remove()
    assert len(base._forward_hooks) == 0
    assert _photo_hook_target(net.proj, True) is base
    assert _photo_hook_target(net.proj, False) is net.proj
    for active in (True, False, True):
        net.proj.enable_adapters(active)
        handle = _photo_hook_target(net.proj, True).register_forward_hook(addition)
        try:
            observed = net(x)
            if active:
                torch.testing.assert_close(observed, first, rtol=0, atol=0)
        finally:
            handle.remove()
        assert not base._forward_hooks
    net.proj.delete_adapter('photomaker')
    peft.inject_adapter_in_model(cfg, net, adapter_name='photomaker')
    assert _photo_hook_target(net.proj, True) is base
    assert not base._forward_hooks


def test_actual_job_registers_on_photo_base_and_removes_hooks_on_guard(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from sdfx.clora import runtime
    from sdfx.protocol import EngineError
    model = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False))
    cfg = peft.LoraConfig(r=1, lora_alpha=1, target_modules=['0'])
    peft.inject_adapter_in_model(cfg, model, adapter_name='photomaker')
    layer, base = model[0], model[0].get_base_layer()
    source = tmp_path / 'adapter.safetensors'; source.touch()
    spec = SimpleNamespace(name='x', dyn=None, lbw=None, start=None, stop=None, weight=1., unet=None)
    gp = SimpleNamespace(hooks={'photomaker': {}}, loras=[spec], prompt='', hires=SimpleNamespace(prompt=''))
    req = SimpleNamespace(manual_loras=None, lora_paths={'x': str(source)})
    patch = SimpleNamespace(name='', cache={})
    monkeypatch.setattr(runtime.patches, 'load', lambda *args: [('unet', '0', layer, patch)])
    original_register = base.register_forward_hook
    recorded = []
    def register(fn):
        recorded.append('base')
        return original_register(fn)
    monkeypatch.setattr(base, 'register_forward_hook', register)
    with pytest.raises(EngineError, match='pending real pipeline parity'):
        runtime.Job(SimpleNamespace(unet=model), gp, req, SimpleNamespace())
    assert recorded == ['base']
    assert not base._forward_hooks and not layer._forward_hooks
