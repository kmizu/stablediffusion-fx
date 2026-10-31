"""Real PEFT inference tensors survive Accelerate offload hook replacement."""
from types import SimpleNamespace
import pytest

torch = pytest.importorskip('torch')
peft = pytest.importorskip('peft')
accelerate = pytest.importorskip('accelerate')


def test_engine_drops_and_reinstalls_sequential_hooks_with_inference_adapter():
    from accelerate.hooks import remove_hook_from_module
    from sdfx.engine import Engine
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.proj = torch.nn.Linear(2, 2)
        def forward(self, value): return self.proj(value)
    model = Model()
    with torch.inference_mode():
        peft.inject_adapter_in_model(peft.LoraConfig(r=1, lora_alpha=1, target_modules=['proj']),
                                     model, adapter_name='photomaker')
        accelerate.cpu_offload(model, execution_device=torch.device('cpu'))
        model(torch.ones(1, 2))
    pipe = SimpleNamespace(remove_all_hooks=lambda: remove_hook_from_module(model, recurse=True),
                           enable_sequential_cpu_offload=lambda: accelerate.cpu_offload(model, execution_device=torch.device('cpu')))
    engine = Engine.__new__(Engine)
    engine.cuda = True; engine.load_opts = SimpleNamespace(vram_mode='sequential_offload')
    engine.hook_owner = pipe
    engine._drop_hooks()
    assert all(not p.is_meta for p in model.parameters()), 'hook detach must not silently leave an inference tensor on meta'
    assert all(not hasattr(m, '_hf_hook') for m in model.modules())
    engine._place(pipe)
    with torch.inference_mode():
        assert torch.isfinite(model(torch.ones(1, 2))).all()
    engine._drop_hooks()
    assert all(not p.is_meta for p in model.parameters())
