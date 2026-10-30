"""Shared #85 inference guard regression, retained for PhotoMaker interaction."""
import pytest
torch = pytest.importorskip("torch")


def test_id_encoder_real_module_preserves_weights_after_interruption():
    from types import SimpleNamespace
    from sdfx.photomaker_adapters import id_encoder_scope
    encoder = torch.nn.Conv2d(3, 4, kernel_size=2).eval()
    weights = {name: value.detach().clone() for name, value in encoder.state_dict().items()}
    identities = [id(parameter) for parameter in encoder.parameters()]
    engine = SimpleNamespace(pipe=object(), device='cpu')
    with pytest.raises(RuntimeError, match='interrupted'):
        with id_encoder_scope(engine, SimpleNamespace(id_encoder=encoder)):
            with torch.inference_mode():
                assert encoder(torch.ones(1, 3, 4, 4)).shape == (1, 4, 3, 3)
            raise RuntimeError('interrupted')
    assert identities == [id(parameter) for parameter in encoder.parameters()]
    for name, value in encoder.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)

def test_offloaded_inference_peft_adapters_reactivate_across_jobs():
    from types import SimpleNamespace
    from accelerate.utils import set_module_tensor_to_device
    from diffusers.utils.peft_utils import set_weights_and_activate_adapters
    from peft import LoraConfig, inject_adapter_in_model
    from sdfx.engine import Engine
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
    inject_adapter_in_model(LoraConfig(r=2, lora_alpha=2, target_modules=['0']), model, 'a')
    name = '0.lora_A.a.weight'
    before = dict(model.named_parameters())[name].detach().clone()
    with torch.inference_mode():
        moved = before.clone()
        set_module_tensor_to_device(model, name, 'cpu', value=moved)
    parameter = dict(model.named_parameters())[name]
    assert parameter.is_inference() and parameter.requires_grad
    def set_adapters(names, adapter_weights):
        set_weights_and_activate_adapters(model, names, adapter_weights)
    pipe = SimpleNamespace(set_adapters=set_adapters,
        enable_lora=lambda: model[0].enable_adapters(True),
        disable_lora=lambda: model[0].enable_adapters(False))
    with pytest.raises(RuntimeError, match='inference tensor'):
        pipe.set_adapters(['a'], [1.])
    for _ in range(2):
        Engine._set_inference_adapters(SimpleNamespace(_offload=True), pipe, ['a'], [.5])
        torch.testing.assert_close(dict(model.named_parameters())[name], before)
        assert not torch.is_inference_mode_enabled()
    assert dict(model.named_parameters())[name].requires_grad
    Engine._set_inference_lora_enabled(SimpleNamespace(_offload=True), pipe, False)
    assert model[0].disable_adapters
    with pytest.raises(RuntimeError, match='inference tensor'):
        pipe.enable_lora()
    Engine._set_inference_lora_enabled(SimpleNamespace(_offload=True), pipe, True)
    assert not model[0].disable_adapters and not torch.is_inference_mode_enabled()
    assert dict(model.named_parameters())[name].requires_grad
