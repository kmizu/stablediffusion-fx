import pytest

torch = pytest.importorskip('torch')
from sdfx.dynamic_lora import truncate_state_dict


def test_conventional_rank_slice_preserves_original_alpha_divisor():
    down = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    up = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    state = {'x.lora_down.weight': down, 'x.lora_up.weight': up, 'x.alpha': torch.tensor(8.)}
    for rank in (1, 2, 9, -1):
        result = truncate_state_dict(state, rank)
        actual_rank = len(range(4)[:rank])
        produced = result['x.lora_up.weight'] @ result['x.lora_down.weight'] * (result['x.alpha'] / actual_rank)
        expected = up[:, :rank] @ down[:rank] * 2
        torch.testing.assert_close(produced, expected)
    assert state['x.lora_down.weight'].shape == (4, 3)


def test_convolutional_and_peft_factor_names_and_cp_ignore():
    down = torch.arange(4 * 3 * 3 * 3, dtype=torch.float32).reshape(4, 3, 3, 3)
    up = torch.ones((2, 4, 1, 1))
    state = {'a.lora_A.weight': down, 'a.lora_B.weight': up,
             'b.lora_down.weight': down, 'b.lora_up.weight': up, 'b.lora_mid.weight': torch.ones(4, 4, 3, 3)}
    result = truncate_state_dict(state, 2)
    assert result['a.lora_A.weight'].shape == (2, 3, 3, 3)
    assert result['a.lora_B.weight'].shape == (2, 2, 1, 1)
    assert result['b.lora_down.weight'] is down


def test_incomplete_pairs_are_rejected_and_empty_rank_has_zero_delta():
    with pytest.raises(ValueError, match='incomplete'):
        truncate_state_dict({'x.lora_down.weight': torch.ones(2, 3)}, 1)
    result = truncate_state_dict({'x.lora_down.weight': torch.ones(2, 3), 'x.lora_up.weight': torch.ones(3, 2)}, 0)
    assert not bool(result['x.lora_down.weight'].any())


def test_engine_loads_distinct_ranks_without_changing_source(tmp_path):
    from safetensors.torch import save_file
    from sdfx.engine import Engine
    from sdfx.params import GenerationParams, GenerationRequest, LoraRef
    from types import SimpleNamespace
    path = tmp_path / 'adapter.safetensors'
    source = {'lora_unet_test.lora_down.weight': torch.arange(12, dtype=torch.float32).reshape(4, 3),
              'lora_unet_test.lora_up.weight': torch.ones(2, 4), 'lora_unet_test.alpha': torch.tensor(8.)}
    save_file(source, str(path)); original = path.read_bytes()
    class Pipe:
        def __init__(self): self.loaded = {}; self.selected = []
        def load_lora_weights(self, value, adapter_name): self.loaded[adapter_name] = value
        def enable_lora(self): pass
        def disable_lora(self): pass
        def set_adapters(self, names, adapter_weights): self.selected = names
    pipe = Pipe(); engine = Engine(); engine.pipe = pipe
    gp = GenerationParams(loras=[LoraRef('A', dyn=2), LoraRef('A', dyn=3)])
    req = GenerationRequest(params=gp, lora_paths={'A': str(path)})
    ctx = SimpleNamespace(warn=lambda msg: pytest.fail(msg))
    engine._apply_loras(gp, req, ctx)
    assert len(pipe.loaded) == 2 and len(set(pipe.selected)) == 2
    assert sorted(s['lora_unet_test.lora_down.weight'].shape[0] for s in pipe.loaded.values()) == [2, 3]
    assert path.read_bytes() == original


def test_diffusers_kohya_conversion_keeps_dynamic_alpha():
    from diffusers.loaders import StableDiffusionLoraLoaderMixin
    source = {'lora_unet_down_blocks_0_attentions_0_to_q.lora_down.weight': torch.ones(4, 3),
              'lora_unet_down_blocks_0_attentions_0_to_q.lora_up.weight': torch.ones(2, 4),
              'lora_unet_down_blocks_0_attentions_0_to_q.alpha': torch.tensor(8.)}
    state, alphas = StableDiffusionLoraLoaderMixin.lora_state_dict(truncate_state_dict(source, 2))
    assert any(value.shape == (2, 3) for value in state.values())
    assert alphas and list(alphas.values()) == [4.]


def test_composable_patch_dynamic_rank_uses_original_alpha(tmp_path):
    from types import SimpleNamespace
    from safetensors.torch import save_file
    from sdfx.clora.patches import load
    linear = torch.nn.Linear(3, 2, bias=False)
    pipe = SimpleNamespace(unet=torch.nn.Sequential(linear), text_encoder=None, text_encoder_2=None)
    down = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    up = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    path = tmp_path / 'adapter.safetensors'
    save_file({'lora_unet_0.lora_down.weight': down, 'lora_unet_0.lora_up.weight': up,
               'lora_unet_0.alpha': torch.tensor(8.)}, str(path))
    loaded = load(path, pipe, dyn=2)
    patch = loaded[0][3]
    x = torch.ones((1, 3))
    torch.testing.assert_close(patch.delta(x), x @ down[:2].T @ up[:, :2].T * 2)


def test_composable_rank_variants_route_per_image_and_remove_hooks(tmp_path):
    from types import SimpleNamespace
    from safetensors.torch import save_file
    from sdfx.clora.runtime import Job
    from sdfx.params import GenerationParams, GenerationRequest, LoraRef
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
    model[0].weight.data.copy_(torch.eye(2))
    pipe = SimpleNamespace(unet=model, text_encoder=None, text_encoder_2=None)
    path = tmp_path / 'adapter.safetensors'
    save_file({'lora_unet_0.lora_down.weight': torch.eye(2),
               'lora_unet_0.lora_up.weight': torch.eye(2)}, str(path))
    gp = GenerationParams(prompt='<lora:A:1:1:1> red',
        loras=[LoraRef('A', dyn=1), LoraRef('A', dyn=2)], hooks={'composable_lora': {}})
    req = GenerationRequest(params=gp, lora_paths={'A': str(path)}, manual_loras=[])
    ctx = SimpleNamespace(check_interrupt=lambda: None, warn=lambda msg: pytest.fail(msg))
    job = Job(pipe, gp, req, ctx)
    try:
        prompts = ['<lora:A:1:1:1> red', '<lora:A:1:1:2> blue']
        with job.using(prompts, [True], 1, 10, full_prompt=prompts, negative_prompt=['', '']):
            result = model(torch.ones(2, 2))
        torch.testing.assert_close(result, torch.tensor([[2., 1.], [2., 2.]]))
    finally:
        job.remove()
    assert not model[0]._forward_hooks


def test_composable_manual_and_tagged_dynamic_rank_are_independent(tmp_path):
    from types import SimpleNamespace
    from safetensors.torch import save_file
    from sdfx.clora.runtime import Job
    from sdfx.params import GenerationParams, GenerationRequest, LoraRef
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False)); model[0].weight.data.copy_(torch.eye(2))
    pipe = SimpleNamespace(unet=model, text_encoder=None, text_encoder_2=None)
    path = tmp_path / 'adapter.safetensors'
    save_file({'lora_unet_0.lora_down.weight': torch.eye(2), 'lora_unet_0.lora_up.weight': torch.eye(2)}, str(path))
    manual = LoraRef('A', weight=.5)
    gp = GenerationParams(prompt='<lora:A:1:1:1> red', loras=[manual, LoraRef('A', dyn=1)],
                          hooks={'composable_lora': {}})
    req = GenerationRequest(params=gp, lora_paths={'A': str(path)}, manual_loras=[manual])
    job = Job(pipe, gp, req, SimpleNamespace(check_interrupt=lambda: None, warn=lambda _: None))
    try:
        with job.using(gp.prompt, [True], 1, 10):
            torch.testing.assert_close(model(torch.ones(1, 2)), torch.tensor([[2.5, 1.5]]))
    finally:
        job.remove()


def test_composable_two_ranks_in_one_prompt_add_independently(tmp_path):
    from types import SimpleNamespace
    from safetensors.torch import save_file
    from sdfx.clora.runtime import Job
    from sdfx.params import GenerationParams, GenerationRequest, LoraRef
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False)); model[0].weight.data.copy_(torch.eye(2))
    pipe = SimpleNamespace(unet=model, text_encoder=None, text_encoder_2=None)
    path = tmp_path / 'adapter.safetensors'
    save_file({'lora_unet_0.lora_down.weight': torch.eye(2), 'lora_unet_0.lora_up.weight': torch.eye(2)}, str(path))
    gp = GenerationParams(prompt='<lora:A:1:1:1> <lora:A:1:1:2> red',
                          loras=[LoraRef('A', dyn=1), LoraRef('A', dyn=2)], hooks={'composable_lora': {}})
    req = GenerationRequest(params=gp, lora_paths={'A': str(path)}, manual_loras=[])
    job = Job(pipe, gp, req, SimpleNamespace(check_interrupt=lambda: None, warn=lambda _: None))
    try:
        with job.using(gp.prompt, [True], 1, 10):
            torch.testing.assert_close(model(torch.ones(1, 2)), torch.tensor([[3., 2.]]))
    finally:
        job.remove()


def test_native_peft_dora_dynamic_delta_matches_a1111_at_unit_weight():
    from peft.tuners.lora.dora import DoraLinearLayer
    base = torch.nn.Linear(3, 2, bias=False)
    base.weight.data.copy_(torch.tensor([[1., 0., 0.], [0., 1., 0.]]))
    down = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    up = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    magnitude = torch.tensor([2., 3.])
    state = {'unet.x.lora_A.weight': down, 'unet.x.lora_B.weight': up,
             'unet.x.lora_magnitude_vector.weight': magnitude}
    transformed = truncate_state_dict(state, 2, allow_peft_dora=True)
    a = torch.nn.Linear(3, 2, bias=False); a.weight.data.copy_(transformed['unet.x.lora_A.weight'])
    b = torch.nn.Linear(2, 2, bias=False); b.weight.data.copy_(transformed['unet.x.lora_B.weight'])
    dora = DoraLinearLayer(fan_in_fan_out=False); dora.weight = torch.nn.Parameter(magnitude)
    x = torch.tensor([[1., 2., 3.]])
    actual = base(x) + dora(x, lora_A=a, lora_B=b, scaling=1., base_layer=base)
    merged = base.weight + up[:, :2] @ down[:2]
    expected = torch.nn.functional.linear(x, merged * (magnitude / merged.norm(dim=1)).view(-1, 1))
    torch.testing.assert_close(actual, expected)
    # A1111 applies the external multiplier after DoRA normalization.
    handle = dora.register_forward_hook(lambda _module, _inputs, output: output * .35)
    weighted = base(x) + dora(x, lora_A=a, lora_B=b, scaling=1., base_layer=base)
    handle.remove()
    torch.testing.assert_close(weighted, base(x) + .35 * (expected - base(x)))


def test_native_dora_cached_zero_alpha_rejects_block_weights(tmp_path):
    from safetensors.torch import save_file
    from types import SimpleNamespace
    from sdfx.engine import Engine
    from sdfx.params import GenerationParams, GenerationRequest, LoraRef
    from sdfx.protocol import EngineError
    path = tmp_path / 'dora.safetensors'
    save_file({'unet.block.lora_A.weight': torch.eye(2), 'unet.block.lora_B.weight': torch.eye(2),
               'unet.block.lora_magnitude_vector.weight': torch.ones(2)}, str(path))
    class Pipe:
        def __init__(self):
            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lora_magnitude_vector = torch.nn.ModuleDict()
            self.unet = Model()
        def load_lora_weights(self, value, adapter_name): pass
        def enable_lora(self): pass
        def disable_lora(self): pass
        def set_adapters(self, names, adapter_weights): pass
    engine = Engine(); engine.pipe = Pipe()
    req = GenerationRequest(lora_paths={'A': str(path)})
    ctx = SimpleNamespace(warn=lambda msg: pytest.fail(msg))
    import hashlib, os
    adapter = 'l' + hashlib.sha1((os.path.abspath(path) + '\0dynamic-rank=1').encode()).hexdigest()[:12]
    engine.pipe.unet.lora_magnitude_vector[adapter] = torch.nn.Identity()
    engine.pipe.unet.lora_alpha = {adapter: 0.}
    engine.pipe.unet.r = {adapter: 1}
    engine.pipe.unet.scaling = {adapter: 0.}
    engine._apply_loras(GenerationParams(loras=[LoraRef('A', dyn=1)]), req, ctx)
    with pytest.raises(EngineError, match='zero alpha'):
        engine._apply_loras(GenerationParams(loras=[LoraRef('A', weight=.5, dyn=1, lbw='ALL')]), req, ctx)


def test_dora_window_changes_unet_delta_without_disabling_text_encoder():
    from types import SimpleNamespace
    from sdfx.engine import Engine
    engine = Engine()
    seen = []
    engine.pipe = SimpleNamespace(set_adapters=lambda names, adapter_weights: seen.append(adapter_weights))
    engine.sdxl = False
    values = {'unet': .4, 'text_encoder': .7}
    engine._lora_state = (['dora', 'ordinary'], [1., .8])
    engine._lora_windows = [(0, .3, .6), (1, .3, .6)]
    engine._dora_step_weights = {0: [(values, .4)]}
    engine._lora_key = None
    engine._set_lora_step(1, 10)
    assert values == {'unet': 0., 'text_encoder': .7} and seen[-1][0] == 1.
    assert seen[-1][1]['unet'] == 0.
    engine._set_lora_step(4, 10)
    assert values == {'unet': .4, 'text_encoder': .7}
    assert seen[-1] == [1., .8]
    engine._set_lora_step(8, 10)
    assert values == {'unet': 0., 'text_encoder': .7}


def test_dora_block_weights_capture_per_layer_and_window(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from sdfx.engine import Engine
    from sdfx.params import GenerationParams, GenerationRequest, LoraRef
    from sdfx import dynamic_lora
    path = tmp_path / 'adapter.pt'
    torch.save({'lora_unet_test.lora_down.weight': torch.eye(2),
                'lora_unet_test.lora_up.weight': torch.eye(2),
                'lora_unet_test.dora_scale': torch.ones(2)}, path)
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.lora_magnitude_vector = torch.nn.ModuleDict()
            self.lora_alpha, self.r, self.scaling = {}, {}, {}
    class Pipe:
        def __init__(self): self.unet = torch.nn.Sequential(Layer(), Layer()); self.calls = []
        def enable_lora(self): pass
        def set_adapters(self, names, adapter_weights):
            self.calls.append(adapter_weights)
            for i, layer in enumerate(self.unet):
                weight = adapter_weights[0]
                value = ([.25, .75][i] if isinstance(weight, dict) else weight)
                layer.scaling[names[0]] = value * layer.lora_alpha[names[0]] / layer.r[names[0]]
    engine = Engine(); engine.pipe = Pipe()
    def load(pipe, state, adapter):
        for layer in pipe.unet:
            layer.lora_magnitude_vector[adapter] = torch.nn.Identity()
            layer.lora_alpha[adapter] = 4.; layer.r[adapter] = 2
    monkeypatch.setattr(dynamic_lora, 'load_kohya_dora', load)
    gp = GenerationParams(loras=[LoraRef('A', weight=.5, lbw='ALL', start=.3, stop=.6)])
    engine._apply_loras(gp, GenerationRequest(params=gp, lora_paths={'A': str(path)}),
                        SimpleNamespace(warn=lambda msg: pytest.fail(msg)))
    adapter = next(iter(engine.dynamic_dora_adapters))
    assert engine.pipe.calls[-1] == [1.]
    assert [layer.scaling[adapter] for layer in engine.pipe.unet] == [2., 2.]
    def outputs(): return [layer.lora_magnitude_vector[adapter](torch.tensor(8.)).item() for layer in engine.pipe.unet]
    assert outputs() == [2., 6.]
    engine._set_lora_step(1, 10); assert outputs() == [0., 0.]
    engine._set_lora_step(4, 10); assert outputs() == [2., 6.]


def test_real_peft_dora_block_capture_applies_after_normalization():
    from diffusers.utils.peft_utils import set_weights_and_activate_adapters
    from peft import LoraConfig, inject_adapter_in_model
    from sdfx.dynamic_lora import capture_dora_block_weights, install_dora_weight_hooks
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.first = torch.nn.Linear(2, 2, bias=False); self.second = torch.nn.Linear(2, 2, bias=False)
        def forward(self, x): return self.first(x), self.second(x)
    model = Net()
    for layer in (model.first, model.second): layer.weight.data.copy_(torch.eye(2))
    inject_adapter_in_model(LoraConfig(r=2, lora_alpha=2, target_modules=['first', 'second'], use_dora=True), model, 'a')
    for layer in (model.first, model.second):
        layer.lora_A['a'].weight.data.copy_(torch.eye(2))
        layer.lora_B['a'].weight.data.copy_(torch.eye(2))
        layer.lora_magnitude_vector['a'].weight.data.fill_(3.)
    set_weights_and_activate_adapters(model, ['a'], [{'first': .25, 'second': .75}])
    captured = capture_dora_block_weights([('unet', model)], ['a'])
    set_weights_and_activate_adapters(model, ['a'], [1.])
    handles = install_dora_weight_hooks([('unet', model)], captured)
    try:
        first, second = model(torch.tensor([[1., 0.]]))
        torch.testing.assert_close(first, torch.tensor([[1.5, 0.]]))
        torch.testing.assert_close(second, torch.tensor([[2.5, 0.]]))
    finally:
        for handle in handles: handle.remove()


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


def test_nondynamic_pt_kohya_dora_is_detected_before_standard_loader(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from sdfx.engine import Engine
    from sdfx.params import GenerationParams, GenerationRequest, LoraRef
    from sdfx import dynamic_lora
    path = tmp_path / 'adapter.pt'
    torch.save({'lora_unet_test.lora_down.weight': torch.eye(2),
                'lora_unet_test.lora_up.weight': torch.eye(2),
                'lora_unet_test.dora_scale': torch.ones(2)}, path)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.lora_magnitude_vector = torch.nn.ModuleDict()
    class Pipe:
        def __init__(self): self.unet = Model(); self.weights = None
        def load_lora_weights(self, value, adapter_name): pytest.fail('standard loader dropped DoRA')
        def enable_lora(self): pass
        def set_adapters(self, names, adapter_weights): self.weights = adapter_weights
    engine = Engine(); engine.pipe = Pipe()
    seen = []
    def load(pipe, state, adapter):
        seen.append(state)
        pipe.unet.lora_magnitude_vector[adapter] = torch.nn.Identity()
    monkeypatch.setattr(dynamic_lora, 'load_kohya_dora', load)
    gp = GenerationParams(loras=[LoraRef('A', weight=.4)])
    engine._apply_loras(gp, GenerationRequest(params=gp, lora_paths={'A': str(path)}),
                        SimpleNamespace(warn=lambda msg: pytest.fail(msg)))
    assert seen and 'lora_unet_test.dora_scale' in seen[0]
    assert engine.pipe.weights == [1.0]
    assert engine.pipe.unet.lora_magnitude_vector[next(iter(engine.dynamic_dora_adapters))](torch.tensor(5.)) == 2.


def test_kohya_dora_magnitude_matches_actual_conv_and_linear_targets():
    from types import SimpleNamespace
    from sdfx.dynamic_lora import align_dora_magnitude_shapes
    unet = torch.nn.Module()
    unet.conv = torch.nn.Conv2d(3, 2, 1, bias=False)
    unet.linear = torch.nn.Linear(3, 2, bias=False)
    unet.attn = torch.nn.Module(); unet.attn.to_out = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False))
    text = torch.nn.Module(); text.attn = torch.nn.Module(); text.attn.k_proj = torch.nn.Linear(3, 2, bias=False)
    source = {'unet.conv.lora_magnitude_vector.weight': torch.tensor([2., 3.]),
              'unet.linear.lora_magnitude_vector.weight': torch.tensor([4., 5.]),
              'unet.attn.processor.to_out.lora_magnitude_vector.weight': torch.tensor([6., 7.]),
              'text_encoder.attn.to_k.lora_magnitude_vector.weight': torch.tensor([8., 9.])}
    output = align_dora_magnitude_shapes(SimpleNamespace(unet=unet, text_encoder=text), source)
    assert output['unet.conv.lora_magnitude_vector.weight'].shape == (1, 2, 1, 1)
    assert output['unet.linear.lora_magnitude_vector.weight'].shape == (2,)
    assert output['unet.attn.processor.to_out.lora_magnitude_vector.weight'].shape == (2,)
    assert output['text_encoder.attn.k_proj.lora_magnitude_vector.weight'].tolist() == [8., 9.]
    assert 'text_encoder.attn.to_k.lora_magnitude_vector.weight' not in output
    torch.testing.assert_close(output['unet.conv.lora_magnitude_vector.weight'].flatten(), source['unet.conv.lora_magnitude_vector.weight'])
    with pytest.raises(ValueError, match='magnitude'):
        align_dora_magnitude_shapes(SimpleNamespace(unet=unet), {'unet.conv.lora_magnitude_vector.weight': torch.ones(3)})


def test_kohya_dora_converter_forwards_magnitude_and_alpha():
    from types import SimpleNamespace
    from sdfx.dynamic_lora import load_kohya_dora
    source = {'lora_unet_down_blocks_0_attentions_0_to_q.lora_down.weight': torch.ones(4, 3),
              'lora_unet_down_blocks_0_attentions_0_to_q.lora_up.weight': torch.ones(2, 4),
              'lora_unet_down_blocks_0_attentions_0_to_q.dora_scale': torch.tensor([2., 3.]),
              'lora_unet_down_blocks_0_attentions_0_to_q.alpha': torch.tensor(8.)}
    calls = []
    pipe = SimpleNamespace(unet=SimpleNamespace(get_submodule=lambda _: SimpleNamespace(weight=torch.zeros(2, 3))), text_encoder=None, text_encoder_2=None)
    pipe.load_lora_into_unet = lambda data, **kw: calls.append((data, kw))
    load_kohya_dora(pipe, truncate_state_dict(source, 2, allow_kohya_dora=True), 'adapter')
    data, options = calls[0]
    magnitude = [value for key, value in data.items() if 'lora_magnitude_vector' in key]
    assert len(magnitude) == 1
    torch.testing.assert_close(magnitude[0], torch.tensor([2., 3.]))
    assert list(options['network_alphas'].values()) == [4.]


def test_composable_kohya_dora_dynamic_rank_obeys_a1111_weight_order(tmp_path):
    from types import SimpleNamespace
    from safetensors.torch import save_file
    from sdfx.clora.runtime import Job
    from sdfx.params import GenerationParams, GenerationRequest, LoraRef
    base = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False))
    base[0].weight.data.copy_(torch.tensor([[1., 0., 0.], [0., 1., 0.]]))
    pipe = SimpleNamespace(unet=base, text_encoder=None, text_encoder_2=None)
    down = torch.eye(3); up = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    magnitude = torch.tensor([2., 3.]); path = tmp_path / 'dora.safetensors'
    save_file({'lora_unet_0.lora_down.weight': down, 'lora_unet_0.lora_up.weight': up,
               'lora_unet_0.dora_scale': magnitude}, str(path))
    gp = GenerationParams(prompt='<lora:A:0.5:0.5:2> test', loras=[LoraRef('A', dyn=2)],
                          hooks={'composable_lora': {}})
    req = GenerationRequest(params=gp, lora_paths={'A': str(path)}, manual_loras=[])
    ctx = SimpleNamespace(check_interrupt=lambda: None, warn=lambda _: None)
    job = Job(pipe, gp, req, ctx)
    try:
        x = torch.tensor([[1., 2., 3.]])
        with job.using(gp.prompt, [True], 1, 10): actual = base(x)
        w = base[0].weight.detach(); merged = w + up[:, :2] @ down[:2]
        normalized = merged * (magnitude / merged.norm(dim=1)).view(-1, 1)
        expected = torch.nn.functional.linear(x, w + .5 * (normalized - w))
        torch.testing.assert_close(actual, expected)
    finally: job.remove()
