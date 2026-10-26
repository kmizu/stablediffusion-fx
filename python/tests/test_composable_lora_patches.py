import pytest
torch = pytest.importorskip('torch')
from types import SimpleNamespace


def test_lora_linear_and_spatial_locon_match_explicit_operations():
    from sdfx.clora.patches import Patch
    torch.manual_seed(7)
    layer = torch.nn.Linear(3, 4, bias=False)
    down, up, x = torch.randn(2, 3), torch.randn(4, 2), torch.randn(5, 3)
    p = Patch({'lora_down.weight': down, 'lora_up.weight': up, 'alpha': torch.tensor(1.)}, layer)
    torch.testing.assert_close(p.delta(x), ((x @ down.T) @ up.T) * .5)
    conv = torch.nn.Conv2d(3, 4, 3, padding=1, bias=False)
    down, up, x = torch.randn(2, 3, 3, 3), torch.randn(4, 2, 1, 1), torch.randn(2, 3, 7, 7)
    p = Patch({'lora_down.weight': down, 'lora_up.weight': up}, conv)
    expected = torch.nn.functional.conv2d(torch.nn.functional.conv2d(x, down, padding=1), up)
    torch.testing.assert_close(p.delta(x), expected)


def test_loha_lokr_full_and_ia3_actual_weight_algebra():
    from sdfx.clora.patches import Patch
    torch.manual_seed(9)
    layer = torch.nn.Linear(4, 4, bias=False)
    x = torch.randn(3, 4)
    a, b, c, d = [torch.randn(4, 4) for _ in range(4)]
    patch = Patch(dict(hada_w1_a=a, hada_w1_b=b, hada_w2_a=c, hada_w2_b=d), layer)
    torch.testing.assert_close(patch.delta(x), x @ ((a @ b) * (c @ d)).T)
    w1, w2 = torch.randn(2, 2), torch.randn(2, 2)
    patch = Patch(dict(lokr_w1=w1, lokr_w2=w2), layer)
    torch.testing.assert_close(patch.delta(x), x @ torch.kron(w1, w2).T)
    patch = Patch(dict(diff=a), layer)
    torch.testing.assert_close(patch.delta(x), x @ a.T)
    w = torch.randn(4)
    patch = Patch(dict(weight=w, on_input=torch.tensor(True)), layer)
    torch.testing.assert_close(patch.delta(x), torch.nn.functional.linear(x * w, layer.weight))


def test_locon_middle_and_loha_cp_kernel_are_real_convolutions():
    from sdfx.clora.patches import Patch
    torch.manual_seed(11)
    layer = torch.nn.Conv2d(3, 4, 3, padding=1, bias=False)
    x = torch.randn(2, 3, 6, 6)
    up, down, middle = torch.randn(4, 2, 1, 1), torch.randn(2, 3, 1, 1), torch.randn(2, 2, 3, 3)
    p = Patch({'lora_up.weight': up, 'lora_down.weight': down, 'lora_mid.weight': middle}, layer)
    expected = torch.nn.functional.conv2d(torch.nn.functional.conv2d(torch.nn.functional.conv2d(x, down), middle, padding=1), up)
    torch.testing.assert_close(p.delta(x), expected)
    a, b = up[:, :, 0, 0].T, down[:, :, 0, 0]
    p = Patch(dict(hada_w1_a=a, hada_w1_b=b, hada_t1=middle,
                   hada_w2_a=a, hada_w2_b=b, hada_t2=middle), layer)
    expected_kernel = torch.zeros(4, 3, 3, 3)
    for o in range(4):
        for i in range(3):
            for h in range(3):
                for w in range(3):
                    expected_kernel[o, i, h, w] = (a[:, o] @ middle[:, :, h, w] @ b[:, i]) ** 2
    torch.testing.assert_close(p.delta(x), torch.nn.functional.conv2d(x, expected_kernel, padding=1))


def test_mapping_finds_native_clip_and_unet_layers_without_parent_imports():
    from sdfx.clora.patches import layer_mapping
    pipe = SimpleNamespace(unet=torch.nn.Sequential(torch.nn.Linear(3, 4)),
                           text_encoder=torch.nn.Sequential(torch.nn.Linear(3, 4)), text_encoder_2=None)
    mapping = layer_mapping(pipe)
    assert mapping['lora_unet_0'][2] is pipe.unet[0]
    assert mapping['lora_te_0'][2] is pipe.text_encoder[0]


def test_sparse_lora_residual_is_added_to_the_reconstructed_weight():
    from sdfx.clora.patches import Patch
    layer = torch.nn.Linear(2, 2, bias=False)
    patch = Patch({'lora_down.weight': torch.eye(2), 'lora_up.weight': torch.eye(2),
                   'bias_indices': torch.tensor([[0], [0]]), 'bias_values': torch.tensor([3.]),
                   'bias_size': torch.tensor([2, 2])}, layer)
    torch.testing.assert_close(patch.delta(torch.ones(1, 2)), torch.tensor([[4., 1.]]))


def test_loader_maps_legacy_sgm_input_convolution(tmp_path):
    from safetensors.torch import save_file
    from sdfx.clora.patches import load
    unet = torch.nn.Module()
    unet.conv_in = torch.nn.Conv2d(4, 8, 3, padding=1, bias=False)
    unet.config = SimpleNamespace(layers_per_block=2)
    pipe = SimpleNamespace(unet=unet, text_encoder=None, text_encoder_2=None)
    file = tmp_path / 'legacy.safetensors'
    save_file({'lora_unet_input_blocks_0_0.lora_down.weight': torch.ones(2, 4, 3, 3),
               'lora_unet_input_blocks_0_0.lora_up.weight': torch.ones(8, 2, 1, 1)}, str(file))
    result = load(str(file), pipe)
    assert len(result) == 1 and result[0][1] == 'conv_in'
    x = torch.ones(1, 4, 5, 5)
    expected = torch.nn.functional.conv2d(x, torch.full((8, 4, 3, 3), 2.), padding=1)
    torch.testing.assert_close(result[0][3].delta(x), expected)


def test_lokr_alpha_applies_only_when_a_decomposed_rank_exists():
    from sdfx.clora.patches import Patch
    layer = torch.nn.Linear(4, 4, bias=False)
    full = Patch(dict(lokr_w1=torch.eye(2), lokr_w2=torch.eye(2), alpha=torch.tensor(4.)), layer)
    torch.testing.assert_close(full.delta(torch.ones(1, 4)), torch.ones(1, 4))
    decomposed = Patch(dict(lokr_w1_a=torch.eye(2), lokr_w1_b=torch.eye(2),
                            lokr_w2=torch.eye(2), alpha=torch.tensor(4.)), layer)
    torch.testing.assert_close(decomposed.delta(torch.ones(1, 4)), torch.full((1, 4), 2.))
