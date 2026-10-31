import types
import pytest
torch = pytest.importorskip("torch")

from sdfx.neutral_ip import prepare, align
from sdfx.adapters import sync_ip_components


@pytest.mark.parametrize("guided", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
def test_prepare_uses_native_negative_features_once_and_preserves_adapter_shapes(guided, batch):
    calls = []
    class Pipe:
        _execution_device = "cpu"
        def prepare_ip_adapter_image_embeds(self, image, embeds, device, num_images_per_prompt, do_classifier_free_guidance):
            calls.append((image, embeds, device, num_images_per_prompt, do_classifier_free_guidance, torch.is_grad_enabled()))
            return [torch.cat([torch.full((1, 1, 4), 3.), torch.full((1, 1, 4), 7.)]),
                    torch.cat([torch.full((1, 1, 9, 4), 5.), torch.full((1, 1, 9, 4), 11.)])]
    canonical, native = prepare(Pipe(), "image", batch, guided)
    assert calls == [("image", None, "cpu", 1, True, False)]
    assert canonical[0].shape == (batch*2, 1, 4)
    assert canonical[1].shape == (batch*2, 1, 9, 4)
    assert native[0].shape[0] == (2 if guided else 1)
    assert native[0][-1, 0, 0] == 7
    got = align(canonical, batch, [False, True, False, True], "cpu", torch.float32)
    assert got[0][:, 0, 0].tolist() == [3.] * batch + [7.] * batch + [3.] * batch + [7.] * batch
    assert got[1][:batch, 0, 0, 0].tolist() == [5.] * batch


def test_derived_image_components_follow_load_and_unload_without_replacing_unet():
    class Pipe:
        def __init__(self, encoder=None, extractor=None):
            self.image_encoder, self.feature_extractor = encoder, extractor
            self.unet = object()
        def register_modules(self, **modules):
            self.__dict__.update(modules)
    base, derived = Pipe(object(), object()), Pipe()
    unet = derived.unet
    assert sync_ip_components(base, [derived])
    assert derived.image_encoder is base.image_encoder
    assert derived.feature_extractor is base.feature_extractor
    assert derived.unet is unet
    assert not sync_ip_components(base, [derived])
    base.image_encoder = None
    assert sync_ip_components(base, [derived])
    assert derived.image_encoder is None


@pytest.mark.parametrize("guided", [False, True])
@pytest.mark.parametrize("pag", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
def test_runtime_uses_canonical_image_features_for_every_text_leaf(guided, pag, batch):
    from sdfx.neutral_runtime import RuntimeHook, Plan
    from sdfx.neutral_prompt import parse
    from diffusers import DDIMScheduler
    layout = [False, True, True] if guided and pag else [False, True, False, True] if pag else [False, True]
    class UNet(torch.nn.Module):
        def forward(self, sample, timestep, encoder_hidden_states=None, added_cond_kwargs=None, **kw):
            image = added_cond_kwargs["image_embeds"][0]
            assert image[:, 0, 0].tolist() == [v for p in layout for v in [7. if p else 3.] * batch]
            value = image[:, 0, 0]
            return (torch.ones_like(sample) * value[:, None, None, None],)
    u = UNet()
    branch = dict(prompt_embeds=torch.ones(batch, 77, 1), negative_prompt_embeds=torch.zeros(batch, 77, 1))
    canonical = [torch.cat([torch.full((batch, 1, 4), 3.), torch.full((batch, 1, 4), 7.)])]
    scheduler = DDIMScheduler(clip_sample=False)
    scheduler.set_timesteps(4)
    hook = RuntimeHook(u, scheduler, Plan(parse("a :0.5 AND b :0.5"), [branch, branch]), 6 if guided else 1, 0,
                       pag=pag, image_embeds=canonical).install()
    groups = 3 if guided and pag else 2 if guided or pag else 1
    try:
        out = u(torch.zeros(batch*groups, 4, 2, 2), scheduler.timesteps[0], encoder_hidden_states=torch.empty(0),
                added_cond_kwargs={"image_embeds": [torch.full((batch*groups, 1, 4), 99.)]})[0]
        assert out.shape[0] == batch*groups
    finally:
        hook.remove()


def test_pag_failure_cleanup_keeps_ip_processors_loaded_after_extension_setup(monkeypatch):
    from sdfx.engine import Engine
    from sdfx.params import GenerationParams
    from sdfx import extensions
    class UNet:
        def __init__(self):
            self.attn_processors = {"self": "ordinary", "cross": "ordinary"}
        def set_attn_processor(self, processors):
            self.attn_processors = dict(processors)
    class Pipe:
        def __init__(self):
            self.unet = UNet()
        def __call__(self, **kwargs):
            self.unet.attn_processors["self"] = "pag"
            raise RuntimeError("interrupted pipeline")
    pipe, engine = Pipe(), Engine.__new__(Engine)
    engine.sdxl = False
    engine._get_pipe = lambda *a: pipe
    engine._scheduler = lambda *a: (types.SimpleNamespace(config={}), None)
    engine._generators = lambda *a: None
    engine._callback = lambda *a: lambda *a: None
    engine._activate = lambda *a: None
    monkeypatch.setattr(extensions, "pre_run", lambda *a, **kw: extensions.RunPlan("txt2img"))
    monkeypatch.setattr(extensions, "special_pipe", lambda *a: None)
    def install(*a, **kw):
        pipe.unet.attn_processors["cross"] = "ip-loaded"
    monkeypatch.setattr(extensions, "install", install)
    with pytest.raises(RuntimeError, match="interrupted pipeline"):
        engine._run("txt2img", GenerationParams(), None, types.SimpleNamespace(check_interrupt=lambda: None),
                    sampler="DDIM", schedule=None, cfg_scale=6, embeds={}, steps=4, w=64, h=64, seeds=[1],
                    pass_="base", image_idx=0, images_total=1, pag={"scale": 2, "adaptive_scale": 0})
    assert pipe.unet.attn_processors == {"self": "ordinary", "cross": "ip-loaded"}


def test_invalid_ip_path_does_not_unload_existing_adapter(tmp_path):
    from sdfx.adapters import IPAdapterState
    state = IPAdapterState()
    state.loaded = "existing"
    unloaded = []
    pipe = types.SimpleNamespace(unload_ip_adapter=lambda: unloaded.append(True))
    from sdfx.protocol import EngineError
    with pytest.raises(EngineError, match="not found"):
        state.ensure(pipe, str(tmp_path / "missing.safetensors"), False)
    assert not unloaded and state.loaded == "existing"


def test_failed_ip_load_removes_partial_modules(tmp_path):
    from sdfx.adapters import IPAdapterState
    path = tmp_path / "invalid.safetensors"
    path.write_bytes(b"invalid")
    state = IPAdapterState()
    class Pipe:
        image_encoder = object()
        def load_ip_adapter(self, *args, **kwargs):
            raise ValueError("bad weights")
        def unload_ip_adapter(self):
            self.image_encoder = None
    pipe = Pipe()
    with pytest.raises(ValueError, match="bad weights"):
        state.ensure(pipe, str(path), False)
    assert state.loaded is None and pipe.image_encoder is None


def test_encoder_override_change_invalidates_same_adapter_cache(tmp_path, monkeypatch):
    from sdfx.adapters import IPAdapterState
    from transformers import CLIPVisionModelWithProjection
    class Encoder:
        def __init__(self, name): self.name = name
        def to(self, *a): return self
    monkeypatch.setattr(CLIPVisionModelWithProjection, "from_pretrained", lambda name, **kw: Encoder(name))
    loaded = []
    class Pipe:
        image_encoder = None
        unet = types.SimpleNamespace(dtype=torch.float32, device="cpu")
        def register_modules(self, **kw): self.__dict__.update(kw)
        def load_ip_adapter(self, *a, **kw): loaded.append(self.image_encoder.name)
        def unload_ip_adapter(self): self.image_encoder = None
    path = tmp_path / "adapter.safetensors"; path.write_bytes(b"fake")
    state, pipe = IPAdapterState(), Pipe()
    for encoder in ("encoderA", "encoderB", "encoderB"):
        state.ensure(pipe, str(path), False, encoder)
    assert loaded == ["encoderA", "encoderB"]


def test_free_model_clears_loaded_ip_state(monkeypatch):
    from sdfx.engine import Engine
    from sdfx.adapters import IPAdapterState
    from sdfx import upscalers
    engine = Engine.__new__(Engine)
    old = engine.ip_state = IPAdapterState(); old.loaded = "adapter"
    engine._drop_hooks = lambda: None
    engine.derived, engine.cn_cache, engine.loaded_loras, engine.loaded_embeddings = {}, {}, {}, set()
    engine._dora_weight_handles, engine._dora_step_weights, engine.dynamic_dora_adapters = [], {}, set()
    engine.loaded_embedding_files = {"old": "old.safetensors"}
    engine.cuda = False
    monkeypatch.setattr(upscalers, "unload_models", lambda: None)
    engine._free()
    assert engine.ip_state is not old and engine.ip_state.loaded is None
    assert engine.loaded_embedding_files == {}


def test_encoder_switch_updates_retained_feature_extractor_size(tmp_path, monkeypatch):
    from PIL import Image
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection, CLIPImageProcessor
    from sdfx.adapters import IPAdapterState
    def encoder(name, **kw):
        return CLIPVisionModelWithProjection(CLIPVisionConfig(image_size=int(name), patch_size=4, hidden_size=16,
            intermediate_size=32, projection_dim=8, num_hidden_layers=1, num_attention_heads=2)).eval()
    monkeypatch.setattr(CLIPVisionModelWithProjection, "from_pretrained", encoder)
    class Pipe:
        image_encoder = None
        safety_checker = None
        feature_extractor = CLIPImageProcessor(size={"shortest_edge":16}, crop_size={"height":16,"width":16})
        unet = types.SimpleNamespace(dtype=torch.float32, device="cpu")
        def register_modules(self, **kw): self.__dict__.update(kw)
        def load_ip_adapter(self, *a, **kw): pass
        def unload_ip_adapter(self): self.image_encoder = None  # native SD1.5 retains feature_extractor
    path=tmp_path/"adapter.safetensors"; path.write_bytes(b"fake")
    state, pipe = IPAdapterState(), Pipe()
    state.ensure(pipe, str(path), False, "16")
    state.ensure(pipe, str(path), False, "32")
    pixels = pipe.feature_extractor(images=Image.new("RGB",(32,32)), return_tensors="pt").pixel_values
    assert pixels.shape[-2:] == (32,32)
    assert pipe.image_encoder(pixels).image_embeds.shape == (1,8)


@pytest.mark.parametrize("cfg", [0, 0.5, 1, 2])
@pytest.mark.parametrize("composed", [False, True])
def test_inverse_loop_supplies_native_image_conditions(cfg, composed):
    from diffusers import DDIMScheduler
    from sdfx.extras import invert_latents
    from sdfx.neutral_runtime import Plan
    from sdfx.neutral_prompt import parse
    class UNet(torch.nn.Module):
        def forward(self, sample, timestep, encoder_hidden_states=None, added_cond_kwargs=None, **kw):
            assert added_cond_kwargs is not None and "image_embeds" in added_cond_kwargs
            image = added_cond_kwargs["image_embeds"][0][:, 0, 0]
            text = encoder_hidden_states[:, 0, 0]
            return types.SimpleNamespace(sample=torch.ones_like(sample) * (image+text)[:, None, None, None])
    def branch(value):
        return dict(prompt_embeds=torch.full((1, 77, 1), value), negative_prompt_embeds=torch.ones(1, 77, 1))
    embeds = branch(4.)
    if composed:
        branches = [branch(3.), branch(5.)]
        embeds = dict(branches[0], _neutral=Plan(parse("a :0.5 AND b :0.5"), branches))
    embeds["_ip_image_embeds"] = [torch.cat([torch.full((1, 1, 4), 3.), torch.full((1, 1, 4), 7.)])]
    pipe = types.SimpleNamespace(unet=UNet(), scheduler=DDIMScheduler(clip_sample=False))
    z = torch.zeros(1, 4, 2, 2)
    got = invert_latents(types.SimpleNamespace(sdxl=False), pipe, z, embeds, 3, cfg, 16, 16, lambda: None)
    # 4 = negative image 3 + negative text 1; 11 = positive image 7 + mean positive text 4.
    class Constant(torch.nn.Module):
        def forward(self, sample, timestep, **kw):
            return types.SimpleNamespace(sample=torch.ones_like(sample) * (4 + cfg*7))
    reference_pipe = types.SimpleNamespace(unet=Constant(), scheduler=pipe.scheduler)
    expected = invert_latents(types.SimpleNamespace(sdxl=False), reference_pipe, z, branch(4.), 3, 1, 16, 16, lambda: None)
    torch.testing.assert_close(got, expected)
