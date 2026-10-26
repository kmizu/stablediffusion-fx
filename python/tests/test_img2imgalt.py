"""Analytic inversion tests without model weights; real model probes live in scripts/dev/gpu_check.py."""
import types
import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from sdfx.extras import invert_latents  # noqa: E402
from sdfx.engine import Engine  # noqa: E402
from sdfx.params import GenerationParams  # noqa: E402


@pytest.mark.parametrize("configured_clip", [None, True, False])
def test_real_inverse_scheduler_does_not_clip_latent_space_to_pixel_range(configured_clip):
    config = {"num_train_timesteps": 10, "beta_start": 0.01, "beta_end": 0.02, "beta_schedule": "linear",
              "prediction_type": "epsilon"}
    if configured_clip is not None:
        config["clip_sample"] = configured_clip
    pipe = types.SimpleNamespace(scheduler=types.SimpleNamespace(config=config),
        unet=lambda x, t, **kw: types.SimpleNamespace(sample=torch.zeros_like(x)))
    z = torch.full((1, 4, 2, 2), 2.5)  # valid latent values can exceed the pixel-space [-1, 1] range
    embeds = {"prompt_embeds": torch.ones(1, 77, 1), "negative_prompt_embeds": torch.zeros(1, 77, 1)}
    out = invert_latents(types.SimpleNamespace(sdxl=False), pipe, z, embeds, 2, 1, 16, 16, lambda: None)
    reference = diffusers.DDIMInverseScheduler.from_config(config, clip_sample=False)
    reference.set_timesteps(2)
    expected = z * reference.alphas_cumprod[reference.timesteps[-1]].sqrt()
    assert torch.allclose(out, expected, atol=1e-6)
    assert out.min() > 2.0


@pytest.mark.parametrize("cfg,expected", [(0.0, 1.0), (0.5, 2.0), (1.0, 3.0), (2.0, 5.0)])
@pytest.mark.parametrize("sdxl", [False, True])
def test_inversion_cfg_supports_entire_slider_range(monkeypatch, cfg, expected, sdxl):
    recorded = []

    class Scheduler:
        @classmethod
        def from_config(cls, *args, **kwargs):
            return cls()

        def set_timesteps(self, steps, device):
            self.timesteps = torch.tensor([0], device=device)

        def step(self, eps, t, x):
            return types.SimpleNamespace(prev_sample=x + eps)

    monkeypatch.setattr(diffusers, "DDIMInverseScheduler", Scheduler)

    def unet(x, t, **kwargs):
        recorded.append(kwargs)
        values = kwargs["encoder_hidden_states"][:, 0, 0]
        return types.SimpleNamespace(sample=torch.ones_like(x) * values[:, None, None, None])

    engine = types.SimpleNamespace(sdxl=sdxl)
    pipe = types.SimpleNamespace(scheduler=types.SimpleNamespace(config={}), unet=unet)
    embeds = {"prompt_embeds": torch.full((1, 77, 1), 3.0), "negative_prompt_embeds": torch.full((1, 77, 1), 1.0),
              "pooled_prompt_embeds": torch.full((1, 4), 30.0), "negative_pooled_prompt_embeds": torch.full((1, 4), 10.0)}
    out = invert_latents(engine, pipe, torch.zeros(1, 4, 2, 2), embeds, 1, cfg, 16, 16, lambda: None)
    assert torch.equal(out, torch.full_like(out, expected))
    if sdxl:
        added = recorded[0]["added_cond_kwargs"]
        assert added["text_embeds"][:, 0].tolist() == ([30.0] if cfg == 1 else [10.0, 30.0])
        assert added["time_ids"].shape[0] == (1 if cfg == 1 else 2)


@pytest.mark.parametrize("randomness", [0.0, 0.5, 1.0])
def test_alternative_preserves_inverse_magnitude_and_exact_random_endpoint(monkeypatch, randomness):
    from sdfx import extras, extensions
    inv = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2) / 10 - 0.75
    rnd = torch.arange(32, dtype=torch.float32).reshape(2, 4, 2, 2) / 20 - 0.9
    engine = Engine.__new__(Engine)
    engine.pipe = types.SimpleNamespace(scheduler=None)
    engine._activate = lambda pipe: None
    engine._scheduler = lambda *args: (types.SimpleNamespace(config={}), None)
    engine._noise = lambda *args: rnd
    captured = {}

    def run(*args, **kwargs):
        captured.update(kwargs)
        return [], None

    engine._run = run
    monkeypatch.setattr(extensions, "encode_image", lambda *args: inv)
    monkeypatch.setattr(extras, "invert_latents", lambda *args: inv)
    gp = GenerationParams()
    engine._img2imgalt(gp, None, types.SimpleNamespace(check_interrupt=lambda: None), {"randomness": randomness},
                      None, {}, [11, 12], 0, 2, 16, 16, None, None)
    expected = ((1 - randomness) * inv.repeat(2, 1, 1, 1) + randomness * rnd) / (
        (1 - randomness)**2 + randomness**2)**0.5
    assert torch.allclose(captured["latents"], expected)
    if randomness == 0:
        assert torch.equal(captured["latents"], inv.repeat(2, 1, 1, 1))
    elif randomness == 1:
        assert torch.equal(captured["latents"], rnd)
