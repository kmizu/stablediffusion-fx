import pytest
from sdfx.protocol import EngineError


def test_exact_string_seed_transport_and_repeat_overflow():
    from sdfx.hires_tweaks import seed_plan
    large = 9007199254740993
    plan = seed_plan(dict(seed_enabled=True, seed=str(large), extras=True, subseed=str(large+1)), [7], [8], 0)
    assert plan.batch(0, 1, 0) == ((large,), (large+1,))
    assert seed_plan(dict(seed_enabled=True, seed=str(2**63-1)), [7], [8], 0).seeds == (2**63-1,)
    with pytest.raises(EngineError):
        seed_plan(dict(seed_enabled=True, seed=str(2**63-1), batch_count=2), [7], [8], 0)


def test_prepare_noise_noop_and_unsupported_pipeline_do_not_mutate():
    from types import SimpleNamespace
    from sdfx.hires_tweaks import prepare_noise
    pipe = SimpleNamespace()
    with prepare_noise(pipe, None):
        assert vars(pipe) == {}
    with pytest.raises(EngineError):
        with prepare_noise(pipe, lambda *a: None):
            pytest.fail('unsupported pipeline was accepted')
    assert vars(pipe) == {}


def test_inherited_prepare_method_restored_without_instance_shadow():
    from sdfx.hires_tweaks import prepare_noise
    # Give the fixture its own globals as a diffusers module would have.
    namespace = dict(randn_tensor=lambda: 'ordinary')
    exec('def prepare_latents(self): return randn_tensor()', namespace)
    cls = type('Pipeline', (), dict(prepare_latents=namespace['prepare_latents']))
    a, b = cls(), cls()
    with prepare_noise(a, lambda: 'outer'):
        assert a.prepare_latents() == 'outer'
        assert b.prepare_latents() == 'ordinary'
        with prepare_noise(a, lambda: 'inner'):
            assert a.prepare_latents() == 'inner'
        assert a.prepare_latents() == 'outer'
    assert vars(a) == {}
    assert a.prepare_latents() == 'ordinary'


@pytest.mark.parametrize('family', ['sd', 'xl', 'inpaint', 'xl_inpaint'])
def test_owned_noise_reaches_real_diffusers_prepare_latents_and_restores(family):
    torch = pytest.importorskip('torch')
    pytest.importorskip('diffusers')
    from types import SimpleNamespace, MethodType
    from diffusers import (StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline,
                           StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline)
    from sdfx.hires_tweaks import prepare_noise
    cls = dict(sd=StableDiffusionImg2ImgPipeline, xl=StableDiffusionXLImg2ImgPipeline,
               inpaint=StableDiffusionInpaintPipeline, xl_inpaint=StableDiffusionXLInpaintPipeline)[family]
    pipe = SimpleNamespace(vae=SimpleNamespace(config=SimpleNamespace()), vae_scale_factor=8,
        scheduler=SimpleNamespace(init_noise_sigma=2, add_noise=lambda image, noise, time: image + noise))
    original = MethodType(cls.prepare_latents, pipe)
    pipe.prepare_latents = original
    globals_noise = original.__func__.__globals__['randn_tensor']
    image = torch.zeros(2, 4, 8, 8)
    calls = []
    def noise(shape, **kwargs):
        calls.append(shape)
        return torch.full(shape, 3., device=kwargs['device'], dtype=kwargs['dtype'])
    with pytest.raises(RuntimeError, match='interrupt'):
        with prepare_noise(pipe, noise):
            if 'inpaint' in family:
                result, returned_noise = pipe.prepare_latents(2, 4, 64, 64, torch.float32,
                    torch.device('cpu'), None, image=image, return_noise=True)
                assert torch.all(result == 6)
                assert torch.all(returned_noise == 3)
            else:
                result = pipe.prepare_latents(image, torch.tensor([1]), 2, 1,
                    torch.float32, torch.device('cpu'))
                assert torch.all(result == 3)
            assert original.__func__.__globals__['randn_tensor'] is globals_noise
            raise RuntimeError('interrupt')
    assert pipe.prepare_latents is original
    assert len(calls) == 1


def test_centered_resize_noise_uses_seeded_background_and_variation_patch():
    torch = pytest.importorskip('torch')
    from sdfx.hires_tweaks import resized_noise
    def draw(seed, shape):
        return torch.arange(math.prod(shape)).reshape(shape).float() + seed * 1000
    import math
    # Strength=1 replaces the resized patch with the subseed; outside stays primary.
    out = resized_noise((1, 4, 4), 1, 2, 1., 16, 16, draw, lambda t, a, b: b)
    expected = draw(1, (1, 4, 4))
    expected[:, 1:3, 1:3] = draw(2, (1, 2, 2))
    assert torch.equal(out, expected)
    large = resized_noise((1, 2, 2), 1, 2, 0, 32, 32, draw, None)
    assert torch.equal(large, draw(1, (1, 4, 4))[:, 1:3, 1:3])


def test_inherited_seed_batches_preserve_base_order():
    from sdfx.hires_tweaks import seed_plan
    plan = seed_plan({'batch_count': 3}, [10, 11, 12, 13], [20, 21, 22, 23], 0)
    assert plan.batch(2, 2, 1) == ((13, 14), (23, 24))
    assert plan.batch(0, 2, 0) == ((10, 11), (20, 21))
    assert plan.strength == 0


def test_variation_holds_primary_seed_across_images_and_repeats():
    from sdfx.hires_tweaks import seed_plan
    plan = seed_plan(dict(batch_count=3, seed_enabled=True, seed=50,
                         extras=True, subseed=60, strength=.4), [10, 11], [20, 21], .7)
    assert plan.batch(0, 2, 2) == ((50, 50), (62, 63))
    assert plan.strength == .4


def test_random_subseed_is_independent_and_resolved_once():
    from sdfx.hires_tweaks import seed_plan
    calls = []
    def random_seed():
        calls.append(1)
        return 77
    plan = seed_plan(dict(batch_count=2, seed_enabled=True, seed=50,
                         extras=True, subseed=-1, strength=.5), [10], [20], 0, random_seed)
    assert plan.batch(0, 1, 0) == ((50,), (77,))
    assert plan.batch(0, 1, 1) == ((50,), (78,))
    assert len(calls) == 1


def test_enabling_seed_without_extras_disables_inherited_variation():
    from sdfx.hires_tweaks import seed_plan
    plan = seed_plan(dict(seed_enabled=True), [10], [20], .6)
    assert plan.strength == 0
    assert plan.batch(0, 1, 0) == ((10,), (20,))
    inherited = seed_plan({}, [10], [20], .6)
    assert inherited.strength == .6


def test_inherited_variation_preserves_distinct_base_seeds():
    from sdfx.hires_tweaks import seed_plan
    plan = seed_plan(dict(batch_count=2), [10, 11], [20, 21], .6)
    assert plan.batch(0, 2, 1) == ((10, 11), (21, 22))


def test_two_random_seeds_and_resize_pair():
    from sdfx.hires_tweaks import seed_plan
    values = iter([100, 200])
    config = dict(seed_enabled=True, seed=-1, extras=True, subseed=-1,
                  resize_width=512, resize_height=768)
    plan = seed_plan(config, [10, 11], [20, 21], 0, lambda: next(values))
    assert plan.batch(0, 2, 0) == ((100, 101), (200, 201))
    assert (plan.resize_width, plan.resize_height) == (512, 768)
    config.update(seed=0, subseed=0, resize_height=0)
    plan = seed_plan(config, [10], [20], 0)
    assert (plan.resize_width, plan.resize_height) == (0, 0)


@pytest.mark.parametrize('config', [dict(batch_count=0), dict(batch_count=17),
    dict(batch_count=True), dict(seed_enabled=1), dict(seed=-2), dict(seed=1.5),
    dict(strength=float('nan')), dict(strength=1.1), dict(resize_width=7)])
def test_invalid_seed_configuration(config):
    from sdfx.hires_tweaks import seed_plan
    with pytest.raises(EngineError):
        seed_plan(config, [10], [20], 0)


def test_batch_bounds_and_input_lengths_are_checked():
    from sdfx.hires_tweaks import seed_plan
    with pytest.raises(EngineError):
        seed_plan({}, [10, 11], [20], 0)
    plan = seed_plan({}, [10], [20], 0)
    for args in [(0, 1, 1), (-1, 1, 0), (0, 2, 0), (0, 0, 0)]:
        with pytest.raises(EngineError):
            plan.batch(*args)
