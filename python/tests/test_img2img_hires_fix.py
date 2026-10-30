"""Pinned post-image Hires behavior; no model, torch or GPU required."""
import pytest
from sdfx.params import GenerationParams
from sdfx.protocol import EngineError


def test_source_defaults_and_effective_inheritance_are_job_local():
    from sdfx.img2img_hires_fix import normalize, effective
    config = normalize({})
    gp = GenerationParams(prompt='cat', negative_prompt='bad', steps=23, cfg_scale=7,
                          sampler_name='Euler', scheduler='Karras')
    actual = effective(config, gp)
    assert (actual.prompt, actual.negative_prompt, actual.steps, actual.cfg_scale) == ('cat', 'bad', 23, 7)
    assert (actual.sampler_name, actual.scheduler) == ('Euler', 'Karras')
    assert (config.ratio, config.denoise, config.upscaler) == (2., .33, 'R-ESRGAN 4x+')
    assert actual is not gp
    assert gp.steps == 23


def test_explicit_pair_overrides_ratio_but_single_dimension_does_not():
    from sdfx.img2img_hires_fix import normalize
    assert normalize({'ratio': 1.5}).target(512, 768) == (768, 1152)
    assert normalize({'ratio': 0, 'width': 1024, 'height': 800}).target(512, 768) == (1024, 800)
    assert normalize({'ratio': 1.5, 'width': 1024}).target(512, 768) == (768, 1152)
    assert normalize({'ratio': 1.01}).target(65, 97) == (64, 96)


def test_prompt_sampler_scheduler_cfg_and_steps_overrides():
    from sdfx.img2img_hires_fix import normalize, effective
    gp = GenerationParams(prompt='cat', negative_prompt='bad', steps=20, cfg_scale=7)
    actual = effective(normalize({'prompt': ' dog ', 'negative_prompt': ' blur ', 'steps': 12,
        'cfg': 3, 'sampler': 'Euler', 'scheduler': 'Karras'}), gp)
    assert (actual.prompt, actual.negative_prompt, actual.steps, actual.cfg_scale,
            actual.sampler_name, actual.scheduler) == ('dog', 'blur', 12, 3, 'Euler', 'Karras')
    assert effective(normalize({'prompt': '   '}), gp).prompt == 'cat'


@pytest.mark.parametrize('values', [
    {'ratio': 0}, {'ratio': float('nan')}, {'ratio': 5}, {'width': -8}, {'height': 2056},
    {'steps': -1}, {'steps': 26}, {'steps': True}, {'denoise': -1}, {'denoise': float('inf')},
    {'cfg': 31}, {'prompt': []}, {'sampler': 3}, {'upscaler': 'Latent'},
])
def test_invalid_inputs_fail_before_sampling(values):
    from sdfx.img2img_hires_fix import normalize
    with pytest.raises(EngineError):
        normalize(values)
