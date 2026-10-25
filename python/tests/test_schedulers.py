import pytest

from sdfx import schedulers as S

EXPECTED_AUTO = {
    "DPM++ 2M": "Karras", "DPM++ 2M SDE": "Exponential", "DPM++ 3M SDE": "Exponential",
    "DPM++ 2M SDE Heun": "Exponential", "DPM2": "Karras", "DPM2 a": "Karras", "DPM++ SDE": "Karras",
    "DPM++ 2S a": "Karras", "Euler": "Uniform", "Euler a": "Uniform", "Heun": "Uniform", "LMS": "Uniform",
    "DDIM": "Uniform", "UniPC": "Uniform", "LCM": "Uniform",
}


def test_table_complete():
    assert {s.name for s in S.SAMPLERS} == set(EXPECTED_AUTO)
    assert S.SCHEDULES == ["Automatic", "Uniform", "Karras", "Exponential", "Polyexponential", "SGM Uniform",
                           "KL Optimal", "Align Your Steps", "Simple", "Normal", "DDIM", "Beta"]
    for s in S.SAMPLERS:
        assert s.cls.endswith("Scheduler")
        assert s.auto_schedule in S.SCHEDULES


@pytest.mark.parametrize("sampler,expected", sorted(EXPECTED_AUTO.items()))
def test_automatic(sampler, expected):
    assert S.resolve_schedule(sampler, "Automatic") == expected
    assert S.resolve_schedule(sampler, None) == expected


def test_explicit_and_aliases():
    assert S.resolve_schedule("Euler", "Karras") == "Karras"
    assert S.resolve_schedule("Euler", "sgm_uniform") == "SGM Uniform"
    assert S.resolve_schedule("DPM++ 2M Karras", "Automatic") == "Karras"
    assert S.sampler_spec("k_euler_a").name == "Euler a"
    with pytest.raises(ValueError):
        S.sampler_spec("Nope")
    with pytest.raises(ValueError):
        S.resolve_schedule("Euler", "Nope")


def test_plan_kwargs():
    p = S.plan("DPM++ 2M SDE Heun", "Automatic")
    assert p.cls == "DPMSolverMultistepScheduler" and p.kwargs["solver_type"] == "heun" and p.schedule == "Exponential"
    p = S.plan("DPM++ 3M SDE", "Karras")
    assert p.kwargs["solver_order"] == 3 and p.schedule == "Karras"


def test_sampler_list_shape():
    rows = S.sampler_list()
    assert all(set(r) == {"name", "supported", "default_scheduler"} for r in rows)
    assert rows[0]["name"] == "DPM++ 2M"


def _sd_base_config():
    diffusers = pytest.importorskip("diffusers")
    return diffusers.EulerDiscreteScheduler(beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012,
                                            timestep_spacing="leading", steps_offset=1).config


def test_build_every_sampler_and_schedule():
    pytest.importorskip("torch")
    base = _sd_base_config()
    import importlib.util
    for s in S.SAMPLERS:
        if s.name == "DPM++ SDE" and importlib.util.find_spec("torchsde") is None:
            continue
        for sch in S.SCHEDULES:
            sched, warns = S.build_scheduler(S.plan(s.name, sch), base)
            assert type(sched).__name__ == s.cls
            sched.set_timesteps(12)
            assert len(sched.timesteps) >= 12, (s.name, sch)


def test_custom_sigmas_handoff_exact():
    """Euler accepts custom sigmas: the scheduler must use exactly the A1111 schedule incl. the trailing zero."""
    pytest.importorskip("torch")
    import numpy as np
    from sdfx import sigmas as SG
    base = _sd_base_config()
    for sch in ("Karras", "Exponential", "Align Your Steps", "Uniform", "SGM Uniform", "Beta", "KL Optimal"):
        sched, warns = S.build_scheduler(S.plan("Euler", sch), base)
        assert warns == []
        sched.set_timesteps(15)
        model = SG.DiscreteModel(SG.model_sigmas_from_alphas(sched.alphas_cumprod.numpy()))
        ref = SG.compute(sch, 15, model)
        got = sched.sigmas.numpy()
        assert len(got) == len(ref), sch
        np.testing.assert_allclose(got, ref, rtol=2e-4, atol=1e-6, err_msg=sch)


def test_config_flag_fallbacks():
    pytest.importorskip("torch")
    base = _sd_base_config()
    sched, w = S.build_scheduler(S.plan("DPM++ 2M", "Automatic"), base)
    assert sched.config.use_karras_sigmas is True and sched.config.algorithm_type == "dpmsolver++" and w == []
    sched, w = S.build_scheduler(S.plan("DPM++ 2M SDE", "Automatic"), base)
    assert sched.config.use_exponential_sigmas is True and sched.config.algorithm_type == "sde-dpmsolver++"
    sched, w = S.build_scheduler(S.plan("DPM++ 2M", "SGM Uniform"), base)
    assert sched.config.timestep_spacing == "trailing" and sched.config.use_karras_sigmas is False
    sched, w = S.build_scheduler(S.plan("DPM++ 2M", "Align Your Steps"), base)
    assert w and "Karras" in w[0]
    sched, w = S.build_scheduler(S.plan("Euler", "Karras"), base, extra_config={"prediction_type": "v_prediction"})
    assert sched.config.prediction_type == "v_prediction"
