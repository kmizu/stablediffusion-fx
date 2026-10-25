"""A1111 sampler / schedule names -> diffusers scheduler classes and configs.

The table and the resolution logic are pure python; `build_scheduler` imports diffusers lazily.
Strategy: when the diffusers scheduler accepts custom `sigmas` in `set_timesteps` (Euler family), the exact
A1111 sigma schedule from `sigmas.py` is used; otherwise the closest diffusers config flag is applied
(use_karras_sigmas / use_exponential_sigmas / use_beta_sigmas / timestep_spacing) and a warning is reported
when no equivalent exists.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Optional



@dataclass(frozen=True)
class SamplerSpec:
    name: str
    cls: str
    kwargs: dict = field(default_factory=dict)
    auto_schedule: str = "Uniform"
    supported: bool = True
    note: str = ""


SAMPLERS: list[SamplerSpec] = [
    SamplerSpec("DPM++ 2M", "DPMSolverMultistepScheduler", {"algorithm_type": "dpmsolver++", "solver_order": 2}, "Karras"),
    SamplerSpec("DPM++ SDE", "DPMSolverSDEScheduler", {}, "Karras", note="requires torchsde"),
    SamplerSpec("DPM++ 2M SDE", "DPMSolverMultistepScheduler", {"algorithm_type": "sde-dpmsolver++", "solver_order": 2}, "Exponential"),
    SamplerSpec("DPM++ 2M SDE Heun", "DPMSolverMultistepScheduler",
                {"algorithm_type": "sde-dpmsolver++", "solver_order": 2, "solver_type": "heun"}, "Exponential"),
    SamplerSpec("DPM++ 2S a", "DPMSolverSinglestepScheduler", {"algorithm_type": "dpmsolver++", "solver_order": 2}, "Karras",
                note="approximation (diffusers has no ancestral single-step DPM++)"),
    SamplerSpec("DPM++ 3M SDE", "DPMSolverMultistepScheduler", {"algorithm_type": "sde-dpmsolver++", "solver_order": 3}, "Exponential"),
    SamplerSpec("Euler a", "EulerAncestralDiscreteScheduler", {}, "Uniform"),
    SamplerSpec("Euler", "EulerDiscreteScheduler", {}, "Uniform"),
    SamplerSpec("LMS", "LMSDiscreteScheduler", {}, "Uniform"),
    SamplerSpec("Heun", "HeunDiscreteScheduler", {}, "Uniform"),
    SamplerSpec("DPM2", "KDPM2DiscreteScheduler", {}, "Karras"),
    SamplerSpec("DPM2 a", "KDPM2AncestralDiscreteScheduler", {}, "Karras"),
    SamplerSpec("DDIM", "DDIMScheduler", {"set_alpha_to_one": False}, "Uniform"),
    SamplerSpec("UniPC", "UniPCMultistepScheduler", {}, "Uniform"),
    SamplerSpec("LCM", "LCMScheduler", {}, "Uniform"),
]

SAMPLER_MAP = {s.name: s for s in SAMPLERS}
# A1111 aliases
ALIASES = {
    "k_dpmpp_2m": "DPM++ 2M", "k_dpmpp_sde": "DPM++ SDE", "k_dpmpp_2m_sde": "DPM++ 2M SDE",
    "k_dpmpp_2m_sde_heun": "DPM++ 2M SDE Heun", "k_dpmpp_2s_a": "DPM++ 2S a", "k_dpmpp_3m_sde": "DPM++ 3M SDE",
    "k_euler_a": "Euler a", "k_euler": "Euler", "k_lms": "LMS", "k_heun": "Heun", "k_dpm_2": "DPM2",
    "k_dpm_2_a": "DPM2 a", "ddim": "DDIM", "unipc": "UniPC", "lcm": "LCM",
    # legacy A1111 combined names
    "DPM++ 2M Karras": "DPM++ 2M", "DPM++ SDE Karras": "DPM++ SDE", "DPM++ 2M SDE Karras": "DPM++ 2M SDE",
    "DPM++ 2M SDE Exponential": "DPM++ 2M SDE", "DPM2 Karras": "DPM2", "DPM2 a Karras": "DPM2 a",
    "DPM++ 2S a Karras": "DPM++ 2S a", "DPM++ 3M SDE Karras": "DPM++ 3M SDE", "DPM++ 3M SDE Exponential": "DPM++ 3M SDE",
}

SCHEDULES = ["Automatic", "Uniform", "Karras", "Exponential", "Polyexponential", "SGM Uniform", "KL Optimal",
             "Align Your Steps", "Simple", "Normal", "DDIM", "Beta"]
SCHEDULE_ALIASES = {s.lower(): s for s in SCHEDULES}
SCHEDULE_ALIASES.update({"sgm_uniform": "SGM Uniform", "kl_optimal": "KL Optimal", "align_your_steps": "Align Your Steps",
                         "ays": "Align Your Steps", "polyexponential": "Polyexponential"})

# diffusers config equivalents used when custom sigmas are not accepted by the scheduler class
CONFIG_EQUIVALENT: dict[str, Optional[dict]] = {
    "Uniform": {"timestep_spacing": "linspace"},
    "Karras": {"use_karras_sigmas": True},
    "Exponential": {"use_exponential_sigmas": True},
    "Beta": {"use_beta_sigmas": True},
    "SGM Uniform": {"timestep_spacing": "trailing"},
    "Simple": {"timestep_spacing": "trailing"},
    "Normal": {"timestep_spacing": "linspace"},
    "DDIM": {"timestep_spacing": "leading"},
    "Polyexponential": None,
    "KL Optimal": None,
    "Align Your Steps": None,
}
SIGMA_FLAGS = ("use_karras_sigmas", "use_exponential_sigmas", "use_beta_sigmas")


def sampler_spec(name: str) -> SamplerSpec:
    n = ALIASES.get(name, name)
    if n not in SAMPLER_MAP:
        low = {k.lower(): k for k in SAMPLER_MAP}
        n = low.get(str(name).lower(), n)
    if n not in SAMPLER_MAP:
        raise ValueError(f"unknown sampler '{name}'")
    return SAMPLER_MAP[n]


def legacy_schedule(name: str) -> Optional[str]:
    """Schedule implied by a legacy combined name like 'DPM++ 2M Karras'."""
    if name.endswith(" Karras"):
        return "Karras"
    if name.endswith(" Exponential"):
        return "Exponential"
    return None


def resolve_schedule(sampler: str, schedule: Optional[str]) -> str:
    """'Automatic' -> the A1111 default schedule for the sampler."""
    spec = sampler_spec(sampler)
    s = (schedule or "Automatic").strip()
    s = SCHEDULE_ALIASES.get(s.lower(), s)
    if s not in SCHEDULES:
        raise ValueError(f"unknown schedule type '{schedule}'")
    if s == "Automatic":
        return legacy_schedule(sampler) or spec.auto_schedule
    return s


@dataclass
class Plan:
    sampler: str
    cls: str
    kwargs: dict
    schedule: str


def plan(sampler: str, schedule: Optional[str]) -> Plan:
    spec = sampler_spec(sampler)
    return Plan(spec.name, spec.cls, dict(spec.kwargs), resolve_schedule(sampler, schedule))


def sampler_list(check_backend: bool = True) -> list[dict]:
    """list_samplers rows; `supported` reflects class availability when diffusers is importable.

    check_backend=False never imports diffusers (dry run)."""
    diffusers = None
    if check_backend:
        try:
            import diffusers  # type: ignore
        except Exception:
            diffusers = None
    rows = []
    for s in SAMPLERS:
        ok = s.supported
        if diffusers is not None:
            ok = ok and hasattr(diffusers, s.cls)
            if s.name == "DPM++ SDE":
                import importlib.util
                ok = ok and importlib.util.find_spec("torchsde") is not None
        rows.append({"name": s.name, "supported": bool(ok), "default_scheduler": s.auto_schedule})
    return rows


# --------------------------------------------------------------------------- diffusers glue

def _accepted_params(fn) -> set[str]:
    try:
        return set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return set()


def install_custom_sigmas(scheduler, sigma_fn) -> None:
    """Patch scheduler.set_timesteps so `set_timesteps(N)` uses sigma_fn(N) (n+1 sigmas ending at 0)."""
    orig = scheduler.set_timesteps

    def set_timesteps(num_inference_steps=None, device=None, timesteps=None, sigmas=None, **kw):
        if num_inference_steps is not None and timesteps is None and sigmas is None:
            return orig(device=device, sigmas=[float(x) for x in sigma_fn(int(num_inference_steps))], **kw)
        return orig(num_inference_steps=num_inference_steps, device=device, timesteps=timesteps, sigmas=sigmas, **kw)

    scheduler.set_timesteps = set_timesteps


def build_scheduler(p: Plan, base_config, sdxl: bool = False, extra_config: Optional[dict] = None,
                    opts: Optional[dict] = None) -> tuple[object, list[str]]:
    """Instantiate the diffusers scheduler for plan `p` from a pipeline's scheduler config."""
    import diffusers  # type: ignore
    from . import sigmas as sigmas_mod

    warnings: list[str] = []
    cls = getattr(diffusers, p.cls, None)
    if cls is None:
        raise ValueError(f"sampler '{p.sampler}' is not available in this diffusers version ({p.cls})")
    accepted = _accepted_params(cls.__init__)
    cfg = dict(base_config)
    # diffusers' from_config resets every key listed in "_use_default_values" to the *new* class default, which would
    # silently drop our overrides (e.g. use_karras_sigmas); strip the bookkeeping keys.
    for k in ("_use_default_values", "_class_name", "_diffusers_version"):
        cfg.pop(k, None)
    # strip settings that belong to other scheduler families / previous choices
    for f in SIGMA_FLAGS:
        cfg.pop(f, None)
        if f in accepted:
            cfg[f] = False
    for k, v in p.kwargs.items():
        if k in accepted:
            cfg[k] = v
    for k, v in (extra_config or {}).items():
        if k in accepted:
            cfg[k] = v

    supports_sigmas = "sigmas" in _accepted_params(cls.set_timesteps)
    if supports_sigmas and p.schedule in sigmas_mod.SCHEDULE_FUNCS:
        if "timestep_spacing" in accepted:
            cfg["timestep_spacing"] = "linspace"
        sched = cls.from_config(cfg)
        ac = getattr(sched, "alphas_cumprod", None)
        model = sigmas_mod.DiscreteModel(sigmas_mod.model_sigmas_from_alphas(ac.cpu().numpy() if hasattr(ac, "cpu") else ac)) \
            if ac is not None else sigmas_mod.DiscreteModel()
        schedule = p.schedule
        install_custom_sigmas(sched, lambda n: sigmas_mod.compute(schedule, n, model, sdxl, opts))
        return sched, warnings

    equiv = CONFIG_EQUIVALENT.get(p.schedule)
    if equiv is None:
        warnings.append(f"schedule '{p.schedule}' is not available for sampler '{p.sampler}'; using Karras")
        equiv = CONFIG_EQUIVALENT["Karras"]
    applied = True
    for k, v in equiv.items():
        if k in accepted:
            cfg[k] = v
        else:
            applied = False
    if not applied:
        warnings.append(f"schedule '{p.schedule}' is not supported by sampler '{p.sampler}'; using the sampler's default spacing")
    elif "timestep_spacing" in accepted and "timestep_spacing" not in equiv:
        cfg["timestep_spacing"] = "linspace"
    if p.sampler == "LCM" and p.schedule not in ("Uniform",):
        warnings.append("LCM ignores the schedule type")
    return cls.from_config(cfg), warnings
