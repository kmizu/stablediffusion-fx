"""Pure numpy ports of the A1111/reForge sigma schedules (modules/sd_schedulers.py, k_diffusion/sampling.py).

Every function returns a float64 array of n+1 sigmas, decreasing, normally ending in 0.
`model_sigmas` is the discrete training schedule sqrt((1 - alphas_cumprod) / alphas_cumprod), ascending.
"""
from __future__ import annotations

import math

import numpy as np

# Default SD1.x/SDXL training schedule (scaled_linear 0.00085 .. 0.012, 1000 steps)
SD_SIGMA_MIN = 0.029167532920837402
SD_SIGMA_MAX = 14.614643096923828


def default_alphas_cumprod(n: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012) -> np.ndarray:
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, n, dtype=np.float64) ** 2
    return np.cumprod(1.0 - betas)


def model_sigmas_from_alphas(alphas_cumprod) -> np.ndarray:
    ac = np.asarray(alphas_cumprod, dtype=np.float64)
    return np.sqrt((1.0 - ac) / ac)


def default_model_sigmas() -> np.ndarray:
    return model_sigmas_from_alphas(default_alphas_cumprod())


def _append_zero(x: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(x, dtype=np.float64), [0.0]])


# --------------------------------------------------------------------------- closed-form schedules

def karras(n: int, sigma_min: float, sigma_max: float, rho: float = 7.0) -> np.ndarray:
    ramp = np.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    return _append_zero((max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho)


def exponential(n: int, sigma_min: float, sigma_max: float, shrink_factor: float = 0.0) -> np.ndarray:
    s = np.exp(np.linspace(math.log(sigma_max), math.log(sigma_min), n))
    s = s * np.exp(shrink_factor * np.linspace(0, 1, n))
    return _append_zero(s)


def polyexponential(n: int, sigma_min: float, sigma_max: float, rho: float = 1.0) -> np.ndarray:
    ramp = np.linspace(1, 0, n) ** rho
    return _append_zero(np.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min)))


def kl_optimal(n: int, sigma_min: float, sigma_max: float) -> np.ndarray:
    """KL-optimal schedule exactly as reForge sd_schedulers.kl_optimal: n+1 sigmas from sigma_max down to
    sigma_min (no trailing zero)."""
    alpha_min = math.atan(sigma_min)
    alpha_max = math.atan(sigma_max)
    idx = np.arange(n + 1, dtype=np.float64)
    return np.tan(idx / n * alpha_min + (1.0 - idx / n) * alpha_max)


AYS_SD15 = [1.0, 1 / 2.257, 1 / 3.785, 1 / 5.418, 1 / 7.749, 1 / 10.469, 1 / 15.176, 1 / 22.415, 1 / 36.629, 1 / 96.151]
AYS_SDXL = [1.0, 1 / 2.314, 1 / 3.875, 1 / 6.701, 1 / 10.89, 1 / 16.954, 1 / 26.333, 1 / 38.46, 1 / 62.457, 1 / 129.336]


def _loglinear_interp(t_steps, num_steps: int) -> np.ndarray:
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(np.asarray(t_steps, dtype=np.float64)[::-1])
    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)
    return np.exp(new_ys)[::-1].copy()


def align_your_steps(n: int, sigma_max: float = SD_SIGMA_MAX, sdxl: bool = False) -> np.ndarray:
    base = AYS_SDXL if sdxl else AYS_SD15
    sigmas = [sigma_max * f for f in base] + [0.029]
    if n != len(sigmas):
        return _append_zero(_loglinear_interp(sigmas, n))
    return _append_zero(np.asarray(sigmas, dtype=np.float64))


# --------------------------------------------------------------------------- discrete-model schedules

class DiscreteModel:
    """Minimal k-diffusion DiscreteSchedule (log-linear sigma<->t interpolation)."""

    def __init__(self, model_sigmas=None):
        self.sigmas = np.asarray(default_model_sigmas() if model_sigmas is None else model_sigmas, dtype=np.float64)
        self.log_sigmas = np.log(self.sigmas)

    @property
    def sigma_min(self) -> float:
        return float(self.sigmas[0])

    @property
    def sigma_max(self) -> float:
        return float(self.sigmas[-1])

    def sigma_to_t(self, sigma: float) -> float:
        ls = math.log(sigma)
        dists = ls - self.log_sigmas
        low_idx = int(np.clip(np.cumsum(dists >= 0).argmax(), 0, len(self.log_sigmas) - 2))
        high_idx = low_idx + 1
        low, high = self.log_sigmas[low_idx], self.log_sigmas[high_idx]
        w = (low - ls) / (low - high)
        w = min(max(w, 0.0), 1.0)
        return (1 - w) * low_idx + w * high_idx

    def t_to_sigma(self, t: float) -> float:
        t = float(t)
        low_idx = int(math.floor(t))
        high_idx = int(math.ceil(t))
        w = t - low_idx
        return float(math.exp((1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]))

    def get_sigmas(self, n: int) -> np.ndarray:
        t_max = len(self.sigmas) - 1
        ts = np.linspace(t_max, 0, n)
        return _append_zero([self.t_to_sigma(t) for t in ts])


def uniform(n: int, model: DiscreteModel) -> np.ndarray:
    return model.get_sigmas(n)


def sgm_uniform(n: int, model: DiscreteModel) -> np.ndarray:
    start = model.sigma_to_t(model.sigma_max)
    end = model.sigma_to_t(model.sigma_min)
    ts = np.linspace(start, end, n + 1)[:-1]
    return _append_zero([model.t_to_sigma(t) for t in ts])


def simple(n: int, model: DiscreteModel) -> np.ndarray:
    ss = len(model.sigmas) / n
    return _append_zero([float(model.sigmas[-(1 + int(x * ss))]) for x in range(n)])


def normal(n: int, model: DiscreteModel) -> np.ndarray:
    start = model.sigma_to_t(model.sigma_max)
    end = model.sigma_to_t(model.sigma_min)
    ts = np.linspace(start, end, n)
    return _append_zero([model.t_to_sigma(t) for t in ts])


def ddim(n: int, model: DiscreteModel) -> np.ndarray:
    sigs = []
    ss = max(len(model.sigmas) // n, 1)
    x = 1
    while x < len(model.sigmas):
        sigs.append(float(model.sigmas[x]))
        x += ss
    return _append_zero(sigs[::-1])


def _beta_cdf(x: np.ndarray, a: float, b: float, grid: int = 20001) -> tuple[np.ndarray, np.ndarray]:
    """Numerically integrated regularized incomplete beta on a grid (used when scipy is missing)."""
    eps = 1e-12
    xs = np.linspace(0.0, 1.0, grid)
    # integrate pdf on midpoints to avoid the endpoint singularities for a, b < 1
    mids = (xs[:-1] + xs[1:]) / 2
    pdf = np.power(np.clip(mids, eps, 1), a - 1) * np.power(np.clip(1 - mids, eps, 1), b - 1)
    cdf = np.concatenate([[0.0], np.cumsum(pdf * np.diff(xs))])
    cdf /= cdf[-1]
    return xs, cdf


def beta_ppf(q, a: float, b: float) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    try:
        from scipy import stats  # type: ignore
        return stats.beta.ppf(q, a, b)
    except Exception:
        xs, cdf = _beta_cdf(q, a, b)
        return np.interp(q, cdf, xs)


def beta(n: int, model: DiscreteModel, alpha: float = 0.6, beta_: float = 0.6) -> np.ndarray:
    total = len(model.sigmas) - 1
    ts = 1 - np.linspace(0, 1, n, endpoint=False)
    ts = np.rint(beta_ppf(ts, alpha, beta_) * total)
    sigs, last = [], -1
    for t in ts:
        if t != last:
            sigs.append(float(model.sigmas[int(t)]))
        last = t
    return _append_zero(sigs)


# --------------------------------------------------------------------------- dispatch

SCHEDULE_FUNCS = ("Uniform", "Karras", "Exponential", "Polyexponential", "SGM Uniform", "KL Optimal",
                  "Align Your Steps", "Simple", "Normal", "DDIM", "Beta")


def compute(schedule: str, n: int, model: DiscreteModel | None = None, sdxl: bool = False, opts: dict | None = None) -> np.ndarray:
    """Compute A1111 schedule `schedule` for n steps against a model sigma table."""
    model = model or DiscreteModel()
    opts = opts or {}
    smin, smax = model.sigma_min, model.sigma_max
    if schedule == "Uniform":
        return uniform(n, model)
    if schedule == "Karras":
        return karras(n, smin, smax, float(opts.get("karras_rho", 7.0)))
    if schedule == "Exponential":
        return exponential(n, smin, smax, float(opts.get("exponential_shrink_factor", 0.0)))
    if schedule == "Polyexponential":
        return polyexponential(n, smin, smax, float(opts.get("polyexponential_rho", 1.0)))
    if schedule == "SGM Uniform":
        return sgm_uniform(n, model)
    if schedule == "KL Optimal":
        return kl_optimal(n, smin, smax)
    if schedule == "Align Your Steps":
        return align_your_steps(n, smax, sdxl)
    if schedule == "Simple":
        return simple(n, model)
    if schedule == "Normal":
        return normal(n, model)
    if schedule == "DDIM":
        return ddim(n, model)
    if schedule == "Beta":
        return beta(n, model, float(opts.get("beta_dist_alpha", 0.6)), float(opts.get("beta_dist_beta", 0.6)))
    raise ValueError(f"unknown schedule '{schedule}'")
