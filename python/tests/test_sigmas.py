"""Compare numpy sigma schedules with the reference formulas of k-diffusion / reForge sd_schedulers.py."""
import math

import numpy as np
import pytest

from sdfx import sigmas as S

SMIN, SMAX = 0.0291675, 14.614642


def ref_karras(n, smin, smax, rho=7.0):
    # k_diffusion.sampling.get_sigmas_karras
    ramp = np.linspace(0, 1, n)
    min_inv_rho = smin ** (1 / rho)
    max_inv_rho = smax ** (1 / rho)
    return np.append((max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho, 0.0)


def ref_exponential(n, smin, smax, shrink=0.0):
    # reForge sd_schedulers.get_sigmas_exponential
    s = np.exp(np.linspace(math.log(smax), math.log(smin), n))
    s = s * np.exp(shrink * np.linspace(0, 1, n))
    return np.append(s, 0.0)


def ref_polyexponential(n, smin, smax, rho=1.0):
    ramp = np.linspace(1, 0, n) ** rho
    return np.append(np.exp(ramp * (math.log(smax) - math.log(smin)) + math.log(smin)), 0.0)


@pytest.mark.parametrize("n", [1, 2, 10, 20, 37])
def test_karras(n):
    np.testing.assert_allclose(S.karras(n, SMIN, SMAX), ref_karras(n, SMIN, SMAX), rtol=1e-12)
    np.testing.assert_allclose(S.karras(n, SMIN, SMAX, 3.5), ref_karras(n, SMIN, SMAX, 3.5), rtol=1e-12)


def test_karras_known_values():
    s = S.karras(20, SMIN, SMAX)
    assert len(s) == 21 and s[-1] == 0.0
    assert s[0] == pytest.approx(SMAX) and s[-2] == pytest.approx(SMIN)
    assert np.all(np.diff(s) < 0)


@pytest.mark.parametrize("n", [2, 10, 25])
def test_exponential(n):
    np.testing.assert_allclose(S.exponential(n, SMIN, SMAX), ref_exponential(n, SMIN, SMAX), rtol=1e-12)
    np.testing.assert_allclose(S.exponential(n, SMIN, SMAX, 0.3), ref_exponential(n, SMIN, SMAX, 0.3), rtol=1e-12)


@pytest.mark.parametrize("n", [2, 10, 25])
def test_polyexponential(n):
    np.testing.assert_allclose(S.polyexponential(n, SMIN, SMAX), ref_polyexponential(n, SMIN, SMAX), rtol=1e-12)
    np.testing.assert_allclose(S.polyexponential(n, SMIN, SMAX, 2.0), ref_polyexponential(n, SMIN, SMAX, 2.0), rtol=1e-12)
    # rho=1 polyexponential == exponential
    np.testing.assert_allclose(S.polyexponential(n, SMIN, SMAX), S.exponential(n, SMIN, SMAX), rtol=1e-12)


def test_ays_matches_nvidia_table():
    # NVIDIA AYS SD1.5 10-step schedule (from the AlignYourSteps how-to)
    nvidia = [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029]
    s = S.align_your_steps(11, 14.615, sdxl=False)
    assert len(s) == 12 and s[-1] == 0.0
    np.testing.assert_allclose(s[:11], nvidia, rtol=2e-3)
    nvidia_xl = [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029]
    np.testing.assert_allclose(S.align_your_steps(11, 14.615, sdxl=True)[:11], nvidia_xl, rtol=2e-3)


def test_ays_interpolated():
    s = S.align_your_steps(20, SMAX)
    assert len(s) == 21 and s[-1] == 0
    assert s[0] == pytest.approx(SMAX) and s[-2] == pytest.approx(0.029)
    assert np.all(np.diff(s[:-1]) < 0)
    # reference loglinear_interp (reForge get_align_your_steps_sigmas)
    base = [SMAX, SMAX / 2.257, SMAX / 3.785, SMAX / 5.418, SMAX / 7.749, SMAX / 10.469, SMAX / 15.176,
            SMAX / 22.415, SMAX / 36.629, SMAX / 96.151, 0.029]
    xs = np.linspace(0, 1, len(base))
    ys = np.log(np.array(base)[::-1])
    ref = np.exp(np.interp(np.linspace(0, 1, 20), xs, ys))[::-1]
    np.testing.assert_allclose(s[:-1], ref, rtol=1e-12)


def test_kl_optimal_matches_reforge():
    # reForge sd_schedulers.kl_optimal: tan(i/n * atan(smin) + (1 - i/n) * atan(smax)), i = 0..n (no trailing zero)
    n = 10
    i = np.arange(n + 1)
    ref = np.tan(i / n * math.atan(SMIN) + (1.0 - i / n) * math.atan(SMAX))
    s = S.kl_optimal(n, SMIN, SMAX)
    np.testing.assert_allclose(s, ref, rtol=1e-12)
    assert s[0] == pytest.approx(SMAX) and s[-1] == pytest.approx(SMIN)


def test_discrete_model_schedules():
    m = S.DiscreteModel()
    assert m.sigma_min == pytest.approx(S.SD_SIGMA_MIN, rel=1e-4)
    assert m.sigma_max == pytest.approx(S.SD_SIGMA_MAX, rel=1e-4)
    assert m.sigma_to_t(m.t_to_sigma(500.25)) == pytest.approx(500.25, abs=1e-6)
    for name in S.SCHEDULE_FUNCS:
        s = S.compute(name, 20, m)
        assert s[-1] == (pytest.approx(m.sigma_min) if name == "KL Optimal" else 0.0), name
        assert np.all(np.diff(s) <= 0), name
        assert s[0] <= m.sigma_max * 1.0001, name
    u = S.compute("Uniform", 20, m)
    assert len(u) == 21 and u[0] == pytest.approx(m.sigma_max)
    assert len(S.compute("DDIM", 20, m)) == 21
    assert len(S.compute("SGM Uniform", 20, m)) == 21
    assert len(S.compute("Simple", 20, m)) == 21


def test_beta_ppf_fallback_close_to_scipy_formula():
    # symmetric beta: ppf(0.5) = 0.5 ; monotone
    q = np.linspace(0.01, 0.99, 50)
    xs, cdf = S._beta_cdf(q, 0.6, 0.6)
    v = np.interp(q, cdf, xs)
    assert np.all(np.diff(v) > 0)
    assert np.interp(0.5, cdf, xs) == pytest.approx(0.5, abs=1e-3)
    b = S.compute("Beta", 20, S.DiscreteModel())
    assert b[-1] == 0 and len(b) <= 21


def test_unknown_schedule():
    with pytest.raises(ValueError):
        S.compute("Nope", 10)
