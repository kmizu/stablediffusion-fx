"""Sampling-time hooks from the extension index (sdfx.sampling_ext)."""
import math

import numpy as np
import pytest

from sdfx import sampling_ext as se


def test_detail_daemon_schedule_shape():
    m = se.make_dd_schedule(20, 0.2, 0.8, 0.5, 0.3, 1.0, 0.0, 0.0, 0.0, False)
    assert len(m) == 20
    assert m[:4].max() == 0 and m[16:].max() == 0
    assert m.max() == pytest.approx(0.3) and int(np.argmax(m)) in (9, 10)
    faded = se.make_dd_schedule(20, 0.2, 0.8, 0.5, 0.3, 1.0, 0.0, 0.0, 0.5, True)
    assert faded.max() == pytest.approx(0.15)
    off = se.make_dd_schedule(10, 0.3, 0.6, 0.5, 0.2, 1.0, 0.05, 0.1, 0.0, False)
    assert off[0] == pytest.approx(0.05) and off[-1] == pytest.approx(0.1)


def test_cads_gamma_and_cfg_curve_and_scaling():
    assert se.cads_gamma(0.0, 0.6, 0.9) == 0.0          # t = 1 >= tau2
    assert se.cads_gamma(0.25, 0.6, 0.9) == pytest.approx((0.9 - 0.75) / 0.3)
    assert se.cads_gamma(0.5, 0.6, 0.9) == 1.0          # t = 0.5 <= tau1
    c = dict(se.CFG_SCHEDULE_DEFAULTS)
    assert se.cfg_at(c, 0.0) == 7.0 and se.cfg_at(c, 1.0) == 3.0 and se.cfg_at(c, 0.5) == pytest.approx(5.0)
    c["curve"] = "cosine"
    assert se.cfg_at(c, 0.25) == pytest.approx(7 - 4 * 0.5 * (1 - math.cos(math.pi / 4)))
    assert se.scaling("Flat", 0.3) == 1.0 and se.scaling("Cos", 1.0) == pytest.approx(0.0, abs=1e-9)
    assert se.scaling("1 - Sin", 0.0) == 1.0


def test_rgb_direction_round_trips_through_the_preview_map():
    for sdxl, f in ((False, se._SD15), (True, se._SDXL)):
        d = se.rgb_to_latent_dir([0.3, -0.1, 0.2], sdxl, ridge=0)
        assert np.allclose(f.T @ d, [0.3, -0.1, 0.2], atol=1e-6)
        for rgb in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]):
            r = se.rgb_to_latent_dir(rgb, sdxl)
            assert np.linalg.norm(r) < 4.0                       # bounded, unlike the pseudo-inverse
            back = f.T @ r
            assert float(np.dot(back, rgb)) / (np.linalg.norm(back) * np.linalg.norm(rgb)) > 0.6


torch = pytest.importorskip("torch")


def test_tiling_patch_matches_circular_conv():
    conv = torch.nn.Conv2d(2, 3, 3, padding=1)
    ref = torch.nn.Conv2d(2, 3, 3, padding=1, padding_mode="circular")
    ref.load_state_dict(conv.state_dict())
    x = torch.randn(1, 2, 8, 10)
    tp = se.TilingPatch([conv], True, True).install()
    assert torch.allclose(conv(x), ref(x), atol=1e-5)
    tp.active = False
    plain = conv(x)
    tp.remove()
    assert torch.allclose(conv(x), plain)
    # X only: columns wrap, rows are zero padded
    se.TilingPatch([conv], True, False).install()
    y = conv(x)
    xp = torch.nn.functional.pad(torch.nn.functional.pad(x, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1))
    assert torch.allclose(y, torch.nn.functional.conv2d(xp, conv.weight, conv.bias), atol=1e-5)


class _Sched:
    def __init__(self, n):
        self.timesteps = torch.linspace(999, 0, n).round()


class _Unet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, sample, timestep, encoder_hidden_states=None):
        self.seen.append((sample.clone(), timestep, None if encoder_hidden_states is None else encoder_hidden_states.clone()))
        return (sample * 0 + 1,)


def _ac():
    betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000) ** 2
    return torch.cumprod(1 - betas, 0)


def test_detail_daemon_lowers_the_sigma_the_model_sees():
    u, sch = _Unet(), _Sched(10)
    c = dict(se.DETAIL_DAEMON_DEFAULTS, amount=0.5, start=0.0, end=1.0, bias=0.5, smooth=False)
    hk = se.DetailDaemonHook(u, sch, _ac(), c, lambda: True).install()
    x = torch.ones(2, 4, 8, 8)
    t = sch.timesteps[5]
    u(x, t)
    s, tt, _ = u.seen[-1]
    assert float(tt.flatten()[0]) < float(t) and s.mean() > 1.0  # lower sigma, larger c_in
    hk.c["mode"] = "cond"
    u(x, t)
    s2, tt2, _ = u.seen[-1]
    assert float(tt2[0]) == pytest.approx(float(t)) and float(tt2[1]) < float(t)
    assert torch.allclose(s2[0], x[0]) and not torch.allclose(s2[1], x[1])
    hk.remove()
    u(x, t)
    assert float(u.seen[-1][1]) == float(t)


def test_cfg_schedule_rescales_the_cond_half():
    class U(torch.nn.Module):
        def forward(self, sample, timestep):
            return (torch.cat([torch.zeros_like(sample[:1]), torch.ones_like(sample[:1])]),)
    u, sch = U(), _Sched(5)
    hk = se.CfgScheduleHook(u, sch, 8.0, dict(se.CFG_SCHEDULE_DEFAULTS, start_cfg=4.0, end_cfg=4.0), lambda: True).install()
    out = u(torch.zeros(2, 1, 2, 2), sch.timesteps[0])[0]
    uncond, cond = out.chunk(2)
    assert torch.allclose(uncond + 8.0 * (cond - uncond), torch.full_like(cond, 4.0))
    hk.remove()


def test_cads_noises_only_the_cond_half_early():
    u, sch = _Unet(), _Sched(10)
    hk = se.CadsHook(u, sch, dict(se.CADS_DEFAULTS), 1, lambda: True).install()
    ehs = torch.randn(2, 77, 16)
    u(torch.zeros(2, 4, 4, 4), sch.timesteps[0], encoder_hidden_states=ehs)
    e = u.seen[-1][2]
    assert torch.allclose(e[0], ehs[0]) and not torch.allclose(e[1], ehs[1])
    u(torch.zeros(2, 4, 4, 4), sch.timesteps[-1], encoder_hidden_states=ehs)  # late: gamma = 1
    assert torch.allclose(u.seen[-1][2], ehs)
    hk.remove()


class _Pipe:
    _num_timesteps = 6
    _interrupt = False


def test_anti_burn_averages_and_stops_early():
    lc = se.LatentCallbacks(False, 6, 0)
    lc.anti_burn({"skip_last": 2, "average": 2})
    p = _Pipe()
    out = None
    for i in range(4):
        out = lc(p, i, None, {"latents": torch.full((1, 4, 2, 2), float(i))})["latents"]
    assert p._interrupt and torch.allclose(out, torch.full((1, 4, 2, 2), 2.5))


def test_latent_edits_change_latents_and_mirroring_is_symmetric():
    lat = torch.randn(1, 4, 8, 8)
    lc = se.LatentCallbacks(False, 4, 0)
    lc.mirroring({"mode": "Blend Average", "style": "Vertical Mirroring", "max_step": 1.0, "strength": 1.0})
    out = lc(_Pipe(), 0, None, {"latents": lat.clone()})["latents"]
    assert torch.allclose(out, torch.flip(out, dims=[3]), atol=1e-6)
    for setup in (lambda l: l.vectorscope(dict(se.VECTORSCOPE_DEFAULTS, brightness=1.0, saturation=1.0)),
                  lambda l: l.color_grading(dict(se.COLOR_GRADING_DEFAULTS, r=1.0, normalize=True))):
        lc = se.LatentCallbacks(False, 4, 0)
        setup(lc)
        assert not torch.allclose(lc(_Pipe(), 0, None, {"latents": lat.clone()})["latents"], lat)
    lc = se.LatentCallbacks(False, 4, 0)
    lc.resharpen({"sharpness": 1.0, "scaling": "Flat"})
    lc(_Pipe(), 0, None, {"latents": lat.clone()})
    out = lc(_Pipe(), 1, None, {"latents": lat + 1})["latents"]
    assert torch.allclose(out, lat + 1.1, atol=1e-5)
