"""Sampling-time hooks re-implemented from extensions in the A1111 extension index.

All of them work on diffusers pipelines through two mechanisms only:
  * UNet forward (pre-)hooks: change the timestep / input / text conditioning the model sees, or the CFG halves;
  * callback_on_step_end on "latents": change the latent between steps.

  hook name          extension                           what
  detail_daemon      muerrilla/sd-webui-detail-daemon    lowers (or raises) the sigma the model is told, over a
                                                         schedule, so it adds (or removes) fine detail
  vectorscope_cc     Haoming02/sd-webui-vectorscope-cc   brightness / contrast / saturation / RGB offsets nudged
                                                         into the latent at every step
  latent_mirroring   dfaker/SD-latent-mirroring          mirrors / rotates the latent during the first steps
  tiling             tjm35/asymmetric-tiling-sd-webui    circular padding on X and/or Y (UNet + VAE) -> tileable
  anti_burn          klimaleksus/...-anti-burn           skips the last steps and averages the ones before
  resharpen          Haoming02/sd-webui-resharpen        pushes each step further along (or back from) its delta
  color_grading      Haoming02/sd-webui-diffusion-cg     recenters latent channel means and stretches the range
  cfg_schedule       guzuligo/CFG-Schedule-for-...       CFG scale that changes over the steps
  cads               v0xie/sd-webui-cads                 scheduled noise on the text conditioning (diversity)

Everything is written from the extensions' documented behaviour; parameter names follow their UIs.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from .adapters import normalize
from .hooks import hook_config

DETAIL_DAEMON_DEFAULTS = {"mode": "both", "amount": 0.1, "start": 0.2, "end": 0.8, "bias": 0.5, "exponent": 1.0,
                          "start_offset": 0.0, "end_offset": 0.0, "fade": 0.0, "smooth": True}
VECTORSCOPE_DEFAULTS = {"brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "r": 0.0, "g": 0.0, "b": 0.0,
                        "method": "Straight", "scaling": "Flat", "strength": 1.0}
MIRRORING_DEFAULTS = {"mode": "Alternate Steps", "style": "Vertical Mirroring", "max_step": 0.25, "strength": 1.0}
TILING_DEFAULTS = {"tile_x": True, "tile_y": True, "start_step": 0, "stop_step": -1}
ANTI_BURN_DEFAULTS = {"skip_last": 2, "average": 3}
RESHARPEN_DEFAULTS = {"sharpness": 0.5, "scaling": "Flat"}
COLOR_GRADING_DEFAULTS = {"recenter": True, "recenter_strength": 1.0, "r": 0.0, "g": 0.0, "b": 0.0,
                          "normalize": False, "normalize_strength": 1.0}
CFG_SCHEDULE_DEFAULTS = {"start_cfg": 7.0, "end_cfg": 3.0, "curve": "linear", "start": 0.0, "end": 1.0}
CADS_DEFAULTS = {"tau1": 0.6, "tau2": 0.9, "noise_scale": 0.25, "mixing_factor": 1.0, "rescale": True,
                 "apply_to_uncond": False}

HOOKS = ("detail_daemon", "vectorscope_cc", "latent_mirroring", "tiling", "anti_burn", "resharpen", "color_grading",
         "cfg_schedule", "cads")

# latent -> RGB linear maps (A1111 cheap approximation), rows = latent channels
_SD15 = np.array([[0.298, 0.207, 0.208], [0.187, 0.286, 0.173], [-0.158, 0.189, 0.264], [-0.184, -0.271, -0.473]])
_SDXL = np.array([[0.3448, 0.4168, 0.4395], [-0.1953, -0.0290, 0.0250], [0.1074, 0.0886, -0.0163],
                  [-0.3730, -0.2499, -0.2088]])


# ---------------------------------------------------------------------------------------------------------------
# helpers

def rgb_to_latent_dir(rgb, sdxl: bool, ridge: float = 0.05) -> np.ndarray:
    """Ridge-regularised latent offset (4,) whose cheap RGB approximation points along `rgb` (3,).

    The plain pseudo-inverse is ill-conditioned for pure colours (|offset| up to ~15 for SDXL green), which pushes
    latents off the VAE's manifold; the ridge term keeps offsets small while preserving the colour direction."""
    f = _SDXL if sdxl else _SD15
    v = np.asarray(rgb, dtype=np.float64)
    return f @ np.linalg.solve(f.T @ f + ridge * np.eye(3), v) if ridge > 0 else np.linalg.pinv(f.T) @ v


def scaling(name: str, progress: float) -> float:
    """Haoming02's step scaling curves (progress 0 -> 1)."""
    p = min(max(progress, 0.0), 1.0)
    n = (name or "Flat").lower().replace(" ", "")
    if n == "cos":
        return math.cos(p * math.pi / 2)
    if n == "sin":
        return math.sin(p * math.pi / 2)
    if n == "1-cos":
        return 1 - math.cos(p * math.pi / 2)
    if n == "1-sin":
        return 1 - math.sin(p * math.pi / 2)
    return 1.0


def make_dd_schedule(steps: int, start: float, end: float, bias: float, amount: float, exponent: float,
                     start_offset: float, end_offset: float, fade: float, smooth: bool) -> np.ndarray:
    """Detail Daemon multiplier per step: ramps start_offset -> amount (at start + bias*(end-start)) -> end_offset."""
    steps = max(int(steps), 1)
    start = min(start, end)
    mid = start + bias * (end - start)
    m = np.zeros(steps)
    si, mi, ei = (int(round(x * (steps - 1))) for x in (start, mid, end))

    def ramp(a, b, n):
        v = np.linspace(a, b, n)
        if smooth:
            v = 0.5 * (1 - np.cos(v * np.pi))
        return v ** exponent

    up = ramp(0, 1, mi - si + 1)
    up = up * (amount - start_offset) + start_offset
    down = ramp(1, 0, ei - mi + 1)
    down = down * (amount - end_offset) + end_offset
    m[si:mi + 1] = up
    m[mi:ei + 1] = down
    m[:si] = start_offset
    m[ei + 1:] = end_offset
    return m * (1 - fade)


def cads_gamma(progress: float, tau1: float, tau2: float) -> float:
    """CADS annealing: t = 1 - progress; gamma = 1 for t <= tau1, 0 for t >= tau2, linear between."""
    t = 1.0 - progress
    if t <= tau1:
        return 1.0
    if t >= tau2:
        return 0.0
    return (tau2 - t) / (tau2 - tau1)


def cads_noise(y, gamma: float, noise_scale: float, psi: float, rescale: bool, gen, *, noise=None):
    import torch
    if gamma >= 1.0:
        return y
    if noise is not None and noise.shape != y.shape:
        raise ValueError("CADS supplied noise must match the conditioning shape")
    n = (torch.randn(y.shape, generator=gen, device="cpu", dtype=torch.float32) if noise is None else noise).to(y.device, y.dtype)
    yh = math.sqrt(gamma) * y + noise_scale * math.sqrt(1 - gamma) * n
    if rescale:
        m0, s0 = y.mean(dim=(1, 2), keepdim=True), y.std(dim=(1, 2), keepdim=True)
        m1, s1 = yh.mean(dim=(1, 2), keepdim=True), yh.std(dim=(1, 2), keepdim=True)
        yr = (yh - m1) / s1.clamp_min(1e-6) * s0 + m0
        yh = psi * yr + (1 - psi) * yh
    return yh


def cads_context(value, gamma, config, generator, *, marked=False, noise=None):
    """Perturb CLIP features while keeping private NegPiP signs and masked padding intact."""
    if gamma >= 1:
        return value
    import torch
    features = value[..., :-1] if marked else value
    result = cads_noise(features, gamma, config["noise_scale"], config["mixing_factor"], config["rescale"], generator,
                        noise=noise)
    if marked:
        signs = value[..., -1:]
        result = torch.cat([torch.where(signs != 0, result, 0.), signs], dim=-1)
    return result


def cfg_at(c: dict, progress: float) -> float:
    a, b = float(c["start_cfg"]), float(c["end_cfg"])
    s, e = float(c["start"]), float(c["end"])
    if progress <= s:
        return a
    if progress >= e or e <= s:
        return b
    x = (progress - s) / (e - s)
    curve = str(c.get("curve", "linear")).lower()
    if curve == "cosine":
        x = 0.5 * (1 - math.cos(x * math.pi))
    elif curve == "square":
        x = x * x
    elif curve == "sqrt":
        x = math.sqrt(x)
    return a + (b - a) * x


def mirror(lat, style: str):
    import torch
    s = (style or "").lower()
    if s.startswith("vertical"):         # mirror across the vertical axis (left <-> right)
        return torch.flip(lat, dims=[3])
    if s.startswith("horizontal"):       # top <-> bottom
        return torch.flip(lat, dims=[2])
    if "90" in s and lat.shape[2] == lat.shape[3]:
        return torch.rot90(lat, 1, dims=[2, 3])
    if "180" in s or "90" in s:
        return torch.rot90(lat, 2, dims=[2, 3])
    if "both" in s:
        return torch.flip(lat, dims=[2, 3])
    return lat


class Progress:
    """Step position of a UNet call, from the scheduler's timestep list (robust to 2nd-order samplers)."""

    def __init__(self, scheduler):
        self.s = scheduler

    def __call__(self, t) -> tuple[int, int, float]:
        import torch
        ts = getattr(self.s, "timesteps", None)
        tv = float(t.flatten()[0]) if isinstance(t, torch.Tensor) else float(t)
        if ts is None or len(ts) == 0:
            return 0, 1, 1.0 - tv / 999.0
        arr = ts.float().cpu().numpy() if isinstance(ts, torch.Tensor) else np.asarray(ts, dtype=np.float64)
        i = int(np.argmin(np.abs(arr - tv)))
        n = len(arr)
        return i, n, i / max(n - 1, 1)


class SigmaTable:
    def __init__(self, alphas_cumprod):
        ac = np.asarray(alphas_cumprod.float().cpu() if hasattr(alphas_cumprod, "cpu") else alphas_cumprod, dtype=np.float64)
        self.sig = np.sqrt((1 - ac) / ac)
        self.log = np.log(self.sig)

    def sigma(self, t: float) -> float:
        return float(np.exp(np.interp(t, np.arange(len(self.sig)), self.log)))

    def t(self, sigma: float) -> float:
        return float(np.interp(math.log(max(sigma, 1e-8)), self.log, np.arange(len(self.sig))))


# ---------------------------------------------------------------------------------------------------------------
# UNet-level hooks

class DetailDaemonHook:
    def __init__(self, unet, scheduler, alphas_cumprod, c: dict, cfg_active: Callable[[], bool]):
        self.unet, self.c, self.cfg_active = unet, c, cfg_active
        self.prog = Progress(scheduler)
        self.tab = SigmaTable(alphas_cumprod)
        self.sched = None
        self.handle = None

    def _mult(self, i: int, n: int) -> float:
        if self.sched is None or len(self.sched) != n:
            c = self.c
            self.sched = make_dd_schedule(n, c["start"], c["end"], c["bias"], c["amount"], c["exponent"],
                                          c["start_offset"], c["end_offset"], c["fade"], c["smooth"])
        return float(self.sched[min(i, n - 1)])

    def pre(self, module, args, kwargs):
        import torch
        sample = args[0] if args else kwargs.get("sample")
        t = args[1] if len(args) > 1 else kwargs.get("timestep")
        if sample is None or t is None:
            return None
        i, n, _ = self.prog(t)
        m = self._mult(i, n)
        if abs(m) < 1e-6:
            return None
        tv = float(t.flatten()[0]) if isinstance(t, torch.Tensor) else float(t)
        s = self.tab.sigma(tv)
        s2 = min(max(s * max(1.0 - m, 1e-3), float(self.tab.sig[0])), float(self.tab.sig[-1]))
        t2 = self.tab.t(s2)
        k = math.sqrt(1 + s * s) / math.sqrt(1 + s2 * s2)
        b = sample.shape[0]
        mode = str(self.c["mode"]).lower()
        sel = torch.ones(b, dtype=torch.bool, device=sample.device)
        if mode in ("cond", "uncond") and self.cfg_active() and b % 2 == 0:
            sel[:] = False
            if mode == "cond":
                sel[b // 2:] = True
            else:
                sel[: b // 2] = True
        scale = torch.where(sel, torch.tensor(k, device=sample.device), torch.tensor(1.0, device=sample.device))
        new_sample = sample * scale.view(-1, 1, 1, 1).to(sample.dtype)
        tt = torch.where(sel.cpu(), torch.tensor(t2), torch.tensor(tv)).to(
            t.device if isinstance(t, torch.Tensor) else sample.device)
        if isinstance(t, torch.Tensor) and t.dtype in (torch.int64, torch.int32):
            tt = tt.round().to(t.dtype)
        if len(args) >= 2:
            args = (new_sample, tt) + tuple(args[2:])
        elif len(args) == 1:
            args, kwargs = (new_sample,), dict(kwargs, timestep=tt)
        else:
            kwargs = dict(kwargs, sample=new_sample, timestep=tt)
        return args, kwargs

    def install(self):
        self.handle = self.unet.register_forward_pre_hook(self.pre, with_kwargs=True)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class CfgScheduleHook:
    """Rewrites native CFG groups to use g_t while preserving PAG's conditional/perturbed difference."""

    def __init__(self, unet, scheduler, g: float, c: dict, cfg_active: Callable[[], bool], *, pag_active=lambda: False):
        self.unet, self.g, self.c, self.cfg_active = unet, float(g), c, cfg_active
        self.prog = Progress(scheduler)
        self.handle = None
        self.seen: list = []
        self.pag_active = pag_active

    def hook(self, module, args, kwargs, output):
        out = output[0] if isinstance(output, tuple) else getattr(output, "sample", output)
        t = args[1] if len(args) > 1 else kwargs.get("timestep")
        if out is None or t is None or not self.cfg_active() or self.g <= 0:
            return None
        pag = self.pag_active()
        groups = 3 if pag else 2
        if out.shape[0] % groups:
            return None
        _, _, p = self.prog(t)
        gt = cfg_at(self.c, p)
        self.seen.append(round(gt, 4))
        if gt == self.g:
            return None
        parts = out.chunk(groups)
        u, c = parts[:2]
        guided = u + (gt / self.g) * (c - u)
        # Native PAG later adds scale * (c-p); move both conditional groups by the same offset.
        new = __import__("torch").cat([u, guided] + ([parts[2] + (guided - c)] if pag else []))
        if isinstance(output, tuple):
            return (new,) + tuple(output[1:])
        if hasattr(output, "sample"):
            output.sample = new
            return output
        return new

    def install(self):
        self.handle = self.unet.register_forward_hook(self.hook, with_kwargs=True)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class CadsHook:
    """Perturbs canonical CFG contexts once before native PAG positive-context duplication."""

    def __init__(self, unet, scheduler, c: dict, seed: int, cfg_active: Callable[[], bool], *, pag_active=lambda: False):
        import torch
        self.unet, self.c, self.cfg_active = unet, c, cfg_active
        self.prog = Progress(scheduler)
        self.gen = torch.Generator("cpu").manual_seed(int(seed) & 0xFFFFFFFF)
        self.handle = None
        self.pag_active = pag_active

    def pre(self, module, args, kwargs):
        import torch
        t = args[1] if len(args) > 1 else kwargs.get("timestep")
        ehs = args[2] if len(args) > 2 else kwargs.get("encoder_hidden_states")
        if t is None or ehs is None:
            return None
        _, _, p = self.prog(t)
        c = self.c
        g = cads_gamma(p, c["tau1"], c["tau2"])
        if g >= 1.0:
            return None
        b = ehs.shape[0]
        dim = getattr(getattr(self.unet, "config", None), "cross_attention_dim", None)
        marked = isinstance(dim, int) and ehs.shape[-1] == dim + 1
        if self.pag_active():
            guided = self.cfg_active()
            groups = 3 if guided else 2
            if b % groups:
                return None
            parts = ehs.chunk(groups)
            positive = parts[1] if guided else parts[0]
            negative = parts[0] if guided else None
            if guided and c["apply_to_uncond"]:
                negative = cads_context(negative, g, c, self.gen, marked=marked)
            positive = cads_context(positive, g, c, self.gen, marked=marked)
            new = torch.cat(([negative] if guided else []) + [positive, positive])
        elif self.cfg_active() and b % 2 == 0 and not c["apply_to_uncond"]:
            u, co = ehs.chunk(2)
            new = torch.cat([u, cads_context(co, g, c, self.gen, marked=marked)])
        else:
            new = cads_context(ehs, g, c, self.gen, marked=marked)
        if len(args) > 2:
            args = tuple(args[:2]) + (new,) + tuple(args[3:])
        else:
            kwargs = dict(kwargs, encoder_hidden_states=new)
        return args, kwargs

    def install(self):
        self.handle = self.unet.register_forward_pre_hook(self.pre, with_kwargs=True)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class TilingPatch:
    """Circular padding per axis on every padded Conv2d of the given modules (asymmetric tiling)."""

    def __init__(self, modules, tile_x: bool, tile_y: bool):
        self.modules = [m for m in modules if m is not None]
        self.x, self.y = tile_x, tile_y
        self.active = True
        self.saved: list = []

    def install(self):
        import torch
        import torch.nn.functional as F
        for root in self.modules:
            for m in root.modules():
                if not isinstance(m, torch.nn.Conv2d):
                    continue
                pad = m.padding if isinstance(m.padding, tuple) else (m.padding, m.padding)
                if isinstance(m.padding, str) or pad == (0, 0):
                    continue
                orig = m._conv_forward

                def fwd(inp, weight, bias, m=m, pad=pad, orig=orig):
                    if not self.active:
                        return orig(inp, weight, bias)
                    py, px = pad
                    x = F.pad(inp, (px, px, 0, 0), mode="circular" if self.x else "constant")
                    x = F.pad(x, (0, 0, py, py), mode="circular" if self.y else "constant")
                    return F.conv2d(x, weight, bias, m.stride, 0, m.dilation, m.groups)
                m._conv_forward = fwd
                self.saved.append((m, orig))
        return self

    def remove(self):
        for m, orig in self.saved:
            m._conv_forward = orig
        self.saved = []


# ---------------------------------------------------------------------------------------------------------------
# latent callbacks (callback_on_step_end, "latents")

class LatentCallbacks:
    """One callback_on_step_end running the enabled latent edits in a fixed order."""

    def __init__(self, sdxl: bool, total_steps: int, seed: int):
        import torch
        self.sdxl = sdxl
        self.total = max(int(total_steps), 1)
        self.gen = torch.Generator("cpu").manual_seed((int(seed) + 17) & 0xFFFFFFFF)
        self.fns: list = []
        self.history: list = []
        self.prev = None

    def add(self, fn):
        self.fns.append(fn)

    def __call__(self, pipe, i, t, d):
        lat = d.get("latents")
        if lat is None:
            return d
        n = int(getattr(pipe, "_num_timesteps", 0) or self.total)
        for fn in self.fns:
            lat = fn(pipe, i, n, lat)
        d["latents"] = lat
        return d

    # --- edits -----------------------------------------------------------------
    def vectorscope(self, c: dict):
        import torch
        rgb = np.array([c["brightness"] + c["r"], c["brightness"] + c["g"], c["brightness"] + c["b"]])
        off = torch.tensor(rgb_to_latent_dir(rgb, self.sdxl), dtype=torch.float32)
        luma = torch.tensor(rgb_to_latent_dir([1.0, 1.0, 1.0], self.sdxl), dtype=torch.float32)
        luma = luma / luma.norm().clamp_min(1e-6)
        method = str(c["method"]).lower()

        def fn(pipe, i, n, lat):
            # per-step share of the total adjustment, so the result does not depend on the step count
            a = float(c["strength"]) * scaling(c["scaling"], (i + 1) / n) / n
            if a == 0:
                return lat
            x = lat.float()
            ch = x.shape[1]
            o = off[:ch].to(x.device).view(1, ch, 1, 1)
            if "random" in method:
                noise = torch.randn(x.shape, generator=self.gen).to(x.device)
                o = o * (noise.abs() if "u." in method else (1 + 0.5 * noise))
            elif method.startswith("cross"):
                o = o * x.abs().mean(dim=(2, 3), keepdim=True)
            x = x + a * 0.15 * o
            if c["contrast"]:
                mean = x.mean(dim=(2, 3), keepdim=True)
                x = mean + (x - mean) * (1 + a * 0.08 * float(c["contrast"]))
            if c["saturation"]:
                lv = luma[:ch].to(x.device).view(1, ch, 1, 1)
                along = (x * lv).sum(dim=1, keepdim=True) * lv
                x = along + (x - along) * (1 + a * 0.08 * float(c["saturation"]))
            return x.to(lat.dtype)
        self.add(fn)

    def mirroring(self, c: dict):
        import torch
        mode = str(c["mode"]).lower()
        s = float(c["strength"])

        def fn(pipe, i, n, lat):
            if (i + 1) > c["max_step"] * n:
                return lat
            m = mirror(lat, c["style"])
            if m.shape != lat.shape:
                return lat
            if mode.startswith("alternate"):
                return torch.lerp(lat, m, s) if i % 2 == 0 else lat
            return torch.lerp(lat, (lat + m) / 2, s)
        self.add(fn)

    def resharpen(self, c: dict):
        def fn(pipe, i, n, lat):
            prev, self.prev = self.prev, lat.detach().clone()
            if prev is None or prev.shape != lat.shape or i + 1 >= n:
                return lat
            k = float(c["sharpness"]) * 0.1 * scaling(c["scaling"], (i + 1) / n)
            out = lat + (lat - prev) * k
            self.prev = out.detach().clone()
            return out
        self.add(fn)

    def color_grading(self, c: dict):
        import torch
        tgt = torch.tensor(0.5 * rgb_to_latent_dir([c["r"], c["g"], c["b"]], self.sdxl), dtype=torch.float32)

        def fn(pipe, i, n, lat):
            x = lat.float()
            ch = x.shape[1]
            p = (i + 1) / n
            if c["recenter"]:
                mean = x.mean(dim=(2, 3), keepdim=True)
                want = tgt[:ch].to(x.device).view(1, ch, 1, 1)
                # luminance channel (0) keeps its mean; colour channels move towards the target
                w = torch.ones(1, ch, 1, 1, device=x.device)
                w[:, 0] = 0
                x = x - (mean - want) * w * float(c["recenter_strength"]) * (1.0 / n) * 2.0
            if c["normalize"] and i + 1 == n:
                lo = x.amin(dim=(2, 3), keepdim=True)
                hi = x.amax(dim=(2, 3), keepdim=True)
                mid = (hi + lo) / 2
                half = ((hi - lo) / 2).clamp_min(1e-6)
                target = 4.0 if not self.sdxl else 3.5
                k = float(c["normalize_strength"])
                x = mid + (x - mid) * ((1 - k) + k * (target / half).clamp(max=1.5))
            return x.to(lat.dtype)
        self.add(fn)

    def anti_burn(self, c: dict):
        import torch
        skip, avg = max(int(c["skip_last"]), 0), max(int(c["average"]), 1)

        def fn(pipe, i, n, lat):
            self.history.append(lat.detach().clone())
            self.history = self.history[-avg:]
            stop_at = n - skip
            if skip > 0 and i + 1 >= stop_at:
                pipe._interrupt = True
                out = torch.stack(self.history).mean(dim=0).to(lat.dtype)
                self.history = []
                return out
            if skip == 0 and i + 1 == n:
                return torch.stack(self.history).mean(dim=0).to(lat.dtype)
            return lat
        self.add(fn)


# ---------------------------------------------------------------------------------------------------------------

def install(engine, pipe, gp, kw: dict, ctx, cleanups: list, *, sched, cfg_scale: float, steps: int, seed: int,
            chain: Callable, composed: bool = False) -> None:
    import torch
    hooks = gp.hooks or {}
    unet = pipe.unet

    def cfg_active():
        return bool(getattr(pipe, "do_classifier_free_guidance", cfg_scale > 1.0))

    def pag_active():
        return bool(getattr(pipe, "do_perturbed_attention_guidance", False))

    ac = getattr(sched, "alphas_cumprod", None)
    if ac is None:
        from .sigmas import default_alphas_cumprod
        ac = torch.tensor(default_alphas_cumprod())

    dd = hook_config(hooks, "detail_daemon", DETAIL_DAEMON_DEFAULTS)
    if dd:
        c = normalize(dd, DETAIL_DAEMON_DEFAULTS)
        if str(c["mode"]).lower() not in ("both", "cond", "uncond"):
            c["mode"] = "both"
        cleanups.append(DetailDaemonHook(unet, sched, ac, c, cfg_active).install().remove)
    cs = hook_config(hooks, "cfg_schedule", CFG_SCHEDULE_DEFAULTS)
    if cs:
        if cfg_scale <= 1.0:
            ctx.warn("CFG schedule needs CFG scale > 1 (CFG is off); ignored")
        elif not composed:
            cleanups.append(CfgScheduleHook(unet, sched, cfg_scale, normalize(cs, CFG_SCHEDULE_DEFAULTS),
                                            cfg_active, pag_active=pag_active).install().remove)
    ca = hook_config(hooks, "cads", CADS_DEFAULTS)
    if ca and not composed:
        c = normalize(ca, CADS_DEFAULTS)
        if c["tau1"] > c["tau2"]:
            c["tau1"], c["tau2"] = c["tau2"], c["tau1"]
        cleanups.append(CadsHook(unet, sched, c, seed, cfg_active, pag_active=pag_active).install().remove)

    tl = hook_config(hooks, "tiling", TILING_DEFAULTS)
    lc = LatentCallbacks(engine.sdxl, steps, seed)
    if tl:
        c = normalize(tl, TILING_DEFAULTS)
        if c["tile_x"] or c["tile_y"]:
            tp = TilingPatch([unet, getattr(pipe, "vae", None)], c["tile_x"], c["tile_y"]).install()
            cleanups.append(tp.remove)
            start, stop = int(c["start_step"]), int(c["stop_step"])
            if start > 0 or stop >= 0:
                tp.active = start <= 0

                def tiling_steps(pipe_, i, n, lat, tp=tp, start=start, stop=stop):
                    nxt = i + 1  # 0-based index of the next step
                    tp.active = nxt >= start and (stop < 0 or nxt < stop)
                    if nxt >= n:
                        tp.active = True  # VAE decode tiles too
                    return lat
                lc.add(tiling_steps)
    vc = hook_config(hooks, "vectorscope_cc", VECTORSCOPE_DEFAULTS)
    if vc:
        lc.vectorscope(normalize(vc, VECTORSCOPE_DEFAULTS))
    lm = hook_config(hooks, "latent_mirroring", MIRRORING_DEFAULTS)
    if lm:
        lc.mirroring(normalize(lm, MIRRORING_DEFAULTS))
    rs = hook_config(hooks, "resharpen", RESHARPEN_DEFAULTS)
    if rs:
        lc.resharpen(normalize(rs, RESHARPEN_DEFAULTS))
    cg = hook_config(hooks, "color_grading", COLOR_GRADING_DEFAULTS)
    if cg:
        lc.color_grading(normalize(cg, COLOR_GRADING_DEFAULTS))
    ab = hook_config(hooks, "anti_burn", ANTI_BURN_DEFAULTS)
    if ab:
        lc.anti_burn(normalize(ab, ANTI_BURN_DEFAULTS))
    if lc.fns:
        chain(kw, lc)


__all__ = ["HOOKS", "install", "make_dd_schedule", "cads_gamma", "cads_noise", "cfg_at", "scaling", "mirror",
           "rgb_to_latent_dir", "SigmaTable", "TilingPatch", "LatentCallbacks"]
