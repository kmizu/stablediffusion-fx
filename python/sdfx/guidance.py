"""Classifier-free-guidance replacements, ported from the reForge built-ins.

  {"apg": {"momentum","adaptive_momentum","norm_threshold","eta","guidance_limiter",
           "guidance_sigma_start","guidance_sigma_end"}}        reForge-APGIsYourCFG (ComfyUI cfg function)
  {"mahiro": {}}                                                 mahiro_reforge (ComfyUI post-cfg function)
  {"rescale_cfg": {"multiplier"}}                                reForge-RescaleCFG -> diffusers guidance_rescale

GuidanceHook is a UNet forward hook. diffusers computes `uncond + g * (cond - uncond)` from the UNet output, so
the hook replaces the cond half with `uncond + (r - uncond) / g`, where `r` is the guided noise prediction we
want. Inside the hook everything is converted to k-diffusion terms (x = z / sqrt(a), sigma = sqrt((1-a)/a),
denoised x0) so the formulas are exactly the ComfyUI ones the reForge extensions use.
"""
from __future__ import annotations

import math
from typing import Any, Optional

APG_DEFAULTS = {"momentum": 0.5, "adaptive_momentum": 0.18, "norm_threshold": 15.0, "eta": 1.0,
                "guidance_limiter": False, "guidance_sigma_start": 5.42, "guidance_sigma_end": 0.28}
MAHIRO_DEFAULTS: dict = {}
RESCALE_CFG_DEFAULTS = {"multiplier": 0.7}


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average: Any = 0

    def update(self, value):
        self.running_average = value + self.momentum * self.running_average


def project(v0, v1):
    import torch
    dtype = v0.dtype
    v1 = torch.nn.functional.normalize(v1.double(), dim=[-1, -2, -3])
    v0d = v0.double()
    par = (v0d * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    return par.to(dtype), (v0d - par).to(dtype)


def normalized_guidance(cond, uncond, scale: float, buf: Optional[MomentumBuffer], eta: float, norm_threshold: float):
    """APG (Sadat et al. 2024) on ComfyUI cfg-function inputs (x - denoised)."""
    import torch
    diff = cond - uncond
    if buf is not None:
        buf.update(diff)
        diff = buf.running_average
    if norm_threshold > 0:
        n = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
        diff = diff * torch.minimum(torch.ones_like(n), norm_threshold / n.clamp_min(1e-12))
    par, orth = project(diff, cond)
    return cond + (scale - 1) * (orth + eta * par)


def apg_momentum(momentum: float, adaptive: float, t: float) -> float:
    """Signal scale schedule of reForge APG for timestep t (0..999)."""
    s = momentum
    if adaptive > 0:
        if momentum < 0:
            s += -momentum * (adaptive ** 4) * (1000 - t)
            s = min(s, 0.0)
        else:
            s -= momentum * (adaptive ** 4) * (1000 - t)
            s = max(s, 0.0)
    return s


def mahiro(cond_x0, uncond_x0, cfg_x0, scale: float):
    """mahiro_normd from ComfyUI/reForge (post-cfg, denoised space)."""
    import torch
    import torch.nn.functional as F
    leap = cond_x0 * scale
    u_leap = uncond_x0 * scale
    merge = (leap + cfg_x0) / 2
    normu = torch.sqrt(u_leap.abs()) * u_leap.sign()
    normm = torch.sqrt(merge.abs()) * merge.sign()
    sim = F.cosine_similarity(normu, normm).mean()
    simsc = 2 * (sim + 1)
    return (simsc * cfg_x0 + (4 - simsc) * leap) / 4


class GuidanceHook:
    """Applies APG (replacing CFG) and/or Mahiro (after CFG) through the UNet output."""

    def __init__(self, unet, alphas_cumprod, prediction_type: str, guidance_scale: float,
                 apg: Optional[dict] = None, use_mahiro: bool = False, post=None):
        self.unet = unet
        self.ac = alphas_cumprod.float().cpu()
        self.pred = prediction_type
        self.g = float(guidance_scale)
        self.apg = apg
        self.use_mahiro = use_mahiro
        self.post = post  # optional fn(x0, progress) -> x0 applied to the guided prediction (latent modifier)
        self.buf = MomentumBuffer(apg["momentum"]) if apg else None
        self.handle = None

    def _alpha(self, t) -> tuple[float, float]:
        import torch
        tv = float(t.flatten()[0]) if isinstance(t, torch.Tensor) else float(t)
        tv = min(max(tv, 0.0), len(self.ac) - 1.0)
        lo, hi = int(math.floor(tv)), int(math.ceil(tv))
        w = tv - lo
        return float(self.ac[lo] * (1 - w) + self.ac[hi] * w), tv

    def _guide(self, x0c, x0u, xk, sigma: float, tv: float):
        import torch
        g = self.g
        if self.apg:
            a = self.apg
            limited = a.get("guidance_limiter") and (
                (a["guidance_sigma_start"] >= 0 and sigma > a["guidance_sigma_start"]) or
                (a["guidance_sigma_end"] >= 0 and sigma <= a["guidance_sigma_end"]))
            cond, uncond = xk - x0c, xk - x0u
            if limited:
                guided = cond
            else:
                rav = self.buf.running_average
                if (torch.is_tensor(rav) and rav.shape != cond.shape) or tv >= 999:
                    self.buf = MomentumBuffer(a["momentum"])
                else:
                    self.buf.momentum = apg_momentum(a["momentum"], a["adaptive_momentum"], tv)
                guided = normalized_guidance(cond, uncond, g, self.buf, a["eta"], a["norm_threshold"])
            x0 = xk - guided
        else:
            x0 = x0u + g * (x0c - x0u)
        if self.use_mahiro:
            x0 = mahiro(x0c, x0u, x0, g)
        if self.post is not None:
            x0 = self.post(x0, 1.0 - min(max(tv, 0.0), 999.0) / 999.0)
        return x0

    def hook(self, module, args, kwargs, output):
        import torch
        out = output[0] if isinstance(output, tuple) else getattr(output, "sample", output)
        sample = args[0] if args else kwargs.get("sample")
        t = args[1] if len(args) > 1 else kwargs.get("timestep")
        if out is None or sample is None or t is None or out.shape[0] % 2 != 0 or self.g <= 1.0:
            return None
        a, tv = self._alpha(t)
        sa, s1a = math.sqrt(a), math.sqrt(max(1.0 - a, 1e-8))
        sigma = s1a / sa
        dt = out.dtype
        u, c = out.float().chunk(2)
        z = sample.float().chunk(2)[0]
        if self.pred == "v_prediction":
            x0u, x0c = sa * z - s1a * u, sa * z - s1a * c
        else:
            x0u, x0c = (z - s1a * u) / sa, (z - s1a * c) / sa
        x0r = self._guide(x0c, x0u, z / sa, sigma, tv)
        r = (sa * z - x0r) / s1a if self.pred == "v_prediction" else (z - sa * x0r) / s1a
        new = torch.cat([u, u + (r - u) / self.g]).to(dt)
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


def normalize_apg(cfg: dict) -> dict:
    from .hooks import coerce_bool
    out = dict(APG_DEFAULTS)
    for k in ("momentum", "adaptive_momentum", "norm_threshold", "eta", "guidance_sigma_start", "guidance_sigma_end"):
        if k in cfg:
            out[k] = float(cfg[k])
    out["guidance_limiter"] = coerce_bool(cfg.get("guidance_limiter"), False)
    return out


__all__ = ["GuidanceHook", "normalized_guidance", "apg_momentum", "mahiro", "project", "MomentumBuffer",
           "normalize_apg", "APG_DEFAULTS", "MAHIRO_DEFAULTS", "RESCALE_CFG_DEFAULTS"]
