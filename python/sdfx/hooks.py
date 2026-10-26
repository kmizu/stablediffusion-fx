"""Optional generation hooks from params.hooks:

  {"freeu": {"b1","b2","s1","s2"}}
  {"dynamic_thresholding": {"mimic_scale","threshold_percentile","mimic_mode","mimic_scale_min","cfg_mode",
                            "cfg_scale_min","sched_val","separate_feature_channels","scaling_startpoint",
                            "variability_measure","interpolate_phi"}}
  {"hypertile": {"tile_size","swap_size","max_depth","scale_depth"}}
  {"pag": {"scale","layers"}}

Any hook object may contain "enabled": false to disable it. The math helpers (interpret_scale, dynthresh,
random_divisor) work on numpy arrays and torch tensors and are unit-tested without torch.
"""
from __future__ import annotations

import math
import random
from typing import Any, Optional

KNOWN_HOOKS = ("freeu", "dynamic_thresholding", "hypertile", "pag", "apg", "mahiro", "rescale_cfg", "model_sampling",
               "kohya_hrfix", "multidiffusion", "latent_modifier", "soft_inpainting", "sag", "neveroom", "reference",
               "revision", "ip_adapter", "controllllite", "fooocus_inpaint", "photomaker", "face_restore", "img2imgalt",
               # sdfx.sampling_ext.HOOKS (extension index)
               "detail_daemon", "vectorscope_cc", "latent_mirroring", "tiling", "anti_burn", "resharpen", "color_grading",
               "cfg_schedule", "cads", "adetailer", "weight_gradient", "negpip", "neutral_prompt", "prompt_fusion", "save_intermediate", "latent_upscale", "composable_lora")

FREEU_DEFAULTS = {
    False: {"b1": 1.5, "b2": 1.6, "s1": 0.9, "s2": 0.2},  # SD1.x
    True: {"b1": 1.3, "b2": 1.4, "s1": 0.9, "s2": 0.2},   # SDXL
}

DYNTHRESH_DEFAULTS = {
    "mimic_scale": 7.0, "threshold_percentile": 1.0, "mimic_mode": "Constant", "mimic_scale_min": 0.0,
    "cfg_mode": "Constant", "cfg_scale_min": 0.0, "sched_val": 1.0, "separate_feature_channels": True,
    "scaling_startpoint": "MEAN", "variability_measure": "AD", "interpolate_phi": 1.0,
}

HYPERTILE_DEFAULTS = {"tile_size": 256, "swap_size": 2, "max_depth": 0, "scale_depth": False}

PAG_DEFAULTS = {"scale": 3.0, "adaptive_scale": 0.0, "layers": ["mid"]}


def hook_config(hooks: dict, name: str, defaults: dict) -> Optional[dict]:
    """Merged config for hook `name`, or None when absent/disabled."""
    if not hooks or name not in hooks:
        return None
    v = hooks.get(name)
    if isinstance(v, (bool, str)):
        if not coerce_bool(v, False):
            return None
        v = {}
    if v is None:
        return None
    if not isinstance(v, dict):
        return None
    if not coerce_bool(v.get("enabled", True), True):
        return None
    out = dict(defaults)
    for k, x in v.items():
        if k != "enabled":
            out[k] = x
    return out


def coerce_bool(v: Any, default: bool = False) -> bool:
    """Lenient bool: JSON bool, numbers, or "enable"/"disable"/"true"/"false"/"on"/"off"/"yes"/"no"."""
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        t = v.strip().lower()
        if t in ("enable", "enabled", "true", "on", "yes", "1"):
            return True
        if t in ("disable", "disabled", "false", "off", "no", "0", ""):
            return False
    return default


_LAYER_ALIASES = {"middle": "mid", "mid": "mid", "down": "down", "up": "up"}


def coerce_layers(v: Any, default: Optional[list] = None) -> list[str]:
    """PAG layers: JSON list or comma separated string; "middle" -> "mid"."""
    if v is None:
        items: list = []
    elif isinstance(v, str):
        items = v.split(",")
    elif isinstance(v, (list, tuple)):
        items = []
        for x in v:
            items.extend(str(x).split(",")) if isinstance(x, str) else items.append(str(x))
    else:
        items = [str(v)]
    out = []
    for x in items:
        t = str(x).strip()
        if not t:
            continue
        out.append(_LAYER_ALIASES.get(t.lower(), t))
    return out or list(default or ["mid"])


def normalize_dynthresh(cfg: dict) -> dict:
    out = {k: cfg.get(k, v) for k, v in DYNTHRESH_DEFAULTS.items()}
    out["separate_feature_channels"] = coerce_bool(out["separate_feature_channels"], True)
    for k in ("mimic_scale", "threshold_percentile", "mimic_scale_min", "cfg_scale_min", "sched_val", "interpolate_phi"):
        out[k] = float(out[k])
    out["scaling_startpoint"] = str(out["scaling_startpoint"]).upper()
    out["variability_measure"] = str(out["variability_measure"]).upper()
    return out


def normalize_hypertile(cfg: dict) -> dict:
    return {"tile_size": int(float(cfg.get("tile_size", 256))), "swap_size": int(float(cfg.get("swap_size", 2))),
            "max_depth": int(float(cfg.get("max_depth", 0))), "scale_depth": coerce_bool(cfg.get("scale_depth"), False)}


def normalize_pag(cfg: dict) -> dict:
    return {"scale": float(cfg.get("scale", 3.0)), "adaptive_scale": float(cfg.get("adaptive_scale", 0.0) or 0.0),
            "layers": coerce_layers(cfg.get("layers"), ["mid"])}


def unknown_hooks(hooks: dict) -> list[str]:
    return [k for k in (hooks or {}) if k not in KNOWN_HOOKS]


# --------------------------------------------------------------------------- dynamic thresholding math

def interpret_scale(scale: float, mode: str, min_: float, frac: float, sched_val: float = 1.0) -> float:
    """Port of sd-dynamic-thresholding DynThresh.interpret_scale with frac = step/(max_steps-1)."""
    scale -= min_
    if mode == "Constant":
        pass
    elif mode == "Linear Down":
        scale *= 1.0 - frac
    elif mode == "Half Cosine Down":
        scale *= math.cos(frac)
    elif mode == "Cosine Down":
        scale *= math.cos(frac * 1.5707)
    elif mode == "Linear Up":
        scale *= frac
    elif mode == "Half Cosine Up":
        scale *= 1.0 - math.cos(frac)
    elif mode == "Cosine Up":
        scale *= 1.0 - math.cos(frac * 1.5707)
    elif mode == "Power Up":
        scale *= math.pow(frac, sched_val)
    elif mode == "Power Down":
        scale *= 1.0 - math.pow(frac, sched_val)
    elif mode == "Linear Repeating":
        portion = (frac * sched_val) % 1.0
        scale *= (0.5 - portion) * 2 if portion < 0.5 else (portion - 0.5) * 2
    elif mode == "Cosine Repeating":
        scale *= math.cos(frac * 6.28318 * sched_val) * 0.5 + 0.5
    elif mode == "Sawtooth":
        scale *= (frac * sched_val) % 1.0
    scale += min_
    return scale


class _Np:
    def __init__(self):
        import numpy as np
        self.np = np

    def mean(self, x, axis):
        return x.mean(axis=axis, keepdims=True)

    def absmax(self, x, axis):
        return self.np.abs(x).max(axis=axis, keepdims=True) if axis is not None else self.np.abs(x).max()

    def quantile(self, x, q, axis):
        return self.np.quantile(self.np.abs(x), q, axis=axis, keepdims=True) if axis is not None \
            else self.np.quantile(self.np.abs(x), q)

    def std(self, x, axis):
        return x.std(axis=axis, ddof=1, keepdims=True) if axis is not None else x.std(ddof=1)

    def maximum(self, a, b):
        return self.np.maximum(a, b)

    def clamp(self, x, m):
        return self.np.clip(x, -m, m)


class _Torch:
    def __init__(self):
        import torch
        self.torch = torch

    def mean(self, x, axis):
        return x.mean(dim=axis, keepdim=True)

    def absmax(self, x, axis):
        return x.abs().amax(dim=axis, keepdim=True) if axis is not None else x.abs().max()

    def quantile(self, x, q, axis):
        a = x.abs()
        if a.dtype in (self.torch.float16, self.torch.bfloat16):
            a = a.float()
        return (self.torch.quantile(a, q, dim=axis, keepdim=True) if axis is not None else self.torch.quantile(a, q)).to(x.dtype)

    def std(self, x, axis):
        return x.std(dim=axis, keepdim=True) if axis is not None else x.std()

    def maximum(self, a, b):
        return self.torch.maximum(a, b) if hasattr(a, "shape") and hasattr(b, "shape") else max(a, b)

    def clamp(self, x, m):
        return self.torch.max(self.torch.min(x, m), -m) if hasattr(m, "shape") else x.clamp(-m, m)


def _ops(x):
    return _Torch() if type(x).__module__.startswith("torch") else _Np()


def dynthresh(cond, uncond, cfg_scale: float, frac: float = 0.0, **cfg: Any):
    """Dynamic thresholding (mcmonkey sd-dynamic-thresholding) for one cond per uncond.

    cond/uncond: (B, C, H, W) denoised predictions. Returns the combined (B, C, H, W) prediction.
    """
    o = dict(DYNTHRESH_DEFAULTS)
    o.update({k: v for k, v in cfg.items() if v is not None})
    ops = _ops(cond)
    sched_val = float(o["sched_val"])
    mimic_scale = interpret_scale(float(o["mimic_scale"]), str(o["mimic_mode"]), float(o["mimic_scale_min"]), frac, sched_val)
    cfg_scale = interpret_scale(float(cfg_scale), str(o["cfg_mode"]), float(o["cfg_scale_min"]), frac, sched_val)
    pct = float(o["threshold_percentile"])

    relative = cond - uncond
    mim_target = uncond + relative * mimic_scale
    cfg_target = uncond + relative * cfg_scale
    shape = tuple(mim_target.shape)
    b, c = shape[0], shape[1]
    mim_flat = mim_target.reshape(b, c, -1)
    cfg_flat = cfg_target.reshape(b, c, -1)
    mim_means = ops.mean(mim_flat, 2)
    cfg_means = ops.mean(cfg_flat, 2)
    mim_c = mim_flat - mim_means
    cfg_c = cfg_flat - cfg_means

    axis = 2 if coerce_bool(o["separate_feature_channels"], True) else None
    if str(o["variability_measure"]).upper() == "STD":
        mim_ref = ops.std(mim_c, axis)
        cfg_ref = ops.std(cfg_c, axis)
    else:
        mim_ref = ops.absmax(mim_c, axis)
        cfg_ref = ops.quantile(cfg_c, pct, axis)

    if str(o["scaling_startpoint"]).upper() == "ZERO":
        result = cfg_flat * (mim_ref / cfg_ref)
    else:
        if str(o["variability_measure"]).upper() == "STD":
            renorm = (cfg_c / cfg_ref) * mim_ref
        else:
            max_ref = ops.maximum(mim_ref, cfg_ref)
            renorm = (ops.clamp(cfg_c, max_ref) / max_ref) * mim_ref
        result = renorm + cfg_means
    res = result.reshape(shape)
    phi = float(o["interpolate_phi"])
    if phi != 1.0:
        res = res * phi + cfg_target * (1.0 - phi)
    return res


# --------------------------------------------------------------------------- hypertile math

def random_divisor(value: int, min_value: int, max_options: int = 1, rng: Optional[random.Random] = None) -> int:
    """HyperTile: number of tiles along an axis (value // a big divisor >= min_value)."""
    min_value = min(min_value, value)
    divisors = [i for i in range(min_value, value + 1) if value % i == 0]
    ns = [value // i for i in divisors[:max_options]]
    if len(ns) > 1:
        return ns[(rng or random).randint(0, len(ns) - 1)]
    return ns[0]


def hypertile_split(seq: int, latent_h: int, latent_w: int, tile_size: int, swap_size: int, max_depth: int,
                    scale_depth: bool, rng: Optional[random.Random] = None) -> Optional[tuple[int, int, int, int]]:
    """Return (nh, nw, h, w) tiling for a self-attention input with `seq` tokens, or None."""
    latent_tile = max(32, int(tile_size)) // 8
    for depth in range(int(max_depth) + 1):
        h, w = latent_h // (2 ** depth), latent_w // (2 ** depth)
        if h * w == seq and latent_h % (2 ** depth) == 0 and latent_w % (2 ** depth) == 0:
            factor = 2 ** depth if scale_depth else 1
            nh = random_divisor(h, latent_tile * factor, int(swap_size), rng)
            nw = random_divisor(w, latent_tile * factor, int(swap_size), rng)
            if nh * nw > 1:
                return nh, nw, h, w
            return None
    return None


# --------------------------------------------------------------------------- torch glue

class DynThreshHook:
    """Forward hook on the UNet that rewrites the cond half of a CFG batch so the pipeline's
    `uncond + g * (cond - uncond)` equals the dynamic-thresholding result (computed in x0 space)."""

    def __init__(self, unet, alphas_cumprod, prediction_type: str, guidance_scale: float, cfg: dict):
        self.unet = unet
        self.ac = alphas_cumprod.float().cpu()
        self.pred = prediction_type
        self.g = float(guidance_scale)
        self.cfg = cfg
        self.handle = None

    def _alpha(self, t) -> float:
        import torch
        tv = float(t.flatten()[0]) if isinstance(t, torch.Tensor) else float(t)
        tv = min(max(tv, 0.0), len(self.ac) - 1.0)
        lo, hi = int(math.floor(tv)), int(math.ceil(tv))
        w = tv - lo
        return float(self.ac[lo] * (1 - w) + self.ac[hi] * w)

    def hook(self, module, args, kwargs, output):
        import torch
        out = output[0] if isinstance(output, tuple) else getattr(output, "sample", output)
        sample = args[0] if args else kwargs.get("sample")
        t = args[1] if len(args) > 1 else kwargs.get("timestep")
        if out is None or sample is None or t is None or out.shape[0] % 2 != 0 or self.g <= 1.0:
            return None
        a = self._alpha(t)
        sa, s1a = math.sqrt(a), math.sqrt(max(1.0 - a, 1e-8))
        dt = out.dtype
        u, c = out.float().chunk(2)
        z = sample.float().chunk(2)[0]
        if self.pred == "v_prediction":
            x0u, x0c = sa * z - s1a * u, sa * z - s1a * c
        else:
            x0u, x0c = (z - s1a * u) / sa, (z - s1a * c) / sa
        frac = 1.0 - min(max(float(t.flatten()[0]) if isinstance(t, torch.Tensor) else float(t), 0.0), 999.0) / 999.0
        x0r = dynthresh(x0c, x0u, self.g, frac, **self.cfg)
        r = (sa * z - x0r) / s1a if self.pred == "v_prediction" else (z - sa * x0r) / s1a
        c_new = u + (r - u) / self.g
        new = torch.cat([u, c_new]).to(dt)
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


class HyperTileHook:
    """Splits UNet self-attention (attn1) inputs into spatial tiles (HyperTile)."""

    def __init__(self, unet, latent_h: int, latent_w: int, cfg: dict, seed: int = 0):
        self.unet = unet
        self.lh, self.lw = latent_h, latent_w
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.handles = []

    def _pre(self, module, args, kwargs):
        hs = args[0] if args else kwargs.get("hidden_states")
        module._sdfx_ht = None
        if hs is None or hs.ndim != 3:
            return None
        sp = hypertile_split(hs.shape[1], self.lh, self.lw, int(self.cfg["tile_size"]), int(self.cfg["swap_size"]),
                             int(self.cfg["max_depth"]), bool(self.cfg["scale_depth"]), self.rng)
        if sp is None:
            return None
        nh, nw, h, w = sp
        b, _, c = hs.shape
        x = hs.reshape(b, nh, h // nh, nw, w // nw, c).permute(0, 1, 3, 2, 4, 5).reshape(b * nh * nw, (h // nh) * (w // nw), c)
        module._sdfx_ht = (b, nh, nw, h, w)
        if args:
            return (x,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = x
        return args, kwargs

    def _post(self, module, args, kwargs, output):
        st = getattr(module, "_sdfx_ht", None)
        if st is None:
            return None
        module._sdfx_ht = None
        b, nh, nw, h, w = st
        c = output.shape[-1]
        y = output.reshape(b, nh, nw, h // nh, w // nw, c).permute(0, 1, 3, 2, 4, 5).reshape(b, h * w, c)
        return y

    def install(self):
        for name, mod in self.unet.named_modules():
            if name.endswith("attn1") and hasattr(mod, "to_q"):
                self.handles.append(mod.register_forward_pre_hook(self._pre, with_kwargs=True))
                self.handles.append(mod.register_forward_hook(self._post, with_kwargs=True))
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
