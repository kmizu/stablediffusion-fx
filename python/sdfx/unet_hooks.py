"""UNet structure hooks ported from reForge built-ins.

  {"kohya_hrfix": {"block_number","downscale_factor","start_percent","end_percent","downscale_after_skip",
                   "downscale_method","upscale_method"}}         sd_forge_kohya_hrfix (Deep Shrink)
  {"multidiffusion": {"method","tile_width","tile_height","tile_overlap","tile_batch_size"}}
                                                                   sd_forge_multidiffusion (tiled diffusion)

Pure helpers (block mapping, tile layout, blend weights) have no torch dependency and are unit tested.
"""
from __future__ import annotations

import math
from typing import Optional

KOHYA_DEFAULTS = {"block_number": 3, "downscale_factor": 2.0, "start_percent": 0.0, "end_percent": 0.35,
                  "downscale_after_skip": True, "downscale_method": "bicubic", "upscale_method": "bicubic"}
MULTIDIFFUSION_DEFAULTS = {"method": "MultiDiffusion", "tile_width": 768, "tile_height": 768, "tile_overlap": 64,
                           "tile_batch_size": 4}
INTERP_METHODS = ("bicubic", "nearest-exact", "bilinear", "area", "bislerp")


def kohya_block_location(block_number: int, layers_per_level: list[int]) -> Optional[tuple[int, int]]:
    """Maps a ComfyUI/ldm input block number (1-based, 0 = conv_in) to (level, index) of the diffusers down path.

    Each level has `layers` resnet(/attention) layers followed by a downsampler (index == layers) except the last.
    Returns None when the number is out of range.
    """
    n = block_number
    if n < 1:
        return None
    for level, layers in enumerate(layers_per_level):
        last = level == len(layers_per_level) - 1
        span = layers + (0 if last else 1)
        if n <= span:
            return level, n - 1
        n -= span
    return None


def timestep_window(start_percent: float, end_percent: float) -> tuple[float, float]:
    """(t_hi, t_lo): the hook is active while t_lo <= t <= t_hi (t in 0..999, 999 = pure noise)."""
    return 999.0 * (1.0 - start_percent), 999.0 * (1.0 - end_percent)


def tile_starts(size: int, tile: int, overlap: int) -> list[int]:
    """Start offsets covering [0, size) with tiles of `tile` and at least `overlap` overlap; last tile flush."""
    if tile >= size:
        return [0]
    stride = max(1, tile - overlap)
    n = math.ceil((size - tile) / stride) + 1
    if n == 1:
        return [0]
    step = (size - tile) / (n - 1)
    return sorted({int(round(i * step)) for i in range(n)})


def tile_grid(h: int, w: int, th: int, tw: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """List of (y0, y1, x0, x1) tiles in latent units."""
    th, tw = min(th, h), min(tw, w)
    return [(y, y + th, x, x + tw) for y in tile_starts(h, th, overlap) for x in tile_starts(w, tw, overlap)]


def gaussian_weights(th: int, tw: int):
    """Mixture-of-Diffusers style weights (list of rows), peak 1 in the centre."""
    var = 0.01
    def g(n):
        mid = (n - 1) / 2
        return [math.exp(-((i - mid) / max(n, 1)) ** 2 / (2 * var)) / math.sqrt(2 * math.pi * var) for i in range(n)]
    gy, gx = g(th), g(tw)
    peak = gy[len(gy) // 2] * gx[len(gx) // 2]
    # floor keeps corners (covered only by tile edges) numerically stable
    return [[max(a * b, peak * 1e-3) for b in gx] for a in gy]


def _interp(x, size, method: str):
    import torch.nn.functional as F
    if method == "bislerp":
        method = "bicubic"
    if method in ("bilinear", "bicubic"):
        return F.interpolate(x.float(), size=size, mode=method, align_corners=False).to(x.dtype)
    return F.interpolate(x.float(), size=size, mode=method).to(x.dtype)


class DeepShrinkHook:
    """Kohya "Deep Shrink" high-res fix on a diffusers UNet2DConditionModel."""

    def __init__(self, unet, cfg: dict):
        self.unet = unet
        self.cfg = cfg
        self.t: float = 999.0
        self.handles = []
        self.t_hi, self.t_lo = timestep_window(float(cfg["start_percent"]), float(cfg["end_percent"]))

    def active(self) -> bool:
        return self.t_lo <= self.t <= self.t_hi

    def _unet_pre(self, module, args, kwargs):
        import torch
        t = args[1] if len(args) > 1 else kwargs.get("timestep")
        self.t = float(t.flatten()[0]) if isinstance(t, torch.Tensor) else float(t)

    def _shrink(self, x):
        f = float(self.cfg["downscale_factor"])
        if not self.active() or f == 1.0 or x.ndim != 4:
            return x
        h, w = max(1, round(x.shape[-2] / f)), max(1, round(x.shape[-1] / f))
        return _interp(x, (h, w), self.cfg["downscale_method"])

    def _post_shrink(self, module, args, output):
        if isinstance(output, tuple):
            return (self._shrink(output[0]),) + tuple(output[1:])
        if hasattr(output, "sample"):
            output.sample = self._shrink(output.sample)
            return output
        return self._shrink(output)

    def _pre_shrink(self, module, args, kwargs):
        if args:
            return (self._shrink(args[0]),) + tuple(args[1:]), kwargs
        if "hidden_states" in kwargs:
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = self._shrink(kwargs["hidden_states"])
        elif "input_tensor" in kwargs:
            kwargs = dict(kwargs)
            kwargs["input_tensor"] = self._shrink(kwargs["input_tensor"])
        return args, kwargs

    def _up_pre(self, module, args, kwargs):
        """Resize hidden state and skip tensors of an up block to the largest skip resolution."""
        hs = args[0] if args else kwargs.get("hidden_states")
        res = args[1] if len(args) > 1 else kwargs.get("res_hidden_states_tuple")
        if hs is None or not res:
            return None
        size = max((r.shape[-2:] for r in res), key=lambda s: s[0] * s[1])
        meth = self.cfg["upscale_method"]
        changed = False
        new_res = []
        for r in res:
            if tuple(r.shape[-2:]) != tuple(size):
                r = _interp(r, tuple(size), meth)
                changed = True
            new_res.append(r)
        if tuple(hs.shape[-2:]) != tuple(size):
            hs = _interp(hs, tuple(size), meth)
            changed = True
        if not changed:
            return None
        kwargs = dict(kwargs)
        if args:
            args = (hs, tuple(new_res)) + tuple(args[2:])
        else:
            kwargs["hidden_states"] = hs
            kwargs["res_hidden_states_tuple"] = tuple(new_res)
        # the upsampler inside the block must produce the next level's size, not 2x of a shrunk map
        kwargs.pop("upsample_size", None)
        return args, kwargs

    def install(self):
        u = self.unet
        layers = [len(b.resnets) for b in u.down_blocks]
        loc = kohya_block_location(int(self.cfg["block_number"]), layers)
        if loc is None:
            raise ValueError(f"block number {self.cfg['block_number']} is out of range for this UNet (max "
                             f"{sum(layers) + len(layers) - 1})")
        level, idx = loc
        block = u.down_blocks[level]
        after_skip = bool(self.cfg["downscale_after_skip"])
        n_layers = len(block.resnets)
        if idx < n_layers:
            attn = getattr(block, "attentions", None)
            target = attn[idx] if attn is not None and len(attn) > idx else block.resnets[idx]
            nxt = None
            if after_skip:
                if idx + 1 < n_layers:
                    nxt = block.resnets[idx + 1]
                elif getattr(block, "downsamplers", None):
                    nxt = block.downsamplers[0]
        else:
            target = block.downsamplers[0]
            nxt = None
            if after_skip:
                nxt = u.down_blocks[level + 1].resnets[0] if level + 1 < len(u.down_blocks) else u.mid_block
        self.handles.append(u.register_forward_pre_hook(self._unet_pre, with_kwargs=True))
        if after_skip and nxt is not None:
            self.handles.append(nxt.register_forward_pre_hook(self._pre_shrink, with_kwargs=True))
        else:
            self.handles.append(target.register_forward_hook(self._post_shrink))
        for ub in u.up_blocks:
            self.handles.append(ub.register_forward_pre_hook(self._up_pre, with_kwargs=True))
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class MultiDiffusionWrapper:
    """Runs the UNet over overlapping latent tiles and blends the predictions (MultiDiffusion / Mixture of
    Diffusers). ControlNet residuals are sliced per tile."""

    def __init__(self, unet, cfg: dict):
        self.unet = unet
        self.cfg = cfg
        self.orig = None
        self._missing = object()
        self._previous = self._missing

    def _call(self, sample, timestep, encoder_hidden_states, *args, **kwargs):
        import torch
        th = max(8, int(self.cfg["tile_height"]) // 8)
        tw = max(8, int(self.cfg["tile_width"]) // 8)
        ov = max(0, int(self.cfg["tile_overlap"]) // 8)
        H, W = sample.shape[-2:]
        if H <= th and W <= tw:
            return self.orig(sample, timestep, encoder_hidden_states, *args, **kwargs)
        tiles = tile_grid(H, W, th, tw, ov)
        gauss = self.cfg["method"].lower().startswith("mixture")
        acc = torch.zeros_like(sample, dtype=torch.float32)
        wsum = torch.zeros((1, 1, H, W), device=sample.device, dtype=torch.float32)
        down = kwargs.get("down_block_additional_residuals")
        mid = kwargs.get("mid_block_additional_residual")
        return_dict = kwargs.pop("return_dict", True)
        for (y0, y1, x0, x1) in tiles:
            kw = dict(kwargs)
            if down is not None:
                kw["down_block_additional_residuals"] = tuple(_slice_feat(r, H, y0, y1, x0, x1) for r in down)
            if mid is not None:
                kw["mid_block_additional_residual"] = _slice_feat(mid, H, y0, y1, x0, x1)
            out = self.orig(sample[..., y0:y1, x0:x1], timestep, encoder_hidden_states, *args, return_dict=False, **kw)[0]
            if gauss:
                wt = torch.tensor(gaussian_weights(y1 - y0, x1 - x0), device=sample.device, dtype=torch.float32)[None, None]
            else:
                wt = torch.ones((1, 1, y1 - y0, x1 - x0), device=sample.device, dtype=torch.float32)
            acc[..., y0:y1, x0:x1] += out.float() * wt
            wsum[..., y0:y1, x0:x1] += wt
        res = (acc / wsum.clamp_min(1e-12)).to(sample.dtype)
        if not return_dict:
            return (res,)
        from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput  # type: ignore
        return UNet2DConditionOutput(sample=res)

    def install(self):
        if self.orig is not None:
            raise RuntimeError('MultiDiffusion wrapper already installed')
        self._previous = self.unet.__dict__.get('forward', self._missing)
        self.orig = self.unet.forward
        self.unet.forward = self._call
        return self

    def remove(self):
        if self.orig is not None:
            if self._previous is self._missing:
                del self.unet.forward
            else:
                self.unet.forward = self._previous
            self.orig = None
            self._previous = self._missing


def _slice_feat(r, latent_h: int, y0: int, y1: int, x0: int, x1: int):
    s = r.shape[-2] / latent_h
    return r[..., int(y0 * s):max(int(y0 * s) + 1, int(round(y1 * s))), int(x0 * s):max(int(x0 * s) + 1, int(round(x1 * s)))]


def normalize_kohya(cfg: dict) -> dict:
    from .hooks import coerce_bool
    out = dict(KOHYA_DEFAULTS)
    out.update({k: cfg[k] for k in KOHYA_DEFAULTS if k in cfg})
    out["block_number"] = int(out["block_number"])
    for k in ("downscale_factor", "start_percent", "end_percent"):
        out[k] = float(out[k])
    out["downscale_after_skip"] = coerce_bool(out["downscale_after_skip"], True)
    for k in ("downscale_method", "upscale_method"):
        if out[k] not in INTERP_METHODS:
            out[k] = "bicubic"
    return out


def normalize_multidiffusion(cfg: dict) -> dict:
    out = dict(MULTIDIFFUSION_DEFAULTS)
    out.update({k: cfg[k] for k in MULTIDIFFUSION_DEFAULTS if k in cfg})
    for k in ("tile_width", "tile_height", "tile_overlap", "tile_batch_size"):
        out[k] = int(float(out[k]))
    out["method"] = str(out["method"])
    return out
