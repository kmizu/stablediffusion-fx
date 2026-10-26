"""LoRA Block Weight (hako-mikan/sd-webui-lora-block-weight), re-implemented from its documented syntax.

  <lora:name:1:lbw=MIDD>                 preset (built-in or from config/lbwpresets.txt, "NAME:v1,v2,...")
  <lora:name:1:lbw=1,0,0,0,1,1,1,1,...>  explicit block weights
  <lora:name:1:lbw=ALL:start=5:stop=0.5> LoRA (U-Net part) active only from step 5 to 50 % of the steps
  <lora:name:1:lbw=ALL:step=5-10>        same as start=5:stop=10

Block layouts (README of the extension):
  SD1.x LoRA (17): BASE IN01 IN02 IN04 IN05 IN07 IN08 MID OUT03 OUT04 OUT05 OUT06 OUT07 OUT08 OUT09 OUT10 OUT11
  SD1.x full (26): BASE IN00..IN11 MID OUT00..OUT11
  SDXL LoRA (12):  BASE IN04 IN05 IN07 IN08 MID OUT00 OUT01 OUT02 OUT03 OUT04 OUT05
  SDXL full (20):  BASE IN00..IN08 MID OUT00..OUT08
Blocks that are not given get weight 0 (as in the extension). BASE scales the text encoder(s).

The weights are turned into diffusers `set_adapters` block scales: attention layers of each U-Net block get their
own scale, blocks without attention get the mean of their entries (only LoCon/LyCORIS layers live there).
"""
from __future__ import annotations

from typing import Optional

SD15_17 = ["BASE", "IN01", "IN02", "IN04", "IN05", "IN07", "IN08", "MID",
           "OUT03", "OUT04", "OUT05", "OUT06", "OUT07", "OUT08", "OUT09", "OUT10", "OUT11"]
SD15_26 = ["BASE"] + [f"IN{i:02d}" for i in range(12)] + ["MID"] + [f"OUT{i:02d}" for i in range(12)]
SDXL_12 = ["BASE", "IN04", "IN05", "IN07", "IN08", "MID", "OUT00", "OUT01", "OUT02", "OUT03", "OUT04", "OUT05"]
SDXL_20 = ["BASE"] + [f"IN{i:02d}" for i in range(9)] + ["MID"] + [f"OUT{i:02d}" for i in range(9)]

_IN_ALL = {f"IN{i:02d}" for i in range(12)}
_OUT_ALL = {f"OUT{i:02d}" for i in range(12)}
# presets as block sets (+BASE), so one definition serves every layout
PRESET_BLOCKS = {
    "NONE": set(), "NOT": set(),
    "ALL": {"BASE", "MID"} | _IN_ALL | _OUT_ALL,
    "INS": {"BASE", "IN00", "IN01", "IN02", "IN03"},               # shallow input blocks
    "IND": {"BASE", "IN04", "IN05", "IN06", "IN07", "IN08"},       # deep input blocks
    "INALL": {"BASE"} | _IN_ALL,
    "MIDD": {"BASE", "IN04", "IN05", "IN06", "IN07", "IN08", "IN09", "IN10", "IN11", "MID",
             "OUT00", "OUT01", "OUT02", "OUT03", "OUT04", "OUT05"},   # the deep middle of the U-Net
    "OUTD": {"BASE", "OUT00", "OUT01", "OUT02", "OUT03", "OUT04", "OUT05"},
    "OUTS": {"BASE", "OUT06", "OUT07", "OUT08", "OUT09", "OUT10", "OUT11"},
    "OUTALL": {"BASE"} | _OUT_ALL,
}
PRESET_SCALE = {"ALL0.5": ("ALL", 0.5)}


def layout(sdxl: bool, n: int) -> list[str]:
    if sdxl:
        return SDXL_20 if n == 20 else SDXL_12
    return SD15_26 if n == 26 else SD15_17


def parse_weights(spec: str, sdxl: bool, presets: Optional[dict] = None) -> dict[str, float]:
    """Block name -> weight for a lbw= value (preset name or comma separated numbers)."""
    s = (spec or "").strip()
    user = {k.strip().upper(): v for k, v in (presets or {}).items()}
    if s.upper() in user:
        s = str(user[s.upper()]).strip()
    key = s.upper()
    if key in PRESET_SCALE:
        base, k = PRESET_SCALE[key]
        names = PRESET_BLOCKS[base]
        return {b: (k if b in names else 0.0) for b in layout(sdxl, 26 if not sdxl else 20)}
    if key in PRESET_BLOCKS:
        names = PRESET_BLOCKS[key]
        return {b: (1.0 if b in names else 0.0) for b in layout(sdxl, 26 if not sdxl else 20)}
    try:
        vals = [float(v) for v in s.replace(" ", "").split(",") if v != ""]
    except ValueError:
        raise ValueError(f"lbw: unknown preset or bad weights '{spec}'")
    names = layout(sdxl, len(vals))
    # other lengths are accepted like in the extension: missing blocks are 0, extra values are ignored
    vals = (vals + [0.0] * len(names))[:len(names)]
    return dict(zip(names, vals))


def _get(w: dict, name: str) -> float:
    return float(w.get(name, 0.0))


def _mean(w: dict, names) -> float:
    vals = [_get(w, n) for n in names if n in w]
    return sum(vals) / len(vals) if vals else 0.0


def diffusers_scales(w: dict[str, float], sdxl: bool, unet_weight: float, te_weight: float) -> dict:
    """set_adapters weight dict for one adapter."""
    g = _get
    if sdxl:
        unet = {
            "down": {"block_0": unet_weight * _mean(w, ["IN00", "IN01", "IN02", "IN03"]),
                     "block_1": [unet_weight * g(w, "IN04"), unet_weight * g(w, "IN05")],
                     "block_2": [unet_weight * g(w, "IN07"), unet_weight * g(w, "IN08")]},
            "mid": unet_weight * g(w, "MID"),
            "up": {"block_0": [unet_weight * g(w, "OUT00"), unet_weight * g(w, "OUT01"), unet_weight * g(w, "OUT02")],
                   "block_1": [unet_weight * g(w, "OUT03"), unet_weight * g(w, "OUT04"), unet_weight * g(w, "OUT05")],
                   "block_2": unet_weight * _mean(w, ["OUT06", "OUT07", "OUT08"])},
        }
        te = te_weight * g(w, "BASE")
        return {"unet": unet, "text_encoder": te, "text_encoder_2": te}
    unet = {
        "down": {"block_0": [unet_weight * g(w, "IN01"), unet_weight * g(w, "IN02")],
                 "block_1": [unet_weight * g(w, "IN04"), unet_weight * g(w, "IN05")],
                 "block_2": [unet_weight * g(w, "IN07"), unet_weight * g(w, "IN08")],
                 "block_3": unet_weight * _mean(w, ["IN10", "IN11"])},
        "mid": unet_weight * g(w, "MID"),
        "up": {"block_0": unet_weight * _mean(w, ["OUT00", "OUT01", "OUT02"]),
               "block_1": [unet_weight * g(w, "OUT03"), unet_weight * g(w, "OUT04"), unet_weight * g(w, "OUT05")],
               "block_2": [unet_weight * g(w, "OUT06"), unet_weight * g(w, "OUT07"), unet_weight * g(w, "OUT08")],
               "block_3": [unet_weight * g(w, "OUT09"), unet_weight * g(w, "OUT10"), unet_weight * g(w, "OUT11")]},
    }
    return {"unet": unet, "text_encoder": te_weight * g(w, "BASE")}


def plain_scales(sdxl: bool, unet_weight: float, te_weight: float) -> dict:
    d = {"unet": unet_weight, "text_encoder": te_weight}
    if sdxl:
        d["text_encoder_2"] = te_weight
    return d


def zero_unet(scales, sdxl: bool):
    """Same adapter scales with the U-Net part switched off (start/stop outside the active window)."""
    if isinstance(scales, dict):
        return dict(scales, unet=0.0)
    return plain_scales(sdxl, 0.0, float(scales))


def step_of(v: Optional[float], total: int) -> Optional[int]:
    """start/stop value -> step number: 0 < v < 1 is a fraction of the steps, otherwise a step number."""
    if v is None:
        return None
    v = float(v)
    if 0 < v < 1:
        return int(round(v * total))
    return int(v)


def active(step: int, total: int, start: Optional[float], stop: Optional[float]) -> bool:
    """Whether the U-Net part is on for 1-based `step`: start <= step < stop (stop=N switches off at step N)."""
    s, e = step_of(start, total), step_of(stop, total)
    if s is not None and step < s:
        return False
    if e is not None and step >= e:
        return False
    return True


__all__ = ["parse_weights", "diffusers_scales", "plain_scales", "zero_unet", "active", "step_of", "layout",
           "PRESET_BLOCKS", "SD15_17", "SDXL_12"]
