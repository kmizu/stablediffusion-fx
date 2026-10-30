"""Request parameter dataclasses mirroring the Scala contract (snake_case JSON keys).

Parsing is lenient: unknown keys are ignored, missing keys and nulls for non-Optional fields take
the same defaults as the Scala case classes.
"""
from __future__ import annotations

import dataclasses
import math
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ParamError(ValueError):
    """Raised for malformed parameters (mapped to error code invalid_params)."""


def _f8(v: int) -> int:
    # Scala HiresParams.targetSize: f8(v) = max(64, v / 8 * 8) on Ints (A1111 latent = size // 8)
    return max(64, int(v) // 8 * 8)


@dataclass
class LoraRef:
    name: str = ""
    weight: float = 1.0          # text encoder multiplier, and U-Net multiplier unless `unet` is given
    unet: Optional[float] = None
    lbw: Optional[str] = None    # LoRA Block Weight: preset name or comma separated block weights
    start: Optional[float] = None
    stop: Optional[float] = None


@dataclass
class HiresParams:
    enabled: bool = False
    scale: float = 2.0
    upscaler: str = "Latent"
    second_pass_steps: int = 0
    resize_x: int = 0
    resize_y: int = 0
    denoising_strength: float = 0.55
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None
    prompt: str = ""
    negative_prompt: str = ""
    cfg_scale: Optional[float] = None

    def _raw_target(self, width: int, height: int) -> tuple[int, int]:
        w, h = max(1, int(width)), max(1, int(height))
        x, y = int(self.resize_x), int(self.resize_y)
        if x == 0 and y == 0:
            return int(w * self.scale), int(h * self.scale)
        if y == 0:
            return x, x * h // w
        if x == 0:
            return y * w // h, y
        return x, y

    def target_size(self, width: int, height: int) -> tuple[int, int]:
        """Final second-pass size; identical to Scala HiresParams.targetSize (A1111 truncation semantics)."""
        tx, ty = self._raw_target(width, height)
        return _f8(tx), _f8(ty)

    def upscale_size(self, width: int, height: int) -> tuple[int, int]:
        """Size the first-pass result is upscaled to before the center crop to target_size (A1111
        hr_upscale_to_x/y). Equals target_size unless both resize_x and resize_y are set, in which case the
        image is upscaled preserving aspect ratio to cover the target and then center-cropped."""
        tw, th = self.target_size(width, height)
        if self.resize_x == 0 or self.resize_y == 0:
            return tw, th
        w, h = max(1, int(width)), max(1, int(height))
        x, y = int(self.resize_x), int(self.resize_y)
        if w / h < x / y:
            ux, uy = x, x * h // w
        else:
            ux, uy = y * w // h, y
        return max(tw, _f8(ux)), max(th, _f8(uy))


def center_crop_box(src_w: int, src_h: int, dst_w: int, dst_h: int, unit: int = 1) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of a dst-sized centered crop, offsets aligned to `unit` (A1111 truncate//2)."""
    tx, ty = (src_w - dst_w) // unit, (src_h - dst_h) // unit
    left, top = (tx // 2) * unit, (ty // 2) * unit
    return left, top, left + dst_w, top + dst_h


@dataclass
class Img2ImgParams:
    init_image: str = ""
    mask: Optional[str] = None
    denoising_strength: float = 0.75
    resize_mode: int = 0
    mask_blur: int = 4
    inpainting_fill: int = 1
    inpaint_full_res: bool = False
    inpaint_full_res_padding: int = 32
    inpainting_mask_invert: bool = False


@dataclass
class ControlNetUnit:
    enabled: bool = True
    model: str = ""
    image: Optional[str] = None
    preprocessor: str = "none"
    weight: float = 1.0
    guidance_start: float = 0.0
    guidance_end: float = 1.0
    preprocessor_args: Dict[str, float] = field(default_factory=dict)


@dataclass
class GenerationParams:
    prompt: str = ""
    negative_prompt: str = ""
    seed: int = -1
    subseed: int = -1
    subseed_strength: float = 0.0
    sampler_name: str = "DPM++ 2M"
    scheduler: str = "Automatic"
    steps: int = 20
    cfg_scale: float = 7.0
    width: int = 512
    height: int = 512
    batch_size: int = 1
    n_iter: int = 1
    clip_skip: int = 1
    eta: Optional[float] = None
    hires: HiresParams = field(default_factory=HiresParams)
    img2img: Optional[Img2ImgParams] = None
    loras: List[LoraRef] = field(default_factory=list)
    controlnet: List[ControlNetUnit] = field(default_factory=list)
    hooks: Dict[str, Any] = field(default_factory=dict)
    styles: List[str] = field(default_factory=list)
    checkpoint: Optional[str] = None
    vae: Optional[str] = None
    extra: Dict[str, str] = field(default_factory=dict)

    @property
    def is_img2img(self) -> bool:
        return self.img2img is not None

    @property
    def is_inpaint(self) -> bool:
        return self.img2img is not None and self.img2img.mask is not None

    @property
    def images_per_job(self) -> int:
        return self.batch_size * self.n_iter

    def validate(self) -> list[str]:
        """Same checks and messages as Scala GenerationParams.validate."""
        p: list[str] = []
        if self.steps < 1 or self.steps > 150:
            p.append(f"steps must be 1..150 (got {self.steps})")
        if self.width < 64 or self.width > 4096 or self.width % 8 != 0:
            p.append(f"width must be a multiple of 8 in 64..4096 (got {self.width})")
        if self.height < 64 or self.height > 4096 or self.height % 8 != 0:
            p.append(f"height must be a multiple of 8 in 64..4096 (got {self.height})")
        if self.batch_size < 1 or self.batch_size > 16:
            p.append(f"batch size must be 1..16 (got {self.batch_size})")
        if self.n_iter < 1 or self.n_iter > 100:
            p.append(f"batch count must be 1..100 (got {self.n_iter})")
        if self.cfg_scale < 0 or self.cfg_scale > 30:
            p.append(f"CFG scale must be 0..30 (got {float(self.cfg_scale)!r})")
        if self.clip_skip < 1 or self.clip_skip > 12:
            p.append(f"clip skip must be 1..12 (got {self.clip_skip})")
        if self.subseed_strength < 0 or self.subseed_strength > 1:
            p.append("variation strength must be 0..1")
        if self.img2img is not None and (self.img2img.denoising_strength < 0 or self.img2img.denoising_strength > 1):
            p.append("denoising strength must be 0..1")
        if self.hires.enabled and (self.hires.denoising_strength < 0 or self.hires.denoising_strength > 1):
            p.append("hires denoising must be 0..1")
        return p


@dataclass
class HardwareInfoOptions:
    enabled: bool = False
    remark: str = ''


@dataclass
class GenerationRequest:
    params: GenerationParams = field(default_factory=GenerationParams)
    output_dir: str = ""
    lora_paths: Dict[str, str] = field(default_factory=dict)
    embedding_paths: List[str] = field(default_factory=list)
    controlnet_paths: Dict[str, str] = field(default_factory=dict)
    upscaler_paths: Dict[str, str] = field(default_factory=dict)
    preview_every: int = 2
    preview_method: str = "approx"
    lbw_presets: Dict[str, str] = field(default_factory=dict)   # LoRA Block Weight user presets (NAME -> weights)
    system_info_fields: List[str] = field(default_factory=list)  # explicit opt-in only
    vae_recovery_dir: str = ""  # persistent recovery artifacts; never overwrite source inputs
    hardware_info: HardwareInfoOptions = field(default_factory=HardwareInfoOptions)
    hires_loras: Optional[List[LoraRef]] = None
    manual_loras: Optional[List[LoraRef]] = None
    img2img_hires_loras: Optional[List[LoraRef]] = None


@dataclass
class LoadModelRequest:
    path: str = ""
    dtype: str = "fp16"
    vram_mode: str = "gpu"
    vae_path: Optional[str] = None
    vae_tiling: bool = False


@dataclass
class UpscaleRequest:
    image_path: str = ""
    upscaler: str = "Lanczos"
    scale: float = 2.0
    output_dir: str = ""
    upscaler_paths: Dict[str, str] = field(default_factory=dict)
    hardware_info: HardwareInfoOptions = field(default_factory=HardwareInfoOptions)


@dataclass
class PreprocessRequest:
    preprocessor: str = "none"
    image_path: str = ""
    output_dir: str = ""
    args: Dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- parsing

class _Missing(Exception):
    pass


def _convert(tp: Any, value: Any, path: str) -> Any:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if tp is Any:
        return value
    if origin is typing.Union:  # Optional[X]
        if value is None:
            return None
        inner = [a for a in args if a is not type(None)]
        return _convert(inner[0], value, path)
    if value is None:
        raise _Missing()
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ParamError(f"{path}: expected object")
        return from_dict(tp, value, path)
    if origin in (list, List):
        if not isinstance(value, (list, tuple)):
            raise ParamError(f"{path}: expected array")
        return [_convert(args[0], v, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin in (dict, Dict):
        if not isinstance(value, dict):
            raise ParamError(f"{path}: expected object")
        out = {}
        for k, v in value.items():
            try:
                out[str(k)] = _convert(args[1], v, f"{path}.{k}")
            except _Missing:
                continue
        return out
    if tp is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str) and value.lower() in ("true", "false", "1", "0"):
            return value.lower() in ("true", "1")
        raise ParamError(f"{path}: expected boolean")
    if tp is int:
        if isinstance(value, bool):
            raise ParamError(f"{path}: expected integer")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                pass
        raise ParamError(f"{path}: expected integer")
    if tp is float:
        if isinstance(value, bool):
            raise ParamError(f"{path}: expected number")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                pass
        raise ParamError(f"{path}: expected number")
    if tp is str:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        raise ParamError(f"{path}: expected string")
    return value


def from_dict(cls, data: Optional[dict], path: str = "params"):
    """Build dataclass `cls` from a JSON object (lenient: unknown keys ignored, null/missing -> default)."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ParamError(f"{path}: expected object")
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        try:
            kwargs[f.name] = _convert(hints[f.name], data[f.name], f"{path}.{f.name}")
        except _Missing:
            continue
    return cls(**kwargs)


def parse_generation_request(data: Optional[dict]) -> GenerationRequest:
    data = data or {}
    if not isinstance(data, dict):
        raise ParamError("params: expected object")
    if not isinstance(data.get("params"), dict):
        raise ParamError("params.params: generation params object is required")
    return from_dict(GenerationRequest, data, "params")


# --------------------------------------------------------------------------- step helpers

def img2img_steps(steps: int, strength: float) -> tuple[int, float, int]:
    """Plain img2img (A1111 without 'fix steps'): runs ~steps*strength steps.

    Returns (num_inference_steps, effective_strength, expected_steps) for a diffusers img2img call.
    """
    steps = max(1, int(steps))
    strength = min(max(float(strength), 0.0), 1.0)
    actual = min(int(steps * strength), steps)
    if actual < 1:
        actual = 1
        strength = 1.0 / steps
    return steps, strength, actual


def hires_steps(hr_steps: int, strength: float) -> tuple[int, float, int]:
    """Hires second pass: A1111 runs exactly hr_steps steps of a schedule of length hr_steps/strength."""
    hr_steps = max(1, int(hr_steps))
    strength = min(max(float(strength), 0.0), 1.0)
    if strength <= 0:
        return hr_steps, 1.0 / hr_steps, 1
    n = int(math.ceil(hr_steps / min(strength, 0.999) - 1e-9))
    n = max(n, hr_steps)
    actual = min(int(n * strength), n)
    if actual < 1:
        actual = 1
        strength = 1.0 / n
    return n, strength, actual


def job_seeds(seed: int, subseed: int, count: int) -> list[tuple[int, int]]:
    """A1111: image k of the job uses seed+k and subseed+k."""
    return [(seed + k, subseed + k) for k in range(count)]


def resolve_seed(seed: int, rng=None) -> int:
    """-1 -> random 32-bit seed (A1111 semantics)."""
    if seed is None or int(seed) < 0:
        import random
        return (rng or random).randrange(0, 2**32 - 1)
    return int(seed)
