"""Checkpoint merging and conversion (engine requests `merge_models`, `convert_model`).

Re-implements the behaviour of the index's merger extensions (SuperMerger, Merge Block Weighted, Merge Board, Model
Mixer, Untitled Merger: weighted sum / add difference, optionally per U-Net block) and converters (Model Converter,
Model toolkit: precision, EMA handling, junk-key pruning, CLIP position_ids fix, VAE baking). Tensors are processed
one at a time from safetensors files, so peak memory is about the size of the output model.

Block names (LDM key layout):
  SD1.x: IN00..IN11 = input_blocks.0..11, M00 = middle_block, OUT00..OUT11 = output_blocks.0..11
  SDXL:  IN00..IN08, M00, OUT00..OUT08
  BASE:  the text encoder(s) and the U-Net parts outside the blocks (time / label embeddings, final layer)
Block weight lists: SD1.x 25 values (IN00..OUT11) or 26 (BASE first); SDXL 19 or 20 (BASE first).
"""
from __future__ import annotations

import json
import os
import re
from typing import Callable, Optional

MODES = ("Weight sum", "Add difference", "Sum twice", "Triple sum")
_UNET = "model.diffusion_model."
_IN = re.compile(r"^model\.diffusion_model\.input_blocks\.(\d+)\.")
_OUT = re.compile(r"^model\.diffusion_model\.output_blocks\.(\d+)\.")


def is_sdxl_keys(keys) -> bool:
    return any(k.startswith("conditioner.") for k in keys)


def block_names(sdxl: bool) -> list[str]:
    n = 9 if sdxl else 12
    return [f"IN{i:02d}" for i in range(n)] + ["M00"] + [f"OUT{i:02d}" for i in range(n)]


def block_of(key: str) -> Optional[str]:
    """Block name of a state-dict key, "BASE" for text encoder / non-block U-Net keys, None for the VAE."""
    if key.startswith("first_stage_model."):
        return None
    m = _IN.match(key)
    if m:
        return f"IN{int(m.group(1)):02d}"
    if key.startswith(_UNET + "middle_block."):
        return "M00"
    m = _OUT.match(key)
    if m:
        return f"OUT{int(m.group(1)):02d}"
    return "BASE"


PRESETS = ("GRAD_V", "GRAD_A", "FLAT_25", "FLAT_50", "FLAT_75", "IN_ONLY", "OUT_ONLY", "MID_ONLY")


def preset_weights(name: str, sdxl: bool) -> list[float]:
    """Block alphas of a named shape: GRAD_V = 1 at both ends and 0 in the middle, GRAD_A the opposite, ..."""
    n = 9 if sdxl else 12
    idx = range(2 * n + 1)
    name = name.upper()
    if name == "GRAD_V":
        return [round(abs(i - n) / n, 4) for i in idx]
    if name == "GRAD_A":
        return [round(1 - abs(i - n) / n, 4) for i in idx]
    if name.startswith("FLAT_"):
        return [int(name[5:]) / 100.0 for _ in idx]
    if name == "IN_ONLY":
        return [1.0 if i < n else 0.0 for i in idx]
    if name == "OUT_ONLY":
        return [1.0 if i > n else 0.0 for i in idx]
    if name == "MID_ONLY":
        return [1.0 if i == n else 0.0 for i in idx]
    raise ValueError(f"unknown block preset '{name}' (one of {', '.join(PRESETS)})")


def parse_block_weights(text: str, sdxl: bool, base: float) -> dict[str, float]:
    names = block_names(sdxl)
    t = (text or "").strip()
    if t.lower().startswith("preset:"):
        return dict(zip(names, preset_weights(t[7:].strip(), sdxl)), BASE=base)
    vals = [float(v) for v in re.split(r"[,\s]+", t) if v != ""]
    if len(vals) == len(names) + 1:
        base, vals = vals[0], vals[1:]
    if len(vals) != len(names):
        raise ValueError(f"block weights need {len(names)} values ({len(names) + 1} with BASE first) for "
                         f"{'SDXL' if sdxl else 'SD1.x'}, got {len(vals)}")
    out = dict(zip(names, vals))
    out["BASE"] = base
    return out


def _open(path: str):
    if not path.lower().endswith(".safetensors"):
        raise ValueError(f"only .safetensors checkpoints can be merged / converted directly: {os.path.basename(path)}")
    from safetensors import safe_open
    return safe_open(path, framework="pt", device="cpu")


def _dtype(name: str):
    import torch
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get((name or "fp16").lower())


def merge(a: str, b: str, out: str, *, c: Optional[str] = None, mode: str = "Weight sum", alpha: float = 0.5,
          beta: float = 0.0, block_weights: str = "", vae_source: str = "merge", precision: str = "fp16",
          progress: Callable[[int, int], None] = lambda i, n: None, check: Callable[[], None] = lambda: None) -> dict:
    """Merges checkpoints A, B (and C) into `out` (.safetensors). Returns a summary."""
    import torch
    from safetensors.torch import save_file
    if mode not in MODES:
        raise ValueError(f"unknown merge mode '{mode}' (one of {', '.join(MODES)})")
    if mode != "Weight sum" and not c:
        raise ValueError(f"'{mode}' needs model C")
    fa, fb = _open(a), _open(b)
    fc = _open(c) if c and mode != "Weight sum" else None
    keys = list(fa.keys())
    kb = set(fb.keys())
    kc = set(fc.keys()) if fc else set()
    sdxl = is_sdxl_keys(keys)
    if is_sdxl_keys(kb) != sdxl:
        raise ValueError("model A and model B are different architectures (SD1.x vs SDXL)")
    bw = parse_block_weights(block_weights, sdxl, alpha) if (block_weights or "").strip() else None
    dt = _dtype(precision)
    res, skipped = {}, 0
    for i, k in enumerate(keys):
        if i % 64 == 0:
            check()
            progress(i, len(keys))
        ta = fa.get_tensor(k)
        if not ta.is_floating_point() or k not in kb:
            res[k] = ta
            skipped += k not in kb
            continue
        blk = block_of(k)
        if blk is None and vae_source in ("A", "B"):
            res[k] = (ta if vae_source == "A" else fb.get_tensor(k)).to(dt or ta.dtype)
            continue
        w = alpha if bw is None or blk is None else bw.get(blk, alpha)
        x, y = ta.float(), fb.get_tensor(k).float()
        if x.shape != y.shape:
            # inpainting / instruct-pix2pix conv_in: merge the channels both models have (A1111 behaviour)
            if x.dim() == 4 and x.shape[0] == y.shape[0] and x.shape[2:] == y.shape[2:]:
                n = min(x.shape[1], y.shape[1])
                y2 = x.clone()
                y2[:, :n] = y[:, :n]
                y = y2
            else:
                res[k] = ta
                skipped += 1
                continue
        if mode == "Weight sum":
            r = x * (1 - w) + y * w
        else:
            z = fc.get_tensor(k).float() if k in kc else None
            if z is not None and z.shape != x.shape:
                z = None
            if mode == "Add difference":
                r = x + (y - z) * w if z is not None else x
            elif mode == "Sum twice":      # (A*(1-a) + B*a) * (1-b) + C*b
                r = x * (1 - w) + y * w
                r = r * (1 - beta) + z * beta if z is not None else r
            else:                          # Triple sum: A*(1-a-b) + B*a + C*b
                r = x * (1 - w - beta) + y * w + z * beta if z is not None else x * (1 - w) + y * w
        res[k] = r.to(dt or ta.dtype)
    recipe = {"type": "sdfx-merge", "mode": mode, "alpha": alpha, "beta": beta, "block_weights": block_weights or None,
              "vae_source": vae_source, "precision": precision,
              "model_a": os.path.basename(a), "model_b": os.path.basename(b), "model_c": os.path.basename(c) if c else None}
    tmp = out + ".part"
    save_file(res, tmp, metadata={"sd_merge_recipe": json.dumps(recipe), "format": "pt"})
    os.replace(tmp, out)
    progress(len(keys), len(keys))
    return {"path": out, "keys": len(keys), "skipped": skipped, "sdxl": sdxl, "recipe": recipe}


_JUNK_PREFIXES = ("optimizer", "model_ema.", "lr_scheduler", "global_step", "epoch", "loss")
_POSITION_IDS = re.compile(r"(^|\.)position_ids$")


def _ema_key(k: str) -> str:
    return "model_ema." + ("diffusion_model" + k[len(_UNET):]).replace(".", "") if k.startswith(_UNET) else ""


def convert(src: str, out: str, *, precision: str = "fp16", ema: str = "no-ema", fix_clip: bool = True,
            drop_junk: bool = True, bake_vae: Optional[str] = None, progress: Callable[[int, int], None] = lambda i, n: None,
            check: Callable[[], None] = lambda: None) -> dict:
    """Converts a checkpoint: precision, EMA (keep / no-ema / ema-only), junk keys, CLIP position_ids, baked VAE."""
    import torch
    from safetensors.torch import save_file
    f = _open(src)
    keys = list(f.keys())
    kset = set(keys)
    dt = _dtype(precision) if precision != "keep" else None
    vae = None
    if bake_vae:
        if bake_vae.lower().endswith(".safetensors"):
            from safetensors.torch import load_file
            vae = load_file(bake_vae, device="cpu")
        else:
            vae = torch.load(bake_vae, map_location="cpu", weights_only=True)
            vae = vae.get("state_dict", vae)
        vae = {k[len("first_stage_model."):] if k.startswith("first_stage_model.") else k: v for k, v in vae.items()
               if not k.startswith(("loss", "model_ema"))}
    res, dropped, ema_used, fixed = {}, 0, 0, 0
    for i, k in enumerate(keys):
        if i % 64 == 0:
            check()
            progress(i, len(keys))
        if k.startswith("model_ema."):
            if ema == "keep":
                t = f.get_tensor(k)
                res[k] = t.to(dt) if dt is not None and t.is_floating_point() else t
            else:
                dropped += 1
            continue
        if drop_junk and k.startswith(_JUNK_PREFIXES):
            dropped += 1
            continue
        t = f.get_tensor(k)
        if ema == "ema-only" and k.startswith(_UNET):
            ek = _ema_key(k)
            if ek in kset:
                t = f.get_tensor(ek)
                ema_used += 1
        if fix_clip and _POSITION_IDS.search(k):
            n = t.shape[-1]
            t = torch.arange(n, dtype=torch.int64).reshape(t.shape)
            fixed += 1
        elif vae is not None and k.startswith("first_stage_model."):
            v = vae.get(k[len("first_stage_model."):])
            if v is not None and v.shape == t.shape:
                t = v
        if dt is not None and t.is_floating_point():
            t = t.to(dt)
        res[k] = t.contiguous()
    if ema == "ema-only" and ema_used == 0:
        raise ValueError("ema-only: this checkpoint has no EMA weights")
    tmp = out + ".part"
    save_file(res, tmp, metadata={"format": "pt", "sdfx_convert": json.dumps(
        {"source": os.path.basename(src), "precision": precision, "ema": ema, "fix_clip": fix_clip,
         "drop_junk": drop_junk, "bake_vae": os.path.basename(bake_vae) if bake_vae else None})})
    os.replace(tmp, out)
    progress(len(keys), len(keys))
    return {"path": out, "keys": len(res), "dropped": dropped, "ema_weights_used": ema_used, "clip_fixed": fixed,
            "vae_baked": bool(vae)}


__all__ = ["merge", "convert", "block_of", "block_names", "parse_block_weights", "MODES"]
