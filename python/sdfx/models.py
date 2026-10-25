"""Model/file scanning, checkpoint kind detection from the safetensors JSON header, and A1111 short hashes.

Nothing here imports torch.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
from pathlib import Path
from typing import Iterable, Optional

CHECKPOINT_EXTS = (".safetensors", ".ckpt")
FILE_EXTS = (".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".patch")
FILE_KINDS = ("vae", "lora", "controlnet", "embeddings", "upscalers", "face_restore", "svd", "zero123", "photomaker")
# diffusers-format model folders are listed as one entry when they contain this file
DIR_MARKERS = {"controlnet": "config.json", "svd": "model_index.json", "zero123": "model_index.json"}

MAX_HEADER = 100 * 1024 * 1024


# --------------------------------------------------------------------------- safetensors header

def read_safetensors_header(path: str | os.PathLike) -> dict:
    """Return the JSON header of a .safetensors file (tensor name -> {dtype, shape, data_offsets})."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("file too short for safetensors")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > MAX_HEADER:
            raise ValueError(f"implausible safetensors header length {n}")
        data = f.read(n)
        if len(data) != n:
            raise ValueError("truncated safetensors header")
    hdr = json.loads(data.decode("utf-8"))
    if not isinstance(hdr, dict):
        raise ValueError("safetensors header is not an object")
    return hdr


def _shape(hdr: dict, key: str) -> Optional[list]:
    v = hdr.get(key)
    if isinstance(v, dict):
        s = v.get("shape")
        if isinstance(s, list):
            return s
    return None


def detect_kind_from_header(hdr: dict) -> str:
    """Classify a checkpoint: sd15, sd15-inpaint, sdxl, sdxl-inpaint, sd2, unsupported, unknown."""
    keys = [k for k in hdr.keys() if k != "__metadata__"]
    ks = set(keys)
    if not keys:
        return "unknown"

    def any_prefix(p: str) -> bool:
        return any(k.startswith(p) for k in keys)

    # Architectures we do not support
    if any_prefix("double_blocks.") or any_prefix("model.diffusion_model.double_blocks.") \
            or any_prefix("transformer.transformer_blocks.") and any_prefix("transformer.single_transformer_blocks."):
        return "unsupported"  # flux
    if any_prefix("model.diffusion_model.joint_blocks.") or any_prefix("joint_blocks."):
        return "unsupported"  # sd3
    if any_prefix("model.diffusion_model.blocks.") or any_prefix("model.diffusion_model.transformer_blocks."):
        return "unsupported"  # pixart / hunyuan / misc DiT
    if any_prefix("lora_unet_") or any_prefix("lora_te") or any(".lora_down." in k or ".lora_A." in k for k in keys[:200]):
        return "unsupported"  # a LoRA in the checkpoint folder

    in_key = "model.diffusion_model.input_blocks.0.0.weight"
    in_shape = _shape(hdr, in_key)
    has_unet = in_shape is not None or any_prefix("model.diffusion_model.")
    if not has_unet:
        return "unsupported" if keys else "unknown"
    in_ch = in_shape[1] if in_shape and len(in_shape) >= 2 else 4
    inpaint = in_ch == 9
    if in_ch not in (4, 9):
        return "unsupported"  # e.g. instruct-pix2pix (8ch) or upscalers

    has_xl_te2 = any_prefix("conditioner.embedders.1.")
    has_label_emb = any_prefix("model.diffusion_model.label_emb.")
    if has_xl_te2 or (has_label_emb and any_prefix("conditioner.embedders.")):
        if not has_xl_te2:
            return "unsupported"  # SDXL refiner (only OpenCLIP bigG)
        return "sdxl-inpaint" if inpaint else "sdxl"
    if has_label_emb:
        # SDXL unet without text encoders, or SD2 unCLIP
        ctx = _shape(hdr, "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight")
        if ctx and len(ctx) == 2 and ctx[1] == 2048:
            return "sdxl-inpaint" if inpaint else "sdxl"
        return "unsupported"
    if any_prefix("cond_stage_model.model.") and not any_prefix("cond_stage_model.transformer."):
        return "sd2"  # OpenCLIP ViT-H text encoder
    ctx = _shape(hdr, "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight")
    if ctx and len(ctx) == 2 and ctx[1] == 1024:
        return "sd2"
    return "sd15-inpaint" if inpaint else "sd15"


def checkpoint_flags(hdr: dict) -> dict:
    """Extra hints stored in some checkpoints (Comfy convention): v-prediction / zero terminal SNR."""
    return {"v_pred": "v_pred" in hdr, "ztsnr": "ztsnr" in hdr}


def detect_kind(path: str | os.PathLike) -> str:
    p = str(path)
    if p.lower().endswith(".safetensors"):
        try:
            return detect_kind_from_header(read_safetensors_header(p))
        except Exception:
            return "unknown"
    return "unknown"


# --------------------------------------------------------------------------- scanning

def _size_mb(p: Path) -> float:
    try:
        return round(p.stat().st_size / (1024 * 1024), 2)
    except OSError:
        return 0.0


def _rel_name(base: Path, p: Path, strip_ext: bool = True) -> str:
    try:
        rel = p.relative_to(base)
    except ValueError:
        rel = Path(p.name)
    parts = list(rel.parts)
    if strip_ext and parts:
        last = parts[-1]
        stem, ext = os.path.splitext(last)
        parts[-1] = stem if ext else last
    return "/".join(parts)


def _walk(base: Path, exts: Iterable[str], dir_marker: Optional[str] = None):
    """Yield (path, is_dir_model) recursively; directories containing `dir_marker` are yielded as models."""
    exts = tuple(e.lower() for e in exts)
    for root, dirs, files in os.walk(base, followlinks=False):
        rootp = Path(root)
        if dir_marker:
            for d in list(dirs):
                if (rootp / d / dir_marker).is_file():
                    yield rootp / d, True
                    dirs.remove(d)
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for fn in sorted(files):
            if fn.lower().endswith(exts):
                yield rootp / fn, False


def scan_models(dirs: Iterable[str]) -> list[dict]:
    out, seen = [], set()
    for d in dirs or []:
        if not d:
            continue
        base = Path(d)
        if not base.is_dir():
            continue
        for p, _ in _walk(base, CHECKPOINT_EXTS):
            key = os.path.normcase(str(p.resolve()))
            if key in seen:
                continue
            seen.add(key)
            out.append({"name": _rel_name(base, p), "path": str(p), "kind": detect_kind(p), "size_mb": _size_mb(p)})
    out.sort(key=lambda e: e["name"].lower())
    return out


def _dir_size_mb(p: Path) -> float:
    total = 0
    for root, _, files in os.walk(p):
        for fn in files:
            try:
                total += (Path(root) / fn).stat().st_size
            except OSError:
                pass
    return round(total / (1024 * 1024), 2)


def scan_files(kind: str, dirs: Iterable[str]) -> list[dict]:
    if kind not in FILE_KINDS:
        raise ValueError(f"unknown file kind '{kind}' (expected one of {', '.join(FILE_KINDS)})")
    out, seen = [], set()
    marker = DIR_MARKERS.get(kind)
    for d in dirs or []:
        if not d:
            continue
        base = Path(d)
        if not base.is_dir():
            continue
        for p, is_dir in _walk(base, FILE_EXTS, marker):
            key = os.path.normcase(str(p.resolve()))
            if key in seen:
                continue
            seen.add(key)
            size = _dir_size_mb(p) if is_dir else _size_mb(p)
            out.append({"name": _rel_name(base, p, strip_ext=not is_dir), "path": str(p), "size_mb": size})
    out.sort(key=lambda e: e["name"].lower())
    return out


# --------------------------------------------------------------------------- hashing

_HASH_LOCK = threading.Lock()


def _cache_file() -> Path:
    env = os.environ.get("SDFX_HASH_CACHE")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / ".cache" / "hashes.json"


def _load_cache(cf: Path) -> dict:
    try:
        with open(cf, "r", encoding="utf-8") as f:
            d = json.load(f)
            return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def sha256_file(path: str | os.PathLike, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def full_hash(path: str | os.PathLike) -> str:
    """sha256 of the whole file, cached in python/.cache/hashes.json keyed by path + mtime + size."""
    p = Path(path)
    st = p.stat()
    key = os.path.normcase(str(p.resolve()))
    cf = _cache_file()
    with _HASH_LOCK:
        cache = _load_cache(cf)
        e = cache.get(key)
        if isinstance(e, dict) and e.get("mtime") == st.st_mtime and e.get("size") == st.st_size and e.get("sha256"):
            return e["sha256"]
    digest = sha256_file(p)
    with _HASH_LOCK:
        cache = _load_cache(cf)
        cache[key] = {"mtime": st.st_mtime, "size": st.st_size, "sha256": digest}
        try:
            cf.parent.mkdir(parents=True, exist_ok=True)
            tmp = cf.with_name(f"{cf.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=0)
            os.replace(tmp, cf)
        except OSError:
            pass
    return digest


def short_hash(path: str | os.PathLike) -> str:
    """A1111 short hash (AutoV2): first 10 hex chars of the whole-file sha256."""
    return full_hash(path)[:10]


def path_hash(path: str | os.PathLike) -> str:
    """Hash of a file, or of a diffusers directory (hash of its weights file / config)."""
    p = Path(path)
    if p.is_dir():
        for cand in ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.fp16.safetensors",
                     "diffusion_pytorch_model.bin", "config.json"):
            if (p / cand).is_file():
                return short_hash(p / cand)
        return hashlib.sha256(str(p).encode()).hexdigest()[:10]
    return short_hash(p)


def model_name(path: str | os.PathLike) -> str:
    return Path(path).stem
