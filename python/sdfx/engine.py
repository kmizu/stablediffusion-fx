"""Real inference engine on diffusers (SD1.x / SD2 / SDXL single-file checkpoints).

Importing this module imports torch and diffusers; server.build_engine falls back to UnavailableEngine
when that fails.
"""
from __future__ import annotations

import gc
from dataclasses import replace
from collections import OrderedDict
import hashlib
import inspect
import logging
import math
import os
import re
import time
import traceback
from typing import Any, Optional

import torch  # noqa: F401  (ImportError here -> degraded engine)
import diffusers  # noqa: F401

from . import controlnet as cn_mod
from . import hires as hires_mod
from . import hooks as hooks_mod
from . import adapters, extras, inpaint, lora_blocks, models, preview, prompt, prompt_schedule, schedulers, upscalers
from . import weight_gradient
from . import negpip
from . import neutral_prompt, neutral_runtime
from . import extensions as ext_mod
from .base import LATENT_UPSCALERS, BaseEngine
from .images import Image, load_mask, load_rgb, next_output_path, resize_image, save_png
from .params import GenerationParams, GenerationRequest, center_crop_box, hires_steps, img2img_steps, resolve_seed
from .protocol import (INTERRUPTED, INVALID_PARAMS, MODEL_LOAD_FAILED, NO_MODEL, UNSUPPORTED, EngineError,
                       JobContext)

log = logging.getLogger("sdfx.engine")

PIPELINE_CLASSES = {
    "sd15": "StableDiffusionPipeline",
    "sd2": "StableDiffusionPipeline",
    "sd15-inpaint": "StableDiffusionInpaintPipeline",
    "sdxl": "StableDiffusionXLPipeline",
    "sdxl-inpaint": "StableDiffusionXLInpaintPipeline",
}

MAX_LORAS = 6        # resident LoRA adapters (LRU)
MAX_CONTROLNETS = 2  # resident ControlNet models (LRU)
MAX_DERIVED = 6      # cached from_pipe pipelines (LRU)

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


class ScheduledEmbeds(dict):
    """Embeddings for step 1, plus `schedule` = [(end_step, embeds), ...] when the prompt changes mid-sampling."""
    schedule: list = []


def slerp(val: float, low, high):
    """A1111 slerp used for variation seeds (per batch row)."""
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm * high_norm).sum(1)
    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    return (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high


def _accepts(fn, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    return name in sig.parameters or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())


def _filter_kwargs(fn, kw: dict) -> dict:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kw
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return kw
    return {k: v for k, v in kw.items() if k in params}


def _hf_hint(e: BaseException) -> str:
    t = str(e).lower()
    if "offline" in t or "connect" in t or "huggingface" in t or "hf_hub" in t:
        return " (diffusers needs base model configs from the Hugging Face cache; run once online, then HF_HUB_OFFLINE=1 works)"
    return ""


PICKLE_REFUSED = "legacy pickle {what} refused for safety; convert to .safetensors"


class _InertStub:
    """Data-only stand-in for harmless training metadata objects pickled into old .ckpt/.pt files."""

    def __setstate__(self, state):
        self.__dict__["_state"] = state


# Globals that old A1111-era checkpoints/VAEs pickle as metadata only (training callbacks). They are mapped to
# inert stubs under torch's weights_only unpickler: no foreign code is imported or executed.
INERT_GLOBALS = ["pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint"]


def _inert_stubs() -> list:
    out = []
    for full in INERT_GLOBALS:
        mod, _, name = full.rpartition(".")
        out.append(type(name, (_InertStub,), {"__module__": mod, "__qualname__": name}))
    return out


def safe_torch_load(path: str, what: str = "checkpoint"):
    """torch.load restricted to tensors/primitives (weights_only=True). Never falls back to full unpickling."""
    try:
        sg = getattr(torch.serialization, "safe_globals", None)
        if sg is not None:
            with sg(_inert_stubs()):
                return torch.load(path, map_location="cpu", weights_only=True)
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise EngineError(UNSUPPORTED, f"{PICKLE_REFUSED.format(what=what)}: {os.path.basename(path)} ({type(e).__name__})")


def _ckpt_kind_from_torch(path: str) -> str:
    """Kind detection for .ckpt/.pt checkpoints (weights_only load; refuses pickles that need code execution)."""
    sd = safe_torch_load(path, "checkpoint")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        return "unknown"
    hdr = {k: {"shape": list(v.shape)} for k, v in sd.items() if hasattr(v, "shape")}
    del sd
    gc.collect()
    return models.detect_kind_from_header(hdr)


class Engine(BaseEngine):
    dry_run = False

    def __init__(self):
        super().__init__()
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.pipe = None
        self.kind: Optional[str] = None
        self.sdxl = False
        self.flags: dict = {}
        self.dtype = torch.float16 if self.cuda else torch.float32
        self.orig_vae = None
        self.vae_info: Optional[dict] = None
        self.base_sched_config: dict = {}
        self.derived: "OrderedDict[tuple, Any]" = OrderedDict()   # LRU of from_pipe pipelines
        self.loaded_loras: "OrderedDict[str, str]" = OrderedDict()  # adapter -> path, LRU
        self.hook_owner = None
        self._pre: Optional[list] = None
        self._sampler_ok: Optional[dict] = None  # pipeline whose accelerate offload hooks are installed (offload modes)
        self.loaded_embeddings: set[str] = set()
        self.cn_cache: "OrderedDict[str, Any]" = OrderedDict()  # path -> ControlNetModel, LRU
        self.taesd = preview.TaesdPreviewer()
        self._extra: dict = {}
        self._ms_overrides: dict = {}
        self.ext_cache: dict = {}
        self.ip_state = adapters.IPAdapterState()
        self.parked = False

    # -- info ---------------------------------------------------------------
    def hello(self) -> dict:
        h = super().hello()
        h["torch"] = torch.__version__
        h["diffusers"] = diffusers.__version__
        h["cuda"] = bool(self.cuda)
        if self.cuda:
            props = torch.cuda.get_device_properties(0)
            h["device"] = f"cuda:0 {props.name}"
            h["vram_mb"] = int(props.total_memory // (1024 * 1024))
        else:
            h["device"] = "cpu"
        return h

    def features(self) -> list[str]:
        f = ["freeu", "dynamic_thresholding", "hypertile"]
        if hasattr(diffusers, "StableDiffusionXLPAGPipeline"):
            f.append("pag")
        if hasattr(diffusers, "ControlNetModel"):
            f.append("controlnet")
        if cn_mod.has_controlnet_aux():
            f.append("controlnet_aux")
        if upscalers.has_spandrel():
            f.append("spandrel")
        if hasattr(diffusers, "AutoencoderTiny"):
            f.append("taesd")
        f += ["apg", "mahiro", "rescale_cfg", "model_sampling", "kohya_hrfix", "multidiffusion", "latent_modifier",
              "soft_inpainting", "neveroom", "reference", "revision", "ip_adapter", "controllllite", "fooocus_inpaint",
              "img2imgalt", "caption", "ldsr", "adetailer", "weight_gradient", "negpip", "neutral_prompt"]
        f += list(ext_mod.sampling_ext.HOOKS)
        if hasattr(diffusers, "StableDiffusionSAGPipeline"):
            f.append("sag")
        if adapters.photomaker_available():
            f.append("photomaker")
        if upscalers.has_spandrel() and cn_mod._has("cv2"):
            f.append("face_restore")
        if hasattr(diffusers, "StableVideoDiffusionPipeline"):
            f.append("img2vid")
        f.append("novel_view")
        if hasattr(diffusers, "MarigoldDepthPipeline"):
            f.append("marigold")
        if extras.code_allowed():
            f.append("custom_code")
        return f

    def preprocessors(self) -> list[str]:
        if self._pre is None:
            self._pre = cn_mod.available_preprocessors()
        return self._pre

    def sampler_supported(self, name: str) -> bool:
        if self._sampler_ok is None:
            self._sampler_ok = {r["name"]: r["supported"] for r in schedulers.sampler_list(True)}
        try:
            return self._sampler_ok.get(schedulers.sampler_spec(name).name, False)
        except ValueError:
            return False

    def check_model_upscaler(self, name: str, upscaler_paths: dict) -> None:
        super().check_model_upscaler(name, upscaler_paths)
        if not upscalers.has_spandrel():
            raise EngineError(UNSUPPORTED, f"upscaler '{name}' needs the spandrel package")

    def memory(self, params: dict, ctx: JobContext) -> dict:
        m = super().memory(params, ctx)
        if self.cuda:
            m["allocated_mb"] = round(torch.cuda.memory_allocated() / 2**20, 1)
            m["reserved_mb"] = round(torch.cuda.memory_reserved() / 2**20, 1)
            m["total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**20, 1)
        return m

    def on_oom(self) -> None:
        gc.collect()
        if self.cuda:
            torch.cuda.empty_cache()

    # -- model management --------------------------------------------------
    def _free(self):
        self.pipe = None
        self.orig_vae = None
        self._drop_hooks()
        self.derived.clear()
        self.cn_cache.clear()
        self.loaded_loras.clear()
        self.loaded_embeddings.clear()
        self.model = None
        self.vae_info = None
        self.taesd = preview.TaesdPreviewer()
        upscalers.unload_models()
        gc.collect()
        if self.cuda:
            torch.cuda.empty_cache()

    def unload(self, params: dict, ctx: JobContext) -> dict:
        self._free()
        return {}

    @property
    def _offload(self) -> bool:
        return self.cuda and self.load_opts.vram_mode in ("model_offload", "sequential_offload")

    def _drop_hooks(self):
        if self.hook_owner is not None:
            try:
                self.hook_owner.remove_all_hooks()
            except Exception:
                pass
            self.hook_owner = None

    def _place(self, pipe):
        """Put a pipeline on the device per vram_mode. Offload hooks live on exactly one pipeline at a time;
        derived (from_pipe) pipelines share the modules, so switching pipelines moves the hooks."""
        mode = self.load_opts.vram_mode
        if not self.cuda:
            pipe.to("cpu")
        elif mode == "model_offload":
            self._drop_hooks()
            pipe.enable_model_cpu_offload()
            self.hook_owner = pipe
        elif mode == "sequential_offload":
            self._drop_hooks()
            pipe.enable_sequential_cpu_offload()
            self.hook_owner = pipe
        else:
            pipe.to(self.device)

    def park(self):
        """Move the SD pipeline off the GPU (gpu mode) so another model (SVD, novel view) can use the VRAM."""
        if self.pipe is not None and self.cuda and not self._offload and not self.parked:
            self.pipe.to("cpu")
            self.parked = True
            torch.cuda.empty_cache()

    def _activate(self, pipe):
        """Make `pipe` ready to run (offload modes: move the accelerate hooks to it; gpu mode: move new modules)."""
        if self.parked and self.cuda and not self._offload:
            self.pipe.to(self.device)
            self.parked = False
        if self._offload:
            if pipe is not self.hook_owner:
                self._place(pipe)
        elif self.cuda:
            for name in ("controlnet",):
                m = getattr(pipe, name, None)
                if m is not None and hasattr(m, "to"):
                    m.to(self.device)

    def load_model(self, params: dict, ctx: JobContext) -> dict:
        t0 = time.monotonic()
        req = self.parse_load(params)
        kind = models.detect_kind(req.path)
        if kind == "unknown" and not req.path.lower().endswith(".safetensors"):
            kind = _ckpt_kind_from_torch(req.path)
        if kind == "unknown":
            kind = "sd15"
            ctx.warn("could not detect the model type; assuming SD1.5")
        if kind not in PIPELINE_CLASSES:
            raise EngineError(MODEL_LOAD_FAILED, f"unsupported model architecture ({kind}): {os.path.basename(req.path)}")
        self._free()
        self.load_opts = req
        self.dtype = DTYPES[req.dtype] if self.cuda else torch.float32
        flags = {}
        if req.path.lower().endswith(".safetensors"):
            try:
                flags = models.checkpoint_flags(models.read_safetensors_header(req.path))
            except Exception:
                flags = {}
        cls = getattr(diffusers, PIPELINE_CLASSES[kind])
        kw: dict = {"torch_dtype": self.dtype}
        if not kind.startswith("sdxl"):
            kw.update(safety_checker=None, requires_safety_checker=False)
        try:
            pipe = cls.from_single_file(req.path, **kw)
        except Exception as e:
            raise EngineError(MODEL_LOAD_FAILED, f"failed to load {os.path.basename(req.path)}: {e}{_hf_hint(e)}",
                              traceback.format_exc())
        pipe.set_progress_bar_config(disable=True)
        self.kind = kind
        self.sdxl = kind.startswith("sdxl")
        self.flags = flags
        self.base_sched_config = dict(pipe.scheduler.config)
        self.pipe = pipe
        self.orig_vae = pipe.vae
        if req.vae_tiling:
            pipe.vae.enable_tiling()
        if req.vae_path:
            self._swap_vae(req.vae_path, ctx)
        try:
            self._place(pipe)
        except Exception as e:
            self._free()
            raise EngineError(MODEL_LOAD_FAILED, f"failed to move model to device: {e}", traceback.format_exc())
        self.model = {"name": models.model_name(req.path), "path": req.path, "kind": kind,
                      "hash": models.short_hash(req.path)}
        return self._loaded(int((time.monotonic() - t0) * 1000))

    def _loaded(self, ms: int = 0) -> dict:
        m = dict(self.model)
        m["loaded_ms"] = ms
        m["vae_name"] = self.vae_info["vae_name"] if self.vae_info else None
        m["vae_hash"] = self.vae_info["vae_hash"] if self.vae_info else None
        return m

    def _load_vae_module(self, path: str):
        from diffusers import AutoencoderKL  # type: ignore
        if os.path.isdir(path):
            return AutoencoderKL.from_pretrained(path, torch_dtype=self.dtype)
        from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint, load_single_file_checkpoint  # type: ignore
        if path.lower().endswith(".safetensors"):
            sd = load_single_file_checkpoint(path)
        else:
            sd = safe_torch_load(path, "VAE")
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        cfg = dict(self.orig_vae.config)
        vae = AutoencoderKL.from_config(cfg)
        if any(k.startswith("encoder.down.") or k.startswith("first_stage_model.") or k.startswith("decoder.up.") for k in sd):
            sd = convert_ldm_vae_checkpoint(sd, vae.config)
        missing, unexpected = vae.load_state_dict(sd, strict=False)
        if missing:
            raise ValueError(f"VAE state dict mismatch ({len(missing)} missing keys, e.g. {missing[:3]})")
        return vae.to(dtype=self.dtype)

    def _swap_vae(self, path: Optional[str], ctx: JobContext):
        pipe = self.pipe
        if path:
            try:
                vae = self._load_vae_module(path)
            except EngineError:
                raise
            except Exception as e:
                raise EngineError(MODEL_LOAD_FAILED, f"failed to load VAE {os.path.basename(path)}: {e}", traceback.format_exc())
            if self.load_opts.vae_tiling:
                vae.enable_tiling()
            self.vae_info = {"vae_name": models.model_name(path), "vae_hash": models.path_hash(path)}
        else:
            vae = self.orig_vae
            self.vae_info = None
        pipe.vae = vae
        self._drop_hooks()
        self.derived.clear()

    def set_vae(self, params: dict, ctx: JobContext) -> dict:
        if self.pipe is None:
            raise EngineError(NO_MODEL, "no model loaded")
        path = params.get("path")
        if path and not os.path.exists(path):
            raise EngineError(MODEL_LOAD_FAILED, f"VAE file not found: {path}")
        self._drop_hooks()
        self._swap_vae(path or None, ctx)
        self.load_opts.vae_path = path or None
        self._place(self.pipe)
        return self._loaded()

    # -- pipelines -----------------------------------------------------------
    def _get_pipe(self, task: str, controlnets: Optional[list] = None, pag: Optional[dict] = None):
        """task: txt2img | img2img | inpaint. Derived pipelines share all components with the base pipe."""
        base_task = "inpaint" if self.kind.endswith("inpaint") else "txt2img"
        cn_paths = tuple(p for p, _ in controlnets or [])
        key = (task, cn_paths, tuple(pag["layers"]) if pag else None)
        if task == base_task and not controlnets and not pag:
            return self.pipe
        if key in self.derived:
            self.derived.move_to_end(key)
            return self.derived[key]
        controlnets = [m for _, m in controlnets or []]
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting, AutoPipelineForText2Image  # type: ignore
        auto = {"txt2img": AutoPipelineForText2Image, "img2img": AutoPipelineForImage2Image,
                "inpaint": AutoPipelineForInpainting}[task]
        kw: dict = {}
        if controlnets:
            kw["controlnet"] = controlnets[0] if len(controlnets) == 1 else list(controlnets)
        if pag:
            kw["enable_pag"] = True
            kw["pag_applied_layers"] = pag["layers"]
        p = auto.from_pipe(self.pipe, **kw)
        p.set_progress_bar_config(disable=True)
        self.derived[key] = p
        while len(self.derived) > MAX_DERIVED:
            _, old = self.derived.popitem(last=False)
            if old is self.hook_owner:
                self._drop_hooks()
        return p

    def _evict_controlnet(self, path: str):
        self.cn_cache.pop(path, None)
        for k in [k for k in self.derived if path in k[1]]:
            if self.derived[k] is self.hook_owner:
                self._drop_hooks()
            del self.derived[k]
        gc.collect()
        if self.cuda:
            torch.cuda.empty_cache()

    # -- loras / embeddings ----------------------------------------------------
    def _apply_loras(self, gp: GenerationParams, req: GenerationRequest, ctx: JobContext) -> dict:
        hashes: dict = {}
        pipe = self.pipe
        want = []
        self._lora_windows = []
        self._lora_key = None
        for lr in gp.loras:
            path = req.lora_paths[lr.name]
            if not os.path.isfile(path):
                ctx.warn(f"LoRA file not found: {path}")
                continue
            te = float(lr.weight)
            unet = float(lr.unet) if lr.unet is not None else te
            if lr.lbw:
                try:
                    bw = lora_blocks.parse_weights(lr.lbw, self.sdxl, req.lbw_presets)
                except ValueError as e:
                    raise EngineError(INVALID_PARAMS, f"LoRA '{lr.name}': {e}")
                scale = lora_blocks.diffusers_scales(bw, self.sdxl, unet, te)
            elif lr.unet is not None or lr.start is not None or lr.stop is not None:
                scale = lora_blocks.plain_scales(self.sdxl, unet, te)
            else:
                scale = te
            want.append((lr.name, path, scale, lr.start, lr.stop))
            hashes[lr.name] = models.short_hash(path)
        if not want:
            if self.loaded_loras:
                try:
                    pipe.disable_lora()
                except Exception:
                    pass
            return hashes
        names, weights = [], []
        for name, path, w, start, stop in want:
            adapter = "l" + hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:12]
            if adapter in self.loaded_loras:
                self.loaded_loras.move_to_end(adapter)
            else:
                try:
                    pipe.load_lora_weights(path, adapter_name=adapter)
                    self.loaded_loras[adapter] = path
                except Exception as e:
                    ctx.warn(f"LoRA '{name}' could not be loaded: {str(e).splitlines()[0] if str(e) else type(e).__name__}")
                    hashes.pop(name, None)
                    continue
            names.append(adapter)
            weights.append(w)
            if start is not None or stop is not None:
                self._lora_windows.append((len(names) - 1, start, stop))
        stale = [a for a in self.loaded_loras if a not in names]
        while len(self.loaded_loras) > MAX_LORAS and stale:
            a = stale.pop(0)
            try:
                pipe.delete_adapters([a])
            except Exception as e:
                log.debug("delete_adapters(%s) failed: %s", a, e)
            self.loaded_loras.pop(a, None)
        if names:
            try:
                pipe.enable_lora()
            except Exception:
                pass
            try:
                pipe.set_adapters(names, adapter_weights=weights)
            except (ValueError, KeyError, TypeError) as e:
                raise EngineError(INVALID_PARAMS, f"LoRA block weights could not be applied: {e}")
            self._lora_state = (names, weights)
        elif self.loaded_loras:
            try:
                pipe.disable_lora()
            except Exception:
                pass
        return hashes

    def _load_embeddings(self, req: GenerationRequest, texts: list[str], ctx: JobContext):
        pipe = self.pipe
        joined = " ".join(texts)
        for path in req.embedding_paths:
            token = os.path.splitext(os.path.basename(path))[0]
            if token in self.loaded_embeddings or not os.path.isfile(path) \
                    or not re.search(r"(?<![\w-])" + re.escape(token) + r"(?![\w-])", joined):
                continue
            try:
                if path.lower().endswith(".safetensors"):
                    from safetensors.torch import load_file  # type: ignore
                    sd = load_file(path)
                else:
                    try:
                        sd = safe_torch_load(path, "embedding")
                    except EngineError as e:
                        ctx.warn(e.message)
                        self.loaded_embeddings.add(token)
                        continue
                if self.sdxl:
                    if not (isinstance(sd, dict) and "clip_g" in sd and "clip_l" in sd):
                        ctx.warn(f"embedding '{token}' is not an SDXL embedding; skipped")
                        self.loaded_embeddings.add(token)
                        continue
                    pipe.load_textual_inversion(sd["clip_g"], token=token, text_encoder=pipe.text_encoder_2,
                                                tokenizer=pipe.tokenizer_2)
                    pipe.load_textual_inversion(sd["clip_l"], token=token, text_encoder=pipe.text_encoder,
                                                tokenizer=pipe.tokenizer)
                else:
                    if isinstance(sd, dict) and "clip_g" in sd:
                        ctx.warn(f"embedding '{token}' is an SDXL embedding; skipped")
                        self.loaded_embeddings.add(token)
                        continue
                    pipe.load_textual_inversion(sd, token=token)
                self.loaded_embeddings.add(token)
            except Exception as e:
                ctx.warn(f"embedding '{token}' could not be loaded: {str(e).splitlines()[0] if str(e) else type(e).__name__}")
                self.loaded_embeddings.add(token)

    # -- prompts -------------------------------------------------------------
    def _encode(self, pos: str, neg: str, gp: GenerationParams, bs: int) -> dict:
        if hooks_mod.hook_config(gp.hooks, "negpip", {}) is not None:
            return negpip.encode(self, prompt.strip_extra_networks(pos), prompt.strip_extra_networks(neg), gp, bs)
        return self._encode_plain(pos, neg, gp, bs)

    def _encode_plain(self, pos: str, neg: str, gp: GenerationParams, bs: int) -> dict:
        pipe = self.pipe
        pos = prompt.strip_extra_networks(pos)
        neg = prompt.strip_extra_networks(neg)
        with torch.no_grad():
            if self.sdxl:
                pe, pp = prompt.encode_sdxl(pipe, pos, gp.clip_skip)
                if neg.strip() == "":
                    ne, npool = torch.zeros_like(pe), torch.zeros_like(pp)
                else:
                    ne, npool = prompt.encode_sdxl(pipe, neg, gp.clip_skip)
                if pe.shape[1] != ne.shape[1]:
                    empty, _ = prompt.encode_sdxl(pipe, "", gp.clip_skip)
                    if neg.strip() == "":
                        empty = torch.zeros_like(empty)
                        ne = torch.zeros((ne.shape[0], pe.shape[1], ne.shape[2]), dtype=ne.dtype, device=ne.device)
                    else:
                        pe, ne = prompt.pad_to_match(pe, ne, empty)
                return {"prompt_embeds": pe.repeat(bs, 1, 1).to(self.dtype),
                        "negative_prompt_embeds": ne.repeat(bs, 1, 1).to(self.dtype),
                        "pooled_prompt_embeds": pp.repeat(bs, 1).to(self.dtype),
                        "negative_pooled_prompt_embeds": npool.repeat(bs, 1).to(self.dtype)}
            oc = self.kind == "sd2"
            pe = prompt.encode_sd15(pipe, pos, gp.clip_skip, oc)
            ne = prompt.encode_sd15(pipe, neg, gp.clip_skip, oc)
            if pe.shape[1] != ne.shape[1]:
                empty = prompt.encode_sd15(pipe, "", gp.clip_skip, oc)
                pe, ne = prompt.pad_to_match(pe, ne, empty)
            return {"prompt_embeds": pe.repeat(bs, 1, 1).to(self.dtype),
                    "negative_prompt_embeds": ne.repeat(bs, 1, 1).to(self.dtype)}

    def _encode_scheduled(self, pos: str, neg: str, gp: GenerationParams, bs: int, steps: int,
                          hires_total: Optional[int] = None, gradient_total: Optional[int] = None) -> dict:
        """_encode plus A1111 prompt editing / alternation: returns the embeddings for step 1; when the prompt
        changes during sampling the result is a ScheduledEmbeds carrying [(end_step, embeds), ...] (equal lengths)."""
        nc = hooks_mod.hook_config(gp.hooks, "neutral_prompt", {"cfg_rescale": 0})
        if nc is not None:
            try:
                phi = float(nc["cfg_rescale"])
                if not math.isfinite(phi) or not 0 <= phi <= 1:
                    raise ValueError("Neutral Prompt CFG rescale must be between 0 and 1")
                tree = neutral_prompt.parse(pos)
            except ValueError as exc:
                raise EngineError(INVALID_PARAMS, str(exc)) from exc
            nodes = neutral_prompt.leaves(tree)
            # Literal ordinary prompts bypass all wrappers and retain original encoding bytes/attention semantics.
            active = phi != 0 or len(nodes) != 1 or any(n.weight != 1 or n.kind != "AND" for n in nodes) or nodes[0].text != pos
            if active:
                sub = replace(gp, hooks={k: v for k, v in gp.hooks.items() if k != "neutral_prompt"})
                branches = [self._encode_scheduled(n.text, neg, sub, bs, steps, hires_total, gradient_total) for n in nodes]
                out = dict(branches[0])
                out["_neutral"] = neutral_runtime.Plan(tree, branches)
                return out
        def schedule(text):
            text = prompt.strip_extra_networks(text)
            if hooks_mod.hook_config(gp.hooks, "weight_gradient", {}) is not None:
                return weight_gradient.schedule(text, steps, hires_total, gradient_total)
            return prompt_schedule.schedule(text, steps, hires_total)
        ps, ns = schedule(pos), schedule(neg)
        if len(ps) == 1 and len(ns) == 1:
            return self._encode(ps[0][1], ns[0][1], gp, bs)
        cache: dict = {}
        variants = []
        for e in sorted({e for e, _ in ps} | {e for e, _ in ns}):
            key = (prompt_schedule.text_at(ps, e), prompt_schedule.text_at(ns, e))
            if key not in cache:
                cache[key] = self._encode(key[0], key[1], gp, bs)
            variants.append((e, cache[key]))
        longest = max(v["prompt_embeds"].shape[1] for _, v in variants)
        if any(v.get("_negpip") for _, v in variants):
            variants = negpip.align_variants(variants)
        elif any(v["prompt_embeds"].shape[1] != longest for _, v in variants):
            with torch.no_grad():
                empty = (prompt.encode_sdxl(self.pipe, "", gp.clip_skip)[0] if self.sdxl
                         else prompt.encode_sd15(self.pipe, "", gp.clip_skip, self.kind == "sd2"))
            for d in cache.values():
                for k in ("prompt_embeds", "negative_prompt_embeds"):
                    t = d[k]
                    if t.shape[1] < longest:
                        fill = torch.zeros_like(empty) if not bool(t.abs().any()) else empty
                        reps = (longest - t.shape[1]) // fill.shape[1]
                        d[k] = torch.cat([t] + [fill.to(t.device, t.dtype).repeat(t.shape[0], 1, 1)] * reps, dim=1)
        out = ScheduledEmbeds(variants[0][1])
        out.schedule = variants
        return out

    def _adjust_embeds(self, gp: GenerationParams, embeds: dict, ctx: JobContext) -> dict:
        if "_neutral" in embeds:
            plan = embeds["_neutral"]
            branches = [self._adjust_embeds(gp, branch, ctx) for branch in plan.branches]
            out = dict(branches[0])
            out["_neutral"] = neutral_runtime.Plan(plan.tree, branches)
            return out
        sch = getattr(embeds, "schedule", None)
        def apply(d):
            return negpip.adjust(d, lambda plain: ext_mod.adjust_embeds(self, gp, plain, ctx))
        if not sch:
            return apply(embeds)
        done: dict = {}
        variants = []
        for e, v in sch:
            if id(v) not in done:
                done[id(v)] = apply(v)
            variants.append((e, done[id(v)]))
        out = ScheduledEmbeds(variants[0][1])
        out.schedule = variants
        return out

    def _set_lora_step(self, step: int, total: int):
        """LoRA Block Weight start/stop: U-Net part of windowed adapters on only inside their window."""
        names, weights = self._lora_state
        cur = list(weights)
        for idx, start, stop in self._lora_windows:
            if not lora_blocks.active(step, total, start, stop):
                cur[idx] = lora_blocks.zero_unet(weights[idx], self.sdxl)
        key = tuple(i for i, w in enumerate(cur) if w is not weights[i])
        if key != getattr(self, "_lora_key", None):
            self.pipe.set_adapters(names, adapter_weights=cur)
            self._lora_key = key

    def _lora_window_callback(self):
        def cb(pipe, i, t, d):
            total = int(getattr(pipe, "_num_timesteps", 0) or 1)
            self._set_lora_step(i + 2, total)  # applies to the next step
            return d
        return cb

    def _schedule_swapper(self, schedule: list):
        """callback_on_step_end that swaps the (CFG / PAG concatenated) text embeddings when the prompt changes."""
        state = {"last": schedule[0][1]}

        def at(step):
            for e, v in schedule:
                if step <= e:
                    return v
            return schedule[-1][1]

        def cb(pipe, i, t, d):
            # Heun / DPM2 / DPM++ SDE evaluate each inference step twice. Keep one conditioning state
            # for both stages instead of finishing a requested-step schedule halfway through the pass.
            order = max(1, int(getattr(getattr(pipe, "scheduler", None), "order", 1)))
            cur = at((int(i) + 1) // order + 1)  # next logical step, 1-based
            if cur is state["last"]:
                return d
            state["last"] = cur
            cfg = bool(getattr(pipe, "do_classifier_free_guidance", False))
            pag = bool(getattr(pipe, "do_perturbed_attention_guidance", False))

            def build(pk, nk, ref):
                p, n = cur[pk], cur[nk]
                parts = [n, p, p] if cfg and pag else [n, p] if cfg else [p, p] if pag else [p]
                x = torch.cat(parts).to(ref.device, ref.dtype)
                return x if x.shape == ref.shape else None
            pe = d.get("prompt_embeds")
            if pe is not None:
                x = build("prompt_embeds", "negative_prompt_embeds", pe)
                if x is not None:
                    d["prompt_embeds"] = x
            te = d.get("add_text_embeds")
            if te is not None and "pooled_prompt_embeds" in cur:
                x = build("pooled_prompt_embeds", "negative_pooled_prompt_embeds", te)
                if x is not None:
                    d["add_text_embeds"] = x
            return d
        return cb

    # -- noise ---------------------------------------------------------------
    def _rng_device(self, gp: GenerationParams) -> str:
        r = (gp.extra or {}).get("rng", "gpu").lower()
        return "cuda" if (r == "gpu" and self.cuda) else "cpu"

    def _noise(self, seeds: list[int], subseeds: list[int], strength: float, w: int, h: int, gp: GenerationParams):
        ch = int(getattr(self.pipe.vae.config, "latent_channels", 4))
        shape = (ch, h // 8, w // 8)
        dev = self._rng_device(gp)
        out = []
        for s, ss in zip(seeds, subseeds):
            g = torch.Generator(dev).manual_seed(int(s) & 0xFFFFFFFFFFFFFFFF)
            n = torch.randn(shape, generator=g, device=dev, dtype=torch.float32)
            if strength > 0:
                g2 = torch.Generator(dev).manual_seed(int(ss) & 0xFFFFFFFFFFFFFFFF)
                sub = torch.randn(shape, generator=g2, device=dev, dtype=torch.float32)
                n = slerp(strength, n.unsqueeze(0), sub.unsqueeze(0))[0]
            out.append(n)
        return torch.stack(out).to(self.device if self.cuda else "cpu", self.dtype)

    def _generators(self, seeds: list[int], gp: GenerationParams):
        dev = self._rng_device(gp)
        return [torch.Generator(dev).manual_seed(int(s) & 0xFFFFFFFFFFFFFFFF) for s in seeds]

    # -- callback ------------------------------------------------------------
    def _callback(self, ctx: JobContext, req: GenerationRequest, pass_: str, image: int, images: int, steps_hint: int):
        every = int(req.preview_every or 0)
        method = req.preview_method
        state = {"started": False}

        def cb(pipe, i, t, kw):
            if not state["started"]:
                ctx.start_pass()
                state["started"] = True
            if ctx.interrupted:
                raise EngineError(INTERRUPTED, "interrupted")
            total = int(getattr(pipe, "_num_timesteps", 0) or steps_hint)
            step = i + 1
            ctx.progress(step, total, pass_, image=image, images=images)
            if every > 0 and method != "none" and step % every == 0 and step < total:
                lat = kw.get("latents")
                if lat is not None:
                    try:
                        b64 = None
                        if method == "taesd":
                            b64 = self.taesd.decode_b64(lat, self.sdxl, self.device, self.dtype)
                        if b64 is None:
                            b64 = preview.latent_preview_b64(lat[:1, :4].float().cpu().numpy(), self.sdxl)
                        if b64:
                            ctx.preview(b64, step)
                    except Exception as e:  # never fail a generation because of a preview
                        log.debug("preview failed: %s", e)
            return kw
        return cb

    # -- generation ------------------------------------------------------------
    def txt2img(self, params: dict, ctx: JobContext) -> dict:
        return self._generate(params, ctx, "txt2img")

    def img2img(self, params: dict, ctx: JobContext) -> dict:
        return self._generate(params, ctx, "img2img")

    def _scheduler(self, sampler: str, schedule: Optional[str], ctx: JobContext):
        plan = schedulers.plan(sampler, schedule)
        extra = {}
        if self.flags.get("v_pred"):
            extra["prediction_type"] = "v_prediction"
        if self.flags.get("ztsnr"):
            extra["rescale_betas_zero_snr"] = True
        extra.update(self._ms_overrides)
        pt = (self._extra.get("prediction_type") or "").strip()
        if pt in ("epsilon", "v_prediction"):
            extra["prediction_type"] = pt
        try:
            sched, warns = schedulers.build_scheduler(plan, self.base_sched_config, self.sdxl, extra)
        except Exception as e:
            raise EngineError(UNSUPPORTED, f"sampler '{sampler}' is not available: {e}")
        for w in warns:
            ctx.warn(w)
        return sched, plan

    def _controlnets(self, gp: GenerationParams, req: GenerationRequest, init_img, w: int, h: int, ctx: JobContext):
        units = [u for u in gp.controlnet if u.enabled]
        if not units:
            return [], [], {}
        models_, images = [], []
        wanted = {req.controlnet_paths.get(u.model, u.model) for u in units}
        for u in units:
            path = req.controlnet_paths.get(u.model, u.model)
            cnm = self.cn_cache.get(path)
            if cnm is not None:
                self.cn_cache.move_to_end(path)
            else:
                for old in [k for k in self.cn_cache if k not in wanted][: max(0, len(self.cn_cache) + 1 - MAX_CONTROLNETS)]:
                    self._evict_controlnet(old)
                try:
                    cnm = cn_mod.load_controlnet(path, self.dtype)
                except Exception as e:
                    raise EngineError(MODEL_LOAD_FAILED, f"failed to load ControlNet {os.path.basename(path)}: {e}{_hf_hint(e)}",
                                      traceback.format_exc())
                # device placement happens in _activate (gpu: .to(cuda); offload modes: accelerate hooks)
                cnm.to(dtype=self.dtype)
                self.cn_cache[path] = cnm
            if u.image:
                src = load_rgb(u.image)
            elif init_img is not None:
                src = init_img
            else:
                raise EngineError(INVALID_PARAMS, f"ControlNet unit '{u.model}' has no image")
            if u.preprocessor not in ("none", "") and u.preprocessor not in self.preprocessors():
                cond = src  # warned in parse_generation
            else:
                try:
                    cond = cn_mod.run_preprocessor(u.preprocessor, src, u.preprocessor_args)
                except Exception as e:
                    raise EngineError(INVALID_PARAMS, f"preprocessor '{u.preprocessor}' failed: {e}", traceback.format_exc())
            models_.append((path, cnm))
            images.append(cond)
        single = len(units) == 1
        kw = {
            "controlnet_conditioning_scale": float(units[0].weight) if single else [float(u.weight) for u in units],
            "control_guidance_start": float(units[0].guidance_start) if single else [float(u.guidance_start) for u in units],
            "control_guidance_end": float(units[0].guidance_end) if single else [float(u.guidance_end) for u in units],
        }
        return models_, images, kw

    @staticmethod
    def _cn_images(images: list, w: int, h: int):
        sized = [resize_image(1, im, w, h) for im in images]
        return sized[0] if len(sized) == 1 else sized

    def _pag_config(self, gp: GenerationParams, ctx: JobContext) -> Optional[dict]:
        cfg = hooks_mod.hook_config(gp.hooks, "pag", hooks_mod.PAG_DEFAULTS)
        if cfg is None:
            return None
        if not hasattr(diffusers, "StableDiffusionXLPAGPipeline"):
            ctx.warn("PAG requires a newer diffusers; ignored")
            return None
        return hooks_mod.normalize_pag(cfg)

    def _run(self, task: str, gp: GenerationParams, req: GenerationRequest, ctx: JobContext, *, sampler: str,
             schedule: Optional[str], cfg_scale: float, embeds: dict, steps: int, w: int, h: int, seeds: list[int],
             pass_: str, image_idx: int, images_total: int, latents=None, image=None, mask=None, strength: float = 1.0,
             expected_steps: int = 0, cn: Optional[tuple] = None, output_type: str = "pil", pag: Optional[dict] = None,
             hook_seed: int = 0):
        cn_models, cn_images, cn_kw = cn if cn else ([], [], {})
        xplan = ext_mod.pre_run(self, task, gp, mask_present=mask is not None, controlnet=bool(cn_models), pag=bool(pag))
        for wmsg in xplan.warnings:
            ctx.warn(wmsg)
        task = xplan.task
        try:
            pipe = ext_mod.special_pipe(self, xplan) or self._get_pipe(task, cn_models or None, pag)
        except Exception as e:
            if pag:
                ctx.warn(f"PAG could not be enabled: {e}")
                pag = None
                pipe = self._get_pipe(task, cn_models or None, None)
            else:
                raise
        sched, plan = self._scheduler(sampler, schedule, ctx)
        if xplan.sag and type(sched).__name__ not in SAG_SCHEDULERS:
            ctx.warn(f"SAG does not work with sampler '{sampler}' (use DDIM, UniPC or a DPM++ sampler); ignored")
            xplan.sag = None
            pipe = self._get_pipe(task, cn_models or None, pag)
        pipe.scheduler = sched
        kw: dict = dict(embeds)
        kw.pop("_negpip", None)
        neutral_plan = kw.pop("_neutral", None)
        if neutral_plan:
            neutral_runtime.validate(gp.hooks, controlnet=bool(cn_models), pag=bool(pag))
        kw.update(num_inference_steps=steps, guidance_scale=cfg_scale, generator=self._generators(seeds, gp),
                  output_type=output_type, callback_on_step_end=self._callback(ctx, req, pass_, image_idx, images_total,
                                                                               expected_steps or steps))
        if gp.eta is not None:
            kw["eta"] = float(gp.eta)
        if task == "txt2img":
            kw.update(width=w, height=h)
            if latents is not None:
                kw["latents"] = latents
            if cn_models:
                kw["image"] = self._cn_images(cn_images, w, h)
        else:
            kw.update(image=image, strength=strength)
            if task == "inpaint":
                kw.update(mask_image=mask, width=w, height=h)
            if cn_models:
                kw["control_image"] = self._cn_images(cn_images, w, h)
        if cn_models:
            kw.update(cn_kw)
        if pag:
            kw["pag_scale"] = pag["scale"]
            kw["pag_adaptive_scale"] = pag["adaptive_scale"]
        if xplan.photomaker:
            pm = (gp.hooks or {}).get("photomaker") or {}
            kw["input_id_images"] = [load_rgb(p) for p in (pm.get("images") or [])]
            if not kw["input_id_images"]:
                raise EngineError(INVALID_PARAMS, "PhotoMaker needs at least one ID image")
            kw["start_merge_step"] = int(xplan.photomaker["start_merge_step"])
            # PhotoMaker locates the trigger word in the token stream, so it needs the prompt text, not embeddings
            for k in ("prompt_embeds", "negative_prompt_embeds", "pooled_prompt_embeds", "negative_pooled_prompt_embeds"):
                kw.pop(k, None)
            kw["prompt"] = prompt.strip_extra_networks(gp.prompt)
            kw["negative_prompt"] = prompt.strip_extra_networks(gp.negative_prompt)
            kw["num_images_per_prompt"] = len(seeds)
            if getattr(embeds, "schedule", None):
                ctx.warn("prompt editing is not available with PhotoMaker; the first prompt is used")
            if not re.search(r"\b" + re.escape(pm.get("trigger_word", "img")) + r"\b", kw["prompt"]):
                raise EngineError(INVALID_PARAMS, f"PhotoMaker: put the trigger word '{pm.get('trigger_word', 'img')}' after "
                                                  "the class word in the prompt, e.g. 'a man img'")

        if getattr(self, "_lora_windows", None):
            ext_mod._chain_callback(kw, self._lora_window_callback())
            self._set_lora_step(1, max(int(expected_steps or steps), 1))  # state for the first step
        sch = getattr(embeds, "schedule", None)
        if sch and "prompt_embeds" in kw:
            ext_mod._chain_callback(kw, self._schedule_swapper(sch))
            avail = list(getattr(pipe, "_callback_tensor_inputs", ["latents"]))
            want = list(kw.get("callback_on_step_end_tensor_inputs") or ["latents"])
            for k in ("prompt_embeds", "add_text_embeds"):
                if k in avail and k not in want:
                    want.append(k)
            kw["callback_on_step_end_tensor_inputs"] = [k for k in want if k in avail]

        cleanups = []
        unet = pipe.unet
        self._activate(pipe)
        if pag:
            # diffusers PAG pipelines swap attention processors and restore them only after a successful call;
            # an interrupt/OOM mid-call would leave the shared UNet with PAG processors.
            saved_procs = dict(unet.attn_processors)
            cleanups.append(lambda: unet.set_attn_processor(saved_procs))
        fu = hooks_mod.hook_config(gp.hooks, "freeu", hooks_mod.FREEU_DEFAULTS[self.sdxl])
        if fu:
            pipe.enable_freeu(s1=float(fu["s1"]), s2=float(fu["s2"]), b1=float(fu["b1"]), b2=float(fu["b2"]))
            cleanups.append(pipe.disable_freeu)
        dt = hooks_mod.hook_config(gp.hooks, "dynamic_thresholding", hooks_mod.DYNTHRESH_DEFAULTS)
        if dt:
            if pag:
                ctx.warn("dynamic thresholding is disabled while PAG is active")
            else:
                ac = getattr(sched, "alphas_cumprod", None)
                if ac is None:
                    from .sigmas import default_alphas_cumprod
                    ac = torch.tensor(default_alphas_cumprod())
                hk = hooks_mod.DynThreshHook(unet, ac, sched.config.get("prediction_type", "epsilon"), cfg_scale,
                                             hooks_mod.normalize_dynthresh(dt))
                cleanups.append(hk.remove)
                hk.install()
        ht = hooks_mod.hook_config(gp.hooks, "hypertile", hooks_mod.HYPERTILE_DEFAULTS)
        if ht:
            hk2 = hooks_mod.HyperTileHook(unet, h // 8, w // 8, hooks_mod.normalize_hypertile(ht), hook_seed)
            cleanups.append(hk2.remove)
        try:
            if ht:
                hk2.install()
            ext_mod.install(self, pipe, task, gp, req, kw, ctx, cleanups, plan=xplan, w=w, h=h, sched=sched,
                            cfg_scale=cfg_scale, image=image, mask=mask, steps=steps, seed=hook_seed)
            if neutral_plan:
                nh = neutral_runtime.RuntimeHook(unet, sched, neutral_plan, cfg_scale,
                        float(hooks_mod.hook_config(gp.hooks, "neutral_prompt", {"cfg_rescale": 0})["cfg_rescale"]),
                        guided=cfg_scale > 1 and getattr(unet.config, "time_cond_proj_dim", None) is None,
                        check_interrupt=ctx.check_interrupt).install()
                cleanups.append(nh.remove)
                ext_mod._chain_callback(kw, nh.callback)
            if embeds.get("_negpip") or (neutral_plan and any(v.get("_negpip") for v in neutral_plan.branches)):
                cleanups.append(negpip.install(pipe))
            ctx.check_interrupt()
            filtered = _filter_kwargs(pipe.__call__, kw)
            for k in ("eta", "pag_scale", "pag_adaptive_scale", "controlnet_conditioning_scale", "guidance_rescale",
                      "sag_scale", "ip_adapter_image", "input_id_images"):
                if k in kw and k not in filtered:
                    ctx.warn(f"'{k}' is not supported by {type(pipe).__name__}; ignored")
            if "eta" in filtered and not _accepts(sched.step, "eta"):
                ctx.warn(f"eta is ignored by sampler '{sampler}'")
            kw = filtered
            with torch.inference_mode():
                out = pipe(**kw)
        finally:
            for c in reversed(cleanups):
                try:
                    c()
                except Exception:
                    pass
        imgs = out.images
        if output_type == "pil":
            imgs = ext_mod.post_images(self, gp, xplan, imgs, image=image, ctx=ctx)
        return imgs, plan

    def _generate(self, params: dict, ctx: JobContext, mode: str) -> dict:
        t0 = time.monotonic()
        req = self.parse_generation(params, mode, ctx)
        gp = req.params
        self._extra = dict(gp.extra or {})
        self._ms_overrides = ext_mod.model_sampling_overrides(gp)
        self.ensure_model(gp, ctx)
        seed = resolve_seed(gp.seed)
        subseed = resolve_seed(gp.subseed)
        bs, n_iter = gp.batch_size, gp.n_iter
        total = bs * n_iter
        w, h = gp.width, gp.height
        is_inpaint_model = self.kind.endswith("inpaint")

        lora_hashes = self._apply_loras(gp, req, ctx)
        hp = gp.hires
        hires = mode == "txt2img" and hp.enabled
        texts = [gp.prompt, gp.negative_prompt] + ([hp.prompt, hp.negative_prompt] if hires else [])
        gradient_enabled = hooks_mod.hook_config(gp.hooks, "weight_gradient", {}) is not None
        if (gradient_enabled and hooks_mod.hook_config(gp.hooks, "photomaker", {}) is not None
                and any(weight_gradient.has_gradient(text) for text in texts)):
            raise EngineError(UNSUPPORTED, "Weight Gradient syntax is not available with PhotoMaker's text-only encoder")
        if (hooks_mod.hook_config(gp.hooks, "negpip", {}) is not None
                and hooks_mod.hook_config(gp.hooks, "photomaker", {}) is not None
                and any(negpip.extract(text)[1] for text in texts)):
            raise EngineError(UNSUPPORTED, "NegPiP negative weights are not available with PhotoMaker's text-only encoder")
        self._load_embeddings(req, texts, ctx)
        pag = self._pag_config(gp, ctx)

        alternative = (mode == "img2img" and gp.img2img.mask is None and not is_inpaint_model
                       and hooks_mod.hook_config(gp.hooks, "img2imgalt", extras.IMG2IMGALT_DEFAULTS) is not None)
        gradient_steps = (img2img_steps(gp.steps, gp.img2img.denoising_strength)[2]
                          if mode == "img2img" and not alternative else gp.steps)
        embeds = self._encode_scheduled(gp.prompt, gp.negative_prompt, gp, bs, gp.steps, gradient_total=gradient_steps)
        hr_pos, hr_neg = (hp.prompt or gp.prompt, hp.negative_prompt or gp.negative_prompt) if hires else ("", "")
        gradient_hr = gradient_enabled and (weight_gradient.has_gradient(hr_pos) or weight_gradient.has_gradient(hr_neg))
        if hires and (hp.prompt or hp.negative_prompt or prompt_schedule.has_schedule(hr_pos)
                      or prompt_schedule.has_schedule(hr_neg) or gradient_hr):
            hr_actual = hires_steps(hp.second_pass_steps or gp.steps, hp.denoising_strength)[2]
            hr_embeds = self._encode_scheduled(hr_pos, hr_neg, gp, bs, gp.steps, hp.second_pass_steps or gp.steps,
                                               gradient_total=hr_actual)
        else:
            hr_embeds = embeds
        same_hr = hr_embeds is embeds
        embeds = self._adjust_embeds(gp, embeds, ctx)
        hr_embeds = embeds if same_hr else self._adjust_embeds(gp, hr_embeds, ctx)

        init_img = None
        job = None
        if mode == "img2img":
            init_img = load_rgb(gp.img2img.init_image)
        cn = self._controlnets(gp, req, init_img, w, h, ctx)
        if gp.subseed_strength > 0 and mode == "img2img":
            ctx.warn("variation seed is only applied in txt2img")

        out_images = []
        plan = None
        for it in range(n_iter):
            idx0 = it * bs
            seeds = [seed + idx0 + j for j in range(bs)]
            subs = [subseed + idx0 + j for j in range(bs)]
            ctx.check_interrupt()
            if mode == "txt2img":
                latent_hr = hires and hires_mod.is_latent(hp.upscaler)
                otype = "latent" if latent_hr else "pil"
                if is_inpaint_model:
                    gray = Image.new("RGB", (w, h), (127, 127, 127))
                    white = Image.new("L", (w, h), 255)
                    imgs, plan = self._run("inpaint", gp, req, ctx, sampler=gp.sampler_name, schedule=gp.scheduler,
                                           cfg_scale=gp.cfg_scale, embeds=embeds, steps=gp.steps, w=w, h=h, seeds=seeds,
                                           pass_="base", image_idx=idx0, images_total=total, image=[gray] * bs,
                                           mask=[white] * bs, strength=1.0, cn=cn, output_type=otype, pag=pag,
                                           hook_seed=seeds[0])
                else:
                    lat = self._noise(seeds, subs, gp.subseed_strength, w, h, gp)
                    imgs, plan = self._run("txt2img", gp, req, ctx, sampler=gp.sampler_name, schedule=gp.scheduler,
                                           cfg_scale=gp.cfg_scale, embeds=embeds, steps=gp.steps, w=w, h=h, seeds=seeds,
                                           pass_="base", image_idx=idx0, images_total=total, latents=lat, cn=cn,
                                           output_type=otype, pag=pag, hook_seed=seeds[0])
                if hires:
                    tw, th = hp.target_size(w, h)
                    uw, uh = hp.upscale_size(w, h)
                    if latent_hr:
                        src = hires_mod.upscale_latents(imgs, uw, uh, hp.upscaler)
                        if (uw, uh) != (tw, th):
                            l, t, r, b = center_crop_box(uw // 8, uh // 8, tw // 8, th // 8)
                            src = src[:, :, t:b, l:r]
                    else:
                        src = []
                        for im in imgs:
                            ctx.check_interrupt()
                            try:
                                up = upscalers.upscale_to(im, uw, uh, hp.upscaler, req.upscaler_paths.get(hp.upscaler),
                                                          self.device, self.dtype, ctx.check_interrupt)
                            except EngineError:
                                raise
                            except ImportError as e:
                                raise EngineError(UNSUPPORTED, f"hires upscaler '{hp.upscaler}' needs spandrel: {e}")
                            except Exception as e:
                                raise EngineError(MODEL_LOAD_FAILED, f"hires upscaler '{hp.upscaler}' failed: {e}",
                                                  traceback.format_exc())
                            if up.size != (tw, th):
                                up = up.crop(center_crop_box(up.width, up.height, tw, th, 8))
                            src.append(up)
                    n, strength, actual = hires_steps(hp.second_pass_steps or gp.steps, hp.denoising_strength)
                    task2 = "inpaint" if is_inpaint_model else "img2img"
                    if task2 == "inpaint" and latent_hr:
                        ctx.warn("latent hires upscale is not supported for inpainting models; decoding first")
                        src = self._decode(src)
                    hr_cn = (cn[0], cn[1], cn[2]) if cn[0] else None
                    imgs, _ = self._run(task2, gp, req, ctx, sampler=hp.sampler_name or gp.sampler_name,
                                        schedule=hp.scheduler or gp.scheduler,
                                        cfg_scale=hp.cfg_scale if hp.cfg_scale is not None else gp.cfg_scale,
                                        embeds=hr_embeds, steps=n, w=tw, h=th, seeds=seeds, pass_="hires", image_idx=idx0,
                                        images_total=total, image=src, mask=[Image.new("L", (tw, th), 255)] * bs,
                                        strength=strength, expected_steps=actual, cn=hr_cn, pag=pag, hook_seed=seeds[0] + 1)
            else:
                ip = gp.img2img
                n, strength, actual = img2img_steps(gp.steps, ip.denoising_strength)
                if ip.mask is not None:
                    job = inpaint.prepare(init_img, load_mask(ip.mask), w, h, ip.resize_mode, ip.mask_blur, ip.inpainting_fill,
                                          ip.inpaint_full_res, ip.inpaint_full_res_padding, ip.inpainting_mask_invert, seeds[0])
                    image, mask, task = job.image, job.mask, "inpaint"
                else:
                    image = resize_image(ip.resize_mode, init_img, w, h)
                    mask = Image.new("L", (w, h), 255)
                    task = "inpaint" if is_inpaint_model else "img2img"
                alt = hooks_mod.hook_config(gp.hooks, "img2imgalt", extras.IMG2IMGALT_DEFAULTS)
                if alt and (ip.mask is not None or is_inpaint_model):
                    ctx.warn("img2img alternative test needs plain img2img with a normal model; ignored")
                    alt = None
                if alt:
                    imgs, plan = self._img2imgalt(gp, req, ctx, alt, image, embeds, seeds, idx0, total, w, h, cn, pag)
                else:
                    imgs, plan = self._run(task, gp, req, ctx, sampler=gp.sampler_name, schedule=gp.scheduler,
                                           cfg_scale=gp.cfg_scale, embeds=embeds, steps=n, w=w, h=h, seeds=seeds,
                                           pass_="base", image_idx=idx0, images_total=total, image=[image] * bs,
                                           mask=[mask] * bs, strength=strength, expected_steps=actual, cn=cn, pag=pag,
                                           hook_seed=seeds[0])
                if job is not None:
                    imgs = [job.finalize(im) for im in imgs]
            ad = hooks_mod.hook_config(gp.hooks, "adetailer", {})
            if ad:
                imgs = self._adetailer(gp, req, ctx, ad, imgs, seeds, idx0, total, pag)
            imgs = ext_mod.post_images(self, gp, None, imgs, ctx=ctx, faces=True)
            for j, im in enumerate(imgs):
                k = idx0 + j
                path = next_output_path(req.output_dir, str(seed + k))
                save_png(im, path)
                out_images.append({"path": path, "seed": seed + k, "subseed": subseed + k,
                                   "width": im.width, "height": im.height})

        m = self.model
        return {
            "images": out_images,
            "timing_s": round(time.monotonic() - t0, 3),
            "model_name": m["name"], "model_hash": m["hash"],
            "vae_name": self.vae_info["vae_name"] if self.vae_info else None,
            "vae_hash": self.vae_info["vae_hash"] if self.vae_info else None,
            "lora_hashes": lora_hashes,
            "scheduler": plan.schedule if plan else schedulers.resolve_schedule(gp.sampler_name, gp.scheduler),
            "warnings": list(ctx.warnings),
        }

    # hooks that must not run again on ADetailer's inpainting passes (whole-image effects, or ADetailer itself)
    _ADETAILER_SKIP_HOOKS = ("adetailer", "tiling", "latent_mirroring", "multidiffusion", "kohya_hrfix", "reference",
                             "revision", "ip_adapter", "controllllite", "photomaker", "img2imgalt", "soft_inpainting")

    def _adetailer(self, gp, req, ctx, cfg, imgs, seeds, idx0, total, pag):
        """ADetailer: detect, then inpaint every detection of every unit (in unit order) at full resolution."""
        import dataclasses

        from . import adetailer as adm
        units = adm.normalize_units(cfg)
        if not units:
            return imgs
        sub = dataclasses.replace(gp, hooks={k: v for k, v in (gp.hooks or {}).items()
                                             if k not in self._ADETAILER_SKIP_HOOKS and not k.endswith("_xyz")})
        dev = "cuda" if self.cuda else "cpu"
        out = []
        for j, im in enumerate(imgs):
            seed = seeds[j] if j < len(seeds) else seeds[0]
            for ui, u in enumerate(units):
                ctx.check_interrupt()
                try:
                    masks = adm.unit_masks(im, u, dev, cfg.get("model_dirs"))
                except ImportError as e:
                    raise EngineError(UNSUPPORTED, f"ADetailer model '{u['model']}' needs the ultralytics package: {e}")
                except FileNotFoundError as e:
                    raise EngineError(INVALID_PARAMS, str(e))
                if not masks:
                    log.info("ADetailer unit %d: nothing detected", ui + 1)
                    continue
                pos = adm.unit_prompts(u["prompt"], gp.prompt, len(masks))
                neg = adm.unit_prompts(u["negative_prompt"], gp.negative_prompt, len(masks))
                if u["use_inpaint_width_height"]:
                    w, h = int(u["inpaint_width"]) // 8 * 8, int(u["inpaint_height"]) // 8 * 8
                else:
                    w, h = gp.width, gp.height
                steps = int(u["steps"]) if u["use_steps"] else gp.steps
                cfg_scale = float(u["cfg_scale"]) if u["use_cfg_scale"] else gp.cfg_scale
                sampler = u["sampler"] if u["use_sampler"] and u["sampler"] else gp.sampler_name
                n, strength, actual = img2img_steps(steps, float(u["denoising_strength"]))
                for k, m in enumerate(masks):
                    ctx.check_interrupt()
                    job = inpaint.prepare(im, m, w, h, 0, int(u["mask_blur"]), 1, bool(u["inpaint_only_masked"]),
                                          int(u["inpaint_padding"]), False, seed)
                    emb = self._adjust_embeds(sub, self._encode_scheduled(pos[k], neg[k], sub, 1, steps,
                                                                         gradient_total=actual), ctx)
                    res, _ = self._run("inpaint", sub, req, ctx, sampler=sampler, schedule=gp.scheduler,
                                       cfg_scale=cfg_scale, embeds=emb, steps=n, w=w, h=h, seeds=[seed],
                                       pass_="adetailer", image_idx=idx0 + j, images_total=total, image=[job.image],
                                       mask=[job.mask], strength=strength, expected_steps=actual, pag=pag,
                                       hook_seed=seed)
                    redrawn = res[0]
                    if u["color_correction"]:
                        redrawn = adm.match_colors(redrawn, job.image)
                    im = job.finalize(redrawn)
            out.append(im)
        return out

    def _img2imgalt(self, gp, req, ctx, alt, image, embeds, seeds, idx0, total, w, h, cn, pag):
        """img2imgalt.py: invert the init image to noise with DDIM, then sample from that noise."""
        if "_neutral" in embeds:
            raise EngineError(UNSUPPORTED, "Neutral Prompt integration pending for img2img Alternative")
        c = adapters.normalize(alt, extras.IMG2IMGALT_DEFAULTS)
        enc = embeds
        if c["override_prompt"] or c["override_negative"]:
            enc = self._encode(c["override_prompt"] or gp.prompt, c["override_negative"] or gp.negative_prompt, gp, 1)
        self._activate(self.pipe)
        sched, _ = self._scheduler(gp.sampler_name, gp.scheduler, ctx)
        self.pipe.scheduler = sched
        lat = ext_mod.encode_image(self, self.pipe, image)
        steps = int(c["inversion_steps"]) or gp.steps
        remove_negpip = negpip.install(self.pipe) if enc.get("_negpip") else lambda: None
        try:
            inv = extras.invert_latents(self, self.pipe, lat, enc, steps, float(c["decode_cfg_scale"]), w, h,
                                        ctx.check_interrupt)
        finally:
            remove_negpip()
        noise = inv.repeat(len(seeds), 1, 1, 1)
        r = float(c["randomness"])
        if r > 0:
            rnd = self._noise(seeds, seeds, 0.0, w, h, gp).to(noise.device, noise.dtype)
            noise = ((1 - r) * noise + r * rnd) / math.sqrt((1 - r)**2 + r**2)
        # DDIM's inverse state has meaningful amplitude. Empirical std normalization changes the recovered
        # signal (and even changes a pure seeded Gaussian at randomness=1), causing excessive contrast.
        return self._run("txt2img", gp, req, ctx, sampler=gp.sampler_name, schedule=gp.scheduler, cfg_scale=gp.cfg_scale,
                         embeds=embeds, steps=gp.steps, w=w, h=h, seeds=seeds, pass_="base", image_idx=idx0,
                         images_total=total, latents=noise, cn=cn, pag=pag, hook_seed=seeds[0])

    def _decode(self, latents):
        pipe = self.pipe
        vae = pipe.vae
        with torch.inference_mode():
            needs_upcast = vae.dtype == torch.float16 and getattr(vae.config, "force_upcast", False)
            if needs_upcast:
                vae.to(torch.float32)
            lat = latents.to(vae.device, vae.dtype) / vae.config.scaling_factor
            img = vae.decode(lat, return_dict=False)[0]
            if needs_upcast:
                vae.to(torch.float16)
        return pipe.image_processor.postprocess(img, output_type="pil")

    # -- extras ----------------------------------------------------------------
    def restore_faces(self, params: dict, ctx: JobContext) -> dict:
        path = params.get("model") or ""
        if not os.path.isfile(path):
            raise EngineError(INVALID_PARAMS, f"face restoration model not found: {path}")
        img = load_rgb(params["image_path"])
        try:
            out = extras.restore_faces_image(img, path, float(params.get("visibility", 1.0)),
                                             float(params.get("codeformer_weight", 0.5)), self.device)
        except ImportError as e:
            raise EngineError(UNSUPPORTED, f"face restoration needs spandrel and opencv: {e}")
        dst = next_output_path(params["output_dir"], "faces")
        save_png(out, dst)
        return {"path": dst}

    def caption(self, params: dict, ctx: JobContext) -> dict:
        img = load_rgb(params["image_path"])
        try:
            text = extras.caption_image(img, str(params.get("method", "blip")), float(params.get("threshold", 0.35)),
                                        self.device)
        except ImportError as e:
            raise EngineError(UNSUPPORTED, f"captioning needs transformers: {e}")
        return {"caption": text}

    def run_code(self, params: dict, ctx: JobContext) -> dict:
        ns = {"engine": self, "pipe": self.pipe, "torch": torch, "Image": Image, "params": params, "ctx": ctx}
        res = extras.run_code(str(params.get("code", "")), ns)
        paths = []
        for i, im in enumerate(res.pop("images")):
            dst = next_output_path(params.get("output_dir") or ".", f"code{i}")
            save_png(im, dst)
            paths.append(dst)
        res["images"] = paths
        return res

    def img2vid(self, params: dict, ctx: JobContext) -> dict:
        return extras.img2vid(self, params, ctx)

    def novel_view(self, params: dict, ctx: JobContext) -> dict:
        return extras.novel_view(self, params, ctx)

    # -- tools -----------------------------------------------------------------
    def preprocess(self, params: dict, ctx: JobContext) -> dict:
        req = self.parse_preprocess(params)
        img = load_rgb(req.image_path)
        try:
            out = cn_mod.run_preprocessor(req.preprocessor, img, req.args)
        except Exception as e:
            raise EngineError(INVALID_PARAMS, f"preprocessor '{req.preprocessor}' failed: {e}", traceback.format_exc())
        path = next_output_path(req.output_dir, f"pre-{req.preprocessor}")
        save_png(out, path)
        return {"path": path}

    def upscale(self, params: dict, ctx: JobContext) -> dict:
        req = self.parse_upscale(params)
        img = load_rgb(req.image_path)
        try:
            out = upscalers.upscale_by(img, req.scale, req.upscaler, req.upscaler_paths.get(req.upscaler),
                                       self.device, self.dtype, ctx.check_interrupt)
        except EngineError:
            raise
        except ImportError as e:
            raise EngineError(UNSUPPORTED, f"upscaler '{req.upscaler}' needs spandrel: {e}")
        except Exception as e:
            raise EngineError(MODEL_LOAD_FAILED, f"upscaler '{req.upscaler}' failed: {e}", traceback.format_exc())
        path = next_output_path(req.output_dir, f"up-{req.upscaler}")
        save_png(out, path)
        return {"path": path}


SAG_SCHEDULERS = ("DDIMScheduler", "PNDMScheduler", "DDPMScheduler", "DEISMultistepScheduler", "UniPCMultistepScheduler",
                  "DPMSolverMultistepScheduler", "DPMSolverSinglestepScheduler")

__all__ = ["Engine", "slerp", "LATENT_UPSCALERS", "math"]
