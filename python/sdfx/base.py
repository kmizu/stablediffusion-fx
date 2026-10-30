"""Engine base class: request handlers shared by the real engine and the dry-run FakeEngine."""
from __future__ import annotations

import os
import platform
import sys
from typing import Optional

from . import ENGINE_VERSION, PROTOCOL_VERSION, models, schedulers
from .params import (GenerationRequest, LoadModelRequest, ParamError, PreprocessRequest, UpscaleRequest, from_dict,
                     parse_generation_request)
from .protocol import (INVALID_PARAMS, MODEL_LOAD_FAILED, NO_MODEL, UNSUPPORTED, EngineError, JobContext)
from .waifu2x import NAMES as WAIFU2X_UPSCALERS

LATENT_UPSCALERS = ["Latent", "Latent (antialiased)", "Latent (bicubic)", "Latent (bicubic antialiased)",
                    "Latent (nearest)", "Latent (nearest-exact)"]
PIXEL_UPSCALERS = ["None", "Lanczos", "Nearest", "LDSR", *WAIFU2X_UPSCALERS]


def ram_used_mb() -> float:
    try:
        import psutil  # type: ignore
        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return round(pmc.WorkingSetSize / (1024 * 1024), 1)
        except Exception:
            return 0.0
        return 0.0
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / 1024.0 if sys.platform != "darwin" else r / (1024.0 * 1024.0), 1)
    except Exception:
        return 0.0


class BaseEngine:
    dry_run = False

    def __init__(self):
        import time
        self.system_started_at = time.time()
        self.model: Optional[dict] = None   # LoadedModel-shaped dict
        self.load_opts = LoadModelRequest()

    # -- info ---------------------------------------------------------------
    def hello(self) -> dict:
        return {
            "protocol": PROTOCOL_VERSION, "engine_version": ENGINE_VERSION,
            "python": platform.python_version(), "torch": None, "diffusers": None, "cuda": False,
            "device": "cpu", "vram_mb": 0, "dry_run": self.dry_run, "features": self.features(),
        }

    def features(self) -> list[str]:
        return ["prevent_interrupt", *self._batch_features()]

    def _batch_features(self) -> list[str]:
        if self.dry_run:
            return []
        service = getattr(self, "_batch_interrogate_service", None)
        if service is None:
            return []
        from .batch_interrogate_worker import BatchInterrogateService
        if isinstance(service, BatchInterrogateService) and service.available():
            return ["batch_interrogate_wd_v3_partial"]
        return []

    def temporal_lab(self, params: dict, ctx: JobContext) -> dict:
        """Run an owned, serialized native frame-processing request."""
        from . import temporal_lab
        return temporal_lab.run(params, ctx)

    def inpaint_difference_preview(self, params: dict, ctx: JobContext) -> dict:
        try:
            from .inpaint_difference import preview_request
        except ImportError as exc:
            raise EngineError(UNSUPPORTED, 'Inpaint Difference requires OpenCV, NumPy and Pillow') from exc
        try:
            return preview_request(params, ctx)
        except (ValueError, OSError) as exc:
            raise EngineError(INVALID_PARAMS, f'Inpaint Difference preview: {exc}') from exc

    def difference_generation(self, params: dict, mode: str, ctx: JobContext):
        try:
            from .inpaint_difference import generation_request
        except ImportError as exc:
            raise EngineError(UNSUPPORTED, 'Inpaint Difference requires OpenCV, NumPy and Pillow') from exc
        return generation_request(params, mode, ctx)

    def _batch_service(self):
        service = getattr(self, "_batch_interrogate_service", None)
        if self.dry_run or service is None:
            raise EngineError(UNSUPPORTED, "no trusted WD batch service is configured")
        from .batch_interrogate_worker import BatchInterrogateService
        if not isinstance(service, BatchInterrogateService):
            raise EngineError(UNSUPPORTED, "no trusted WD batch service is configured")
        return service

    def batch_interrogate_prepare(self, params: dict, ctx: JobContext) -> dict:
        return self._batch_service().prepare(self, params, ctx)

    def batch_interrogate_execute(self, params: dict, ctx: JobContext) -> dict:
        return self._batch_service().execute(self, params, ctx)

    def preprocessors(self) -> list[str]:
        return ["none"]

    # -- listing -----------------------------------------------------------
    def list_models(self, params: dict, ctx: JobContext) -> dict:
        dirs = _str_list(params.get("dirs"), "dirs")
        return {"models": models.scan_models(dirs)}

    def list_files(self, params: dict, ctx: JobContext) -> dict:
        kind = params.get("kind")
        if kind not in models.FILE_KINDS:
            raise EngineError(INVALID_PARAMS, f"kind must be one of {', '.join(models.FILE_KINDS)} (got {kind!r})")
        return {"files": models.scan_files(kind, _str_list(params.get("dirs"), "dirs"))}

    def list_samplers(self, params: dict, ctx: JobContext) -> dict:
        return {"samplers": schedulers.sampler_list(check_backend=not self.dry_run), "schedulers": list(schedulers.SCHEDULES),
                "upscalers": LATENT_UPSCALERS + PIXEL_UPSCALERS, "preprocessors": self.preprocessors()}

    def unload(self, params: dict, ctx: JobContext) -> dict:
        self.model = None
        return {}

    def on_oom(self) -> None:
        pass

    # extras: engines that cannot do them answer `unsupported`
    def _no_extra(self, what: str):
        raise EngineError(UNSUPPORTED, f"{what} is not available in this engine")

    # -- model tools (CPU, safetensors; work without a loaded model) -----------------------------------------
    def _model_tool(self, what: str, inputs: list, out: str, overwrite: bool, run):
        import os
        for p in inputs:
            if not p or not os.path.isfile(p):
                raise EngineError(INVALID_PARAMS, f"{what}: model not found: {p}")
        if not out or not out.lower().endswith(".safetensors"):
            raise EngineError(INVALID_PARAMS, f"{what}: output must be a .safetensors path")
        if any(os.path.abspath(out) == os.path.abspath(p) for p in inputs):
            raise EngineError(INVALID_PARAMS, f"{what}: output would overwrite an input model")
        if os.path.exists(out) and not overwrite:
            raise EngineError(INVALID_PARAMS, f"{what}: {os.path.basename(out)} already exists")
        try:
            return run()
        except ImportError as e:
            raise EngineError(UNSUPPORTED, f"{what} needs torch and safetensors: {e}")
        except (ValueError, KeyError) as e:
            raise EngineError(INVALID_PARAMS, f"{what}: {e}")
        finally:
            if os.path.exists(out + ".part"):
                try:
                    os.remove(out + ".part")
                except OSError:
                    pass

    def tokenize(self, params: dict, ctx: JobContext) -> dict:
        if self.dry_run:
            raise EngineError(UNSUPPORTED, "token inspection requires a real loaded model; dry-run has no tokenizer")
        if self.model is None or getattr(self, "pipe", None) is None:
            raise EngineError(NO_MODEL, "load a checkpoint before inspecting its tokenizer")
        from .tokenizer_tool import inspect
        return inspect(self.pipe, params, ctx)

    def vae_tools(self, params: dict, ctx: JobContext) -> dict:
        if self.dry_run:
            raise EngineError(UNSUPPORTED, 'VAE Tools requires a real loaded VAE; dry-run cannot encode or decode latents')
        if self.model is None or getattr(self, 'pipe', None) is None:
            raise EngineError(NO_MODEL, 'load a checkpoint before using VAE Tools')
        from .vae_tools import run
        try:
            return run(self, params, ctx)
        except (ValueError, TypeError, OSError) as exc:
            raise EngineError(INVALID_PARAMS, f'VAE Tools: {exc}') from exc

    def image_tool(self, params: dict, ctx: JobContext) -> dict:
        """Non-diffusion image tools (sdfx.image_tools): {tool, image_path, output_dir, ...tool params}."""
        from . import image_tools
        from .images import next_output_path, save_png
        tool = str(params.get("tool", ""))
        if tool not in image_tools.TOOLS:
            raise EngineError(INVALID_PARAMS, f"unknown image tool '{tool}'")
        src = params.get("image_path")
        if not src:
            raise EngineError(INVALID_PARAMS, "image_path is required")
        from PIL import Image, ImageOps
        with Image.open(src) as source:
            oriented = ImageOps.exif_transpose(source)
            img = oriented.convert("RGBA" if oriented.mode in ("RGBA", "LA", "P") else "RGB")
        dev = "cuda" if getattr(self, "cuda", False) else "cpu"
        try:
            if tool == "waifu2x":
                if self.dry_run:
                    raise EngineError(UNSUPPORTED, "Waifu2x requires the real engine; dry-run does not execute neural upscaling")
                ctx.check_interrupt()
                out, extra = image_tools.run(tool, img, params, dev, check_interrupt=ctx.check_interrupt,
                                            progress=lambda done, total: ctx.progress(done, total, 'waifu2x'))
                ctx.check_interrupt()
            else:
                out, extra = image_tools.run(tool, img if tool != "remove_background" else img.convert("RGB"), params, dev)
        except ImportError as e:
            raise EngineError(UNSUPPORTED, f"{tool} needs an optional package: {e}")
        except (ValueError, FileNotFoundError) as e:
            raise EngineError(INVALID_PARAMS, f"{tool}: {e}")
        od = params.get("output_dir") or "."
        res = {"path": save_png(out, next_output_path(od, tool)), "outputs": {}, "info": {}}
        for k, v in extra.items():
            if hasattr(v, "save"):
                res["outputs"][k] = save_png(v, next_output_path(od, f"{tool}-{k}"))
            else:
                res["info"][k] = v
        return res

    def merge_models(self, params: dict, ctx: JobContext) -> dict:
        from . import model_tools
        a, b, c = params.get("model_a"), params.get("model_b"), params.get("model_c") or None
        mode = str(params.get("mode", "Weight sum"))
        out = str(params.get("output", ""))
        inputs = [a, b] + ([c] if c and mode != "Weight sum" else [])
        return self._model_tool("merge", inputs, out, bool(params.get("overwrite", False)), lambda: model_tools.merge(
            a, b, out, c=c, mode=mode, alpha=float(params.get("alpha", 0.5)), beta=float(params.get("beta", 0.0)),
            block_weights=str(params.get("block_weights", "") or ""), vae_source=str(params.get("vae_source", "merge")),
            precision=str(params.get("precision", "fp16")),
            progress=lambda i, n: ctx.progress(i, n, "merge"), check=ctx.check_interrupt))

    def convert_model(self, params: dict, ctx: JobContext) -> dict:
        from . import model_tools
        src, out = params.get("model"), str(params.get("output", ""))
        vae = params.get("bake_vae") or None
        return self._model_tool("convert", [src] + ([vae] if vae else []), out, bool(params.get("overwrite", False)),
                                lambda: model_tools.convert(
            src, out, precision=str(params.get("precision", "fp16")), ema=str(params.get("ema", "no-ema")),
            fix_clip=bool(params.get("fix_clip", True)), drop_junk=bool(params.get("drop_junk", True)), bake_vae=vae,
            progress=lambda i, n: ctx.progress(i, n, "convert"), check=ctx.check_interrupt))

    def restore_faces(self, params: dict, ctx: JobContext) -> dict:
        self._no_extra("face restoration")

    def caption(self, params: dict, ctx: JobContext) -> dict:
        self._no_extra("captioning")

    def run_code(self, params: dict, ctx: JobContext) -> dict:
        self._no_extra("custom code")

    def img2vid(self, params: dict, ctx: JobContext) -> dict:
        self._no_extra("img2vid")

    def novel_view(self, params: dict, ctx: JobContext) -> dict:
        self._no_extra("novel view")

    def memory(self, params: dict, ctx: JobContext) -> dict:
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "total_mb": 0.0, "ram_used_mb": ram_used_mb()}

    def system_status(self, params: dict, ctx: JobContext) -> dict:
        from .system_info import snapshot
        from .hardware_info import cached_profile
        data = snapshot(self)
        data['hardware_info'] = cached_profile(self)
        return data

    def system_info(self, params: dict, ctx: JobContext) -> dict:
        from .system_info import snapshot, inventory
        from .hardware_info import cached_profile
        data = snapshot(self, full=True)
        data['hardware_info'] = cached_profile(self)
        if 'dirs' in params:
            data['networks'] = inventory(self, params['dirs'], params.get('hashes', False), ctx)
        return data

    def vram_estimator(self, params: dict, ctx: JobContext) -> dict:
        from .vram_estimator import dispatch
        return dispatch(self, params, ctx)

    def system_benchmark(self, params: dict, ctx: JobContext) -> dict:
        from .system_info import benchmark
        previous = getattr(self, '_benchmark_no_save', False)
        self._benchmark_no_save = True
        try:
            return benchmark(self, params, ctx)
        finally:
            self._benchmark_no_save = previous

    def system_submission_preview(self, params: dict, ctx: JobContext) -> dict:
        from .system_info import submission_preview
        return submission_preview(self, params)

    def system_submit(self, params: dict, ctx: JobContext) -> dict:
        from .system_info import submit
        return submit(self, params)

    # -- helpers -----------------------------------------------------------
    def parse_load(self, params: dict) -> LoadModelRequest:
        req = from_dict(LoadModelRequest, params, "params")
        if not req.path:
            raise EngineError(INVALID_PARAMS, "path is required")
        if not os.path.exists(req.path):
            raise EngineError(MODEL_LOAD_FAILED, f"model file not found: {req.path}")
        if req.dtype not in ("fp16", "bf16", "fp32"):
            raise EngineError(INVALID_PARAMS, f"dtype must be fp16|bf16|fp32 (got {req.dtype})")
        if req.vram_mode not in ("gpu", "model_offload", "sequential_offload"):
            raise EngineError(INVALID_PARAMS, f"vram_mode must be gpu|model_offload|sequential_offload (got {req.vram_mode})")
        if req.vae_path and not os.path.exists(req.vae_path):
            raise EngineError(MODEL_LOAD_FAILED, f"VAE file not found: {req.vae_path}")
        return req

    def parse_generation(self, params: dict, mode: str, ctx: JobContext) -> GenerationRequest:
        try:
            req = parse_generation_request(params)
        except ParamError as e:
            raise EngineError(INVALID_PARAMS, str(e))
        gp = req.params
        problems = gp.validate()
        if problems:
            raise EngineError(INVALID_PARAMS, "; ".join(problems))
        from .hooks import hook_config
        latent = hook_config(gp.hooks, "latent_upscale", {})
        if latent is not None:
            from . import latent_upscale
            latent_upscale.normalize(latent)
            if mode != "img2img" or gp.img2img is None:
                raise EngineError(INVALID_PARAMS, "Latent Upscale needs an img2img input image")
            latent_upscale.full_steps(gp.steps, gp.img2img.denoising_strength)
            if hook_config(gp.hooks, "img2imgalt", {}) is not None:
                raise EngineError(UNSUPPORTED, "Latent Upscale cannot be combined with img2img Alternative inversion")
        if mode == "img2img":
            if gp.img2img is None:
                raise EngineError(INVALID_PARAMS, "img2img params are required for img2img")
            if not gp.img2img.init_image or not os.path.isfile(gp.img2img.init_image):
                raise EngineError(INVALID_PARAMS, f"init image not found: {gp.img2img.init_image}")
            if gp.img2img.mask is not None and not os.path.isfile(gp.img2img.mask):
                raise EngineError(INVALID_PARAMS, f"mask image not found: {gp.img2img.mask}")
            if gp.img2img.resize_mode not in (0, 1, 2):
                raise EngineError(INVALID_PARAMS, f"resize_mode must be 0, 1 or 2 (got {gp.img2img.resize_mode})")
            if gp.img2img.inpainting_fill not in (0, 1, 2, 3):
                raise EngineError(INVALID_PARAMS, f"inpainting_fill must be 0..3 (got {gp.img2img.inpainting_fill})")
        try:
            schedulers.sampler_spec(gp.sampler_name)
            schedulers.resolve_schedule(gp.sampler_name, gp.scheduler)
            if gp.hires.enabled and gp.hires.sampler_name:
                schedulers.sampler_spec(gp.hires.sampler_name)
            if gp.hires.enabled and gp.hires.scheduler:
                schedulers.resolve_schedule(gp.hires.sampler_name or gp.sampler_name, gp.hires.scheduler)
        except ValueError as e:
            raise EngineError(INVALID_PARAMS, str(e))
        samplers = [gp.sampler_name] + ([gp.hires.sampler_name] if gp.hires.enabled and gp.hires.sampler_name else [])
        for sn in samplers:
            if not self.sampler_supported(sn):
                raise EngineError(UNSUPPORTED, f"sampler '{sn}' is not available in this engine")
        if gp.hires.enabled and mode == "txt2img":
            up = gp.hires.upscaler
            if up not in LATENT_UPSCALERS and up not in PIXEL_UPSCALERS:
                self.check_model_upscaler(up, req.upscaler_paths)
        for lr in gp.loras + (req.hires_loras or []) + (req.manual_loras or []):
            if lr.name not in req.lora_paths:
                raise EngineError(INVALID_PARAMS, f"no path for LoRA '{lr.name}' in lora_paths")
            if not os.path.isfile(req.lora_paths[lr.name]):
                raise EngineError(INVALID_PARAMS, f"LoRA file not found: {req.lora_paths[lr.name]}")
        for u in gp.controlnet:
            if not u.enabled:
                continue
            path = req.controlnet_paths.get(u.model, u.model)
            if u.model not in req.controlnet_paths and not os.path.exists(u.model):
                raise EngineError(INVALID_PARAMS, f"no path for ControlNet model '{u.model}' in controlnet_paths")
            if not os.path.exists(path):
                raise EngineError(INVALID_PARAMS, f"ControlNet model not found: {path}")
            if u.image and not os.path.isfile(u.image):
                raise EngineError(INVALID_PARAMS, f"ControlNet image not found: {u.image}")
            if not u.image and mode != "img2img":
                raise EngineError(INVALID_PARAMS, f"ControlNet unit '{u.model}' has no image")
            if u.preprocessor not in ("none", "") and u.preprocessor not in self.preprocessors():
                ctx.warn(f"ControlNet preprocessor '{u.preprocessor}' is not available; using the image as-is")
        from .prompt import unsupported_syntax_warnings
        for index, t in enumerate((gp.prompt, gp.negative_prompt)):
            if index == 0:
                from .hooks import hook_config
                if any(hook_config(gp.hooks, name, {}) is not None
                       for name in ("neutral_prompt", "composable_lora")):
                    continue
            for w in unsupported_syntax_warnings(t):
                ctx.warn(w)
        if req.preview_method not in ("approx", "taesd", "none"):
            ctx.warn(f"unknown preview_method '{req.preview_method}'; using approx")
        if not req.output_dir:
            raise EngineError(INVALID_PARAMS, "output_dir is required")
        if mode == "img2img" and gp.hires.enabled:
            ctx.warn("hires fix is ignored for img2img")
        return req

    def ensure_model(self, gp, ctx: JobContext) -> None:
        """Auto-load params.checkpoint when it is an existing path that differs from the loaded model."""
        ck = gp.checkpoint
        if ck and os.path.isfile(ck):
            if self.model is None or os.path.normcase(os.path.abspath(self.model["path"])) != os.path.normcase(os.path.abspath(ck)):
                ctx.log("info", f"loading checkpoint {os.path.basename(ck)}")
                opts = self.load_opts
                self.load_model({"path": ck, "dtype": opts.dtype, "vram_mode": opts.vram_mode,
                                 "vae_path": opts.vae_path, "vae_tiling": opts.vae_tiling}, ctx)
        if self.model is None:
            raise EngineError(NO_MODEL, "no model loaded (send load_model first)")

    def sampler_supported(self, name: str) -> bool:
        return True

    def check_model_upscaler(self, name: str, upscaler_paths: dict) -> None:
        """Model upscalers (ESRGAN etc.) must have an existing file in upscaler_paths."""
        if name not in upscaler_paths:
            raise EngineError(INVALID_PARAMS, f"unknown upscaler '{name}'")
        if not os.path.isfile(upscaler_paths[name]):
            raise EngineError(INVALID_PARAMS, f"upscaler model not found: {upscaler_paths[name]}")

    def parse_upscale(self, params: dict) -> UpscaleRequest:
        req = from_dict(UpscaleRequest, params, "params")
        if not req.image_path or not os.path.isfile(req.image_path):
            raise EngineError(INVALID_PARAMS, f"image not found: {req.image_path}")
        if not (0 < req.scale <= 16):
            raise EngineError(INVALID_PARAMS, f"scale must be in (0, 16] (got {req.scale})")
        if not req.output_dir:
            raise EngineError(INVALID_PARAMS, "output_dir is required")
        if req.upscaler not in PIXEL_UPSCALERS:
            self.check_model_upscaler(req.upscaler, req.upscaler_paths)
        return req

    def parse_preprocess(self, params: dict) -> PreprocessRequest:
        req = from_dict(PreprocessRequest, params, "params")
        if not req.image_path or not os.path.isfile(req.image_path):
            raise EngineError(INVALID_PARAMS, f"image not found: {req.image_path}")
        if not req.output_dir:
            raise EngineError(INVALID_PARAMS, "output_dir is required")
        if req.preprocessor not in self.preprocessors():
            raise EngineError(UNSUPPORTED, f"preprocessor '{req.preprocessor}' is not available")
        return req


def _str_list(v, name: str) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise EngineError(INVALID_PARAMS, f"{name} must be an array of strings")
    return v
