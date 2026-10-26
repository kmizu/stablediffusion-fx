"""Dry-run engine: same request surface as the real engine, never imports torch/diffusers.

Generation writes deterministic seed-based gradient+noise PNGs (Pillow, or a pure-python PNG writer),
emits progress for each step of each pass, 1-2 previews per pass, and honours interrupt.
"""
from __future__ import annotations

import os
import random
import shutil
import time
from typing import Optional

from . import inpaint, models, schedulers

try:  # warm optional imports on the main thread: a first numpy import inside the worker thread can stall
    import numpy  # noqa: F401  (for many seconds on Windows when stdio are pipes)
except ImportError:
    pass
from .base import LATENT_UPSCALERS, BaseEngine
from .images import (HAVE_PIL, Image, ImageFilter, ImageOps, load_mask, load_rgb, next_output_path, png_size,
                     resize_image, save_png, to_jpeg_b64, write_png_rgb)
from . import sampling_ext
from .params import hires_steps, img2img_steps, resolve_seed
from .protocol import INVALID_PARAMS, MODEL_LOAD_FAILED, UNSUPPORTED, EngineError, JobContext


def _colors(seed: int):
    rnd = random.Random(seed)
    return tuple(rnd.randrange(256) for _ in range(3)), tuple(rnd.randrange(256) for _ in range(3)), rnd


def fake_image(width: int, height: int, seed: int, subseed: int = -1, subseed_strength: float = 0.0):
    """Deterministic colourful gradient + noise image (PIL)."""
    c1, c2, rnd = _colors(seed)
    grad = Image.linear_gradient("L").rotate(rnd.choice([0, 45, 90, 135, 180, 225, 270, 315]))
    grad = grad.resize((width, height), Image.BILINEAR)
    img = ImageOps.colorize(grad, c1, c2)
    nw, nh = max(1, width // 8), max(1, height // 8)
    noise = Image.frombytes("RGB", (nw, nh), rnd.randbytes(nw * nh * 3)).resize((width, height), Image.BICUBIC)
    img = Image.blend(img, noise, 0.25)
    if subseed_strength > 0 and subseed >= 0:
        img = Image.blend(img, fake_image(width, height, subseed), min(1.0, subseed_strength))
    return img


def fake_rows(width: int, height: int, seed: int) -> list[bytes]:
    """Pure-python fallback pixels (no Pillow)."""
    c1, c2, rnd = _colors(seed)
    rows = []
    for y in range(height):
        t = y / max(1, height - 1)
        base = [int(c1[i] * (1 - t) + c2[i] * t) for i in range(3)]
        row = bytearray()
        noise = rnd.randbytes(width)
        for x in range(width):
            n = noise[x] // 8 - 16
            row += bytes((min(255, max(0, base[0] + n)), min(255, max(0, base[1] + n)), min(255, max(0, base[2] + n))))
        rows.append(bytes(row))
    return rows


class FakeEngine(BaseEngine):
    dry_run = True

    def __init__(self, delay_ms: int = 0):
        super().__init__()
        self.delay = max(0, int(delay_ms)) / 1000.0
        self.vae: Optional[dict] = None

    def features(self) -> list[str]:
        return ["freeu", "dynamic_thresholding", "hypertile", "pag", "controlnet", "apg", "mahiro", "rescale_cfg",
                "model_sampling", "kohya_hrfix", "multidiffusion", "latent_modifier", "soft_inpainting", "sag", "neveroom",
                "reference", "revision", "ip_adapter", "controllllite", "fooocus_inpaint", "photomaker", "img2imgalt",
                "face_restore", "caption", "img2vid", "novel_view", "ldsr", "marigold", "adetailer", "weight_gradient"] + list(sampling_ext.HOOKS) + (
                    ["custom_code"] if os.environ.get("SDFX_ALLOW_CODE") == "1" else [])

    def preprocessors(self) -> list[str]:
        return (["none", "canny", "depth", "invert", "inpaint", "recolor_luminance", "recolor_intensity", "tile_resample",
                 "tile_colorfix", "blur_gaussian", "depth_marigold", "normal_marigold"] if HAVE_PIL else ["none"])

    # -- models ------------------------------------------------------------
    def load_model(self, params: dict, ctx: JobContext) -> dict:
        t0 = time.monotonic()
        req = self.parse_load(params)
        kind = models.detect_kind(req.path)
        if kind == "unsupported":
            raise EngineError(MODEL_LOAD_FAILED, f"unsupported model architecture: {os.path.basename(req.path)}")
        self.load_opts = req
        self.vae = None
        if req.vae_path:
            self.vae = {"vae_name": models.model_name(req.vae_path), "vae_hash": models.path_hash(req.vae_path)}
        self.model = {"name": models.model_name(req.path), "path": req.path,
                      "kind": "sd15" if kind == "unknown" else kind, "hash": models.short_hash(req.path)}
        return self._loaded(int((time.monotonic() - t0) * 1000))

    def _loaded(self, ms: int = 0) -> dict:
        m = dict(self.model)
        m["loaded_ms"] = ms
        m["vae_name"] = self.vae["vae_name"] if self.vae else None
        m["vae_hash"] = self.vae["vae_hash"] if self.vae else None
        return m

    def set_vae(self, params: dict, ctx: JobContext) -> dict:
        if self.model is None:
            raise EngineError("no_model", "no model loaded")
        p = params.get("path")
        if p:
            if not os.path.exists(p):
                raise EngineError(MODEL_LOAD_FAILED, f"VAE file not found: {p}")
            self.vae = {"vae_name": models.model_name(p), "vae_hash": models.path_hash(p)}
        else:
            self.vae = None
        self.load_opts.vae_path = p or None
        return self._loaded()

    def unload(self, params: dict, ctx: JobContext) -> dict:
        self.model = None
        self.vae = None
        return {}

    # -- generation --------------------------------------------------------
    def txt2img(self, params: dict, ctx: JobContext) -> dict:
        return self._generate(params, ctx, "txt2img")

    def img2img(self, params: dict, ctx: JobContext) -> dict:
        return self._generate(params, ctx, "img2img")

    def _sleep_step(self, ctx: JobContext):
        ctx.check_interrupt()
        if self.delay:
            end = time.monotonic() + self.delay
            while time.monotonic() < end:
                ctx.check_interrupt()
                time.sleep(min(0.01, max(0.0, end - time.monotonic())))
        ctx.check_interrupt()

    def _run_pass(self, ctx: JobContext, steps: int, pass_: str, image: int, images: int, seed: int,
                  preview_every: int, size: tuple[int, int]):
        ctx.start_pass()
        previews = 0
        for step in range(1, steps + 1):
            self._sleep_step(ctx)
            ctx.progress(step, steps, pass_, image=image, images=images)
            if HAVE_PIL and preview_every > 0 and previews < 2 and (step % preview_every == 0 or step == steps):
                w, h = size
                f = 64 / max(w, h)
                pv = fake_image(max(8, int(w * f)), max(8, int(h * f)), seed + step)
                ctx.preview(to_jpeg_b64(pv), step)
                previews += 1

    def _generate(self, params: dict, ctx: JobContext, mode: str) -> dict:
        t0 = time.monotonic()
        req = self.parse_generation(params, mode, ctx)
        gp = req.params
        try:
            self.ensure_model(gp, ctx)
            m = self.model
        except EngineError as e:
            if e.code != "no_model":
                raise
            # dry run stays usable without any checkpoint (UI development / smoke tests)
            m = {"name": "dry-run", "path": "", "kind": "sd15", "hash": "0000000000"}
            ctx.log("info", "dry run: no model loaded, using placeholder model 'dry-run'")
        seed = resolve_seed(gp.seed)
        subseed = resolve_seed(gp.subseed)
        bs, n_iter = gp.batch_size, gp.n_iter
        total = bs * n_iter
        w, h = gp.width, gp.height
        hires = mode == "txt2img" and gp.hires.enabled
        out_w, out_h = gp.hires.target_size(w, h) if hires else (w, h)

        if mode == "img2img":
            base_steps = img2img_steps(gp.steps, gp.img2img.denoising_strength)[2]
        else:
            base_steps = gp.steps
        hr_steps = hires_steps(gp.hires.second_pass_steps or gp.steps, gp.hires.denoising_strength)[2] if hires else 0


        init = job = None
        if mode == "img2img" and HAVE_PIL:
            init = load_rgb(gp.img2img.init_image)

        images = []
        for it in range(n_iter):
            idx0 = it * bs
            self._run_pass(ctx, base_steps, "base", idx0, total, seed + idx0, req.preview_every, (w, h))
            if hires:
                self._run_pass(ctx, hr_steps, "hires", idx0, total, seed + idx0 + 7, req.preview_every, (out_w, out_h))
            for j in range(bs):
                k = idx0 + j
                s, ss = seed + k, subseed + k
                path = next_output_path(req.output_dir, str(s))
                if HAVE_PIL:
                    img = fake_image(out_w, out_h, s, ss, gp.subseed_strength)
                    if init is not None:
                        ip = gp.img2img
                        if ip.mask is not None:
                            job = inpaint.prepare(init, load_mask(ip.mask), w, h, ip.resize_mode, ip.mask_blur,
                                                  ip.inpainting_fill, ip.inpaint_full_res, ip.inpaint_full_res_padding,
                                                  ip.inpainting_mask_invert, s)
                            img = job.finalize(Image.blend(job.image, img, ip.denoising_strength))
                        else:
                            base = resize_image(ip.resize_mode, init, w, h)
                            img = Image.blend(base, img, ip.denoising_strength)
                    save_png(img, path)
                    iw, ih = img.size
                else:
                    write_png_rgb(path, out_w, out_h, fake_rows(out_w, out_h, s))
                    iw, ih = out_w, out_h
                images.append({"path": path, "seed": s, "subseed": ss, "width": iw, "height": ih})

        return {
            "images": images,
            "timing_s": round(time.monotonic() - t0, 3),
            "model_name": m["name"], "model_hash": m["hash"],
            "vae_name": self.vae["vae_name"] if self.vae else None,
            "vae_hash": self.vae["vae_hash"] if self.vae else None,
            "lora_hashes": {lr.name: models.short_hash(req.lora_paths[lr.name]) for lr in gp.loras},
            "scheduler": schedulers.resolve_schedule(gp.sampler_name, gp.scheduler),
            "warnings": list(ctx.warnings),
        }

    # -- tools ---------------------------------------------------------------
    def preprocess(self, params: dict, ctx: JobContext) -> dict:
        req = self.parse_preprocess(params)
        out = next_output_path(req.output_dir, f"pre-{req.preprocessor}")
        if not HAVE_PIL:
            shutil.copyfile(req.image_path, out)
            return {"path": out}
        img = load_rgb(req.image_path)
        if req.preprocessor == "canny":
            img = img.convert("L").filter(ImageFilter.FIND_EDGES).point(
                lambda v: 255 if v > int(req.args.get("low_threshold", 100)) // 4 else 0).convert("RGB")
        elif req.preprocessor == "depth":
            img = img.convert("L").filter(ImageFilter.GaussianBlur(4)).convert("RGB")
        elif req.preprocessor == "invert":
            img = ImageOps.invert(img)
        elif req.preprocessor in ("recolor_luminance", "recolor_intensity", "tile_resample", "tile_colorfix", "blur_gaussian"):
            from .controlnet import pil_preprocess
            img = pil_preprocess(req.preprocessor, img, req.args)
        elif req.preprocessor in ("depth_marigold", "normal_marigold"):
            img = img.convert("L").filter(ImageFilter.GaussianBlur(6)).convert("RGB")
        save_png(img, out)
        return {"path": out}

    def upscale(self, params: dict, ctx: JobContext) -> dict:
        req = self.parse_upscale(params)
        out = next_output_path(req.output_dir, f"up-{req.upscaler}")
        if not HAVE_PIL:
            shutil.copyfile(req.image_path, out)
            return {"path": out}
        img = load_rgb(req.image_path)
        if req.upscaler != "None":
            nw, nh = max(1, round(img.width * req.scale)), max(1, round(img.height * req.scale))
            img = img.resize((nw, nh), Image.NEAREST if req.upscaler == "Nearest" else Image.LANCZOS)
        save_png(img, out)
        return {"path": out}


    # -- extras (fake but shape-correct outputs) ----------------------------------------------------------------------
    def _require(self, params: dict, key: str) -> str:
        v = params.get(key)
        if not v:
            raise EngineError(INVALID_PARAMS, f"'{key}' is required")
        return str(v)

    def restore_faces(self, params: dict, ctx: JobContext) -> dict:
        src = self._require(params, "image_path")
        model = self._require(params, "model")
        if not os.path.isfile(model):
            raise EngineError(INVALID_PARAMS, f"face restoration model not found: {model}")
        out = next_output_path(self._require(params, "output_dir"), "faces")
        if HAVE_PIL:
            save_png(load_rgb(src).filter(ImageFilter.SMOOTH), out)
        else:
            shutil.copyfile(src, out)
        return {"path": out}

    def caption(self, params: dict, ctx: JobContext) -> dict:
        self._require(params, "image_path")
        method = str(params.get("method", "blip"))
        if method not in ("blip", "wd14"):
            raise EngineError(INVALID_PARAMS, f"unknown caption method '{method}' (blip, wd14)")
        return {"caption": "a dry-run image" if method == "blip" else "dry run, noise, colorful"}

    def run_code(self, params: dict, ctx: JobContext) -> dict:
        from .extras import run_code
        res = run_code(str(params.get("code", "")), {"engine": self, "params": params, "ctx": ctx})
        res["images"] = []
        return res

    def img2vid(self, params: dict, ctx: JobContext) -> dict:
        src = self._require(params, "image_path")
        model = self._require(params, "model")
        if not os.path.exists(model):
            raise EngineError(INVALID_PARAMS, f"SVD model not found: {model}")
        out_dir = self._require(params, "output_dir")
        n = max(2, min(int(params.get("frames", 14)), 64))
        steps = int(params.get("steps", 25))
        for i in range(steps):
            ctx.check_interrupt()
            ctx.progress(i + 1, steps, "base")
        if not HAVE_PIL:
            raise EngineError(UNSUPPORTED, "img2vid dry run needs Pillow")
        base = load_rgb(src).resize((int(params.get("width", 1024)) // 4, int(params.get("height", 576)) // 4))
        frames = [base.rotate(i * 360 / n) for i in range(n)]
        from .extras import save_gif
        gif = save_gif(frames, next_output_path(out_dir, "svd", ".gif"), int(params.get("fps", 6)))
        first = next_output_path(out_dir, "svd-frame0")
        save_png(frames[0], first)
        return {"path": gif, "frames": n, "preview": first}

    def novel_view(self, params: dict, ctx: JobContext) -> dict:
        src = self._require(params, "image_path")
        out = next_output_path(self._require(params, "output_dir"), "novel-view")
        if HAVE_PIL:
            save_png(ImageOps.mirror(load_rgb(src)), out)
        else:
            shutil.copyfile(src, out)
        return {"path": out}


__all__ = ["FakeEngine", "fake_image", "LATENT_UPSCALERS", "png_size"]
