"""Engine requests and helpers for the non-diffusion built-ins.

  restore_faces {image_path, model, visibility, codeformer_weight, output_dir}       postprocessing_gfpgan/codeformer
  caption       {image_path, method: blip|wd14, threshold}                          postprocessing_caption
  run_code      {code, output_dir}                                                   custom_code (needs SDFX_ALLOW_CODE=1)
  img2vid       {model, image_path, frames, fps, motion_bucket_id, noise_aug_strength, steps, seed,
                 min_cfg, max_cfg, decode_chunk_size, width, height, output_dir}     sd_forge_svd
  novel_view    {model, image_path, elevation, azimuth, steps, cfg_scale, seed, output_dir}  sd_forge_z123
  img2imgalt    hook used by the engine: noise inversion of the init image (img2imgalt.py)
"""
from __future__ import annotations

import io
import logging
import os
import traceback
from contextlib import redirect_stdout
from typing import Optional


from .protocol import INVALID_PARAMS, MODEL_LOAD_FAILED, UNSUPPORTED, EngineError

log = logging.getLogger("sdfx.extras")

IMG2IMGALT_DEFAULTS = {"override_prompt": "", "override_negative": "", "decode_cfg_scale": 1.0, "randomness": 0.0,
                       "inversion_steps": 0}

_face_cache: dict = {}
_caption_cache: dict = {}
_video_cache: dict = {}


# ------------------------------------------------------------------------------------------------- face restoration
def face_boxes(img) -> list[tuple[int, int, int, int]]:
    """Frontal face boxes (x, y, w, h) using OpenCV's bundled Haar cascade."""
    import cv2  # type: ignore
    import numpy as np
    gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    casc = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    min_side = max(24, min(img.size) // 20)
    faces = casc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side))
    return [tuple(int(v) for v in f) for f in faces]


def expand_box(box, img_w: int, img_h: int, scale: float = 1.6) -> tuple[int, int, int, int]:
    """Square crop around a face box, clamped to the image (left, top, right, bottom)."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    side = int(max(w, h) * scale)
    side = min(side, img_w, img_h)
    left = int(min(max(cx - side / 2, 0), img_w - side))
    top = int(min(max(cy - side / 2, 0), img_h - side))
    return left, top, left + side, top + side


def _face_model(path: str, device):
    m = _face_cache.get(path)
    if m is None:
        from spandrel import ModelLoader  # type: ignore
        try:
            import spandrel_extra_arches  # type: ignore
            spandrel_extra_arches.install()
        except Exception:
            pass
        m = ModelLoader().load_from_file(path)
        m.model.eval()
        _face_cache.clear()
        _face_cache[path] = m
    m.model.to(device)
    return m


def restore_faces_image(img, model_path: str, visibility: float = 1.0, codeformer_weight: float = 0.5, device="cpu"):
    """GFPGAN / CodeFormer (any spandrel FaceSR model) applied to every detected face, feathered back in."""
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFilter
    desc = _face_model(model_path, device)
    is_codeformer = "codeformer" in type(desc.model).__name__.lower() or "codeformer" in os.path.basename(model_path).lower()
    out = img.convert("RGB").copy()
    for box in face_boxes(out):
        l, t, r, b = expand_box(box, out.width, out.height)
        crop = out.crop((l, t, r, b)).resize((512, 512), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(crop, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
        x = (x * 2 - 1).to(device)
        with torch.no_grad():
            y = desc.model(x, weight=codeformer_weight) if is_codeformer else desc.model(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        y = ((y.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).float().cpu().numpy()
        face = Image.fromarray((y * 255).round().astype("uint8")).resize((r - l, b - t), Image.LANCZOS)
        m = Image.new("L", face.size, 0)
        pad = max(2, face.width // 12)
        ImageDraw.Draw(m).rectangle((pad, pad, face.width - pad, face.height - pad), fill=int(255 * visibility))
        m = m.filter(ImageFilter.GaussianBlur(pad))
        out.paste(face, (l, t), m)
    desc.model.to("cpu")
    return out


# ------------------------------------------------------------------------------------------------------- captioning
def caption_image(img, method: str = "blip", threshold: float = 0.35, device="cpu") -> str:
    import torch
    if method == "blip":
        from transformers import BlipForConditionalGeneration, BlipProcessor  # type: ignore
        repo = os.environ.get("SDFX_BLIP_MODEL", "Salesforce/blip-image-captioning-base")
        c = _caption_cache.get(repo)
        if c is None:
            c = (BlipProcessor.from_pretrained(repo), BlipForConditionalGeneration.from_pretrained(repo))
            _caption_cache[repo] = c
        proc, model = c
        model.to(device)
        try:
            inp = proc(images=img.convert("RGB"), return_tensors="pt").to(device)
            with torch.no_grad():
                ids = model.generate(**inp, max_new_tokens=60)
            return proc.decode(ids[0], skip_special_tokens=True).strip()
        finally:
            model.to("cpu")
    if method == "wd14":
        return wd14_tags(img, threshold)
    raise EngineError(INVALID_PARAMS, f"unknown caption method '{method}' (blip, wd14)")


def wd14_tags(img, threshold: float) -> str:
    """Danbooru-style tags (the deepbooru role) with SmilingWolf's WD tagger via onnxruntime."""
    try:
        import csv

        import numpy as np
        import onnxruntime as ort  # type: ignore
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as e:
        raise EngineError(UNSUPPORTED, f"wd14 tagging needs onnxruntime and huggingface_hub: {e}")
    repo = os.environ.get("SDFX_WD14_MODEL", "SmilingWolf/wd-vit-tagger-v3")
    c = _caption_cache.get(repo)
    if c is None:
        sess = ort.InferenceSession(hf_hub_download(repo, "model.onnx"), providers=["CPUExecutionProvider"])
        with open(hf_hub_download(repo, "selected_tags.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        c = (sess, rows)
        _caption_cache[repo] = c
    sess, rows = c
    size = sess.get_inputs()[0].shape[1]
    from PIL import Image
    im = img.convert("RGB")
    side = max(im.size)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    arr = np.asarray(canvas.resize((size, size), Image.BICUBIC), dtype=np.float32)[:, :, ::-1][None]
    probs = sess.run(None, {sess.get_inputs()[0].name: arr})[0][0]
    tags = [(r["name"], float(p)) for r, p in zip(rows, probs) if r.get("category") == "0" and p >= threshold]
    tags.sort(key=lambda x: -x[1])
    return ", ".join(t.replace("_", " ") for t, _ in tags)


# ------------------------------------------------------------------------------------------------------ custom code
def code_allowed() -> bool:
    return os.environ.get("SDFX_ALLOW_CODE", "") == "1"


def run_code(code: str, namespace: dict) -> dict:
    """Executes user Python like A1111's custom_code script. Only when the app was started with custom code
    allowed (SDFX_ALLOW_CODE=1). stdout is captured; `result` and `images` variables are returned."""
    if not code_allowed():
        raise EngineError(UNSUPPORTED, "custom code is disabled (enable 'Allow custom code' in Settings and restart)")
    buf = io.StringIO()
    ns = dict(namespace)
    try:
        with redirect_stdout(buf):
            exec(compile(code, "<custom code>", "exec"), ns)  # noqa: S102 - explicit opt-in feature
    except Exception as e:
        raise EngineError(INVALID_PARAMS, f"custom code failed: {e}", traceback.format_exc())
    res = ns.get("result")
    return {"stdout": buf.getvalue()[-20000:], "result": None if res is None else str(res)[:20000],
            "images": list(ns.get("images") or [])}


# ------------------------------------------------------------------------------------------------------------ video
def fp16_variant(model_dir: str, subfolder: str = "") -> Optional[str]:
    """'fp16' when the folder (or one component subfolder) has *.fp16.* weights but no plain weights, e.g. a
    snapshot downloaded with allow_patterns; diffusers then needs variant="fp16" to find them."""
    d = os.path.join(model_dir, subfolder) if subfolder else model_dir
    names = [f for _, _, fs in os.walk(d) for f in fs if f.endswith((".safetensors", ".bin")) and "ema" not in f]
    if subfolder:
        return "fp16" if names and all(".fp16." in n for n in names) else None
    # whole pipeline: fp16 if some component only has fp16 weights (diffusers falls back to plain files elsewhere)
    comps = [c for c in os.listdir(d) if os.path.isdir(os.path.join(d, c))]
    return "fp16" if any(fp16_variant(d, c) for c in comps) else None


def save_gif(frames: list, path: str, fps: int) -> str:
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=max(1, int(1000 / max(fps, 1))), loop=0)
    return path


def img2vid(engine, params: dict, ctx) -> dict:
    """Stable Video Diffusion image-to-video. The SD pipeline is moved off the GPU while it runs."""
    import torch
    from .images import load_rgb, next_output_path, resize_image, save_png
    model = params.get("model") or ""
    if not model or not os.path.exists(model):
        raise EngineError(INVALID_PARAMS, f"SVD model not found: {model}")
    try:
        from diffusers import StableVideoDiffusionPipeline  # type: ignore
    except ImportError as e:
        raise EngineError(UNSUPPORTED, f"img2vid needs diffusers with StableVideoDiffusionPipeline: {e}")
    pipe = _video_cache.get(model)
    if pipe is None:
        try:
            if os.path.isdir(model):
                pipe = StableVideoDiffusionPipeline.from_pretrained(model, torch_dtype=engine.dtype, variant=fp16_variant(model))
            else:
                pipe = StableVideoDiffusionPipeline.from_single_file(model, torch_dtype=engine.dtype)
        except Exception as e:
            raise EngineError(MODEL_LOAD_FAILED, f"failed to load SVD model: {e}", traceback.format_exc())
        pipe.set_progress_bar_config(disable=True)
        _video_cache.clear()
        _video_cache[model] = pipe
    engine.park()
    w, h = int(params.get("width", 1024)), int(params.get("height", 576))
    img = resize_image(1, load_rgb(params["image_path"]), w, h)
    frames_n = int(params.get("frames", 14))
    steps = int(params.get("steps", 25))
    if engine.cuda:
        pipe.enable_model_cpu_offload()

    def cb(p, i, t, d):
        ctx.check_interrupt()
        ctx.progress(i + 1, steps, "base")
        return d
    g = torch.Generator("cpu").manual_seed(int(params.get("seed", 0)) & 0xFFFFFFFF)
    with torch.inference_mode():
        frames = pipe(img, width=w, height=h, num_frames=frames_n, num_inference_steps=steps,
                      motion_bucket_id=int(params.get("motion_bucket_id", 127)), fps=int(params.get("fps", 6)),
                      noise_aug_strength=float(params.get("noise_aug_strength", 0.02)),
                      min_guidance_scale=float(params.get("min_cfg", 1.0)), max_guidance_scale=float(params.get("max_cfg", 3.0)),
                      decode_chunk_size=int(params.get("decode_chunk_size", 4)), generator=g,
                      callback_on_step_end=cb).frames[0]
    out_dir = params["output_dir"]
    gif = save_gif(frames, next_output_path(out_dir, "svd", ".gif"), int(params.get("fps", 6)))
    first = next_output_path(out_dir, "svd-frame0")
    save_png(frames[0], first)
    return {"path": gif, "frames": len(frames), "preview": first}


def zero123_camera(elevation: float, azimuth: float, distance: float = 0.0) -> list[float]:
    """Relative camera embedding of Zero123: [d_elevation (rad), sin(d_azimuth), cos(d_azimuth), d_distance]."""
    import math
    return [math.radians(elevation), math.sin(math.radians(azimuth)), math.cos(math.radians(azimuth)), distance]


def _is_zero123_dir(model: str) -> bool:
    return any(os.path.isdir(os.path.join(model, d)) for d in ("clip_camera_projection", "cc_projection"))


def _load_projection(model: str, dtype):
    import torch
    sub = "clip_camera_projection" if os.path.isdir(os.path.join(model, "clip_camera_projection")) else "cc_projection"
    d = os.path.join(model, sub)
    files = sorted(f for f in os.listdir(d) if f.endswith((".safetensors", ".bin")) and "ema" not in f)
    if not files:
        raise EngineError(MODEL_LOAD_FAILED, f"no weights in {d}")
    path = os.path.join(d, next((f for f in files if f.endswith(".safetensors")), files[0]))
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    w = next(v for k, v in sd.items() if k.endswith("weight"))
    b = next((v for k, v in sd.items() if k.endswith("bias")), None)
    lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=b is not None)
    with torch.no_grad():
        lin.weight.copy_(w.float())
        if b is not None:
            lin.bias.copy_(b.float())
    return lin.to(dtype)


def zero123_generate(engine, model: str, image, elevation: float, azimuth: float, steps: int, cfg: float, seed: int, ctx,
                     size: int = 256):
    """Zero123 / Zero123-XL novel view synthesis from a diffusers-format folder (threestudio layout): CLIP image
    embedding + relative camera -> projection -> UNet cross-attention; the input image latents are concatenated to
    the noisy latents (8-channel UNet). Classifier-free guidance uses zeros for both conditions."""
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel  # type: ignore
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection  # type: ignore
    dev, dt = engine.device, engine.dtype
    c = _video_cache.get(("zero123", model))
    if c is None:
        v = lambda sub: fp16_variant(model, sub)  # noqa: E731
        c = {"unet": UNet2DConditionModel.from_pretrained(model, subfolder="unet", torch_dtype=dt, variant=v("unet")),
             "vae": AutoencoderKL.from_pretrained(model, subfolder="vae", torch_dtype=dt, variant=v("vae")),
             "enc": CLIPVisionModelWithProjection.from_pretrained(model, subfolder="image_encoder", torch_dtype=dt,
                                                                  variant=v("image_encoder")),
             "proc": CLIPImageProcessor.from_pretrained(model, subfolder="feature_extractor"),
             "proj": _load_projection(model, dt),
             "sched": DDIMScheduler.from_pretrained(model, subfolder="scheduler")}
        _video_cache.clear()
        _video_cache[("zero123", model)] = c
    for k in ("unet", "vae", "enc", "proj"):
        c[k].to(dev)
    try:
        from PIL import Image
        img = image.convert("RGB").resize((size, size), Image.LANCZOS)
        with torch.inference_mode():
            px = c["proc"](images=img, return_tensors="pt").pixel_values.to(dev, dt)
            emb = c["enc"](px).image_embeds[:, None, :]
            cam = torch.tensor([[zero123_camera(elevation, azimuth)]], device=dev, dtype=dt)
            cond = c["proj"](torch.cat([emb, cam], dim=-1))
            x = torch.from_numpy(__import__("numpy").asarray(img, dtype="float32") / 127.5 - 1.0).permute(2, 0, 1)[None]
            img_lat = c["vae"].encode(x.to(dev, dt)).latent_dist.mode()
            sched = c["sched"]
            sched.set_timesteps(steps, device=dev)
            g = torch.Generator("cpu").manual_seed(int(seed) & 0xFFFFFFFF)
            lat = torch.randn((1, 4, size // 8, size // 8), generator=g).to(dev, dt) * sched.init_noise_sigma
            ehs = torch.cat([torch.zeros_like(cond), cond])
            concat = torch.cat([torch.zeros_like(img_lat), img_lat])
            for i, t in enumerate(sched.timesteps):
                ctx.check_interrupt()
                inp = sched.scale_model_input(torch.cat([lat, lat]), t)
                eps = c["unet"](torch.cat([inp, concat], dim=1), t, encoder_hidden_states=ehs).sample
                u, cc = eps.chunk(2)
                lat = sched.step(u + cfg * (cc - u), t, lat).prev_sample
                ctx.progress(i + 1, steps, "base")
            out = c["vae"].decode(lat / c["vae"].config.scaling_factor).sample
        arr = ((out[0].float().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).round().byte().cpu().numpy()
        return Image.fromarray(arr)
    finally:
        for k in ("unet", "vae", "enc", "proj"):
            c[k].to("cpu")
        if engine.cuda:
            torch.cuda.empty_cache()


def novel_view(engine, params: dict, ctx) -> dict:
    """Zero123-style novel view synthesis from a diffusers-format model folder (e.g. a Stable Zero123 conversion)."""
    import torch
    from .images import load_rgb, next_output_path, save_png
    model = params.get("model") or ""
    if not os.path.isdir(model):
        raise EngineError(INVALID_PARAMS, f"novel view needs a diffusers model folder (model_index.json): {model}")
    if _is_zero123_dir(model):
        engine.park()
        try:
            out_img = zero123_generate(engine, model, load_rgb(params["image_path"]), float(params.get("elevation", 0.0)),
                                       float(params.get("azimuth", 30.0)), int(params.get("steps", 50)),
                                       float(params.get("cfg_scale", 3.0)), int(params.get("seed", 0)), ctx)
        except EngineError:
            raise
        except Exception as e:
            raise EngineError(MODEL_LOAD_FAILED, f"Zero123 failed: {e}", traceback.format_exc())
        path = next_output_path(params["output_dir"], "novel-view")
        save_png(out_img, path)
        return {"path": path}
    try:
        from diffusers import DiffusionPipeline  # type: ignore
        pipe = _video_cache.get(model) or DiffusionPipeline.from_pretrained(model, torch_dtype=engine.dtype,
                                                                            custom_pipeline=model if os.path.isfile(
                                                                                os.path.join(model, "pipeline.py")) else None)
    except Exception as e:
        raise EngineError(MODEL_LOAD_FAILED, f"failed to load novel-view model: {e}", traceback.format_exc())
    _video_cache.clear()
    _video_cache[model] = pipe
    engine.park()
    if engine.cuda:
        pipe.to(engine.device)
    from .engine import _filter_kwargs
    kw = _filter_kwargs(pipe.__call__, {
        "image": load_rgb(params["image_path"]), "input_imgs": load_rgb(params["image_path"]),
        "prompt_imgs": load_rgb(params["image_path"]),
        "elevation": float(params.get("elevation", 0.0)), "polar_angle": float(params.get("elevation", 0.0)),
        "azimuth": float(params.get("azimuth", 30.0)), "azimuth_angle": float(params.get("azimuth", 30.0)),
        "num_inference_steps": int(params.get("steps", 50)), "guidance_scale": float(params.get("cfg_scale", 3.0)),
        "generator": torch.Generator("cpu").manual_seed(int(params.get("seed", 0)) & 0xFFFFFFFF),
    })
    with torch.inference_mode():
        out = pipe(**kw)
    pipe.to("cpu")
    imgs = getattr(out, "images", out)
    path = next_output_path(params["output_dir"], "novel-view")
    save_png(imgs[0], path)
    return {"path": path}


# ----------------------------------------------------------------------------------------------- img2img alternative
def invert_latents(engine, pipe, latents, embeds: dict, steps: int, cfg: float, w: int, h: int, check_interrupt,
                   *, branch_setup=None, sampling_setup=None):
    """DDIM-invert scaled VAE latents, preserving the amplitude of the terminal noisy state."""
    import torch
    from diffusers import DDIMInverseScheduler  # type: ignore
    spacing = pipe.scheduler.config.get("timestep_spacing", "leading")
    inv = DDIMInverseScheduler.from_config(pipe.scheduler.config,
                                           clip_sample=False,  # latent x0 is not a pixel image bounded by [-1, 1]
                                           timestep_spacing=spacing if spacing in ("leading", "trailing") else "leading")
    inv.set_timesteps(steps, device=latents.device)
    pe, ne = embeds["prompt_embeds"][:1], embeds["negative_prompt_embeds"][:1]
    guided = cfg != 1.0  # inversion also exposes meaningful unconditional/interpolated values in [0, 1)
    ehs = torch.cat([ne, pe]) if guided else pe
    added = None
    if engine.sdxl:
        pp, npp = embeds["pooled_prompt_embeds"][:1], embeds["negative_pooled_prompt_embeds"][:1]
        tid = torch.tensor([[h, w, 0, 0, h, w]], device=latents.device, dtype=pe.dtype)
        added = {"text_embeds": torch.cat([npp, pp]) if guided else pp,
                 "time_ids": tid.repeat(2 if guided else 1, 1)}
    x = latents
    ip_embeds = embeds.get("_ip_image_embeds")
    if ip_embeds is not None:
        from .neutral_ip import align
        added = dict(added or {}, image_embeds=align(ip_embeds, 1, [False, True] if guided else [True], x.device, x.dtype))
    runtime = None
    try:
        if "_neutral" in embeds:
            from .neutral_runtime import RuntimeHook
            plan = embeds["_neutral"]
            branch_hooks = branch_setup(inv) if branch_setup else ()
            runtime = RuntimeHook(pipe.unet, inv, plan, cfg, plan.phi, guided=guided, image_embeds=ip_embeds,
                                  branch_hooks=branch_hooks,
                                  sampling=sampling_setup(inv) if sampling_setup else None,
                                  check_interrupt=check_interrupt).install()
        with torch.no_grad():
            for index, t in enumerate(inv.timesteps):
                check_interrupt()
                if runtime:
                    runtime.step = index + 1
                xin = torch.cat([x, x]) if guided else x
                kw = {"encoder_hidden_states": ehs}
                if added is not None:
                    kw["added_cond_kwargs"] = added
                eps = pipe.unet(xin, t, **kw).sample
                if guided:
                    u, c = eps.chunk(2)
                    eps = u + cfg * (c - u)
                x = inv.step(eps, t, x).prev_sample
    finally:
        if runtime:
            runtime.remove()
    return x


__all__ = ["restore_faces_image", "face_boxes", "expand_box", "caption_image", "run_code", "code_allowed", "img2vid",
           "novel_view", "invert_latents", "save_gif", "IMG2IMGALT_DEFAULTS"]
