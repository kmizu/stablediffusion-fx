"""Wires the optional hooks in params.hooks into a diffusers pipeline call.

The engine calls, per pipeline run:
  1. pre_run()         may switch the task (soft inpainting -> differential img2img) or pick a special pipeline
                       (SAG, PhotoMaker)
  2. adjust_embeds()   prompt-embedding level hooks (Revision)
  3. install()         UNet / attention / callback hooks; returns cleanups the engine runs in `finally`
  4. post_images()     pixel-space post processing (soft inpainting composite, face restoration)

See guidance.py, unet_hooks.py, latent_ops.py, adapters.py for the individual implementations.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from . import adapters, guidance, latent_ops, sampling_ext, unet_hooks
from .hooks import coerce_bool, hook_config
from .protocol import INVALID_PARAMS, UNSUPPORTED, EngineError

log = logging.getLogger("sdfx.extensions")

# hooks handled here (the engine handles freeu, dynamic_thresholding, hypertile, pag itself)
EXTENSION_HOOKS = ("apg", "mahiro", "rescale_cfg", "model_sampling", "kohya_hrfix", "multidiffusion", "latent_modifier",
                   "soft_inpainting", "sag", "neveroom", "reference", "revision", "ip_adapter", "controllllite",
                   "fooocus_inpaint", "photomaker", "face_restore", "img2imgalt") + sampling_ext.HOOKS

MODEL_SAMPLING_DEFAULTS = {"sampling": "default", "zsnr": False}
SAG_DEFAULTS = {"scale": 0.5, "blur_sigma": 2.0}
NEVEROOM_DEFAULTS = {"unet_offload": False, "vae_tiling": False, "encoder_tile_size": 1024, "decoder_tile_size": 128}
FACE_RESTORE_DEFAULTS = {"model": "", "visibility": 1.0, "codeformer_weight": 0.5}


@dataclass
class RunPlan:
    task: str
    soft_mask: Any = None          # numpy HxW float mask for soft inpainting
    soft_cfg: Optional[dict] = None
    sag: Optional[dict] = None
    photomaker: Optional[dict] = None
    warnings: list = field(default_factory=list)


def cfg(gp, name: str, defaults: dict) -> Optional[dict]:
    return hook_config(gp.hooks or {}, name, defaults)


def model_sampling_overrides(gp) -> dict:
    """advanced_model_sampling -> scheduler config overrides."""
    c = cfg(gp, "model_sampling", MODEL_SAMPLING_DEFAULTS)
    if not c:
        return {}
    out: dict = {}
    s = str(c.get("sampling", "default")).lower()
    if s in ("eps", "epsilon"):
        out["prediction_type"] = "epsilon"
    elif s in ("v_prediction", "v", "v_prediction_zsnr"):
        out["prediction_type"] = "v_prediction"
    elif s not in ("default", ""):
        raise EngineError(UNSUPPORTED, f"model sampling '{s}' is not supported (use default, eps, v_prediction)")
    if coerce_bool(c.get("zsnr"), False) or s == "v_prediction_zsnr":
        out["rescale_betas_zero_snr"] = True
        out["timestep_spacing"] = "trailing"
    return out


def pre_run(engine, task: str, gp, *, mask_present: bool, controlnet: bool, pag: bool) -> RunPlan:
    plan = RunPlan(task=task)
    si = cfg(gp, "soft_inpainting", latent_ops.SOFT_INPAINT_DEFAULTS)
    if si and task == "inpaint" and mask_present and not engine.kind.endswith("inpaint"):
        plan.task = "img2img"
        plan.soft_cfg = latent_ops.normalize_soft_inpaint(si)
    elif si and task != "inpaint":
        plan.warnings.append("soft inpainting only applies to inpainting; ignored")
    sg = cfg(gp, "sag", SAG_DEFAULTS)
    if sg:
        if task != "txt2img" or engine.sdxl or controlnet or pag:
            plan.warnings.append("SAG is only available for SD1.x txt2img without ControlNet/PAG; ignored")
        else:
            plan.sag = {"scale": float(sg["scale"]), "blur_sigma": float(sg["blur_sigma"])}
    pm = cfg(gp, "photomaker", adapters.PHOTOMAKER_DEFAULTS)
    if pm:
        if not engine.sdxl or task != "txt2img":
            raise EngineError(UNSUPPORTED, "PhotoMaker needs an SDXL model and txt2img")
        if not adapters.photomaker_available():
            raise EngineError(UNSUPPORTED, "PhotoMaker needs the 'photomaker' package (pip install photomaker)")
        plan.photomaker = adapters.normalize(pm, adapters.PHOTOMAKER_DEFAULTS)
    return plan


def special_pipe(engine, plan: RunPlan):
    """Returns a pipeline for SAG / PhotoMaker, sharing the base components, or None."""
    if plan.sag:
        from diffusers import StableDiffusionSAGPipeline  # type: ignore
        key = ("sag", (), None)
        p = engine.derived.get(key)
        if p is None:
            p = StableDiffusionSAGPipeline.from_pipe(engine.pipe)
            p.set_progress_bar_config(disable=True)
            engine.derived[key] = p
        return p
    if plan.photomaker:
        from photomaker import PhotoMakerStableDiffusionXLPipeline  # type: ignore
        pm = plan.photomaker
        key = ("photomaker", (pm["model"],), None)
        p = engine.derived.get(key)
        if p is None:
            from .embedding_merge_pipeline import clone_tokenizers
            p = PhotoMakerStableDiffusionXLPipeline(**clone_tokenizers(engine.pipe.components))
            # Pass the weights as a dict: loaded weights-only by us (no pickle code execution), and it avoids the
            # package's file-download helper, which uses diffusers arguments newer diffusers no longer accept.
            from .engine import safe_torch_load
            if "v2" in os.path.basename(pm["model"]).lower():
                raise EngineError(UNSUPPORTED, "PhotoMaker v2 needs InsightFace face embeddings; use photomaker-v1.bin")
            sd = safe_torch_load(pm["model"], "PhotoMaker adapter")
            from .photomaker_adapters import load_owned_adapter
            load_owned_adapter(engine, p, sd, weight_name=os.path.basename(pm["model"]),
                               trigger_word=pm["trigger_word"], pm_version="v1")
            # upstream sets num_tokens = 2 for every version; v1 fuses one ID token per image
            p.num_tokens = 1
            p.set_progress_bar_config(disable=True)
            engine.derived[key] = p
        else:
            # Ordinary TI loading updates base tokenizers without evicting this
            # cached adapter. Refresh private maps before adding its trigger.
            from .embedding_merge_pipeline import clone_tokenizers
            tokenizers = clone_tokenizers({name: getattr(engine.pipe, name, None)
                                          for name in ('tokenizer', 'tokenizer_2')})
            for name, tokenizer in tokenizers.items():
                if tokenizer is not None:
                    tokenizer.add_tokens([pm['trigger_word']], special_tokens=True)
                setattr(p, name, tokenizer)
            p.trigger_word = pm['trigger_word']
        return p
    return None


def adjust_embeds(engine, gp, embeds: dict, ctx) -> dict:
    rv = cfg(gp, "revision", adapters.REVISION_DEFAULTS)
    if not rv:
        return embeds
    if not engine.sdxl:
        ctx.warn("Revision needs an SDXL model; ignored")
        return embeds
    if not rv.get("image"):
        raise EngineError(INVALID_PARAMS, "Revision needs an image")
    from .images import load_rgb
    c = adapters.normalize(rv, adapters.REVISION_DEFAULTS)
    ie = adapters.revision_embeds(load_rgb(rv["image"]), engine.device, engine.dtype, engine.ext_cache)
    return adapters.apply_revision(embeds, ie, c["strength"], c["ignore_prompt"])


def _chain_callback(kw: dict, fn: Callable):
    prev = kw.get("callback_on_step_end")

    def cb(pipe, i, t, d):
        if prev is not None:
            d = prev(pipe, i, t, d)
        return fn(pipe, i, t, d)
    kw["callback_on_step_end"] = cb


def configure_ip_adapter(engine, pipe, gp, req):
    """Synchronize IP components for both sampling and Alternative inversion; return the reference image."""
    ipa = cfg(gp, "ip_adapter", adapters.IP_ADAPTER_DEFAULTS)
    ip_path = req.controlnet_paths.get(ipa.get("model"), ipa.get("model")) if ipa else None
    encoder = (ipa or {}).get("image_encoder") or ""
    ip_changed = engine.ip_state.loaded != ip_path or engine.ip_state.encoder != encoder
    if ip_changed and engine._offload:
        engine._drop_hooks()
    try:
        engine.ip_state.ensure(engine.pipe, ip_path, engine.sdxl, encoder)
    finally:
        if ip_changed or ipa:
            synced = adapters.sync_ip_components(engine.pipe, [pipe, *engine.derived.values()])
            if engine._offload and (ip_changed or synced):
                engine._place(pipe)
    if ipa:
        if not ipa.get("image"):
            raise EngineError(INVALID_PARAMS, "IP-Adapter needs an image")
        from .images import load_rgb
        c = adapters.normalize(ipa, adapters.IP_ADAPTER_DEFAULTS)
        pipe.set_ip_adapter_scale(c["scale"])
        return load_rgb(ipa["image"])
    return None


def install(engine, pipe, task: str, gp, req, kw: dict, ctx, cleanups: list, *, plan: RunPlan, w: int, h: int, sched,
            cfg_scale: float, image=None, mask=None, steps: int = 0, seed: int = 0, branch_hooks=None,
            forward_wrappers=None) -> list:
    """Installs every requested hook, appending cleanup callables to `cleanups` as it goes (so a failure half way
    still lets the engine undo what was installed)."""
    import torch
    unet = pipe.unet
    hooks = gp.hooks or {}

    # --- guidance ------------------------------------------------------------------------------------------------
    rc = cfg(gp, "rescale_cfg", guidance.RESCALE_CFG_DEFAULTS)
    if rc:
        kw["guidance_rescale"] = float(rc.get("multiplier", 0.7))
    apg = cfg(gp, "apg", guidance.APG_DEFAULTS)
    mah = cfg(gp, "mahiro", guidance.MAHIRO_DEFAULTS)
    lm = cfg(gp, "latent_modifier", latent_ops.LATENT_MODIFIER_DEFAULTS)
    post = None
    if lm:
        lmc = latent_ops.normalize_latent_modifier(lm)
        gen = torch.Generator("cpu").manual_seed(int(seed) & 0xFFFFFFFF)
        post = lambda x0, progress, lmc=lmc, gen=gen: latent_ops.modify_latents(x0, lmc, progress, gen)  # noqa: E731
        if cfg_scale <= 1.0:
            ctx.warn("latent modifier needs CFG scale > 1; ignored")
    if apg or mah or post is not None:
        if "dynamic_thresholding" in hooks and hook_config(hooks, "dynamic_thresholding", {}):
            ctx.warn("APG/Mahiro and Dynamic Thresholding both rewrite CFG; APG/Mahiro is applied on top")
        ac = getattr(sched, "alphas_cumprod", None)
        if ac is None:
            from .sigmas import default_alphas_cumprod
            ac = torch.tensor(default_alphas_cumprod())
        gh = guidance.GuidanceHook(unet, ac, sched.config.get("prediction_type", "epsilon"), cfg_scale,
                                   guidance.normalize_apg(apg) if apg else None, bool(mah), post).install()
        cleanups.append(gh.remove)
    sampling_ext.install(engine, pipe, gp, kw, ctx, cleanups, sched=sched, cfg_scale=cfg_scale, steps=steps, seed=seed,
                         chain=_chain_callback, composed=branch_hooks is not None)
    if plan.sag:
        kw["sag_scale"] = plan.sag["scale"]
        if plan.sag["blur_sigma"] != SAG_DEFAULTS["blur_sigma"]:
            ctx.warn("SAG blur sigma is fixed in diffusers; the value is ignored")

    # --- UNet structure --------------------------------------------------------------------------------------------
    kh = cfg(gp, "kohya_hrfix", unet_hooks.KOHYA_DEFAULTS)
    if kh:
        try:
            ds = unet_hooks.DeepShrinkHook(unet, unet_hooks.normalize_kohya(kh)).install()
        except ValueError as e:
            raise EngineError(INVALID_PARAMS, f"Kohya HRFix: {e}")
        cleanups.append(ds.remove)
    md = cfg(gp, "multidiffusion", unet_hooks.MULTIDIFFUSION_DEFAULTS)
    if md:
        mw = unet_hooks.MultiDiffusionWrapper(unet, unet_hooks.normalize_multidiffusion(md)).install()
        cleanups.append(mw.remove)
        if forward_wrappers is not None:
            forward_wrappers.append(mw)

    # --- image conditioning ---------------------------------------------------------------------------------------
    install_reference(engine, pipe, gp, cleanups, w=w, h=h, scheduler=sched, branch_hooks=branch_hooks)
    ip_image = configure_ip_adapter(engine, pipe, gp, req)
    if ip_image is not None:
        kw["ip_adapter_image"] = ip_image
    install_lllite(engine, pipe, gp, req, cleanups, w=w, h=h, image=image, branch_hooks=branch_hooks)
    fo = cfg(gp, "fooocus_inpaint", adapters.FOOOCUS_DEFAULTS)
    if fo:
        if not engine.sdxl or task not in ("inpaint", "img2img") or mask is None:
            raise EngineError(UNSUPPORTED, "Fooocus inpaint needs an SDXL model and an inpaint mask")
        c = adapters.normalize(fo, adapters.FOOOCUS_DEFAULTS)
        patch = req.controlnet_paths.get(fo.get("model"), fo.get("model"))
        head = c["head"] or os.path.join(os.path.dirname(patch or ""), "fooocus_inpaint_head.pth")
        for f in (patch, head):
            if not f or not os.path.isfile(f):
                raise EngineError(INVALID_PARAMS, f"Fooocus inpaint file not found: {f}")
        im0 = image[0] if isinstance(image, list) else image
        mk0 = mask[0] if isinstance(mask, list) else mask
        lat = encode_image(engine, pipe, im0)
        lmask = latent_mask(mk0, lat.shape[-2], lat.shape[-1])
        fh = adapters.FooocusInpaintHook(unet, patch, head, lat, lmask, c["start_percent"], c["end_percent"]).install()
        cleanups.append(fh.remove)

    # --- per-step latent ops -----------------------------------------------------------------------------------------
    if plan.soft_cfg is not None:
        install_soft_inpaint(engine, pipe, kw, plan, image, mask, seed)

    # --- memory ------------------------------------------------------------------------------------------------------
    nv = cfg(gp, "neveroom", NEVEROOM_DEFAULTS)
    if nv:
        c = adapters.normalize(nv, NEVEROOM_DEFAULTS)
        if c["vae_tiling"]:
            pipe.vae.enable_tiling()
            cleanups.append(pipe.vae.disable_tiling)
        if c["unet_offload"]:
            pipe.enable_attention_slicing("max")
            pipe.vae.enable_slicing()
            cleanups.append(pipe.disable_attention_slicing)
            cleanups.append(pipe.vae.disable_slicing)
            if engine.cuda:
                cleanups.append(torch.cuda.empty_cache)
    return cleanups


def encode_image(engine, pipe, img):
    """PIL -> scaled latents (1, 4, h/8, w/8) on the engine device."""
    import torch
    vae = pipe.vae
    with torch.no_grad():
        upcast = vae.dtype == torch.float16 and getattr(vae.config, "force_upcast", False)
        if upcast:
            vae.to(torch.float32)
        x = pipe.image_processor.preprocess(img).to(vae.device, vae.dtype)
        lat = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
        if upcast:
            vae.to(torch.float16)
    return lat.to(engine.device, engine.dtype)


def latent_mask(mask_img, lh: int, lw: int):
    """PIL L mask (white = repaint) -> (1, 1, lh, lw) float tensor via 8x8 max pooling (Fooocus)."""
    import numpy as np
    import torch
    m = torch.from_numpy(np.asarray(mask_img.convert("L"), dtype=np.float32) / 255.0)[None, None]
    m = torch.nn.functional.adaptive_max_pool2d(m, (lh, lw))
    return m.round()


def install_reference(engine, pipe, gp, cleanups, *, w, h, scheduler, branch_hooks=None):
    ref = cfg(gp, "reference", adapters.REFERENCE_DEFAULTS)
    if ref is None:
        return
    if not ref.get("image"):
        raise EngineError(INVALID_PARAMS, "reference needs an image")
    from .images import load_rgb, resize_image
    c = adapters.normalize(ref, adapters.REFERENCE_DEFAULTS)
    rimg = resize_image(1, load_rgb(ref["image"]), w, h)
    ref_lat = encode_image(engine, pipe, rimg)
    hook = adapters.ReferenceOnlyHook(pipe.unet, ref_lat, scheduler, c["style_fidelity"],
                                      c["attention_auto_machine_weight"])
    cleanups.append(hook.remove)
    hook.install(managed=branch_hooks is not None)
    if branch_hooks is not None:
        branch_hooks.append(hook)


def install_lllite(engine, pipe, gp, req, cleanups, *, w, h, image=None, branch_hooks=None):
    ll = cfg(gp, "controllllite", adapters.LLLITE_DEFAULTS)
    if ll is None:
        return
    if not engine.sdxl:
        raise EngineError(UNSUPPORTED, "ControlNet-LLLite models are SDXL only")
    from .images import load_rgb, resize_image
    c = adapters.normalize(ll, adapters.LLLITE_DEFAULTS)
    path = req.controlnet_paths.get(ll.get("model"), ll.get("model"))
    if not path or not os.path.isfile(path):
        raise EngineError(INVALID_PARAMS, f"ControlNet-LLLite model not found: {ll.get('model')}")
    if not ll.get("image") and image is None:
        raise EngineError(INVALID_PARAMS, "ControlNet-LLLite needs an image")
    src = load_rgb(ll["image"]) if ll.get("image") else (image[0] if isinstance(image, list) else image)
    hook = adapters.ControlLLLiteHook(pipe.unet, path, resize_image(1, src, w, h), c["strength"], c["start_percent"],
                                     c["end_percent"])
    cleanups.append(hook.remove)
    hook.install(managed=branch_hooks is not None)
    if branch_hooks is not None:
        branch_hooks.append(hook)


def install_soft_inpaint(engine, pipe, kw: dict, plan: RunPlan, image, mask, seed: int):
    """Differential-diffusion callback for soft inpainting (task already switched to img2img)."""
    import numpy as np
    import torch
    im0 = image[0] if isinstance(image, list) else image
    mk0 = mask[0] if isinstance(mask, list) else mask
    orig = encode_image(engine, pipe, im0)
    soft = np.asarray(mk0.convert("L"), dtype=np.float32) / 255.0
    plan.soft_mask = soft
    lh, lw = orig.shape[-2:]
    sm = torch.nn.functional.interpolate(torch.from_numpy(soft)[None, None], size=(lh, lw), mode="bilinear",
                                         align_corners=False).to(engine.device)
    g = torch.Generator("cpu").manual_seed(int(seed) & 0xFFFFFFFF)
    noise = torch.randn(orig.shape, generator=g, dtype=torch.float32).to(engine.device, orig.dtype)
    sc = plan.soft_cfg

    def soft_cb(p, i, t, d):
        lat = d.get("latents")
        if lat is None:
            return d
        ts = p.scheduler.timesteps
        total = len(ts)
        if i + 1 < total:
            nxt = ts[i + 1:i + 2].to(orig.device)
            # add_noise returns latents in the scheduler's own space (z_t for DDPM-style, x0 + sigma*eps for Euler-style)
            ref = p.scheduler.add_noise(orig, noise, nxt) if hasattr(p.scheduler, "add_noise") else orig
        else:
            ref = orig
        ref = ref.repeat(lat.shape[0] // ref.shape[0], 1, 1, 1) if ref.shape[0] != lat.shape[0] else ref
        d["latents"] = latent_ops.soft_inpaint_blend(lat, ref, sm, i, total, sc)
        return d
    _chain_callback(kw, soft_cb)


def post_images(engine, gp, plan: Optional[RunPlan], images: list, image=None, ctx=None, faces: bool = False) -> list:
    """Soft-inpainting composite (plan given) and AlwaysOn face restoration (faces=True, after the whole job)."""
    out = images
    if plan is not None and plan.soft_cfg is not None and plan.soft_mask is not None and image is not None:
        import numpy as np
        from PIL import Image
        im0 = image[0] if isinstance(image, list) else image
        orig = np.asarray(im0.convert("RGB"), dtype=np.float32) / 255.0
        res = []
        for im in out:
            gen = np.asarray(im.convert("RGB").resize(im0.size), dtype=np.float32) / 255.0
            comp = latent_ops.composite_mask(orig, gen, plan.soft_mask, plan.soft_cfg)
            res.append(Image.fromarray((np.clip(comp, 0, 1) * 255).astype("uint8")))
        out = res
    fr = cfg(gp, "face_restore", FACE_RESTORE_DEFAULTS) if faces else None
    if fr:
        from . import extras
        c = adapters.normalize(fr, FACE_RESTORE_DEFAULTS)
        if not fr.get("model") or not os.path.isfile(fr["model"]):
            if ctx is not None:
                ctx.warn("face restoration model not found; skipped")
        else:
            out = [extras.restore_faces_image(im, fr["model"], c["visibility"], c["codeformer_weight"], engine.device)
                   for im in out]
    return out


__all__ = ["EXTENSION_HOOKS", "RunPlan", "pre_run", "special_pipe", "adjust_embeds", "install", "post_images",
           "model_sampling_overrides", "encode_image", "latent_mask"]
