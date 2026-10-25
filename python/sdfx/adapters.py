"""Image-conditioning extensions ported from reForge built-ins.

  {"reference":   {"image","style_fidelity","attention_auto_machine_weight"}}   forge_preprocessor_reference
  {"revision":    {"image","strength","ignore_prompt"}}                          forge_preprocessor_revision (SDXL)
  {"ip_adapter":  {"model","image","scale","image_encoder"}}                     sd_forge_ipadapter
  {"controllllite": {"model","image","strength","start_percent","end_percent"}}  sd_forge_controlllite (SDXL)
  {"fooocus_inpaint": {"model","head","start_percent","end_percent"}}           sd_forge_fooocus_inpaint (SDXL)
  {"photomaker":  {"model","images","trigger_word","start_merge_step"}}         sd_forge_photomaker (SDXL)

Every class has install()/remove() so the engine can undo it in a finally block; nothing persists on the shared
pipeline between jobs except caches keyed by file path.
"""
from __future__ import annotations

import os
from typing import Optional

from .protocol import UNSUPPORTED, EngineError

REFERENCE_DEFAULTS = {"style_fidelity": 0.5, "attention_auto_machine_weight": 1.0}
REVISION_DEFAULTS = {"strength": 1.0, "ignore_prompt": False}
IP_ADAPTER_DEFAULTS = {"scale": 0.7, "image_encoder": ""}
LLLITE_DEFAULTS = {"strength": 1.0, "start_percent": 0.0, "end_percent": 1.0}
FOOOCUS_DEFAULTS = {"head": "", "start_percent": 0.0, "end_percent": 1.0}
PHOTOMAKER_DEFAULTS = {"trigger_word": "img", "start_merge_step": 10}


def ldm_block_to_diffusers(kind: str, n: int) -> tuple[str, int, int]:
    """ComfyUI/ldm block id -> (diffusers block list, block index, attention index).
    input_blocks N -> down_blocks[(N-1)//3].attentions[(N-1)%3]; output_blocks N -> up_blocks[N//3].attentions[N%3]."""
    if kind == "input":
        return "down_blocks", (n - 1) // 3, (n - 1) % 3
    if kind == "output":
        return "up_blocks", n // 3, n % 3
    if kind == "middle":
        return "mid_block", 0, 0
    raise ValueError(kind)


def lllite_target(name: str) -> Optional[tuple[str, int, int, int, str, str]]:
    """Parses `lllite_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q` into
    (block list, block idx, attn idx, transformer idx, attn1|attn2, to_q|to_k|to_v)."""
    import re
    m = re.match(r"lllite_unet_(input_blocks_(\d+)|middle_block|output_blocks_(\d+))_1_transformer_blocks_(\d+)_(attn[12])_(to_[qkv])$", name)
    if not m:
        return None
    if m.group(2) is not None:
        blk, bi, ai = ldm_block_to_diffusers("input", int(m.group(2)))
    elif m.group(3) is not None:
        blk, bi, ai = ldm_block_to_diffusers("output", int(m.group(3)))
    else:
        blk, bi, ai = "mid_block", 0, 0
    return blk, bi, ai, int(m.group(4)), m.group(5), m.group(6)


def _attention_module(unet, blk: str, bi: int, ai: int, ti: int, attn: str):
    if blk == "mid_block":
        tr = unet.mid_block.attentions[0]
    else:
        tr = getattr(unet, blk)[bi].attentions[ai]
    return getattr(tr.transformer_blocks[ti], attn)


def _load_state(path: str) -> dict:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    from .engine import safe_torch_load  # weights_only loader
    return safe_torch_load(path)


# ---------------------------------------------------------------------------------------------------------- reference
class ReferenceOnlyHook:
    """Reference-only: self-attention also attends to the reference image's hidden states at the same timestep."""

    def __init__(self, unet, ref_latents, scheduler, style_fidelity: float, weight: float):
        self.unet = unet
        self.ref = ref_latents
        self.sched = scheduler
        self.sf = float(style_fidelity)
        self.w = float(weight)
        self.mode = "off"
        self.bank: dict = {}
        self.handles = []
        self.noise = None

    def _attn_pre(self, module, args, kwargs):
        hs = args[0] if args else kwargs.get("hidden_states")
        if hs is None or kwargs.get("encoder_hidden_states") is not None:
            return None
        if self.mode == "write":
            self.bank[id(module)] = hs.detach()
            return None
        if self.mode == "read" and id(module) in self.bank:
            import torch
            ref = self.bank[id(module)]
            if ref.shape[0] != hs.shape[0]:
                ref = ref.repeat(hs.shape[0] // max(ref.shape[0], 1), 1, 1)
            kwargs = dict(kwargs)
            kwargs["encoder_hidden_states"] = torch.cat([hs, ref * self.w], dim=1)
            if args:
                return (hs,) + tuple(args[1:]), kwargs
            kwargs["hidden_states"] = hs
            return args, kwargs
        return None

    def _attn_post(self, module, args, kwargs, output):
        if self.mode != "read" or self.sf <= 0 or id(module) not in self.bank:
            return None
        # style fidelity: blend with the plain self-attention result for the cond half
        hs = args[0] if args else kwargs.get("hidden_states")
        prev, self.mode = self.mode, "off"
        try:
            plain = module(hs)
        finally:
            self.mode = prev
        b = output.shape[0] // 2
        if b == 0:
            return None
        import torch
        cond = output[b:] * (1 - self.sf) + plain[b:] * self.sf
        return torch.cat([output[:b], cond])

    def _unet_pre(self, module, args, kwargs):
        import torch
        if self.mode != "off":
            return None
        sample = args[0] if args else kwargs["sample"]
        t = args[1] if len(args) > 1 else kwargs["timestep"]
        ehs = args[2] if len(args) > 2 else kwargs.get("encoder_hidden_states")
        if self.noise is None or self.noise.shape != self.ref.shape:
            g = torch.Generator(device="cpu").manual_seed(0)
            self.noise = torch.randn(self.ref.shape, generator=g, dtype=torch.float32).to(self.ref.device, self.ref.dtype)
        # the UNet input is z_t = sqrt(a) x0 + sqrt(1-a) eps for every scheduler (k-diffusion style ones scale it with
        # scale_model_input before the UNet), so noise the reference the same way from alphas_cumprod
        tv = float(t.flatten()[0]) if isinstance(t, torch.Tensor) else float(t)
        ac = self.sched.alphas_cumprod
        tv = min(max(tv, 0.0), len(ac) - 1.0)
        lo, hi = int(tv), min(int(tv) + 1, len(ac) - 1)
        a = float(ac[lo] * (1 - (tv - lo)) + ac[hi] * (tv - lo))
        noisy = self.ref * a ** 0.5 + self.noise * (1 - a) ** 0.5
        ref_in = noisy.repeat(sample.shape[0] // noisy.shape[0], 1, 1, 1) if noisy.shape[0] != sample.shape[0] else noisy
        self.bank = {}
        self.mode = "write"
        try:
            kw = {k: v for k, v in kwargs.items() if k not in ("sample", "timestep", "encoder_hidden_states")}
            kw.pop("down_block_additional_residuals", None)
            kw.pop("mid_block_additional_residual", None)
            with torch.no_grad():
                module(ref_in.to(sample.dtype), t, ehs, **kw)
        finally:
            self.mode = "read"
        return None

    def _unet_post(self, module, args, kwargs, output):
        if self.mode == "read":
            self.mode = "off"
        return None

    def install(self):
        for name, mod in self.unet.named_modules():
            if name.endswith("attn1") and hasattr(mod, "to_q"):
                self.handles.append(mod.register_forward_pre_hook(self._attn_pre, with_kwargs=True))
                self.handles.append(mod.register_forward_hook(self._attn_post, with_kwargs=True))
        self.handles.append(self.unet.register_forward_pre_hook(self._unet_pre, with_kwargs=True))
        self.handles.append(self.unet.register_forward_hook(self._unet_post, with_kwargs=True))
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.bank = {}


# ----------------------------------------------------------------------------------------------------------- revision
# vision-only CLIP-ViT-bigG/14 with the 1280-d projection (same weights SDXL Revision uses), ~3.7 GB
REVISION_ENCODER = ("h94/IP-Adapter", "sdxl_models/image_encoder")


def revision_embeds(image, device, dtype, cache: dict):
    """CLIP-G image embeds (1280-d) used as SDXL pooled conditioning (Revision)."""
    import torch
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection  # type: ignore
    enc = cache.get("revision")
    if enc is None:
        repo, sub = REVISION_ENCODER
        enc = (CLIPVisionModelWithProjection.from_pretrained(repo, subfolder=sub, torch_dtype=dtype),
               CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14"))
        cache["revision"] = enc
    model, proc = enc
    model.to(device)
    px = proc(images=image, return_tensors="pt").pixel_values.to(device, dtype)
    with torch.no_grad():
        e = model(px).image_embeds
    model.to("cpu")
    return e


def apply_revision(embeds: dict, image_embeds, strength: float, ignore_prompt: bool) -> dict:
    """forge_preprocessor_revision: the positive SDXL pooled vector (`y[:, :1280]`) is replaced by the CLIP-G image
    embedding times the weight, the negative pooled vector becomes zeros, and "ignore prompt" also zeroes the
    positive text tokens (noise augmentation 0, as in the forge default)."""
    import torch
    out = dict(embeds)
    pooled = out.get("pooled_prompt_embeds")
    if pooled is None:
        raise EngineError(UNSUPPORTED, "Revision needs an SDXL model")
    ie = image_embeds.to(pooled.device, pooled.dtype).repeat(pooled.shape[0] // image_embeds.shape[0] or 1, 1)
    out["pooled_prompt_embeds"] = ie * strength
    if out.get("negative_pooled_prompt_embeds") is not None:
        out["negative_pooled_prompt_embeds"] = torch.zeros_like(out["negative_pooled_prompt_embeds"])
    if ignore_prompt:
        out["prompt_embeds"] = torch.zeros_like(out["prompt_embeds"])
    return out


# -------------------------------------------------------------------------------------------------------- ip-adapter
IP_ENCODERS = {False: ("h94/IP-Adapter", "models/image_encoder"), True: ("h94/IP-Adapter", "sdxl_models/image_encoder")}


class IPAdapterState:
    """Keeps one IP-Adapter loaded on the shared base pipeline; derived pipes share the UNet so they see it."""

    def __init__(self):
        self.loaded: Optional[str] = None

    def ensure(self, pipe, path: Optional[str], sdxl: bool, encoder: str = ""):
        if path == self.loaded:
            return
        if self.loaded is not None:
            try:
                pipe.unload_ip_adapter()
            except Exception:
                pass
            self.loaded = None
        if path is None:
            return
        if not os.path.isfile(path):
            raise EngineError(UNSUPPORTED, f"IP-Adapter file not found: {path}")
        import torch
        from transformers import CLIPVisionModelWithProjection  # type: ignore
        if getattr(pipe, "image_encoder", None) is None:
            repo, sub = (encoder, "") if encoder else IP_ENCODERS[sdxl]
            enc = (CLIPVisionModelWithProjection.from_pretrained(repo, subfolder=sub or None, torch_dtype=pipe.unet.dtype)
                   if not os.path.isdir(repo) else CLIPVisionModelWithProjection.from_pretrained(repo, torch_dtype=pipe.unet.dtype))
            pipe.register_modules(image_encoder=enc.to(pipe.unet.device))
        pipe.load_ip_adapter(os.path.dirname(path), subfolder="", weight_name=os.path.basename(path),
                             image_encoder_folder=None)
        self.loaded = path


# ------------------------------------------------------------------------------------------------------ controllllite
class LLLiteModule:
    """Builder for kohya ControlNet-LLLite modules (torch.nn) from a module state dict."""

    @staticmethod
    def build(weights: dict, multiplier: float):
        import torch
        nn = torch.nn
        if "conditioning1.4.weight" in weights:
            depth = 3
        elif weights["conditioning1.2.weight"].shape[-1] == 4:
            depth = 2
        else:
            depth = 1
        conv = weights["down.0.weight"].ndim == 4
        in_dim = weights["down.0.weight"].shape[1]
        cond_dim = weights["conditioning1.0.weight"].shape[0] * 2
        mlp = weights["down.0.weight"].shape[0]

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                mods = [nn.Conv2d(3, cond_dim // 2, 4, 4)]
                if depth == 1:
                    mods += [nn.ReLU(inplace=True), nn.Conv2d(cond_dim // 2, cond_dim, 2, 2)]
                elif depth == 2:
                    mods += [nn.ReLU(inplace=True), nn.Conv2d(cond_dim // 2, cond_dim, 4, 4)]
                else:
                    mods += [nn.ReLU(inplace=True), nn.Conv2d(cond_dim // 2, cond_dim // 2, 4, 4),
                             nn.ReLU(inplace=True), nn.Conv2d(cond_dim // 2, cond_dim, 2, 2)]
                self.conditioning1 = nn.Sequential(*mods)
                L = (lambda i, o: nn.Conv2d(i, o, 1)) if conv else nn.Linear
                self.down = nn.Sequential(L(in_dim, mlp), nn.ReLU(inplace=True))
                self.mid = nn.Sequential(L(mlp + cond_dim, mlp), nn.ReLU(inplace=True))
                self.up = nn.Sequential(L(mlp, in_dim))
                self.cond_image = None
                self.cond_emb = None
                self.active = True

            def forward(self, x):
                if not self.active:
                    return torch.zeros_like(x)
                if self.cond_emb is None or self.cond_emb.device != x.device:
                    cx = self.conditioning1(self.cond_image.to(x.device, x.dtype))
                    if not conv:
                        n, c, h, w = cx.shape
                        cx = cx.view(n, c, h * w).permute(0, 2, 1)
                    self.cond_emb = cx
                cx = self.cond_emb
                if x.shape[0] != cx.shape[0]:
                    cx = cx.repeat(x.shape[0] // cx.shape[0], *([1] * (cx.ndim - 1)))
                if not conv and cx.shape[1] != x.shape[1]:
                    return torch.zeros_like(x)  # trained for another resolution / block
                cx = torch.cat([cx, self.down(x)], dim=1 if conv else 2)
                return self.up(self.mid(cx)) * multiplier

        m = M()
        m.load_state_dict(weights)
        return m


class ControlLLLiteHook:
    def __init__(self, unet, path: str, cond_image, strength: float, start: float, end: float):
        self.unet = unet
        self.path = path
        self.cond = cond_image  # PIL RGB at output size
        self.strength = strength
        self.t_hi, self.t_lo = 999.0 * (1 - start), 999.0 * (1 - end)
        self.modules: dict = {}
        self.handles = []

    def _unet_pre(self, module, args, kwargs):
        import torch
        t = args[1] if len(args) > 1 else kwargs.get("timestep")
        tv = float(t.flatten()[0]) if isinstance(t, torch.Tensor) else float(t)
        on = self.t_lo <= tv <= self.t_hi
        for m in self.modules.values():
            m.active = on

    def install(self):
        import numpy as np
        import torch
        sd = _load_state(self.path)
        groups: dict = {}
        for k, v in sd.items():
            mod, _, w = k.partition(".")
            groups.setdefault(mod, {})[w] = v
        img = torch.from_numpy(np.asarray(self.cond, dtype=np.float32) / 255.0).permute(2, 0, 1)[None] * 2 - 1
        dev, dt = self.unet.device, self.unet.dtype
        found = 0
        for name, w in groups.items():
            tgt = lllite_target(name)
            if tgt is None:
                continue
            blk, bi, ai, ti, attn, proj = tgt
            try:
                lin = getattr(_attention_module(self.unet, blk, bi, ai, ti, attn), proj)
            except (AttributeError, IndexError):
                continue
            m = LLLiteModule.build(w, self.strength).to(dev, dt)
            m.cond_image = img
            self.modules[name] = m

            def pre(module, args, m=m):
                x = args[0]
                return (x + m(x),) + tuple(args[1:])
            self.handles.append(lin.register_forward_pre_hook(pre))
            found += 1
        if not found:
            raise EngineError(UNSUPPORTED, f"{os.path.basename(self.path)} is not a ControlNet-LLLite model for this UNet")
        self.handles.append(self.unet.register_forward_pre_hook(self._unet_pre, with_kwargs=True))
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.modules = {}


# ---------------------------------------------------------------------------------------------------- fooocus inpaint
class FooocusInpaintHook:
    """Fooocus inpaint patch: dequantised full-weight deltas (ldm keys) + an inpaint head added after conv_in."""

    def __init__(self, unet, patch_path: str, head_path: str, latent_image, latent_mask, start: float, end: float):
        self.unet = unet
        self.patch_path = patch_path
        self.head_path = head_path
        self.latent_image = latent_image
        self.latent_mask = latent_mask
        self.t_hi, self.t_lo = 999.0 * (1 - start), 999.0 * (1 - end)
        self.applied: list = []
        self.handles = []
        self.feature = None
        self.t = 999.0

    @staticmethod
    def dequant(v):
        import torch
        w, lo, hi = v
        return (w.float() / 255.0) * (hi.float() - lo.float()) + lo.float()

    def _deltas_diffusers(self) -> dict:
        from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint  # type: ignore
        raw = _load_state(self.patch_path) if not self.patch_path.endswith(".patch") else _load_patch(self.patch_path)
        ldm = {}
        for k, v in raw.items():
            key = k if k.startswith("model.diffusion_model.") else ("model." + k if k.startswith("diffusion_model.") else k)
            ldm[key] = self.dequant(v) if isinstance(v, (tuple, list)) else v.float()
        return convert_ldm_unet_checkpoint(ldm, self.unet.config)

    def _unet_pre(self, module, args, kwargs):
        import torch
        t = args[1] if len(args) > 1 else kwargs.get("timestep")
        self.t = float(t.flatten()[0]) if isinstance(t, torch.Tensor) else float(t)

    def _conv_in_post(self, module, args, output):
        if self.feature is None or not (self.t_lo <= self.t <= self.t_hi):
            return None
        f = self.feature.to(output.device, output.dtype)
        if f.shape[0] != output.shape[0]:
            f = f.repeat(output.shape[0] // f.shape[0], 1, 1, 1)
        if f.shape[-2:] != output.shape[-2:]:
            return None
        return output + f

    def install(self):
        import torch
        params = dict(self.unet.named_parameters())
        for k, d in self._deltas_diffusers().items():
            p = params.get(k)
            if p is not None and p.shape == d.shape:
                with torch.no_grad():
                    p.add_(d.to(p.device, p.dtype))
                self.applied.append((p, k))
        if not self.applied:
            raise EngineError(UNSUPPORTED, "the Fooocus inpaint patch did not match this UNet (SDXL only)")
        head = _load_state(self.head_path)["head"].float()
        feed = torch.cat([self.latent_mask.float().cpu(), self.latent_image.float().cpu()], dim=1)
        feed = torch.nn.functional.pad(feed, (1, 1, 1, 1), "replicate")
        self.feature = torch.nn.functional.conv2d(feed, head)
        self.handles.append(self.unet.register_forward_pre_hook(self._unet_pre, with_kwargs=True))
        self.handles.append(self.unet.conv_in.register_forward_hook(self._conv_in_post))
        return self

    def remove(self):
        import torch
        for h in self.handles:
            h.remove()
        self.handles = []
        if self.applied:
            deltas = self._deltas_diffusers()
            for p, k in self.applied:
                with torch.no_grad():
                    p.sub_(deltas[k].to(p.device, p.dtype))
        self.applied = []


def _load_patch(path: str) -> dict:
    from .engine import safe_torch_load
    return safe_torch_load(path)


# ---------------------------------------------------------------------------------------------------------- photomaker
def photomaker_available() -> bool:
    try:
        import photomaker  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def normalize(cfg: dict, defaults: dict) -> dict:
    from .hooks import coerce_bool
    out = dict(defaults)
    out.update(cfg)
    for k, d in defaults.items():
        if isinstance(d, bool):
            out[k] = coerce_bool(out.get(k), d)
        elif isinstance(d, float):
            out[k] = float(out.get(k, d))
        elif isinstance(d, int):
            out[k] = int(float(out.get(k, d)))
    return out


__all__ = ["ReferenceOnlyHook", "revision_embeds", "apply_revision", "IPAdapterState", "ControlLLLiteHook",
           "FooocusInpaintHook", "lllite_target", "ldm_block_to_diffusers", "photomaker_available", "normalize"]
