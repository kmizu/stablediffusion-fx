"""Diffusers Neutral Prompt runtime. Scheduler conversions use the current native denoising stage.

This module is torch-free at import time. Composition math and MIT attribution live in neutral_prompt.py.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from . import neutral_prompt
from .protocol import EngineError, UNSUPPORTED


@dataclass
class Plan:
    tree: neutral_prompt.Node
    branches: list[dict]
    phi: float = 0
    image_trees: list[neutral_prompt.Node] | None = None


def validate(hooks, *, controlnet=False, pag=False, allow_photomaker=False):
    """Temporary explicit compatibility boundary while branch-dependent integrations are unfinished."""
    from .hooks import hook_config
    blocked = ("photomaker", "sag",
               "apg", "mahiro", "latent_modifier", "dynamic_thresholding", "detail_daemon")
    conflicts = [name for name in blocked if not (name == 'photomaker' and allow_photomaker)
                 and hook_config(hooks, name, {}) is not None]
    if conflicts:
        raise EngineError(UNSUPPORTED, "Neutral Prompt integration pending for: " + ", ".join(conflicts))


def noise_level(scheduler, timestep):
    """Return native UNet-input alpha, noise and sigma, including intermediate solver stages."""
    name = type(scheduler).__name__
    if name in ("DDIMScheduler", "DDIMInverseScheduler", "LCMScheduler"):
        index = int(timestep)
        if name == "DDIMInverseScheduler":
            index = min(index - scheduler.config.num_train_timesteps // scheduler.num_inference_steps,
                        scheduler.config.num_train_timesteps - 1)
        product = scheduler.initial_alpha_cumprod if index < 0 else scheduler.alphas_cumprod[index]
        alpha = math.sqrt(float(product))
        noise = math.sqrt(max(1 - alpha**2, 0))
        sigma = noise / alpha if alpha else math.inf
    elif name in ("EulerDiscreteScheduler", "EulerAncestralDiscreteScheduler", "LMSDiscreteScheduler",
                  "HeunDiscreteScheduler", "KDPM2DiscreteScheduler", "KDPM2AncestralDiscreteScheduler",
                  "DPMSolverSDEScheduler", "DPMSolverMultistepScheduler", "DPMSolverSinglestepScheduler",
                  "UniPCMultistepScheduler"):
        index = scheduler.step_index
        if index is None:
            index = scheduler.index_for_timestep(timestep)
        sigma = scheduler.sigmas[index]
        if name == "KDPM2DiscreteScheduler" and not scheduler.state_in_first_order:
            sigma = scheduler.sigmas_interpol[index]
        elif name == "KDPM2AncestralDiscreteScheduler" and not scheduler.state_in_first_order:
            sigma = scheduler.sigmas_interpol[index - 1]
        elif name == "DPMSolverSDEScheduler" and not scheduler.state_in_first_order:
            sigma = scheduler.mid_point_sigma
        sigma = float(sigma)
        alpha = 1 / math.sqrt(1 + sigma**2)
        noise = 1.0 if math.isinf(sigma) else sigma * alpha
    else:
        raise EngineError(UNSUPPORTED, f"Neutral Prompt: unsupported scheduler {name}")
    return alpha, noise, sigma


def coefficients(scheduler, timestep):
    """Return A,B where x0 = A * normalized UNet input + B * model prediction."""
    alpha, noise, sigma = noise_level(scheduler, timestep)
    pred = scheduler.config.prediction_type
    if pred == "epsilon":
        if alpha == 0:
            raise EngineError(UNSUPPORTED, "Neutral Prompt: epsilon prediction is undefined at zero SNR")
        return 1 / alpha, -sigma
    if pred == "v_prediction":
        return alpha, -noise
    if pred == "sample":
        return 0.0, 1.0
    raise EngineError(UNSUPPORTED, f"Neutral Prompt: unsupported prediction type {pred}")


def to_x0(model_input, prediction, ab):
    a, b = ab
    return a * model_input.float() + b * prediction.float()


def from_x0(model_input, x0, ab):
    a, b = ab
    if b == 0:
        raise EngineError(UNSUPPORTED, "Neutral Prompt cannot compose a zero-noise model evaluation")
    return (x0.float() - a * model_input.float()) / b


def at(branch, step):
    schedule = getattr(branch, "schedule", None)
    if schedule:
        return next((d for end, d in schedule if step <= end), schedule[-1][1])
    return branch


def zero_noise_guidance(z, unconditional, delta, cfg, phi):
    """Continuous positive-noise limit of x0 guidance/rescale when DDIM inversion starts at alpha=1.

    Mean/std rescale becomes a projection on the shared image's constant/centered components. If that image
    is constant, its variance comes entirely from the prediction, so noise-space rescale is the exact limit.
    """
    import torch
    u, d = unconditional.float(), delta.float()
    guided = u + cfg * d
    if not phi or cfg == 1:
        return guided
    center = z.float() - z.float().flatten(1).mean(1).view(-1, *([1] * (z.ndim - 1)))
    mean = d.flatten(1).mean(1).view(-1, *([1] * (d.ndim - 1)))
    projection = mean + d - neutral_prompt.perpendicular(center, d)
    varied = (center.flatten(1).abs().amax(1) > 0).view(-1, *([1] * (z.ndim - 1)))
    return torch.where(varied, guided + phi * (1 - cfg) * projection,
                       neutral_prompt.rescale(guided, u + d, phi))


class RuntimeHook:
    """Evaluate each positive leaf against the same input, then present a CFG-compatible composed prediction.

    Only the enclosing UNet call invokes root hooks; leaf evaluations use its saved forward method. Explicit
    branch_hooks prepare branch-dependent image conditioning. Engine guards integrations not yet wired into that
    lifecycle. Attention/module hooks still run.
    """
    def __init__(self, unet, scheduler, plan: Plan, cfg: float, phi: float, *, guided=None, pag=False, controlnet=None, image_embeds=None,
                 check_interrupt=lambda: None, branch_hooks=(), sampling=None, branch_transform=None):
        self.unet, self.scheduler, self.plan = unet, scheduler, plan
        self.cfg, self.phi = cfg, phi
        self.guided = cfg > 1 if guided is None else guided
        self.pag = pag
        self.controlnet = controlnet
        self.image_embeds = image_embeds
        self.branch_hooks = sorted(branch_hooks, key=lambda h: h.priority)
        self.sampling = sampling
        self.step = 1
        self.raw_step = 1
        self.branch_transform = branch_transform
        self.check_interrupt = check_interrupt
        self._missing = object()
        self._installed = False

    def callback(self, pipe, i, t, tensors):
        order = max(1, int(getattr(self.scheduler, "order", 1)))
        self.step = (int(i) + 1) // order + 1
        self.raw_step = int(i) + 2
        return tensors

    def forward(self, sample, timestep, *args, **kwargs):
        import torch
        self.check_interrupt()
        branches = [at(branch, self.step) for branch in self.plan.branches]
        if self.branch_transform is not None:
            branches = [self.branch_transform(index, branch, self.raw_step)
                        for index, branch in enumerate(branches)]
        if self.sampling is not None:
            branches = self.sampling.branches(branches, timestep)
        batch = branches[0]["prompt_embeds"].shape[0]
        # A PAG pipeline/config can be inactive (scale=0 or no applied layers). Its actual native batch is
        # authoritative; do not infer activation solely from a nonempty configuration dictionary.
        pag = self.pag and sample.shape[0] == batch * (3 if self.guided else 2)
        native_groups = 3 if self.guided and pag else 2 if self.guided or pag else 1
        if sample.shape[0] != batch * native_groups:
            raise EngineError(UNSUPPORTED, "Neutral Prompt received an unsupported guidance batch")
        indices = list(range(native_groups)) if self.guided else [0, 0, 1, 1] if pag else [0, 0]
        inputs = torch.cat([sample.chunk(native_groups)[i] for i in indices])
        layout = [False, True, True] if self.guided and pag else [False, True, False, True] if pag else [False, True]

        def expand(value):
            if torch.is_tensor(value) and value.ndim:
                if value.shape[0] == sample.shape[0]:
                    return torch.cat([value.chunk(native_groups)[i] for i in indices])
                if value.shape[0] == batch:
                    return torch.cat([value] * len(indices))
            if isinstance(value, dict):
                return {k: expand(v) for k, v in value.items()}
            if isinstance(value, list):
                return [expand(v) for v in value]
            return value

        common = dict(kwargs)
        for key in ("timestep_cond", "class_labels", "added_cond_kwargs", "attention_mask", "encoder_attention_mask"):
            if key in common:
                common[key] = expand(common[key])
        if self.image_embeds is not None:
            from .neutral_ip import align
            common["added_cond_kwargs"] = dict(common.get("added_cond_kwargs") or {},
                image_embeds=align(self.image_embeds, batch, layout, inputs.device, inputs.dtype))
        results, perturbed, template = [], [], None
        for index, branch in enumerate(branches):
            self.check_interrupt()
            kw = dict(common)
            kw["encoder_hidden_states"] = torch.cat([branch["prompt_embeds" if positive else "negative_prompt_embeds"] for positive in layout]).to(inputs.device, inputs.dtype)
            added = dict(kw.get("added_cond_kwargs") or {})
            if "pooled_prompt_embeds" in branch:
                added["text_embeds"] = torch.cat([branch["pooled_prompt_embeds" if positive else "negative_pooled_prompt_embeds"] for positive in layout]).to(inputs.device, inputs.dtype)
            if added:
                kw["added_cond_kwargs"] = added
            if self.controlnet is not None:
                down, mid = self.controlnet.residuals(kw, batch, layout)
                kw["down_block_additional_residuals"] = down
                kw["mid_block_additional_residual"] = mid
            prepared = []
            try:
                for hook in self.branch_hooks:
                    prepared.append(hook)
                    if hasattr(hook, "set_branch"):
                        hook.set_branch(index, branch, self.step)
                    hook.before_branch(self.original, inputs, timestep, kw, layout)
                output = self.original(inputs, timestep, *args, **kw)
            finally:
                for hook in reversed(prepared):
                    hook.after_branch()
            prediction = output[0] if isinstance(output, tuple) else output.sample
            if template is None:
                template, unconditional = output, prediction[:batch]
            results.append(prediction[batch:2 * batch])
            if pag:
                perturbed.append(prediction[-batch:])
        z = inputs[:batch, :unconditional.shape[1]]
        ab = coefficients(self.scheduler, timestep)

        def compose_predictions(values, cfg, phi):
            differences = [v.float() - unconditional.float() for v in values]
            def combine(inputs, zero_noise_limit=False):
                if self.plan.image_trees is None:
                    return neutral_prompt.combine_deltas(self.plan.tree, inputs, zero_noise_limit=zero_noise_limit)
                from .model_keyword import compose_per_item
                return compose_per_item(self.plan.image_trees, inputs, lambda tree: len(neutral_prompt.leaves(tree)),
                    lambda tree, branches: neutral_prompt.combine_deltas(tree, branches, zero_noise_limit=zero_noise_limit),
                    lambda parts: torch.cat(parts, dim=0))
            if ab[1] == 0:
                delta = combine(differences, zero_noise_limit=True)
                return unconditional.float() + delta, zero_noise_guidance(z, unconditional, delta, cfg, phi)
            delta = combine([ab[1] * v for v in differences])
            # Keep the common x0 baseline out of subtraction when rescale is off; late low-noise steps otherwise
            # amplify float32 cancellation when converting composed x0 back into model predictions.
            conditional = unconditional.float() + delta / ab[1]
            guided = unconditional.float() + cfg * delta / ab[1]
            if phi:
                x0u = to_x0(z, unconditional, ab)
                x0 = neutral_prompt.rescale(x0u + cfg * delta, x0u + delta, phi)
                guided = from_x0(z, x0, ab)
            return conditional, guided

        cfg = self.sampling.cfg(self.cfg, timestep, guided=self.guided) if self.sampling is not None else self.cfg
        raw_conditional, guided = compose_predictions(results, cfg, self.phi)
        conditional = (unconditional.float() + (guided - unconditional.float()) / self.cfg
                       if (self.phi or cfg != self.cfg) and self.cfg else raw_conditional)
        if pag:
            raw_perturbed, _ = compose_predictions(perturbed, 1, 0)
            normal = conditional if self.guided else guided
            perturb = normal - (raw_conditional - raw_perturbed)
            guided = torch.cat(([unconditional if self.cfg else guided] if self.guided else []) + [normal, perturb])
        elif self.guided:
            guided = torch.cat([unconditional if self.cfg else guided, conditional])
        guided = guided.to(unconditional.dtype)
        if isinstance(template, tuple):
            return (guided,) + template[1:]
        template.sample = guided
        return template

    def install(self):
        if self._installed:
            raise RuntimeError("Neutral Prompt hook already installed")
        self.previous = self.unet.__dict__.get("forward", self._missing)
        self.original = self.unet.forward
        self.unet.forward = self.forward
        self._installed = True
        return self

    def remove(self):
        if self._installed:
            if self.previous is self._missing:
                del self.unet.forward
            else:
                self.unet.forward = self.previous
            self._installed = False
