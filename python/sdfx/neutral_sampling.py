"""Per-pass sampling state for composed prompts; ordinary root hooks stay in sampling_ext."""
from __future__ import annotations

from . import sampling_ext as native
from .hooks import hook_config


class Controller:
    def __init__(self, scheduler, hooks, seed):
        self.progress = native.Progress(scheduler)
        cs = hook_config(hooks, "cfg_schedule", native.CFG_SCHEDULE_DEFAULTS)
        ca = hook_config(hooks, "cads", native.CADS_DEFAULTS)
        self.schedule = native.normalize(cs, native.CFG_SCHEDULE_DEFAULTS) if cs is not None else None
        self.cads = native.normalize(ca, native.CADS_DEFAULTS) if ca is not None else None
        if self.cads is not None:
            import torch
            if self.cads["tau1"] > self.cads["tau2"]:
                self.cads["tau1"], self.cads["tau2"] = self.cads["tau2"], self.cads["tau1"]
            self.generator = torch.Generator("cpu").manual_seed(int(seed) & 0xFFFFFFFF)

    def cfg(self, base, timestep, *, guided):
        if self.schedule is None or not guided or base <= 1:
            return base
        return native.cfg_at(self.schedule, self.progress(timestep)[2])

    def _noise(self, value, gamma, marked, noise=None):
        return native.cads_context(value, gamma, self.cads, self.generator, marked=marked, noise=noise)

    def branches(self, branches, timestep):
        if self.cads is None:
            return branches
        c = self.cads
        gamma = native.cads_gamma(self.progress(timestep)[2], c["tau1"], c["tau2"])
        if gamma >= 1:
            return branches
        result, negatives = [], {}
        for branch in branches:
            out = dict(branch)
            marked = bool(branch.get("_negpip"))
            # All leaves share the request's negative prompt; keep its stochastic condition shared too.
            if c["apply_to_uncond"]:
                import torch
                value = branch["negative_prompt_embeds"]
                shape = value[..., :-1].shape if marked else value.shape
                key = tuple(shape)
                if key not in negatives:
                    negatives[key] = torch.randn(shape, generator=self.generator, device="cpu", dtype=torch.float32)
                # Share randomness, not the transformed tensor: each leaf retains its own valid-token metadata.
                out["negative_prompt_embeds"] = self._noise(value, gamma, marked, negatives[key])
            out["prompt_embeds"] = self._noise(branch["prompt_embeds"], gamma, marked)
            result.append(out)
        return result
