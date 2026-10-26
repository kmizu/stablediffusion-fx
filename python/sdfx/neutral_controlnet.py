"""Capture native ControlNet step inputs and rebuild residuals for each Neutral Prompt leaf.

The original pipeline owns preprocessing, guidance windows, multi-model scales and control images. Only text/pooled
conditioning changes between leaves. PAG changes UNet self-attention, so its ordinary/perturbed groups share the
corresponding text-conditioned ControlNet residuals. Torch is imported locally for the torch-free engine.
"""
from .protocol import EngineError, UNSUPPORTED


class ControlNetFrame:
    def __init__(self, model):
        self.model = model
        self.latest = None
        self.replaying = False
        self.handle = None

    def capture(self, module, args, kwargs):
        if not self.replaying:
            self.latest = (args, dict(kwargs))

    def install(self):
        self.handle = self.model.register_forward_pre_hook(self.capture, with_kwargs=True)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.latest = None

    def residuals(self, branch_kwargs, batch, layout):
        import torch
        if self.latest is None:
            raise EngineError(UNSUPPORTED, "Neutral Prompt did not receive a native ControlNet call for this step")
        args, kw = self.latest
        kw = dict(kw)
        sample = args[0] if args else kw.pop("sample")
        timestep = args[1] if len(args) > 1 else kw.pop("timestep")
        if len(args) > 2:
            raise EngineError(UNSUPPORTED, "Neutral Prompt expects native ControlNet conditioning as keyword arguments")
        source_batch = sample.shape[0]
        guess = bool(kw.get("guess_mode", False))
        count = 1 if guess else 2

        def hint_batch(value):
            if torch.is_tensor(value) and value.ndim and value.shape[0] == source_batch:
                return torch.cat([value[:batch]] * count)
            if isinstance(value, dict):
                return {k: hint_batch(v) for k, v in value.items()}
            if isinstance(value, list):
                return [hint_batch(v) for v in value]
            return value

        def text_batch(value):
            # Runtime layout always starts with ordinary negative, ordinary positive.
            return value[batch:2 * batch] if guess else value[:2 * batch]

        kw = {k: hint_batch(v) for k, v in kw.items()}
        inputs = torch.cat([sample[:batch]] * count)
        kw["encoder_hidden_states"] = text_batch(branch_kwargs["encoder_hidden_states"])
        if "added_cond_kwargs" in kw:
            added = dict(kw["added_cond_kwargs"] or {})
            for key in ("text_embeds", "time_ids"):
                value = (branch_kwargs.get("added_cond_kwargs") or {}).get(key)
                if value is not None:
                    added[key] = text_batch(value)
            kw["added_cond_kwargs"] = added
        self.replaying = True
        try:
            result = self.model(inputs, timestep, **kw)
        finally:
            self.replaying = False
        down, mid = result[:2] if isinstance(result, tuple) else (result.down_block_res_samples, result.mid_block_res_sample)

        def align(value):
            if guess:
                negative, positive = torch.zeros_like(value), value
            else:
                negative, positive = value.chunk(2)
            return torch.cat([positive if p else negative for p in layout])

        return [align(v) for v in down], align(mid)
