"""Engine used when torch/diffusers cannot be imported: listing works, model/generation requests fail clearly."""
from __future__ import annotations

from .base import BaseEngine
from .protocol import MODEL_LOAD_FAILED, NO_MODEL, UNSUPPORTED, EngineError, JobContext


class UnavailableEngine(BaseEngine):
    def __init__(self, reason: str):
        super().__init__()
        self.reason = reason

    def _fail(self, code: str):
        raise EngineError(code, f"inference backend unavailable: {self.reason} (install requirements into python/.venv)")

    def load_model(self, params: dict, ctx: JobContext):
        self._fail(MODEL_LOAD_FAILED)

    def set_vae(self, params: dict, ctx: JobContext):
        self._fail(NO_MODEL)

    def txt2img(self, params: dict, ctx: JobContext):
        self._fail(NO_MODEL)

    def img2img(self, params: dict, ctx: JobContext):
        self._fail(NO_MODEL)

    def preprocess(self, params: dict, ctx: JobContext):
        self._fail(UNSUPPORTED)

    def upscale(self, params: dict, ctx: JobContext):
        self._fail(UNSUPPORTED)

    def on_oom(self):
        pass
