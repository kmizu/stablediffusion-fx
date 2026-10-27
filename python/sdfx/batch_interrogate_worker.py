"""Internal WD-only worker service; registered only by trusted worker setup.

Public JSON never creates registries, asset pins or snapshot identities. A
prepare request inspects local images and returns a short-lived, owner-bound,
single-use token. The worker queue must serialize both service methods. No
production provider is configured by this module and no model is downloaded.
The token is a bearer secret; owner_id is a routing label, not authentication.
Once execute starts and passes its cancellation check, it consumes the token
even if cancellation or failure occurs immediately afterward; retry requires a
new preparation. A cancellation before that check leaves the token available.
"""
from dataclasses import dataclass
import re
import time
import uuid

from .batch_face_manifest import build_manifest
from .batch_interrogate import BatchInterrogateOptions
from .batch_interrogate_backends import BackendRegistry
from .batch_interrogate_request import execute_prepared, prepare_request
from .batch_interrogate_runner import BatchInput
from .protocol import EngineError, INVALID_PARAMS, UNSUPPORTED

MAX_PREPARED = 8
PREPARED_TTL_SECONDS = 300
_TOKEN = re.compile(r"[0-9a-f]{32}\Z")


@dataclass(frozen=True)
class _Entry:
    owner: str
    expires: float
    request: object


class BatchInterrogateService:
    def __init__(self, registry, *, ready=None):
        if not isinstance(registry, BackendRegistry):
            raise ValueError("trusted internal backend registry required")
        from .wd_v3_provider import WdV3Provider
        from .wd_v3_runtime import _MODEL_IDENTITY, _TAGS_IDENTITY
        provider = registry._selection(("wd_ext",))[0]
        if (type(provider) is not WdV3Provider
                or (provider._model.size, provider._model.sha256) != _MODEL_IDENTITY
                or (provider._tags.size, provider._tags.sha256) != _TAGS_IDENTITY):
            raise ValueError("the fixed WD v3 provider and candidate pins are required")
        # Copy the selection table so callers cannot replace the provider
        # through their original registry after construction.
        self._registry = BackendRegistry({"wd_ext": provider})
        self._entries = {}
        if ready is not None and not callable(ready):
            raise ValueError("invalid service readiness check")
        self._ready = ready or (lambda: True)

    def available(self):
        return bool(self._ready())

    def _owner(self, ctx):
        owner = getattr(ctx, "owner_id", None)
        if not isinstance(owner, str) or not 1 <= len(owner) <= 256:
            raise EngineError(INVALID_PARAMS, "batch owner is required")
        return owner

    def _prune(self):
        now = time.monotonic()
        for key in tuple(self._entries):
            if self._entries[key].expires <= now:
                del self._entries[key]

    def prepare(self, engine, params, ctx):
        owner = self._owner(ctx)
        if not self.available():
            raise EngineError(UNSUPPORTED, "configured local WD assets are unavailable")
        allowed = {"images_root", "generation", "output_parent", "output_name", "checkpoint_token"}
        if not isinstance(params, dict) or set(params) != allowed:
            raise EngineError(INVALID_PARAMS, "invalid batch preparation fields")
        for key in ("images_root", "output_parent", "output_name", "checkpoint_token"):
            if not isinstance(params[key], str) or not 1 <= len(params[key]) <= 4096:
                raise EngineError(INVALID_PARAMS, f"invalid {key}")
        self._prune()
        if len(self._entries) >= MAX_PREPARED:
            raise EngineError(INVALID_PARAMS, "too many pending batch preparations")
        try:
            self._registry._selection(("wd_ext",))
            model = getattr(engine, "model", None)
            if not isinstance(model, dict) or model.get("model_token") != params["checkpoint_token"]:
                raise ValueError("loaded checkpoint changed or unavailable")
            manifest = build_manifest(params["images_root"], check_cancel=ctx.check_interrupt)
            inputs = tuple(BatchInput(str(item.image.path), item.image) for item in manifest.items)
            request = prepare_request(params["checkpoint_token"], inputs, params["generation"], ("wd_ext",),
                                      BatchInterrogateOptions(), params["output_parent"], params["output_name"])
        except ValueError as exc:
            raise EngineError(INVALID_PARAMS, str(exc)) from exc
        ctx.check_interrupt()
        token = uuid.uuid4().hex
        entry = _Entry(owner, time.monotonic() + PREPARED_TTL_SECONDS, request)
        ctx.publish(lambda: self._entries.__setitem__(token, entry))
        return {"token": token, "image_count": len(inputs), "backend": "wd_ext", "partial": True}

    def execute(self, engine, params, ctx):
        owner = self._owner(ctx)
        if not self.available():
            raise EngineError(UNSUPPORTED, "configured local WD assets are unavailable")
        if (not isinstance(params, dict) or set(params) != {"token"} or not isinstance(params["token"], str)
                or not _TOKEN.fullmatch(params["token"])):
            raise EngineError(INVALID_PARAMS, "invalid prepared batch token")
        self._prune()
        entry = self._entries.get(params["token"])
        if entry is None or entry.owner != owner:
            raise EngineError(INVALID_PARAMS, "prepared batch unavailable or expired")
        ctx.check_interrupt()
        del self._entries[params["token"]]
        return {"rows": execute_prepared(engine, entry.request, self._registry, ctx), "partial": True}
