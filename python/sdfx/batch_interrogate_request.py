"""Internal frozen batch submission; not a registered worker operation.

Only trusted code can construct FileSnapshot values and the local registry;
never deserialize either from a public request. Queue serialization and owned
cancellation remain external obligations. This does not advertise capability.
"""
from dataclasses import dataclass
import json
import math
import os

from .batch_face_manifest import FileSnapshot
from .batch_interrogate import BatchInterrogateOptions, compose
from .batch_interrogate_runner import BatchInput, MAX_ITEMS, TransactionFailure, run_registered_batch
from .temporal_lab_io import OutputTransaction, _absolute

MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_DEPTH = 24
MAX_VALUES = 20_000


@dataclass(frozen=True)
class InternalSupport:
    public_capability: bool
    full_parity: bool
    candidate: str
    unavailable: tuple[str, ...]


def internal_support():
    return InternalSupport(False, False, "wd-vit-tagger-v3", ("clip", "deepbooru", "clip_ext", "other_wd_models"))


@dataclass(frozen=True)
class PreparedRequest:
    checkpoint_token: str
    inputs: tuple[BatchInput, ...]
    generation_json: str
    models: tuple[str, ...]
    options: BatchInterrogateOptions
    output_parent: str
    output_name: str

    def generation(self):
        return json.loads(self.generation_json)


def _json_value(value, depth, budget):
    budget[0] += 1
    if depth > MAX_DEPTH or budget[0] > MAX_VALUES:
        raise ValueError("generation request exceeds structural limit")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value): raise ValueError("generation request contains nonfinite value")
        return
    if type(value) is list:
        for item in value: _json_value(item, depth + 1, budget)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values(): _json_value(item, depth + 1, budget)
        return
    raise ValueError("generation request must contain JSON values")


def prepare_request(checkpoint_token, inputs, generation, models, options, output_parent, output_name):
    if not isinstance(checkpoint_token, str) or not 1 <= len(checkpoint_token) <= 256:
        raise ValueError("checkpoint identity required")
    if not isinstance(inputs, (tuple, list)) or not 1 <= len(inputs) <= MAX_ITEMS:
        raise ValueError("bounded internal image snapshots required")
    checked = tuple(inputs)
    if any(not isinstance(item, BatchInput) or not isinstance(item.identity, FileSnapshot)
           or not isinstance(item.path, str) or _absolute(item.path) != _absolute(item.identity.path) for item in checked):
        raise ValueError("internal image snapshots are invalid")
    if len({os.path.normcase(str(_absolute(item.path))) for item in checked}) != len(checked):
        raise ValueError("duplicate image paths are unsupported")
    if not isinstance(models, (tuple, list)) or tuple(models) != ("wd_ext",):
        raise ValueError("only the internal WD v3 candidate is supported")
    if not isinstance(options, BatchInterrogateOptions) or not isinstance(generation, dict) or not isinstance(generation.get("params"), dict):
        raise ValueError("invalid generation or composer options")
    _json_value(generation, 0, [0])
    try:
        frozen = json.dumps(generation, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("invalid generation request") from exc
    try:
        if len(frozen.encode("utf-8")) > MAX_REQUEST_BYTES:
            raise ValueError("generation request exceeds size limit")
    except UnicodeError as exc:
        raise ValueError("invalid generation request text") from exc
    params = generation["params"]
    compose(params.get("prompt", ""), params.get("negative_prompt", ""), (), options)
    hires = params.get("hires") or {}
    if not isinstance(hires, dict) or generation.get("vae_recovery_dir") or params.get("hooks") or hires.get("enabled"):
        raise ValueError("external output, hooks and hires are unsupported")
    transaction = OutputTransaction(output_parent, output_name, [item.path for item in checked])
    return PreparedRequest(checkpoint_token, checked, frozen, tuple(models), options, str(transaction.parent), output_name)


def _check_model(engine, token):
    model = getattr(engine, "model", None)
    if not isinstance(model, dict) or model.get("model_token") != token:
        raise TransactionFailure("loaded checkpoint changed; batch aborted")


class _CheckpointGuard:
    def __init__(self, engine, token): self.engine, self.token = engine, token
    def img2img(self, request, ctx):
        _check_model(self.engine, self.token)
        result = self.engine.img2img(request, ctx)
        _check_model(self.engine, self.token)
        return result


def execute_prepared(engine, request, registry, ctx):
    if not isinstance(request, PreparedRequest):
        raise ValueError("internal prepared request required")
    _check_model(engine, request.checkpoint_token)
    proxy = _CheckpointGuard(engine, request.checkpoint_token)
    return run_registered_batch(proxy, request.inputs, request.generation(), request.models, request.options,
                                registry, request.output_parent, request.output_name, ctx,
                                _before_publish=lambda: _check_model(engine, request.checkpoint_token))
