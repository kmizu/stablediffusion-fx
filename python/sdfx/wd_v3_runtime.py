"""Experimental internal WD v3 CPU inference; not a worker capability.

The local candidate digests are not independently authenticated publisher
digests. The parser and native ORT calls may allocate beyond source bytes and
cannot be interrupted while executing; no hard native RAM limit is promised.
"""
from dataclasses import dataclass
import hashlib
import os

from .wd_v3_assets import AssetPin, MAX_TAG_BYTES, owned_assets, parse_tags, prepare_image
from .wd_v3_onnx_audit import WD_V3_LOCAL_CANDIDATE_POLICY, inspect_model_bytes, read_model_bytes

_MODEL_IDENTITY = (378_536_310, "35f23693620b668f4d53fd3c62bf65e40af739bc52c7eb0fbc49258b58d065b6")
_TAGS_IDENTITY = (308_468, "298633d94d0031d2081c0893f29c82eab7f0df00b08483ba8f29d1e979441217")
_POLICY = WD_V3_LOCAL_CANDIDATE_POLICY


@dataclass(frozen=True)
class ScoredTag:
    name: str
    category: int
    score: float


def _session_shape(info, expected):
    if info.type != "tensor(float)" or len(info.shape) != len(expected):
        raise ValueError("runtime model signature differs from audited model")
    for actual, wanted in zip(info.shape, expected):
        if wanted is None:
            if not (actual is None or isinstance(actual, str)):
                raise ValueError("runtime model signature differs from audited model")
        elif type(actual) is not int or actual != wanted:
            raise ValueError("runtime model signature differs from audited model")
    if not isinstance(info.name, str) or not 1 <= len(info.name) <= 512:
        raise ValueError("invalid runtime tensor name")


def run_internal(model, tags, temp_parent, image, ctx, *, _ort=None):
    """Run one local pinned model against one image, returning unfiltered scores.

    `_ort` is an internal fake-injection seam, never exposed in a worker request.
    Deleting the owned session reference initiates release; provider-native
    allocation reclamation and cancellation inside native calls need real tests.
    """
    if (not isinstance(model, AssetPin) or not isinstance(tags, AssetPin)
            or (model.size, model.sha256) != _MODEL_IDENTITY
            or (tags.size, tags.sha256) != _TAGS_IDENTITY):
        raise ValueError("WD v3 assets differ from the internal candidate pins")
    with owned_assets(model, tags, temp_parent, ctx) as bound:
        with bound.tags.open("rb") as source:
            tag_bytes = source.read(MAX_TAG_BYTES + 1)
        if len(tag_bytes) != tags.size or hashlib.sha256(tag_bytes).hexdigest() != tags.sha256:
            raise ValueError("WD tag asset changed after staging")
        rows = parse_tags(tag_bytes)
        del tag_bytes
        if len(rows) != _POLICY.output_shape[-1]:
            raise ValueError("WD tag count does not match model output")
        ctx.check_interrupt()
        bound_pin = AssetPin(bound.model, model.size, model.sha256)
        session = None
        model_bytes = None
        try:
            try:
                model_bytes = read_model_bytes(bound_pin, check_interrupt=ctx.check_interrupt)
                inspect_model_bytes(model_bytes, bound_pin, _POLICY, check_interrupt=ctx.check_interrupt)
                ctx.check_interrupt()
            except BaseException as exc:
                # Reader/parser frames can retain both the source bytes and
                # the decoded graph. Keep the classification and message.
                exc.__traceback__ = None
                exc.__context__ = None
                exc.__cause__ = None
                raise exc from None
            try:
                import onnxruntime as ort
            except ImportError as exc:
                if _ort is None:
                    raise ValueError("local ONNX Runtime dependency unavailable") from exc
                ort = _ort
            else:
                if _ort is not None:
                    ort = _ort
            import numpy as np
            options = ort.SessionOptions()
            options.enable_mem_pattern = False
            options.enable_cpu_mem_arena = False
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            try:
                session = ort.InferenceSession(model_bytes, sess_options=options, providers=["CPUExecutionProvider"])
            except BaseException as exc:
                # Constructor traceback frames can hold the entire model as an
                # argument. Preserve the exception type and message, while
                # dropping those frames and chained references to them.
                exc.__traceback__ = None
                exc.__context__ = None
                exc.__cause__ = None
                raise exc from None
            ctx.check_interrupt()
            inputs, outputs = session.get_inputs(), session.get_outputs()
            if len(inputs) != 1 or len(outputs) != 1 or session.get_providers() != ["CPUExecutionProvider"]:
                raise ValueError("runtime provider or I/O count mismatch")
            _session_shape(inputs[0], _POLICY.input_shape)
            _session_shape(outputs[0], _POLICY.output_shape)
            pixels = prepare_image(image, _POLICY.input_shape[1])
            ctx.check_interrupt()
            predictions = session.run([outputs[0].name], {inputs[0].name: pixels})
            ctx.check_interrupt()
            if (not isinstance(predictions, list) or len(predictions) != 1
                    or not isinstance(predictions[0], np.ndarray)
                    or predictions[0].dtype != np.float32 or predictions[0].shape != (1, len(rows))
                    or not np.isfinite(predictions[0]).all()
                    or (predictions[0] < 0).any() or (predictions[0] > 1).any()):
                raise ValueError("invalid WD model scores")
            return tuple(ScoredTag(row.name, row.category, float(score))
                         for row, score in zip(rows, predictions[0][0]))
        finally:
            session = None
            model_bytes = None
