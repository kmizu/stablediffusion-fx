"""Internal static ONNX screening, never an inference-safety guarantee.

Policy/pin come from reviewed internal configuration, not worker requests.
Full protobuf parsing uses substantial RAM and cannot be interrupted while
the native parser executes; admission bounds source bytes but not parser peak
RAM. Screening does not validate operator semantics, native runtime memory,
model compatibility or license. No model session is created here.
"""
from dataclasses import dataclass
import hashlib
import os
import stat

from .wd_v3_assets import AssetPin, _signature, _validate_pin

MAX_AUDIT_BYTES = 400_000_000
MAX_NODES = 10_000
MAX_INITIALIZERS = 20_000
MAX_ATTRIBUTES = 100_000


@dataclass(frozen=True)
class AuditPolicy:
    opset: int
    operators: tuple[str, ...]
    input_shape: tuple[int | None, ...]
    output_shape: tuple[int | None, ...]


@dataclass(frozen=True)
class AuditResult:
    operators: tuple[str, ...]
    input_shape: tuple[int | None, ...]
    output_shape: tuple[int | None, ...]


# Candidate policy from local read-only protobuf inspection, not a claim of
# runtime safety, publisher authentication, or extension parity.
WD_V3_LOCAL_CANDIDATE_POLICY = AuditPolicy(
    17,
    ("Add", "Cast", "Concat", "Conv", "Div", "Gather", "Gemm", "LayerNormalization",
     "MatMul", "Mul", "ReduceMean", "Reshape", "Shape", "Sigmoid", "Slice", "Softmax",
     "Split", "Sqrt", "Squeeze", "Sub", "Tanh", "Transpose", "Unsqueeze"),
    (None, 448, 448, 3), (None, 10861),
)


def _shape(value, onnx):
    typ = value.type
    if not typ.HasField("tensor_type") or typ.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("model signature requires float tensor")
    dims = typ.tensor_type.shape.dim
    shape = []
    for dimension in dims:
        if dimension.HasField("dim_value") and dimension.dim_value > 0:
            shape.append(dimension.dim_value)
        elif dimension.HasField("dim_param") and dimension.dim_param:
            shape.append(None)
        else:
            raise ValueError("model signature has unknown dimension")
    return tuple(shape)


def _tensor(tensor, onnx):
    if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data:
        raise ValueError("external tensor data is unsupported")


def _validate_policy(pin, policy, check_interrupt):
    if (not isinstance(pin, AssetPin) or not isinstance(policy, AuditPolicy)
            or type(policy.opset) is not int or policy.opset <= 0
            or not isinstance(policy.operators, tuple) or not policy.operators
            or any(not isinstance(op, str) or not op for op in policy.operators)
            or not callable(check_interrupt)):
        raise ValueError("invalid internal audit policy or pin")


def read_model_bytes(pin, *, check_interrupt=lambda: None):
    path, before = _validate_pin(pin, MAX_AUDIT_BYTES)
    check_interrupt()
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as source:
        if _signature(os.fstat(source.fileno())) != _signature(before):
            raise ValueError("model identity changed")
        data = source.read(pin.size + 1)
        if (len(data) != pin.size or hashlib.sha256(data).hexdigest() != pin.sha256
                or _signature(os.fstat(source.fileno())) != _signature(before)):
            raise ValueError("model digest or identity mismatch")
    check_interrupt()
    return data


def inspect_model(pin, policy, *, check_interrupt=lambda: None):
    """Validate exact verified bytes and a reviewed graph/signature policy."""
    _validate_policy(pin, policy, check_interrupt)
    data = read_model_bytes(pin, check_interrupt=check_interrupt)
    return inspect_model_bytes(data, pin, policy, check_interrupt=check_interrupt)


def inspect_model_bytes(data, pin, policy, *, check_interrupt=lambda: None):
    """Inspect immutable bytes, allowing the caller to pass those same bytes to a loader."""
    _validate_policy(pin, policy, check_interrupt)
    if (not isinstance(data, bytes) or not 0 < len(data) <= MAX_AUDIT_BYTES
            or len(data) != pin.size or hashlib.sha256(data).hexdigest() != pin.sha256):
        raise ValueError("model digest or identity mismatch")
    check_interrupt()
    try:
        import onnx
    except ImportError as exc:
        raise ValueError("offline ONNX structure inspection dependency unavailable") from exc
    try:
        graph_model = onnx.load_model_from_string(data)
    except Exception as exc:
        raise ValueError(f"invalid ONNX model: {exc}") from exc
    check_interrupt()
    if graph_model.functions or graph_model.training_info or not graph_model.HasField("graph"):
        raise ValueError("model functions or training graphs are unsupported")
    imports = [item.version for item in graph_model.opset_import if item.domain in ("", "ai.onnx")]
    if imports != [policy.opset]:
        raise ValueError("model default opset differs from policy")
    graph = graph_model.graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise ValueError("model signature must have one input and output")
    input_shape, output_shape = _shape(graph.input[0], onnx), _shape(graph.output[0], onnx)
    if input_shape != policy.input_shape or output_shape != policy.output_shape:
        raise ValueError("model signature shape differs from policy")
    if len(graph.node) > MAX_NODES or len(graph.initializer) > MAX_INITIALIZERS or graph.sparse_initializer:
        raise ValueError("model graph exceeds limits or contains sparse tensors")
    for tensor in graph.initializer:
        _tensor(tensor, onnx)
    found = set()
    attribute_count = 0
    for node in graph.node:
        check_interrupt()
        if node.domain not in ("", "ai.onnx") or node.op_type not in policy.operators:
            raise ValueError("model operator or domain is unsupported")
        found.add(node.op_type)
        attribute_count += len(node.attribute)
        if attribute_count > MAX_ATTRIBUTES:
            raise ValueError("model attributes exceed limit")
        for attr in node.attribute:
            if attr.HasField("g") or attr.graphs or attr.HasField("sparse_tensor") or attr.sparse_tensors:
                raise ValueError("graph or sparse attributes are unsupported")
            if attr.HasField("t"):
                _tensor(attr.t, onnx)
            for tensor in attr.tensors:
                _tensor(tensor, onnx)
    return AuditResult(tuple(sorted(found)), input_shape, output_shape)
