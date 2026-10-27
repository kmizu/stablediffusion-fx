"""Small ONNX structural fixtures; no ONNX Runtime sessions or inference."""
import hashlib
import pytest
onnx = pytest.importorskip("onnx")
from onnx import helper, TensorProto
from sdfx.wd_v3_assets import AssetPin
from sdfx.wd_v3_onnx_audit import AuditPolicy, inspect_model


def model(node=None, initializer=()):
    node = node or helper.make_node("Identity", ["image"], ["scores"])
    graph = helper.make_graph([node], "test", [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 2, 2, 3])],
                              [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 4])], initializer=list(initializer))
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def audit(tmp_path, graph, policy=None):
    path = tmp_path / "model.onnx"; data = graph.SerializeToString(); path.write_bytes(data)
    pin = AssetPin(path, len(data), hashlib.sha256(data).hexdigest())
    return inspect_model(pin, policy or AuditPolicy(17, ("Identity",), (1, 2, 2, 3), (1, 4)))


def test_declared_signature_and_operator_are_checked_without_inference(tmp_path):
    result = audit(tmp_path, model())
    assert result.operators == ("Identity",)
    assert result.input_shape == (1, 2, 2, 3) and result.output_shape == (1, 4)
    with pytest.raises(ValueError, match="(?i)(signature|shape)"):
        audit(tmp_path, model(), AuditPolicy(17, ("Identity",), (1, 3, 3, 3), (1, 4)))


def test_custom_domain_unknown_operator_and_function_are_rejected(tmp_path):
    for bad in (helper.make_node("Identity", ["image"], ["scores"], domain="custom"),
                helper.make_node("Add", ["image", "image"], ["scores"])):
        with pytest.raises(ValueError): audit(tmp_path, model(bad))
    graph = model()
    graph.functions.append(helper.make_function("custom", "func", ["x"], ["y"],
                             [helper.make_node("Identity", ["x"], ["y"])], [helper.make_opsetid("", 17)]))
    with pytest.raises(ValueError): audit(tmp_path, graph)


def test_external_data_and_subgraph_are_rejected(tmp_path):
    external = TensorProto(name="external", data_type=TensorProto.FLOAT, dims=[1], data_location=TensorProto.EXTERNAL)
    external.external_data.add(key="location", value="elsewhere.bin")
    with pytest.raises(ValueError, match="(?i)external"):
        audit(tmp_path, model(initializer=(external,)))
    nested = helper.make_graph([], "nested", [], [])
    node = helper.make_node("Identity", ["image"], ["scores"], nested=nested)
    with pytest.raises(ValueError, match="(?i)(graph|attribute)"):
        audit(tmp_path, model(node))


def test_wrong_digest_rejected_before_parser(tmp_path, monkeypatch):
    path = tmp_path / "model.onnx"; path.write_bytes(b"not a model")
    called = []
    monkeypatch.setattr(onnx, "load_model_from_string", lambda data: called.append(True))
    with pytest.raises(ValueError, match="(?i)(digest|identity)"):
        inspect_model(AssetPin(path, path.stat().st_size, "0" * 64), AuditPolicy(17, ("Identity",), (1, 2, 2, 3), (1, 4)))
    assert called == []
