import pytest
torch = pytest.importorskip("torch")

from sdfx.embedding_inspector import mix_vectors


def test_weighted_mixing_pads_vectors_and_removes_zero_rows():
    a = torch.tensor([[1., 2.], [3., 4.]])
    b = torch.tensor([[2., 4.]])
    actual = mix_vectors([(a, 1.), (b, -0.5)], concat=False)
    assert torch.equal(actual, torch.tensor([[3., 4.]]))
    assert torch.equal(a, torch.tensor([[1., 2.], [3., 4.]]))


def test_concat_and_combine_are_ordered_and_weighted():
    a = torch.tensor([[1., 2.], [3., 4.]])
    b = torch.tensor([[4., 6.]])
    actual = mix_vectors([(a, 0.5), (b, -1.)], concat=True, combine=True)
    assert torch.equal(actual, torch.tensor([[-2., -3.]]))


def test_incompatible_widths_are_explicit():
    with pytest.raises(ValueError, match='dimension'):
        mix_vectors([(torch.ones(1, 2), 1.), (torch.ones(1, 3), 1.)])


def test_dual_encoder_mixing_preserves_both_widths_and_weights():
    from sdfx.embedding_inspector import mix_encoders
    a = {"clip_l": torch.tensor([[1., 2.]]), "clip_g": torch.tensor([[3., 4., 5.]])}
    b = {"clip_l": torch.tensor([[2., 2.]]), "clip_g": torch.tensor([[1., 1., 1.]])}
    result = mix_encoders([(a, 1.), (b, -0.5)])
    assert torch.equal(result["clip_l"], torch.tensor([[0., 1.]]))
    assert torch.equal(result["clip_g"], torch.tensor([[2.5, 3.5, 4.5]]))


def test_dual_encoder_missing_pair_rejected():
    from sdfx.embedding_inspector import mix_encoders
    a = {"clip_l": torch.ones(1, 2), "clip_g": torch.ones(1, 3)}
    b = {"clip_l": torch.ones(1, 2)}
    with pytest.raises(ValueError, match="encoder"):
        mix_encoders([(a, 1.), (b, 1.)])


def test_dual_encoder_zero_rows_must_stay_aligned():
    from sdfx.embedding_inspector import mix_encoders
    paired = {"clip_l": torch.tensor([[1., 0.], [0., 0.]]),
              "clip_g": torch.tensor([[0., 0., 0.], [0., 1., 0.]])}
    mixed = mix_encoders([(paired, 1.)])
    assert torch.equal(mixed["clip_l"], paired["clip_l"])
    assert torch.equal(mixed["clip_g"], paired["clip_g"])


def test_eval_vector_and_item_modes_match_upstream_variables():
    from sdfx.embedding_inspector import evaluate_vectors
    source = torch.tensor([[1., 2.], [3., 4.]])
    assert torch.equal(evaluate_vectors(source, "v * 2", allow_custom_code=True), source * 2)
    assert torch.equal(evaluate_vectors(source, "=v + i + n", allow_custom_code=True),
                       torch.tensor([[1., 3.], [4., 6.]]))
    assert torch.equal(source, torch.tensor([[1., 2.], [3., 4.]]))


def test_eval_requires_explicit_custom_code_permission():
    from sdfx.embedding_inspector import evaluate_vectors
    with pytest.raises(PermissionError):
        evaluate_vectors(torch.ones(1, 2), "v * 2", allow_custom_code=False)


def test_save_sdxl_preserves_two_encoders_and_step(tmp_path):
    from safetensors import safe_open
    from sdfx.embedding_inspector import save_embedding
    path = tmp_path / "pair.safetensors"
    values = {"clip_l": torch.tensor([[1., 2.]]), "clip_g": torch.tensor([[3., 4., 5.]])}
    save_embedding(path, values, step=37)
    with safe_open(path, framework="pt", device="cpu") as f:
        assert set(f.keys()) == {"clip_l", "clip_g"}
        assert torch.equal(f.get_tensor("clip_g"), values["clip_g"])
        assert f.metadata()["step"] == "37"


def test_save_collision_preserves_bytes_until_overwrite_enabled(tmp_path):
    from sdfx.embedding_inspector import save_embedding
    path = tmp_path / "existing.safetensors"
    path.write_bytes(b"keep original")
    values = {"clip_l": torch.ones(1, 2)}
    with pytest.raises(FileExistsError):
        save_embedding(path, values)
    assert path.read_bytes() == b"keep original"
    save_embedding(path, values, overwrite=True)
    from safetensors.torch import load_file
    assert len(load_file(str(path))) == 1


def test_nearest_tokens_uses_cosine_and_stable_id_order():
    from sdfx.embedding_inspector import nearest_tokens
    matrix = torch.tensor([[0., 1.], [2., 0.], [1., 0.], [-1., 0.]])
    matches = nearest_tokens(torch.tensor([1., 0.]), matrix, limit=3)
    assert [row[0] for row in matches] == [1, 2, 0]
    assert matches[0][1] == pytest.approx(1.)


def test_nearest_tokens_rejects_zero_query_and_width_mismatch():
    from sdfx.embedding_inspector import nearest_tokens
    with pytest.raises(ValueError, match="zero"):
        nearest_tokens(torch.zeros(2), torch.ones(2, 2))
    with pytest.raises(ValueError, match="dimension"):
        nearest_tokens(torch.ones(3), torch.ones(2, 2))


def test_eval_occurs_before_zero_row_removal():
    from sdfx.embedding_inspector import mix_encoders, evaluate_vectors
    source = {"clip_l": torch.tensor([[0., 0.], [1., 2.]])}
    mixed = mix_encoders([(source, 1.)], remove_zeros=False)
    assert torch.equal(evaluate_vectors(mixed["clip_l"], "v + 1", allow_custom_code=True),
                       torch.tensor([[1., 1.], [2., 3.]]))


class TinyTokenizer:
    vocab = {"cat": 0, "dog": 1}
    def encode(self, text, add_special_tokens=False):
        return [self.vocab.get(t, 2) for t in text.split()]
    def convert_ids_to_tokens(self, value):
        return ["cat", "dog", "other"][value]
    def get_added_vocab(self):
        return {}
    def get_vocab(self):
        return dict(self.vocab)


class TinyPipe:
    tokenizer = TinyTokenizer()
    text_encoder = type("Encoder", (), {"get_input_embeddings": lambda self: type("Weights", (), {
        "weight": torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])})()})()


def test_worker_inspection_uses_loaded_model_vectors_and_tokenizer():
    from sdfx.embedding_inspector import run_inspector
    result = run_inspector(TinyPipe(), {"operation": "inspect", "text": "cat", "limit": 2})
    assert result["encoders"]["clip_l"]["token"] == "cat"
    assert result["encoders"]["clip_l"]["nearest"][0]["id"] == 0


def test_worker_eval_is_guarded_by_environment_not_request(monkeypatch, tmp_path):
    from sdfx.embedding_inspector import run_inspector
    monkeypatch.delenv("SDFX_ALLOW_CODE", raising=False)
    request = {"operation": "save", "rows": [{"name": "cat", "weight": 1}],
               "output": str(tmp_path / "cat.safetensors"), "eval": "v*2", "allow_custom_code": True}
    with pytest.raises(PermissionError):
        run_inspector(TinyPipe(), request)
    assert not (tmp_path / "cat.safetensors").exists()


def test_worker_entry_rejects_eval_even_if_request_claims_permission(monkeypatch, tmp_path):
    from sdfx.base import BaseEngine
    engine = BaseEngine()
    engine.model = "test"
    engine.pipe = TinyPipe()
    engine.dry_run = False
    import io, threading
    from sdfx.protocol import Emitter, JobContext
    monkeypatch.delenv("SDFX_ALLOW_CODE", raising=False)
    ctx = JobContext(Emitter(io.BytesIO()), "save", publication_lock=threading.Lock())
    with pytest.raises(Exception, match="Allow custom code"):
        engine.embedding_inspector({"operation": "save", "rows": [{"name": "cat", "weight": 1}],
                                    "output": str(tmp_path / "new.safetensors"), "eval": "v*2",
                                    "allow_custom_code": True}, ctx)


def test_worker_uses_named_safetensors_not_yet_loaded_into_pipeline(tmp_path):
    from safetensors.torch import save_file, load_file
    from sdfx.embedding_inspector import run_inspector
    path = tmp_path / "custom.safetensors"
    save_file({"custom": torch.tensor([[0.25, 0.75]])}, str(path))
    out = tmp_path / "mixed.safetensors"
    result = run_inspector(TinyPipe(), {"operation": "save", "rows": [{"name": "custom", "weight": 2}],
                                        "embedding_paths": [str(path)], "output": str(out)})
    assert result["encoders"]["clip_l"]["shape"] == [1, 2]
    assert torch.equal(load_file(str(out))["clip_l"], torch.tensor([[0.5, 1.5]]))


def test_save_all_presets_exports_base_and_each_named_variant(monkeypatch, tmp_path):
    from sdfx.embedding_inspector import run_inspector
    monkeypatch.setenv("SDFX_ALLOW_CODE", "1")
    output = tmp_path / "mixed.safetensors"
    pipe = TinyPipe()
    pipe.text_encoder = type("Encoder", (), {"get_input_embeddings": lambda self: type("Weights", (), {
        "weight": torch.tensor([[1., 1.], [0., 1.], [-1., 0.]])})()})()
    result = run_inspector(pipe, {"operation": "save", "rows": [{"name": "cat", "weight": 1}],
        "output": str(output), "eval": "", "batch_presets": True})
    assert len(result["paths"]) == 9
    assert output.exists() and (tmp_path / "mixed_Boost.safetensors").exists()


def test_saved_sd1_embedding_uses_diffusers_token_override(tmp_path):
    from safetensors.torch import load_file
    from diffusers.loaders import TextualInversionLoaderMixin
    from sdfx.embedding_inspector import save_embedding
    path = tmp_path / "newconcept.safetensors"
    save_embedding(path, {"clip_l": torch.tensor([[1., 2.]])})
    class Tokenizer:
        def get_vocab(self):
            return {}
    tokens, values = TextualInversionLoaderMixin._retrieve_tokens_and_embeddings(
        ["newconcept"], [load_file(str(path))], Tokenizer())
    assert tokens == ["newconcept"]
    assert torch.equal(values[0], torch.tensor([[1., 2.]]))


def test_sdxl_mixer_accepts_distinct_encoder_specific_ids(tmp_path):
    from safetensors.torch import load_file
    from sdfx.embedding_inspector import run_inspector
    pipe = TinyPipe()
    pipe.tokenizer_2 = TinyTokenizer()
    pipe.text_encoder_2 = type("Encoder", (), {"get_input_embeddings": lambda self: type("Weights", (), {
        "weight": torch.tensor([[10., 0., 0.], [0., 20., 0.], [0., 0., 30.]])})()})()
    path = tmp_path / "dual.safetensors"
    run_inspector(pipe, {"operation": "save", "output": str(path), "rows": [
        {"name": "token", "ids": {"clip_l": 0, "clip_g": 1}, "weight": 1}]})
    result = load_file(str(path))
    assert torch.equal(result["clip_l"], torch.tensor([[1., 0.]]))
    assert torch.equal(result["clip_g"], torch.tensor([[0., 20., 0.]]))


def test_sdxl_rejects_ambiguous_single_numeric_token_id(tmp_path):
    from sdfx.embedding_inspector import run_inspector
    pipe = TinyPipe(); pipe.tokenizer_2 = TinyTokenizer(); pipe.text_encoder_2 = pipe.text_encoder
    with pytest.raises(ValueError, match="encoder"):
        run_inspector(pipe, {"operation": "save", "output": str(tmp_path / "dual.safetensors"),
                             "rows": [{"name": "#1", "weight": 1}]})


def test_export_vector_text_does_not_overwrite_existing_file(tmp_path):
    from sdfx.embedding_inspector import run_inspector
    path = tmp_path / "vector.txt"
    run_inspector(TinyPipe(), {"operation": "export", "text": "cat", "output": str(path)})
    assert "clip_l" in path.read_text(encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        run_inspector(TinyPipe(), {"operation": "export", "text": "dog", "output": str(path)})
    assert path.read_bytes() == before


def test_server_admits_embedding_inspector_to_serial_worker_queue():
    from sdfx.server import QUEUED_TYPES
    assert "embedding_inspector" in QUEUED_TYPES


def test_duplicate_embedding_stem_only_rejected_if_selected(tmp_path):
    from safetensors.torch import save_file
    from sdfx.embedding_inspector import run_inspector
    a, b = tmp_path / "one", tmp_path / "two"
    a.mkdir(); b.mkdir()
    for folder in (a, b):
        save_file({"same": torch.ones(1, 2)}, str(folder / "same.safetensors"))
    paths = [str(a / "same.safetensors"), str(b / "same.safetensors")]
    result = run_inspector(TinyPipe(), {"operation": "inspect", "text": "cat", "embedding_paths": paths})
    assert result["encoders"]["clip_l"]["token"] == "cat"
    with pytest.raises(ValueError, match="duplicate"):
        run_inspector(TinyPipe(), {"operation": "inspect", "text": "same", "embedding_paths": paths})


def test_saved_sdxl_both_sides_parse_with_distinct_dimensions(tmp_path):
    from safetensors.torch import load_file
    from diffusers.loaders import TextualInversionLoaderMixin
    from sdfx.embedding_inspector import save_embedding
    path = tmp_path / "dual.safetensors"
    save_embedding(path, {"clip_l": torch.ones(2, 3), "clip_g": torch.full((2, 5), 2.)})
    tensors = load_file(str(path))
    class Tokenizer:
        def get_vocab(self):
            return {}
    for key, width in (("clip_l", 3), ("clip_g", 5)):
        tokens, values = TextualInversionLoaderMixin._retrieve_tokens_and_embeddings(
            ["dual"], [tensors[key]], Tokenizer())
        assert tokens == ["dual"] and values[0].shape == (2, width)


def test_serial_worker_protocol_routes_real_inspector_operation():
    import io, json, time
    from sdfx.base import BaseEngine
    from sdfx.protocol import Emitter
    from sdfx.server import Server
    engine = BaseEngine(); engine.model = "test"; engine.pipe = TinyPipe(); engine.dry_run = False
    stream = io.BytesIO(); server = Server(engine, Emitter(stream)); server.worker.start()
    try:
        server.handle_line(json.dumps({"id": "inspect", "type": "embedding_inspector",
                                       "params": {"operation": "inspect", "text": "cat"}}))
        deadline = time.monotonic() + 10
        while b'"id":"inspect"' not in stream.getvalue() and time.monotonic() < deadline:
            time.sleep(0.02)
        messages = [json.loads(line) for line in stream.getvalue().splitlines()]
        result = next(item for item in messages if item.get("id") == "inspect")
        assert result["type"] == "result"
        assert result["result"]["encoders"]["clip_l"]["token"] == "cat"
    finally:
        server.q.put(None); server.worker.join(timeout=5)


def test_batch_skips_zero_preset_but_keeps_other_successful_outputs(monkeypatch, tmp_path):
    from sdfx.embedding_inspector import run_inspector
    monkeypatch.setenv("SDFX_ALLOW_CODE", "1")
    result = run_inspector(TinyPipe(), {"operation": "save", "rows": [{"name": "cat"}],
        "output": str(tmp_path / "mixed.safetensors"), "batch_presets": True})
    assert (tmp_path / "mixed.safetensors").exists()
    assert not (tmp_path / "mixed_Crop_low.safetensors").exists()
    assert any("Crop_low" in text for text in result["warnings"])


def test_save_rejects_file_with_wrong_loaded_encoder_width(tmp_path):
    from safetensors.torch import save_file
    from sdfx.embedding_inspector import run_inspector
    source = tmp_path / "wrong.safetensors"
    save_file({"wrong": torch.ones(1, 3)}, str(source))
    output = tmp_path / "out.safetensors"
    with pytest.raises(ValueError, match="dimension"):
        run_inspector(TinyPipe(), {"operation": "save", "rows": [{"name": "wrong"}],
                                  "embedding_paths": [str(source)], "output": str(output)})
    assert not output.exists()


def test_eval_exposes_documented_whole_original_vector():
    from sdfx.embedding_inspector import evaluate_vectors
    source = torch.tensor([[1., 2.], [3., 4.]])
    result = evaluate_vectors(source, "=v + tot_vec[0, i]", allow_custom_code=True)
    assert torch.equal(result, torch.tensor([[2., 4.], [4., 6.]]))


def test_legacy_pt_save_keeps_training_step_in_loadable_state(tmp_path):
    from sdfx.embedding_inspector import save_embedding
    path = tmp_path / "concept.pt"
    save_embedding(path, {"clip_l": torch.tensor([[1., 2.]])}, step=37)
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["name"] == "concept" and state["step"] == 37
    assert torch.equal(state["string_to_param"]["*"], torch.tensor([[1., 2.]]))


def test_legacy_pt_sdxl_round_trip_preserves_encoders_and_step(tmp_path):
    from sdfx.embedding_inspector import save_embedding
    path = tmp_path / "dual.pt"
    original = {"clip_l": torch.ones(2, 3), "clip_g": torch.ones(2, 5)}
    save_embedding(path, original, step=73)
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["step"] == 73
    assert torch.equal(state["clip_l"], original["clip_l"])
    assert torch.equal(state["clip_g"], original["clip_g"])



def test_multivector_inspection_reports_nearest_tokens_for_each_vector(tmp_path):
    from safetensors.torch import save_file
    from sdfx.embedding_inspector import run_inspector
    path = tmp_path / "multi.safetensors"
    save_file({"multi": torch.tensor([[1., 0.], [0., 1.]])}, str(path))
    result = run_inspector(TinyPipe(), {"operation": "inspect", "text": "multi", "limit": 1,
                                       "embedding_paths": [str(path)]})
    rows = result["encoders"]["clip_l"]["per_vector"]
    assert [entry["nearest"][0]["id"] for entry in rows] == [0, 1]


def test_inspect_zero_vector_reports_zero_cosines(tmp_path):
    from safetensors.torch import save_file
    from sdfx.embedding_inspector import run_inspector
    path = tmp_path / "zero.safetensors"
    save_file({"zero": torch.zeros(1, 2)}, str(path))
    result = run_inspector(TinyPipe(), {"operation": "inspect", "text": "zero", "limit": 2,
                                       "embedding_paths": [str(path)]})
    assert all(v["score"] == 0 for v in result["encoders"]["clip_l"]["nearest"])


def test_inspect_file_reports_saved_training_step(tmp_path):
    from sdfx.embedding_inspector import run_inspector, save_embedding
    path = tmp_path / "trained.safetensors"
    save_embedding(path, {"clip_l": torch.ones(1, 2)}, step=123)
    result = run_inspector(TinyPipe(), {"operation": "inspect", "text": "trained", "limit": 1,
                                       "embedding_paths": [str(path)]})
    assert result["metadata"]["step"] == "123"
    assert result["metadata"]["path"] == str(path)


def test_eval_error_saves_original_vectors_with_visible_warning(monkeypatch, tmp_path):
    from safetensors.torch import load_file
    from sdfx.embedding_inspector import run_inspector
    monkeypatch.setenv("SDFX_ALLOW_CODE", "1")
    output = tmp_path / "mixed.safetensors"
    result = run_inspector(TinyPipe(), {"operation": "save", "rows": [{"name": "cat"}],
                                      "output": str(output), "eval": "1 / 0"})
    assert torch.equal(load_file(str(output))["clip_l"], torch.tensor([[1., 0.]]))
    assert any("eval" in warning.lower() for warning in result["warnings"])


def test_simultaneous_no_overwrite_saves_have_one_winner(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from sdfx.embedding_inspector import save_embedding
    target = tmp_path / "race.safetensors"
    def attempt(value):
        try:
            save_embedding(target, {"clip_l": torch.full((1, 2), float(value))})
            return "saved"
        except FileExistsError:
            return "exists"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, [1, 2]))
    assert sorted(results) == ["exists", "saved"]
    from safetensors.torch import load_file
    assert float(load_file(str(target))["clip_l"][0, 0]) in (1., 2.)


def test_minitokenizer_reports_truncation(tmp_path):
    from sdfx.embedding_inspector import run_inspector
    result = run_inspector(TinyPipe(), {"operation": "tokenize", "text": " ".join(["cat"] * 129)})
    assert len(result["encoders"]["clip_l"]) == 128
    assert result["warnings"] and "128" in result["warnings"][0]


def test_saved_legacy_sd1_state_is_accepted_by_real_diffusers_parser(tmp_path):
    from diffusers.loaders import TextualInversionLoaderMixin
    from sdfx.embedding_inspector import save_embedding
    from sdfx.engine import safe_torch_load
    path = tmp_path / "legacy.pt"
    save_embedding(path, {"clip_l": torch.tensor([[1., 2.]])}, step=15)
    state = safe_torch_load(str(path), "embedding")
    class Tokenizer:
        def get_vocab(self):
            return {}
    names, vectors = TextualInversionLoaderMixin._retrieve_tokens_and_embeddings(["legacy"], [state], Tokenizer())
    assert names == ["legacy"]
    assert torch.equal(vectors[0], torch.tensor([[1., 2.]]))


def test_zero_weight_row_is_ignored_before_token_or_file_lookup(tmp_path):
    from sdfx.embedding_inspector import run_inspector
    output = tmp_path / "mixed.safetensors"
    result = run_inspector(TinyPipe(), {"operation": "save", "output": str(output),
        "rows": [{"name": "#999999", "weight": 0}, {"name": "cat", "weight": 1}]})
    assert result["path"] == str(output)


def test_incompatible_embedding_is_skipped_with_warning_when_other_rows_valid(tmp_path):
    from safetensors.torch import save_file, load_file
    from sdfx.embedding_inspector import run_inspector
    path = tmp_path / "wrong.safetensors"
    save_file({"wrong": torch.ones(1, 3)}, str(path))
    output = tmp_path / "mixed.safetensors"
    result = run_inspector(TinyPipe(), {"operation": "save", "output": str(output),
        "embedding_paths": [str(path)], "rows": [{"name": "wrong"}, {"name": "cat"}]})
    assert torch.equal(load_file(str(output))["clip_l"], torch.tensor([[1., 0.]]))
    assert any("wrong" in warning and "dimension" in warning for warning in result["warnings"])


def test_inspect_embedding_name_ignores_outer_whitespace(tmp_path):
    from safetensors.torch import save_file
    from sdfx.embedding_inspector import run_inspector
    path = tmp_path / "custom.safetensors"
    save_file({"custom": torch.tensor([[0., 1.]])}, str(path))
    result = run_inspector(TinyPipe(), {"operation": "inspect", "text": " custom ",
                                       "embedding_paths": [str(path)]})
    assert result["encoders"]["clip_l"]["token"] == "custom"
    assert result["encoders"]["clip_l"]["vectors"] == [[0., 1.]]


def test_ids_only_mixer_row_is_not_treated_as_blank(tmp_path):
    from safetensors.torch import load_file
    from sdfx.embedding_inspector import run_inspector
    path = tmp_path / "id.safetensors"
    run_inspector(TinyPipe(), {"operation": "save", "output": str(path),
                              "rows": [{"ids": {"clip_l": 1}, "weight": 1}]})
    assert torch.equal(load_file(str(path))["clip_l"], torch.tensor([[0., 1.]]))


class RefreshTokenizer:
    def __init__(self, tokens):
        self.tokens = dict(tokens)
    def get_added_vocab(self):
        return dict(self.tokens)
    def get_vocab(self):
        return {"base": 0, **self.tokens}
    def encode(self, text, add_special_tokens=False):
        return [self.get_vocab()[text]]


class RefreshEncoder:
    def __init__(self, width, rows):
        self.embedding = torch.nn.Embedding(rows, width)
        self.config = type("Config", (), {})()
        self.config.vocab_size = rows
        self.vocab_size = rows
    def get_input_embeddings(self):
        return self.embedding
    def set_input_embeddings(self, value):
        self.embedding = value


class RefreshPipe:
    def __init__(self, fail_second=False):
        self.tokenizer = RefreshTokenizer({"old": 1})
        self.tokenizer_2 = RefreshTokenizer({"old": 1})
        self.text_encoder = RefreshEncoder(2, 2)
        self.text_encoder_2 = RefreshEncoder(3, 2)
        self.fail_second = fail_second
        self.derived = {}
    def unload_textual_inversion(self, tokens, tokenizer, text_encoder):
        for name in tokens:
            tokenizer.tokens.pop(name, None)
        text_encoder.embedding.weight.data.fill_(9)
    def load_textual_inversion(self, value, token, tokenizer, text_encoder):
        tokenizer.tokens[token] = 1
        text_encoder.vocab_size = 999
        text_encoder.embedding.weight.data[1] = value[0]
        if self.fail_second and tokenizer is self.tokenizer_2:
            raise RuntimeError("second encoder failed")


def test_refresh_updates_both_encoder_tokenizers_and_engine_tracking(tmp_path):
    from sdfx.embedding_inspector import save_embedding
    from sdfx.embedding_refresh import refresh_saved_embeddings
    from types import SimpleNamespace
    path = tmp_path / "old.safetensors"
    save_embedding(path, {"clip_l": torch.tensor([[1., 2.]]), "clip_g": torch.tensor([[3., 4., 5.]])})
    pipe = RefreshPipe()
    engine = SimpleNamespace(pipe=pipe, sdxl=True, loaded_embeddings={"old"}, loaded_embedding_files={"old": str(path)},
                             embedding_snapshots={}, derived={}, embedding_token_owners={"old": {"clip_l": ["old"], "clip_g": ["old"]}})
    refresh_saved_embeddings(engine, [path])
    assert torch.equal(pipe.text_encoder.embedding.weight[1], torch.tensor([1., 2.]))
    assert torch.equal(pipe.text_encoder_2.embedding.weight[1], torch.tensor([3., 4., 5.]))
    assert engine.loaded_embedding_files["old"] == str(path)


def test_refresh_failure_restores_both_encoder_states(tmp_path):
    from sdfx.embedding_inspector import save_embedding
    from sdfx.embedding_refresh import refresh_saved_embeddings
    from types import SimpleNamespace
    path = tmp_path / "old.safetensors"
    save_embedding(path, {"clip_l": torch.tensor([[1., 2.]]), "clip_g": torch.tensor([[3., 4., 5.]])})
    pipe = RefreshPipe(fail_second=True)
    before = [p.embedding.weight.detach().clone() for p in (pipe.text_encoder, pipe.text_encoder_2)]
    engine = SimpleNamespace(pipe=pipe, sdxl=True, loaded_embeddings={"old"}, loaded_embedding_files={"old": str(path)},
                             embedding_snapshots={}, derived={}, embedding_token_owners={"old": {"clip_l": ["old"], "clip_g": ["old"]}})
    with pytest.raises(RuntimeError, match="second encoder"):
        refresh_saved_embeddings(engine, [path])
    assert pipe.tokenizer.tokens == {"old": 1} and pipe.tokenizer_2.tokens == {"old": 1}
    assert torch.equal(pipe.text_encoder.embedding.weight, before[0])
    assert torch.equal(pipe.text_encoder_2.embedding.weight, before[1])
    assert pipe.text_encoder.vocab_size == 2 and pipe.text_encoder_2.vocab_size == 2


def test_inspector_save_refreshes_live_token_and_claims_publication(tmp_path):
    import io, threading
    from sdfx.base import BaseEngine
    from sdfx.protocol import Emitter, JobContext
    engine = BaseEngine(); engine.model = "test"; engine.dry_run = False; engine.sdxl = True
    engine.pipe = RefreshPipe(); engine.loaded_embeddings = {"old"}
    engine.loaded_embedding_files = {}; engine.embedding_snapshots = {}; engine.derived = {}; engine.embedding_token_owners = {"old": {"clip_l": ["old"], "clip_g": ["old"]}}
    path = tmp_path / "old.safetensors"
    ctx = JobContext(Emitter(io.BytesIO()), "save", publication_lock=threading.Lock())
    result = engine.embedding_inspector({"operation": "save", "rows": [{"name": "base"}], "output": str(path)}, ctx)
    assert result["path"] == str(path) and ctx.publication_state == "committed"
    assert engine.loaded_embedding_files["old"] == str(path)


def test_interrupted_before_inspector_save_never_publishes(tmp_path):
    import io, threading
    from sdfx.base import BaseEngine
    from sdfx.protocol import Emitter, JobContext, EngineError
    engine = BaseEngine(); engine.model = "test"; engine.dry_run = False; engine.sdxl = True
    engine.pipe = RefreshPipe(); engine.loaded_embeddings = set(); engine.loaded_embedding_files = {}; engine.derived = {}
    path = tmp_path / "new.safetensors"
    interrupted = threading.Event(); interrupted.set()
    ctx = JobContext(Emitter(io.BytesIO()), "save", interrupted, publication_lock=threading.Lock())
    with pytest.raises(EngineError, match="interrupted"):
        engine.embedding_inspector({"operation": "save", "rows": [{"name": "base"}], "output": str(path)}, ctx)
    assert not path.exists()


def test_live_refresh_failure_preserves_model_and_blocks_generation(tmp_path):
    import io, threading
    from sdfx.base import BaseEngine
    from sdfx.engine import Engine
    from sdfx.protocol import Emitter, JobContext, EngineError
    engine = BaseEngine(); engine.model = "test"; engine.dry_run = False; engine.sdxl = True
    engine.pipe = RefreshPipe(fail_second=True); engine.loaded_embeddings = {"old"}
    engine.loaded_embedding_files = {}; engine.embedding_snapshots = {}; engine.derived = {}; engine.embedding_token_owners = {"old": {"clip_l": ["old"], "clip_g": ["old"]}}
    before = [x.embedding.weight.detach().clone() for x in (engine.pipe.text_encoder, engine.pipe.text_encoder_2)]
    path = tmp_path / "old.safetensors"
    ctx = JobContext(Emitter(io.BytesIO()), "save", publication_lock=threading.Lock())
    with pytest.raises(EngineError, match="saved but live refresh failed"):
        engine.embedding_inspector({"operation": "save", "rows": [{"name": "base"}], "output": str(path)}, ctx)
    assert path.exists() and ctx.publication_state == "open"
    assert torch.equal(engine.pipe.text_encoder.embedding.weight, before[0])
    assert torch.equal(engine.pipe.text_encoder_2.embedding.weight, before[1])
    assert engine._embedding_reload_error
    with pytest.raises(EngineError, match="reload the checkpoint"):
        Engine._generate_recorded(engine, {}, ctx, "txt2img")


def test_partial_batch_save_failure_blocks_generation_until_reload(monkeypatch, tmp_path):
    import io, threading
    from sdfx.base import BaseEngine
    from sdfx.engine import Engine
    from sdfx.protocol import Emitter, JobContext, EngineError
    import sdfx.embedding_inspector as inspector
    engine = BaseEngine(); engine.model = "test"; engine.dry_run = False; engine.pipe = TinyPipe()
    path = tmp_path / "partial.safetensors"
    def partial(pipe, params):
        path.write_bytes(b"partially published")
        raise OSError("disk failure")
    monkeypatch.setattr(inspector, "run_inspector", partial)
    ctx = JobContext(Emitter(io.BytesIO()), "save", publication_lock=threading.Lock())
    with pytest.raises(EngineError, match="may have been saved"):
        engine.embedding_inspector({"operation": "save", "output": str(path)}, ctx)
    assert path.exists() and engine._embedding_reload_error
    with pytest.raises(EngineError, match="reload the checkpoint"):
        Engine._generate_recorded(engine, {}, ctx, "txt2img")


def test_real_diffusers_refresh_updates_multivector_without_changing_other_tokens(tmp_path):
    from types import SimpleNamespace
    import json
    from transformers import CLIPTokenizer, CLIPTextConfig, CLIPTextModel
    from diffusers.loaders import TextualInversionLoaderMixin
    from sdfx.embedding_inspector import save_embedding
    from sdfx.embedding_refresh import refresh_saved_embeddings
    vocab = tmp_path / "vocab.json"; merges = tmp_path / "merges.txt"
    vocab.write_text(json.dumps({"base</w>": 0, "<|startoftext|>": 1, "<|endoftext|>": 2}), encoding="utf-8")
    merges.write_text("#version: 0.2\n", encoding="utf-8")
    tokenizer = CLIPTokenizer(str(vocab), str(merges))
    encoder = CLIPTextModel(CLIPTextConfig(vocab_size=3, hidden_size=4, intermediate_size=8,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8))
    class Pipe(TextualInversionLoaderMixin):
        hf_device_map = None
        components = {}
    pipe = Pipe(); pipe.tokenizer = tokenizer; pipe.text_encoder = encoder
    pipe.tokenizer_2 = None; pipe.text_encoder_2 = None
    pipe.load_textual_inversion(torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]]), token="old")
    pipe.load_textual_inversion(torch.tensor([[0.5, 0.5, 0.5, 0.5]]), token="other")
    path = tmp_path / "old.safetensors"
    desired = torch.tensor([[7., 8., 9., 10.], [10., 9., 8., 7.], [1., 3., 5., 7.]])
    save_embedding(path, {"clip_l": desired})
    engine = SimpleNamespace(pipe=pipe, sdxl=False, loaded_embeddings={"old", "other"},
        loaded_embedding_files={"old": str(path)}, embedding_snapshots={}, derived={},
        embedding_token_owners={"old": {"clip_l": ["old", "old_1"]}})
    refresh_saved_embeddings(engine, [path])
    ids = [tokenizer.convert_tokens_to_ids(n) for n in ("old", "old_1", "old_2")]
    assert torch.equal(encoder.get_input_embeddings().weight[ids], desired)
    other_id = tokenizer.convert_tokens_to_ids("other")
    assert torch.equal(encoder.get_input_embeddings().weight[other_id], torch.full((4,), 0.5))


def test_real_diffusers_dual_refresh_failure_restores_tokenizers_and_vectors(tmp_path):
    import json
    from types import SimpleNamespace
    from transformers import CLIPTokenizer, CLIPTextConfig, CLIPTextModel
    from diffusers.loaders import TextualInversionLoaderMixin
    from sdfx.embedding_inspector import save_embedding
    from sdfx.embedding_refresh import refresh_saved_embeddings
    vocab = tmp_path / "vocab.json"; merges = tmp_path / "merges.txt"
    vocab.write_text(json.dumps({"base</w>": 0, "<|startoftext|>": 1, "<|endoftext|>": 2}), encoding="utf-8")
    merges.write_text("#version: 0.2\n", encoding="utf-8")
    def make_stream():
        tok = CLIPTokenizer(str(vocab), str(merges))
        encoder = CLIPTextModel(CLIPTextConfig(vocab_size=3, hidden_size=4, intermediate_size=8,
            num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8))
        return tok, encoder
    class Pipe(TextualInversionLoaderMixin):
        hf_device_map = None
        components = {}
        inject_failure = False
        def load_textual_inversion(self, *args, **kwargs):
            result = super().load_textual_inversion(*args, **kwargs)
            if self.inject_failure and kwargs.get("tokenizer") is self.tokenizer_2:
                raise RuntimeError("second encoder failure")
            return result
    pipe = Pipe(); pipe.tokenizer, pipe.text_encoder = make_stream(); pipe.tokenizer_2, pipe.text_encoder_2 = make_stream()
    for tok, encoder in ((pipe.tokenizer, pipe.text_encoder), (pipe.tokenizer_2, pipe.text_encoder_2)):
        pipe.load_textual_inversion(torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]]), token="old", tokenizer=tok, text_encoder=encoder)
        pipe.load_textual_inversion(torch.full((1, 4), 0.5), token="other", tokenizer=tok, text_encoder=encoder)
    before = [(dict(tok.get_added_vocab()), enc.get_input_embeddings().weight.detach().clone(), enc.config.vocab_size)
              for tok, enc in ((pipe.tokenizer, pipe.text_encoder), (pipe.tokenizer_2, pipe.text_encoder_2))]
    path = tmp_path / "old.safetensors"
    save_embedding(path, {"clip_l": torch.ones(3, 4), "clip_g": torch.ones(3, 4)})
    engine = SimpleNamespace(pipe=pipe, sdxl=True, loaded_embeddings={"old", "other"},
        loaded_embedding_files={"old": str(path)}, embedding_snapshots={}, derived={},
        embedding_token_owners={"old": {"clip_l": ["old", "old_1"], "clip_g": ["old", "old_1"]}})
    pipe.inject_failure = True
    with pytest.raises(RuntimeError, match="second encoder failure"):
        refresh_saved_embeddings(engine, [path])
    for (tok, enc), (vocab_before, weights_before, size_before) in zip(
            ((pipe.tokenizer, pipe.text_encoder), (pipe.tokenizer_2, pipe.text_encoder_2)), before):
        assert tok.get_added_vocab() == vocab_before
        assert torch.equal(enc.get_input_embeddings().weight, weights_before)
        assert enc.config.vocab_size == size_before and enc.vocab_size == size_before


def test_interrupt_during_save_publication_is_too_late_and_refresh_completes(monkeypatch, tmp_path):
    import io, json, threading, time
    from sdfx.base import BaseEngine
    from sdfx.protocol import Emitter
    from sdfx.server import Server
    import sdfx.embedding_inspector as inspector
    engine = BaseEngine(); engine.model = "test"; engine.dry_run = False; engine.sdxl = True
    engine.pipe = RefreshPipe(); engine.loaded_embeddings = {"old"}
    engine.loaded_embedding_files = {}; engine.embedding_snapshots = {}; engine.derived = {}; engine.embedding_token_owners = {"old": {"clip_l": ["old"], "clip_g": ["old"]}}
    path = tmp_path / "old.safetensors"
    entered, release = threading.Event(), threading.Event()
    def blocking_save(pipe, params):
        inspector.save_embedding(path, {"clip_l": torch.ones(1, 2), "clip_g": torch.ones(1, 3)})
        entered.set()
        assert release.wait(5)
        return {"path": str(path), "paths": [str(path)]}
    monkeypatch.setattr(inspector, "run_inspector", blocking_save)
    stream = io.BytesIO(); server = Server(engine, Emitter(stream)); server.worker.start()
    try:
        server.handle_line(json.dumps({"id": "save", "type": "embedding_inspector", "params": {"operation": "save", "output": str(path)}}))
        assert entered.wait(5)
        server.handle_line(json.dumps({"id": "interrupt", "type": "interrupt", "params": {"target_id": "save"}}))
        responses = [json.loads(line) for line in stream.getvalue().splitlines()]
        reply = next(row for row in responses if row.get("id") == "interrupt")
        assert reply["result"]["too_late"] is True
        release.set()
        deadline = time.monotonic() + 5
        while b'"id":"save"' not in stream.getvalue() and time.monotonic() < deadline:
            time.sleep(0.01)
        responses = [json.loads(line) for line in stream.getvalue().splitlines()]
        assert next(row for row in responses if row.get("id") == "save")["type"] == "result"
        assert engine.loaded_embedding_files["old"] == str(path)
    finally:
        release.set(); server.q.put(None); server.worker.join(timeout=5)


def test_save_rejects_duplicate_token_name_in_other_catalog_directory(tmp_path):
    from safetensors.torch import save_file
    from sdfx.embedding_inspector import run_inspector
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    existing = a / "same.safetensors"; target = b / "same.safetensors"
    save_file({"same": torch.ones(1, 2)}, str(existing))
    with pytest.raises(ValueError, match="duplicate"):
        run_inspector(TinyPipe(), {"operation": "save", "output": str(target),
            "embedding_paths": [str(existing)], "rows": [{"name": "cat"}]})
    assert not target.exists()


def test_save_rejects_builtin_token_name_before_publication(tmp_path):
    import io, threading
    from sdfx.base import BaseEngine
    from sdfx.protocol import Emitter, JobContext, EngineError
    engine = BaseEngine(); engine.model = "test"; engine.dry_run = False; engine.sdxl = True
    engine.pipe = RefreshPipe(); engine.loaded_embeddings = set(); engine.loaded_embedding_files = {}; engine.derived = {}
    output = tmp_path / "base.safetensors"
    ctx = JobContext(Emitter(io.BytesIO()), "save", publication_lock=threading.Lock())
    with pytest.raises(EngineError, match="built-in token"):
        engine.embedding_inspector({"operation": "save", "rows": [{"name": "base"}], "output": str(output)}, ctx)
    assert not output.exists() and not getattr(engine, "_embedding_reload_error", None)


def test_refresh_restoration_failure_invalidates_model(tmp_path):
    from types import SimpleNamespace
    from sdfx.embedding_inspector import save_embedding
    from sdfx.embedding_refresh import refresh_saved_embeddings
    path = tmp_path / "old.safetensors"
    save_embedding(path, {"clip_l": torch.ones(1, 2), "clip_g": torch.ones(1, 3)})
    pipe = RefreshPipe(fail_second=True)
    def cannot_restore(value):
        raise RuntimeError("restore failed")
    pipe.text_encoder.set_input_embeddings = cannot_restore
    called = []
    engine = SimpleNamespace(pipe=pipe, sdxl=True, loaded_embeddings={"old"}, loaded_embedding_files={},
        embedding_snapshots={}, derived={}, _free=lambda: called.append("unloaded"),
        embedding_token_owners={"old": {"clip_l": ["old"], "clip_g": ["old"]}})
    with pytest.raises(RuntimeError, match="checkpoint must be reloaded"):
        refresh_saved_embeddings(engine, [path])
    assert called == ["unloaded"]


def test_baseexception_during_refresh_marks_saved_file_inactive(monkeypatch, tmp_path):
    import io, threading
    from sdfx.base import BaseEngine
    from sdfx.protocol import Emitter, JobContext, EngineError
    import sdfx.embedding_refresh as refresh
    engine = BaseEngine(); engine.model = "test"; engine.dry_run = False; engine.sdxl = True
    engine.pipe = RefreshPipe(); engine.loaded_embeddings = {"old"}; engine.loaded_embedding_files = {}; engine.derived = {}
    engine.embedding_token_owners = {"old": {"clip_l": ["old"], "clip_g": ["old"]}}
    monkeypatch.setattr(refresh, "refresh_saved_embeddings", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("abort")))
    path = tmp_path / "old.safetensors"
    ctx = JobContext(Emitter(io.BytesIO()), "save", publication_lock=threading.Lock())
    with pytest.raises(EngineError, match="saved but live refresh failed"):
        engine.embedding_inspector({"operation": "save", "rows": [{"name": "base"}], "output": str(path)}, ctx)
    assert path.exists() and engine._embedding_reload_error


def test_real_diffusers_dual_new_and_overwrite_reloads_two_tokenizers(tmp_path):
    import json
    from types import SimpleNamespace
    from transformers import CLIPTokenizer, CLIPTextConfig, CLIPTextModel
    from diffusers.loaders import TextualInversionLoaderMixin
    from sdfx.embedding_inspector import save_embedding
    from sdfx.embedding_refresh import refresh_saved_embeddings
    vocab = tmp_path / "vocab.json"; merges = tmp_path / "merges.txt"
    vocab.write_text(json.dumps({"base</w>": 0, "<|startoftext|>": 1, "<|endoftext|>": 2}), encoding="utf-8")
    merges.write_text("#version: 0.2\n", encoding="utf-8")
    def stream():
        return CLIPTokenizer(str(vocab), str(merges)), CLIPTextModel(CLIPTextConfig(vocab_size=3,
            hidden_size=4, intermediate_size=8, num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8))
    class Pipe(TextualInversionLoaderMixin):
        hf_device_map = None
        components = {}
    pipe = Pipe(); pipe.tokenizer, pipe.text_encoder = stream(); pipe.tokenizer_2, pipe.text_encoder_2 = stream()
    engine = SimpleNamespace(pipe=pipe, sdxl=True, loaded_embeddings=set(), loaded_embedding_files={},
                             embedding_snapshots={}, derived={})
    path = tmp_path / "concept.safetensors"
    save_embedding(path, {"clip_l": torch.ones(1, 4), "clip_g": torch.full((1, 4), 2.)})
    refresh_saved_embeddings(engine, [path])
    save_embedding(path, {"clip_l": torch.full((2, 4), 3.), "clip_g": torch.full((2, 4), 4.)}, overwrite=True)
    refresh_saved_embeddings(engine, [path])
    for tok, enc, value in ((pipe.tokenizer, pipe.text_encoder, 3.), (pipe.tokenizer_2, pipe.text_encoder_2, 4.)):
        ids = [tok.convert_tokens_to_ids("concept"), tok.convert_tokens_to_ids("concept_1")]
        assert torch.equal(enc.get_input_embeddings().weight[ids], torch.full((2, 4), value))


def test_refresh_preflight_rejects_case_only_alias_for_loaded_token(tmp_path):
    from types import SimpleNamespace
    from sdfx.embedding_refresh import validate_saved_names
    pipe = RefreshPipe()
    engine = SimpleNamespace(pipe=pipe)
    with pytest.raises(ValueError, match="case"):
        validate_saved_names(engine, [tmp_path / "OLD.safetensors"])


def test_refresh_rejects_file_changed_during_live_update(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from sdfx.embedding_inspector import save_embedding
    from sdfx.embedding_refresh import refresh_saved_embeddings
    from sdfx.resource_hashes import FileSnapshot
    path = tmp_path / "old.safetensors"
    save_embedding(path, {"clip_l": torch.ones(1, 2), "clip_g": torch.ones(1, 3)})
    pipe = RefreshPipe()
    before = [x.embedding.weight.detach().clone() for x in (pipe.text_encoder, pipe.text_encoder_2)]
    original_identity = FileSnapshot.identity_of
    original_load = pipe.load_textual_inversion
    changed = {"value": False}
    def identity(value):
        original = original_identity(value)
        return (*original[:-1], original[-1] + int(changed["value"]))
    def changing(*args, **kwargs):
        result = original_load(*args, **kwargs)
        changed["value"] = True
        return result
    monkeypatch.setattr(FileSnapshot, "identity_of", staticmethod(identity))
    pipe.load_textual_inversion = changing
    engine = SimpleNamespace(pipe=pipe, sdxl=True, loaded_embeddings={"old"}, loaded_embedding_files={},
                             embedding_snapshots={}, derived={}, embedding_token_owners={"old": {"clip_l": ["old"], "clip_g": ["old"]}})
    with pytest.raises(ValueError, match="changed"):
        refresh_saved_embeddings(engine, [path])
    assert torch.equal(pipe.text_encoder.embedding.weight, before[0])
    assert torch.equal(pipe.text_encoder_2.embedding.weight, before[1])


def test_preflight_rejects_special_token_even_when_in_added_vocab(tmp_path):
    from types import SimpleNamespace
    from sdfx.embedding_refresh import validate_saved_names
    pipe = RefreshPipe()
    pipe.tokenizer.tokens["special"] = 2
    pipe.tokenizer_2.tokens["special"] = 2
    pipe.tokenizer.all_special_tokens = ["special"]
    pipe.tokenizer_2.all_special_tokens = ["special"]
    with pytest.raises(ValueError, match="built-in token"):
        validate_saved_names(SimpleNamespace(pipe=pipe), [tmp_path / "special.safetensors"])


def test_inspection_temporarily_suspends_and_restores_offload_hooks(monkeypatch):
    from sdfx.base import BaseEngine
    import sdfx.embedding_inspector as inspector
    calls = []
    engine = BaseEngine(); engine.model = "test"; engine.dry_run = False; engine.pipe = TinyPipe()
    engine._offload = True
    engine._drop_hooks = lambda: calls.append("drop")
    engine._place = lambda pipe: calls.append("restore")
    monkeypatch.setattr(inspector, "run_inspector", lambda pipe, params: calls.append("inspect") or {"ok": True})
    assert engine.embedding_inspector({"operation": "inspect"}, None) == {"ok": True}
    assert calls == ["drop", "inspect", "restore"]


def test_inspection_restores_offload_hooks_on_failure(monkeypatch):
    from sdfx.base import BaseEngine
    import sdfx.embedding_inspector as inspector
    calls = []
    engine = BaseEngine(); engine.model = "test"; engine.dry_run = False; engine.pipe = TinyPipe()
    engine._offload = True
    engine._drop_hooks = lambda: calls.append("drop")
    engine._place = lambda pipe: calls.append("restore")
    def fail(*args):
        calls.append("inspect")
        raise ValueError("bad embedding")
    monkeypatch.setattr(inspector, "run_inspector", fail)
    with pytest.raises(Exception, match="bad embedding"):
        engine.embedding_inspector({"operation": "inspect"}, None)
    assert calls == ["drop", "inspect", "restore"]


def test_real_accelerate_hook_restores_original_execution_device(monkeypatch):
    from types import SimpleNamespace
    from accelerate.hooks import CpuOffload, add_hook_to_module, remove_hook_from_module
    from sdfx.base import BaseEngine
    import sdfx.embedding_inspector as inspector
    encoder = torch.nn.Linear(2, 2)
    add_hook_to_module(encoder, CpuOffload(execution_device="cuda:0"))
    engine = BaseEngine(); engine.model = "test"; engine.dry_run = False
    engine.pipe = SimpleNamespace(encoder=encoder); engine._offload = True
    engine._drop_hooks = lambda: remove_hook_from_module(encoder)
    engine._place = lambda pipe: add_hook_to_module(encoder, CpuOffload(execution_device="cuda:0"))
    monkeypatch.setattr(inspector, "run_inspector", lambda pipe, params: {"ok": True})
    assert engine.embedding_inspector({"operation": "inspect"}, None) == {"ok": True}
    assert str(encoder._hf_hook.execution_device) == "cuda:0"


def test_offload_hook_restore_failure_unloads_and_reports_recovery(monkeypatch):
    from sdfx.base import BaseEngine
    from sdfx.protocol import EngineError
    import sdfx.embedding_inspector as inspector
    engine = BaseEngine(); engine.model = "test"; engine.dry_run = False; engine.pipe = TinyPipe()
    engine._offload = True
    engine._drop_hooks = lambda: None
    def fail(pipe):
        raise RuntimeError("hook failure")
    engine._place = fail
    unloaded = []
    engine._free = lambda: unloaded.append(True)
    monkeypatch.setattr(inspector, "run_inspector", lambda pipe, params: {"ok": True})
    with pytest.raises(EngineError, match="reload the checkpoint"):
        engine.embedding_inspector({"operation": "inspect"}, None)
    assert unloaded and engine._embedding_reload_error == "hook failure"


def test_offload_encoder_meta_weights_are_rejected_and_hooks_restored(monkeypatch):
    from types import SimpleNamespace
    from sdfx.base import BaseEngine
    import sdfx.embedding_inspector as inspector
    calls = []
    meta_encoder = SimpleNamespace(get_input_embeddings=lambda: SimpleNamespace(weight=torch.empty((2, 2), device="meta")))
    engine = BaseEngine(); engine.model = "test"; engine.dry_run = False
    engine.pipe = SimpleNamespace(text_encoder=meta_encoder, text_encoder_2=None)
    engine._offload = True; engine._drop_hooks = lambda: calls.append("drop"); engine._place = lambda pipe: calls.append("restore")
    monkeypatch.setattr(inspector, "run_inspector", lambda *_: calls.append("inspect"))
    with pytest.raises(Exception, match="materializ"):
        engine.embedding_inspector({"operation": "inspect"}, None)
    assert calls == ["drop", "restore"]


def test_actual_diffusers_save_keeps_accelerate_execution_device(monkeypatch, tmp_path):
    import io, json, threading
    from accelerate.hooks import CpuOffload, add_hook_to_module, remove_hook_from_module
    from transformers import CLIPTokenizer, CLIPTextConfig, CLIPTextModel
    from diffusers.loaders import TextualInversionLoaderMixin
    from sdfx.base import BaseEngine
    from sdfx.protocol import Emitter, JobContext
    vocab = tmp_path / "vocab.json"; merges = tmp_path / "merges.txt"
    vocab.write_text(json.dumps({"base</w>": 0, "<|startoftext|>": 1, "<|endoftext|>": 2}), encoding="utf-8")
    merges.write_text("#version: 0.2\n", encoding="utf-8")
    tok = CLIPTokenizer(str(vocab), str(merges))
    encoder = CLIPTextModel(CLIPTextConfig(vocab_size=3, hidden_size=4, intermediate_size=8,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8))
    class Pipe(TextualInversionLoaderMixin):
        hf_device_map = None
        components = {"text_encoder": encoder}
        tokenizer = tok
        text_encoder = encoder
        tokenizer_2 = None
        text_encoder_2 = None
    pipe = Pipe()
    add_hook_to_module(encoder, CpuOffload(execution_device="cuda:0"))
    engine = BaseEngine(); engine.model = "test"; engine.dry_run = False; engine.pipe = pipe; engine.sdxl = False
    engine._offload = True; engine.loaded_embeddings = set(); engine.loaded_embedding_files = {}; engine.embedding_snapshots = {}; engine.derived = {}; engine.embedding_token_owners = {"old": {"clip_l": ["old"], "clip_g": ["old"]}}
    engine._drop_hooks = lambda: remove_hook_from_module(encoder)
    engine._place = lambda pipe: add_hook_to_module(encoder, CpuOffload(execution_device="cuda:0"))
    path = tmp_path / "new.safetensors"
    ctx = JobContext(Emitter(io.BytesIO()), "save", publication_lock=threading.Lock())
    result = engine.embedding_inspector({"operation": "save", "output": str(path), "rows": [{"name": "#0"}]}, ctx)
    assert result["reloaded"] and "new" in tok.get_added_vocab()
    assert str(encoder._hf_hook.execution_device) == "cuda:0"


def test_interrupted_inspection_returns_interrupted_after_restoring_hooks(monkeypatch):
    import io, threading
    from sdfx.base import BaseEngine
    from sdfx.protocol import Emitter, JobContext, EngineError
    import sdfx.embedding_inspector as inspector
    calls = []
    event = threading.Event()
    engine = BaseEngine(); engine.model = "test"; engine.dry_run = False; engine.pipe = TinyPipe()
    engine._offload = True; engine._drop_hooks = lambda: calls.append("drop"); engine._place = lambda _: calls.append("restore")
    def inspect(*_):
        event.set()
        return {"ok": True}
    monkeypatch.setattr(inspector, "run_inspector", inspect)
    ctx = JobContext(Emitter(io.BytesIO()), "inspect", event, publication_lock=threading.Lock())
    with pytest.raises(EngineError, match="interrupted"):
        engine.embedding_inspector({"operation": "inspect"}, ctx)
    assert calls == ["drop", "restore"]


def test_vector_export_claims_publication_before_writing(monkeypatch, tmp_path):
    import io, threading
    from sdfx.base import BaseEngine
    from sdfx.protocol import Emitter, JobContext
    import sdfx.embedding_inspector as inspector
    path = tmp_path / "vec.txt"
    engine = BaseEngine(); engine.model = "test"; engine.dry_run = False; engine.pipe = TinyPipe()
    event = threading.Event()
    ctx = JobContext(Emitter(io.BytesIO()), "export", event, publication_lock=threading.Lock())
    def export(*_):
        assert ctx.publication_state == "committing"
        path.write_text("vector", encoding="utf-8")
        event.set()
        return {"path": str(path)}
    monkeypatch.setattr(inspector, "run_inspector", export)
    result = engine.embedding_inspector({"operation": "export"}, ctx)
    assert result["path"] == str(path) and ctx.publication_state == "committed"

@pytest.mark.parametrize("base_loaded", [False, True])
def test_refresh_does_not_unload_independent_suffixed_token(tmp_path, base_loaded):
    import json
    from types import SimpleNamespace
    from transformers import CLIPTokenizer, CLIPTextConfig, CLIPTextModel
    from diffusers.loaders import TextualInversionLoaderMixin
    from sdfx.embedding_inspector import save_embedding
    from sdfx.embedding_refresh import refresh_saved_embeddings
    vocab = tmp_path / 'vocab.json'; merges = tmp_path / 'merges.txt'
    vocab.write_text(json.dumps({'base</w>': 0, '<|startoftext|>': 1, '<|endoftext|>': 2}), encoding='utf-8')
    merges.write_text('#version: 0.2\n', encoding='utf-8')
    tokenizer = CLIPTokenizer(str(vocab), str(merges))
    encoder = CLIPTextModel(CLIPTextConfig(vocab_size=3, hidden_size=4, intermediate_size=8,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8))
    class Pipe(TextualInversionLoaderMixin):
        hf_device_map = None
        components = {}
    pipe = Pipe(); pipe.tokenizer = tokenizer; pipe.text_encoder = encoder
    pipe.tokenizer_2 = None; pipe.text_encoder_2 = None
    if base_loaded:
        pipe.load_textual_inversion(torch.ones(1, 4), token='cat')
    pipe.load_textual_inversion(torch.full((1, 4), 0.5), token='cat_1')
    path = tmp_path / 'cat.safetensors'; save_embedding(path, {'clip_l': torch.full((1, 4), 2.)})
    ownership = {'cat': {'clip_l': ['cat']}} if base_loaded else {}
    engine = SimpleNamespace(pipe=pipe, sdxl=False, loaded_embeddings={'cat_1'} | ({'cat'} if base_loaded else set()),
        loaded_embedding_files={}, embedding_snapshots={}, embedding_token_owners=ownership, derived={})
    with pytest.raises(ValueError, match="embedding token collision: cat_1"):
        refresh_saved_embeddings(engine, [path])
    assert 'cat_1' in tokenizer.get_added_vocab()
    assert torch.equal(encoder.get_input_embeddings().weight[tokenizer.convert_tokens_to_ids('cat_1')], torch.full((4,), 0.5))


def test_preflight_rejects_unproven_ownership_and_preserves_existing_token(tmp_path):
    from types import SimpleNamespace
    from sdfx.embedding_refresh import validate_saved_names
    pipe = RefreshPipe()
    with pytest.raises(ValueError, match="ownership is unknown"):
        validate_saved_names(SimpleNamespace(pipe=pipe), [tmp_path / "old.safetensors"])
    assert pipe.tokenizer.tokens == {"old": 1}

def test_engine_records_actual_diffusers_multivector_ownership(tmp_path):
    import json
    from types import SimpleNamespace
    from transformers import CLIPTokenizer, CLIPTextConfig, CLIPTextModel
    from diffusers.loaders import TextualInversionLoaderMixin
    from safetensors.torch import save_file
    from sdfx.engine import Engine
    vocab = tmp_path / 'vocab.json'; merges = tmp_path / 'merges.txt'
    vocab.write_text(json.dumps({'base</w>': 0, '<|startoftext|>': 1, '<|endoftext|>': 2}), encoding='utf-8')
    merges.write_text('#version: 0.2\n', encoding='utf-8')
    class Pipe(TextualInversionLoaderMixin):
        hf_device_map = None
        components = {}
    pipe = Pipe(); pipe.tokenizer = CLIPTokenizer(str(vocab), str(merges))
    pipe.text_encoder = CLIPTextModel(CLIPTextConfig(vocab_size=3, hidden_size=4, intermediate_size=8,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8))
    path = tmp_path / 'concept.safetensors'; save_file({'emb_params': torch.ones(2, 4)}, str(path))
    engine = SimpleNamespace(pipe=pipe, sdxl=False, loaded_embeddings=set(), loaded_embedding_files={},
                             embedding_token_owners={}, embedding_snapshots={})
    warnings = []
    Engine._load_embeddings(engine, SimpleNamespace(embedding_paths=[str(path)]), ['concept'], SimpleNamespace(warn=warnings.append))
    assert not warnings
    assert engine.embedding_token_owners == {'concept': {'clip_l': ['concept', 'concept_1']}}

def test_ordinary_embedding_load_preserves_accelerate_execution_device(tmp_path):
    import json
    from types import SimpleNamespace
    from accelerate.hooks import CpuOffload, add_hook_to_module, remove_hook_from_module
    from transformers import CLIPTokenizer, CLIPTextConfig, CLIPTextModel
    from diffusers.loaders import TextualInversionLoaderMixin
    from safetensors.torch import save_file
    from sdfx.engine import Engine
    vocab = tmp_path / 'vocab.json'; merges = tmp_path / 'merges.txt'
    vocab.write_text(json.dumps({'base</w>': 0, '<|startoftext|>': 1, '<|endoftext|>': 2}), encoding='utf-8')
    merges.write_text('#version: 0.2\n', encoding='utf-8')
    tokenizer = CLIPTokenizer(str(vocab), str(merges))
    encoder = CLIPTextModel(CLIPTextConfig(vocab_size=3, hidden_size=4, intermediate_size=8,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8))
    class Pipe(TextualInversionLoaderMixin):
        hf_device_map = None
        components = {'text_encoder': encoder}
        text_encoder = encoder
        tokenizer_2 = None
        text_encoder_2 = None
        def enable_model_cpu_offload(self, device=None):
            add_hook_to_module(encoder, CpuOffload(execution_device=device))
    pipe = Pipe(); pipe.tokenizer = tokenizer
    add_hook_to_module(encoder, CpuOffload(execution_device='cuda:0'))
    path = tmp_path / 'concept.safetensors'; save_file({'emb_params': torch.ones(1, 4)}, str(path))
    engine = SimpleNamespace(pipe=pipe, sdxl=False, loaded_embeddings=set(), loaded_embedding_files={},
                             embedding_token_owners={}, embedding_snapshots={}, _offload=True)
    engine._drop_hooks = lambda: remove_hook_from_module(encoder)
    engine._place = lambda _: add_hook_to_module(encoder, CpuOffload(execution_device='cuda:0'))
    warnings = []
    Engine._load_embeddings(engine, SimpleNamespace(embedding_paths=[str(path)]), ['concept'], SimpleNamespace(warn=warnings.append))
    assert not warnings
    assert str(encoder._hf_hook.execution_device) == 'cuda:0'

def test_ordinary_load_offload_restore_failure_aborts_generation(tmp_path):
    from types import SimpleNamespace
    from safetensors.torch import save_file
    from sdfx.engine import Engine
    from sdfx.protocol import EngineError
    path = tmp_path / 'new.safetensors'; save_file({'emb_params': torch.ones(1, 2)}, str(path))
    pipe = RefreshPipe()
    pipe.load_textual_inversion = lambda *args, **kwargs: None
    unloaded = []
    def fail_placement(_):
        raise RuntimeError('cannot restore offload')
    engine = SimpleNamespace(pipe=pipe, sdxl=False, loaded_embeddings=set(), loaded_embedding_files={},
        embedding_token_owners={}, embedding_snapshots={}, _offload=True, _drop_hooks=lambda: None,
        _place=fail_placement, _free=lambda: unloaded.append(True))
    with pytest.raises(EngineError, match='reload the checkpoint'):
        Engine._load_embeddings(engine, SimpleNamespace(embedding_paths=[str(path)]), ['new'], SimpleNamespace(warn=lambda _: None))
    assert unloaded and 'new' not in engine.loaded_embedding_files
    assert 'cannot restore offload' in engine._embedding_reload_error

def test_ordinary_dual_embedding_partial_load_failure_invalidates_model(tmp_path):
    import json
    from types import SimpleNamespace
    from transformers import CLIPTokenizer, CLIPTextConfig, CLIPTextModel
    from diffusers.loaders import TextualInversionLoaderMixin
    from safetensors.torch import save_file
    from sdfx.engine import Engine
    from sdfx.protocol import EngineError
    vocab = tmp_path / 'vocab.json'; merges = tmp_path / 'merges.txt'
    vocab.write_text(json.dumps({'base</w>': 0, '<|startoftext|>': 1, '<|endoftext|>': 2}), encoding='utf-8')
    merges.write_text('#version: 0.2\n', encoding='utf-8')
    def stream():
        return CLIPTokenizer(str(vocab), str(merges)), CLIPTextModel(CLIPTextConfig(vocab_size=3, hidden_size=4,
            intermediate_size=8, num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8))
    class Pipe(TextualInversionLoaderMixin):
        hf_device_map = None
        components = {}
        def load_textual_inversion(self, *args, **kwargs):
            if kwargs.get('tokenizer') is self.tokenizer:
                raise RuntimeError('injected second encoder failure')
            return super().load_textual_inversion(*args, **kwargs)
    pipe = Pipe(); pipe.tokenizer, pipe.text_encoder = stream(); pipe.tokenizer_2, pipe.text_encoder_2 = stream()
    path = tmp_path / 'concept.safetensors'
    save_file({'clip_l': torch.ones(1, 4), 'clip_g': torch.ones(1, 4)}, str(path))
    engine = SimpleNamespace(pipe=pipe, sdxl=True, loaded_embeddings=set(), loaded_embedding_files={},
                             embedding_token_owners={}, embedding_snapshots={})
    engine._free = lambda: setattr(engine, 'pipe', None)
    with pytest.raises(EngineError, match='reload the checkpoint'):
        Engine._load_embeddings(engine, SimpleNamespace(embedding_paths=[str(path)]), ['concept'], SimpleNamespace(warn=lambda _: None))
    assert 'concept' in pipe.tokenizer_2.get_added_vocab() # first encoder really mutated
    assert engine.pipe is None and engine._embedding_reload_error
    assert not engine.loaded_embedding_files and not engine.embedding_token_owners

def test_list_distinguishes_loaded_from_catalog_and_shows_loaded_shape(tmp_path):
    from sdfx.embedding_inspector import run_inspector, save_embedding
    loaded = tmp_path / 'loaded.safetensors'; available = tmp_path / 'available.safetensors'
    save_embedding(loaded, {'clip_l': torch.ones(1, 2)}, step=42)
    save_embedding(available, {'clip_l': torch.ones(1, 2)})
    result = run_inspector(TinyPipe(), {'operation': 'list', 'embedding_paths': [str(loaded), str(available)],
        '_list_loaded': {'loaded': {'path': str(loaded), 'owners': {'clip_l': ['loaded']}, 'metadata_valid': True}}})
    assert [item['name'] for item in result['loaded']] == ['loaded']
    assert result['loaded'][0]['encoders']['clip_l'] == [1, 2]
    assert result['loaded'][0]['file_metadata']['step'] == '42'
    assert result['available'] == ['available']


def test_inspect_reports_vector_count_and_width():
    from sdfx.embedding_inspector import run_inspector
    data = run_inspector(TinyPipe(), {'operation': 'inspect', 'text': 'cat', 'limit': 2})
    assert data['encoders']['clip_l']['shape'] == [1, 2]

def test_list_flags_loaded_file_changed_since_loading(tmp_path):
    from sdfx.base import BaseEngine
    from sdfx.embedding_inspector import save_embedding
    from sdfx.resource_hashes import FileSnapshot
    path = tmp_path / 'loaded.safetensors'
    save_embedding(path, {'clip_l': torch.ones(1, 2)}, step=42)
    engine = BaseEngine(); engine.model = 'test'; engine.dry_run = False; engine.pipe = TinyPipe()
    engine.loaded_embedding_files = {'loaded': str(path)}
    engine.embedding_token_owners = {'loaded': {'clip_l': ['loaded']}}
    engine.embedding_snapshots = {'loaded': FileSnapshot.capture(str(path))}
    before = engine.embedding_inspector({'operation': 'list', 'embedding_paths': [str(path)]}, None)
    assert before['loaded'][0]['file_metadata']['step'] == '42'
    save_embedding(path, {'clip_l': torch.ones(1, 2)}, step=43, overwrite=True)
    after = engine.embedding_inspector({'operation': 'list', 'embedding_paths': [str(path)]}, None)
    assert after['loaded'][0]['file_metadata_status'] == 'file changed or could not be verified'
    assert 'file_metadata' not in after['loaded'][0]

def test_inspection_uses_loaded_token_ownership_without_catalog():
    from sdfx.embedding_inspector import run_inspector
    class Tok(TinyTokenizer):
        def get_added_vocab(self): return {'concept': 0, 'concept_1': 1}
    class Pipe:
        tokenizer = Tok()
        text_encoder = TinyPipe.text_encoder
    result = run_inspector(Pipe(), {'operation': 'inspect', 'text': 'concept', 'limit': 2,
        '_loaded_owners': {'concept': {'clip_l': ['concept']}, 'concept_1': {'clip_l': ['concept_1']}}})
    assert result['encoders']['clip_l']['ids'] == [0]
    assert result['encoders']['clip_l']['shape'] == [1, 2]

def test_base_inspection_resolves_owned_tokens_in_id_order_and_rejects_unknown():
    from sdfx.base import BaseEngine
    class Tok(TinyTokenizer):
        def get_added_vocab(self): return {'concept': 0, 'concept_1': 1}
    class Pipe:
        tokenizer = Tok()
        text_encoder = TinyPipe.text_encoder
    engine = BaseEngine(); engine.model = 'test'; engine.dry_run = False; engine.pipe = Pipe()
    engine.embedding_token_owners = {'concept': {'clip_l': ['concept']}}
    result = engine.embedding_inspector({'operation': 'inspect', 'text': 'concept', 'limit': 2}, None)
    assert result['encoders']['clip_l']['ids'] == [0]
    engine.embedding_token_owners = {'concept': {'clip_l': ['concept_1', 'concept']}}
    result = engine.embedding_inspector({'operation': 'inspect', 'text': 'concept', 'limit': 2}, None)
    assert result['encoders']['clip_l']['ids'] == [0, 1]
    engine.embedding_token_owners = {}
    with pytest.raises(Exception, match='ownership is unknown'):
        engine.embedding_inspector({'operation': 'inspect', 'text': 'concept', 'limit': 2}, None)

def test_inspect_and_mix_prefer_active_loaded_vector_over_changed_catalog_file(tmp_path):
    from sdfx.embedding_inspector import run_inspector, save_embedding
    from safetensors.torch import load_file
    class Tok(TinyTokenizer):
        def get_added_vocab(self): return {'concept': 0}
    class Pipe:
        tokenizer = Tok()
        text_encoder = TinyPipe.text_encoder
    path = tmp_path / 'concept.safetensors'; save_embedding(path, {'clip_l': torch.tensor([[0., 1.]])})
    params = {'embedding_paths': [str(path)], '_loaded_owners': {'concept': {'clip_l': ['concept']}}}
    inspected = run_inspector(Pipe(), {**params, 'operation': 'inspect', 'text': 'concept', 'limit': 2})
    assert inspected['encoders']['clip_l']['vectors'] == [[1., 0.]]
    output = tmp_path / 'mixed.safetensors'
    run_inspector(Pipe(), {**params, 'operation': 'save', 'output': str(output), 'rows': [{'name': 'concept'}]})
    assert torch.equal(load_file(str(output))['clip_l'], torch.tensor([[1., 0.]]))
