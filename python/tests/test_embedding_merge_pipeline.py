from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from test_embedding_merge_backend import Engine
from sdfx.embedding_merge_pipeline import run_pipeline, clone_tokenizers


def setup():
    engine = Engine()
    original = engine.pipe
    values = dict(vars(original))
    values = clone_tokenizers(values)
    pipeline = SimpleNamespace(**values)
    pipeline.tokenizer.vocab['img'] = 99
    pipeline.tokenizer_2.vocab['img'] = 99
    events = []
    def access(operation):
        events.append('guard-enter')
        try: return operation()
        finally: events.append('guard-exit')
    engine._with_embedding_encoder_access = access
    engine._activate = lambda pipe: events.append('activate')
    return engine, pipeline, events


def test_pipeline_tokens_live_across_all_text_passes_with_hooks_active_and_restore():
    engine, pipeline, events = setup()
    old = pipeline.text_encoder.module
    def actual(texts):
        assert events == ['guard-enter', 'guard-exit', 'activate']
        name = texts[0].split()[0]
        assert name in pipeline.tokenizer.vocab and name in pipeline.tokenizer_2.vocab
        assert name not in engine.pipe.tokenizer.vocab
        for _ in range(2):
            np.testing.assert_array_equal(pipeline.text_encoder.module.weight[-1], [2, 4])
        return 'pipeline output'
    result = run_pipeline(engine, pipeline, ["<'a'*2> img", ''], actual,
                          check=lambda: None, tensor_factory=lambda x: x)
    assert result == 'pipeline output'
    assert events == ['guard-enter', 'guard-exit', 'activate', 'guard-enter', 'guard-exit']
    assert pipeline.tokenizer.vocab == {'a': 0, 'img': 99}
    assert engine.pipe.tokenizer.vocab == {'a': 0}
    assert pipeline.text_encoder.module is old
    np.testing.assert_array_equal(old.weight, [[1, 2]])


def test_pipeline_exception_restores_both_encoders_and_is_not_swallowed():
    engine, pipeline, events = setup()
    with pytest.raises(RuntimeError, match='cancelled'):
        run_pipeline(engine, pipeline, ["<'a'> img", ''],
            lambda texts: (_ for _ in ()).throw(RuntimeError('cancelled')), check=lambda: None,
            tensor_factory=lambda x: x)
    assert pipeline.tokenizer_2.vocab == {'a': 0, 'img': 99}
    assert pipeline.text_encoder_2.module.weight.shape == (1, 2)
    assert events[-2:] == ['guard-enter', 'guard-exit']


def test_plain_pipeline_bypasses_new_lifecycle():
    engine, pipeline, events = setup()
    assert run_pipeline(engine, pipeline, ['a img', ''], lambda texts: texts, check=lambda: None) == ('a img', '')
    assert events == []


def test_long_plain_photomaker_prompt_bypasses_merge_lifecycle():
    text = 'ordinary words ' * 6000
    assert run_pipeline(object(), object(), (text, ''), lambda prompts: prompts,
                        check=lambda: None) == (text, '')


def test_invalidated_pipeline_preserves_no_model_without_reentering_guard():
    from sdfx.protocol import EngineError, NO_MODEL
    engine, pipeline, events = setup()
    discarded = []
    def invalidated(texts):
        discarded.append(pipeline.text_encoder.module)
        engine.pipe = None
        raise EngineError(NO_MODEL, 'model invalidated')
    original_access = engine._with_embedding_encoder_access
    def access(operation):
        assert engine.pipe is not None, 'must not restore discarded model'
        return original_access(operation)
    engine._with_embedding_encoder_access = access
    with pytest.raises(EngineError) as error:
        run_pipeline(engine, pipeline, ["<'a'> img", ''], invalidated,
                     check=lambda: None, tensor_factory=lambda x: x)
    assert error.value.code == NO_MODEL
    del error
    import gc
    gc.collect()
    assert pipeline.text_encoder.module is discarded[0], 'discarded model must not be revived during scope finalization'


def test_actual_pipeline_call_routes_photomaker_both_prompts_without_mutating_raw(monkeypatch):
    import ast
    from contextlib import nullcontext
    from pathlib import Path
    from sdfx import embedding_merge_pipeline
    tree = ast.parse((Path(__file__).parents[1] / 'sdfx/engine.py').read_text(encoding='utf-8'))
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == '_run_unscoped')
    block = next(node for node in ast.walk(method) if isinstance(node, ast.With)
                 and any(isinstance(item.context_expr, ast.Call)
                         and isinstance(item.context_expr.func, ast.Attribute)
                         and item.context_expr.func.attr == 'inference_mode' for item in node.items))
    raw = {'prompt': "<'a'> img", 'negative_prompt': "<'a'*0.5>", 'steps': 5}
    seen = []
    def lifecycle(engine, pipe, prompts, operation, **kwargs):
        seen.append(tuple(prompts))
        return operation(('MERGED img', 'NEGATIVE'))
    monkeypatch.setattr(embedding_merge_pipeline, 'run_pipeline', lifecycle)
    parameter = SimpleNamespace(device='cpu')
    class Pipeline:
        id_encoder = SimpleNamespace(parameters=lambda: iter([parameter]),
                                     to=lambda device: setattr(parameter, 'device', device))
        def __call__(self, **kwargs):
            assert parameter.device == 'cuda', 'actual call must align the dedicated ID encoder'
            return kwargs
    scope = {'__package__': 'sdfx', 'torch': SimpleNamespace(inference_mode=nullcontext),
             'recovery': nullcontext(), 'prepare_noise': lambda *args: nullcontext(),
             'pipe': Pipeline(), 'noise_factory': None, 'kw': raw,
             'self': SimpleNamespace(pipe=object(), device='cuda'), 'xplan': SimpleNamespace(photomaker={'enabled': True}),
             'ctx': SimpleNamespace(check_interrupt=lambda: None)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), '<actual-call>', 'exec'), scope)
    assert seen == [("<'a'> img", "<'a'*0.5>")]
    assert scope['out'] == {'prompt': 'MERGED img', 'negative_prompt': 'NEGATIVE', 'steps': 5}
    assert raw['prompt'] == "<'a'> img"
    assert parameter.device == 'cpu'


def test_actual_photomaker_constructor_does_not_leak_trigger_into_base(monkeypatch):
    import ast
    import os
    import sys
    from pathlib import Path
    tree = ast.parse((Path(__file__).parents[1] / 'sdfx/extensions.py').read_text(encoding='utf-8'))
    method = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'special_pipe')
    class Pipeline:
        def __init__(self, **components): self.__dict__.update(components)
        def load_photomaker_adapter(self, *args, **kwargs):
            self.tokenizer.vocab['img'] = 99
            self.tokenizer_2.vocab['img'] = 99
        def set_progress_bar_config(self, **kwargs): pass
    monkeypatch.setitem(sys.modules, 'photomaker', SimpleNamespace(PhotoMakerStableDiffusionXLPipeline=Pipeline))
    monkeypatch.setitem(sys.modules, 'sdfx.engine', SimpleNamespace(safe_torch_load=lambda *args: {}))
    scope = {'__package__': 'sdfx', 'RunPlan': object, 'os': os}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<actual-special>', 'exec'), scope)
    engine = Engine()
    engine.derived = {}
    engine.pipe.components = dict(vars(engine.pipe))
    plan = SimpleNamespace(sag=None, photomaker={'model': 'photomaker-v1.bin', 'trigger_word': 'img'})
    pipe = scope['special_pipe'](engine, plan)
    assert 'img' in pipe.tokenizer.vocab and 'img' not in engine.pipe.tokenizer.vocab
    assert 'img' in pipe.tokenizer_2.vocab and 'img' not in engine.pipe.tokenizer_2.vocab
    assert pipe.text_encoder is engine.pipe.text_encoder
    engine.pipe.tokenizer.vocab['later_ti'] = 1
    engine.pipe.tokenizer_2.vocab['later_ti'] = 1
    for tokenizer in (engine.pipe.tokenizer, engine.pipe.tokenizer_2):
        tokenizer.add_tokens = None
    # Model a tokenizer method without capturing the original instance.
    token_type = type(engine.pipe.tokenizer)
    monkeypatch.setattr(token_type, 'add_tokens', lambda self, names, **kw: self.vocab.update({name: len(self.vocab) for name in names}), raising=False)
    del engine.pipe.tokenizer.add_tokens
    del engine.pipe.tokenizer_2.add_tokens
    cached = scope['special_pipe'](engine, plan)
    assert cached is pipe
    assert 'later_ti' in cached.tokenizer.vocab and 'later_ti' in cached.tokenizer_2.vocab
    assert 'img' not in engine.pipe.tokenizer.vocab
