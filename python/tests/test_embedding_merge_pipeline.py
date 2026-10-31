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


@pytest.mark.parametrize('fail', [False, True])
def test_photo_merge_uses_real_eos_only_during_pipeline_and_restores(fail):
    engine, pipe, _ = setup()
    pipe.id_encoder = object()
    pipe.text_encoder_2.text_model = SimpleNamespace(eos_token_id=2)
    pipe.tokenizer_2.eos_token_id = 49407
    def operation(texts):
        assert pipe.text_encoder_2.text_model.eos_token_id == 49407
        if fail:
            raise RuntimeError('cancelled')
        return 'ok'
    if fail:
        with pytest.raises(RuntimeError, match='cancelled'):
            run_pipeline(engine, pipe, ["<'a'> img"], operation, check=lambda: None, tensor_factory=lambda x: x)
    else:
        assert run_pipeline(engine, pipe, ["<'a'> img"], operation, check=lambda: None, tensor_factory=lambda x: x) == 'ok'
    assert pipe.text_encoder_2.text_model.eos_token_id == 2


def test_photo_plain_and_nonlegacy_eos_remain_unchanged():
    engine, pipe, _ = setup()
    pipe.id_encoder = object()
    pipe.text_encoder_2.text_model = SimpleNamespace(eos_token_id=17)
    pipe.tokenizer_2.eos_token_id = 49407
    def observe(texts):
        assert pipe.text_encoder_2.text_model.eos_token_id == 17
    run_pipeline(engine, pipe, ["<'a'> img"], observe, check=lambda: None, tensor_factory=lambda x: x)
    pipe.text_encoder_2.text_model.eos_token_id = 2
    run_pipeline(engine, pipe, ['a img'], lambda texts: None, check=lambda: None)
    assert pipe.text_encoder_2.text_model.eos_token_id == 2
    del pipe.id_encoder
    def non_photo(texts):
        assert pipe.text_encoder_2.text_model.eos_token_id == 2
    run_pipeline(engine, pipe, ["<'a'> img"], non_photo, check=lambda: None, tensor_factory=lambda x: x)


@pytest.mark.parametrize('failure', [None, 'prepare', 'operation'])
def test_offload_rebinds_neutral_after_accelerate_hook_moves_and_cleans_up(failure):
    torch = pytest.importorskip('torch')
    from accelerate.hooks import ModelHook, add_hook_to_module, remove_hook_from_module
    from sdfx.neutral_runtime import RuntimeHook
    engine, pipe, _ = setup()
    engine._offload = True
    unet = torch.nn.Linear(1, 1)
    engine.pipe.unet = pipe.unet = unet
    add_hook_to_module(unet, ModelHook())
    nh = RuntimeHook(unet, None, SimpleNamespace(branches=[]), 1, 0).install()
    def place():
        remove_hook_from_module(unet)
        add_hook_to_module(unet, ModelHook())
    def access(operation):
        remove_hook_from_module(unet)
        try:
            if failure == 'prepare':
                raise RuntimeError('prepare error')
            return operation()
        finally:
            add_hook_to_module(unet, ModelHook())
    engine._with_embedding_encoder_access = access
    engine._activate = lambda _: place()
    def operation(texts):
        assert unet.forward == nh.forward, 'Neutral must be active during actual pipeline'
        if failure == 'operation':
            raise RuntimeError('operation error')
        return 'ok'
    if failure:
        with pytest.raises(RuntimeError, match=failure + ' error'):
            run_pipeline(engine, pipe, ["<'a'> img"], operation, check=lambda: None,
                         tensor_factory=lambda x: x, runtime_hook=nh)
    else:
        assert run_pipeline(engine, pipe, ["<'a'> img"], operation, check=lambda: None,
                            tensor_factory=lambda x: x, runtime_hook=nh) == 'ok'
    nh.remove()  # enclosing Engine cleanup must be harmless after both success and failure
    assert hasattr(unet, '_hf_hook') and hasattr(unet, '_old_forward')
    torch.testing.assert_close(unet(torch.ones(1, 1)), unet._old_forward(torch.ones(1, 1)))


@pytest.mark.parametrize('offload,expression', [(True, False), (False, True)])
def test_neutral_hook_is_untouched_without_both_offload_and_merge(offload, expression):
    engine, pipe, _ = setup()
    engine._offload = offload
    hook = SimpleNamespace(remove=lambda: pytest.fail('unexpected removal'),
                           install=lambda: pytest.fail('unexpected install'))
    text = "<'a'> img" if expression else 'a img'
    assert run_pipeline(engine, pipe, [text], lambda texts: 'ok', check=lambda: None,
                        tensor_factory=lambda x: x, runtime_hook=hook) == 'ok'


@pytest.mark.parametrize('failure', [False, True])
@pytest.mark.parametrize('with_multidiffusion', [False, True])
def test_sequential_offload_rebinds_negpip_child_and_neutral_in_order(failure, with_multidiffusion):
    torch = pytest.importorskip('torch')
    from accelerate.hooks import ModelHook, add_hook_to_module, remove_hook_from_module
    from sdfx import negpip
    from sdfx.neutral_runtime import RuntimeHook
    from sdfx.unet_hooks import MultiDiffusionWrapper
    class Attention(torch.nn.Module):
        is_cross_attention = True
        def forward(self, x): return x
    class Root(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.attn = Attention(); self.config = SimpleNamespace(encoder_hid_dim_type=None)
        def forward(self, x): return self.attn(x)
    engine, pipe, _ = setup(); engine._offload = True
    unet = engine.pipe.unet = pipe.unet = Root()
    def install_accelerate():
        add_hook_to_module(unet, ModelHook()); add_hook_to_module(unet.attn, ModelHook())
    def remove_accelerate():
        remove_hook_from_module(unet, recurse=True)
    install_accelerate()
    multi = MultiDiffusionWrapper(unet, {}).install() if with_multidiffusion else None
    nh = RuntimeHook(unet, None, SimpleNamespace(branches=[]), 1, 0).install()
    negative = negpip.Rebindable(pipe).install()
    def access(operation):
        remove_accelerate()
        try: return operation()
        finally: install_accelerate()
    engine._with_embedding_encoder_access = access
    def activate(_):
        remove_accelerate(); install_accelerate()
    engine._activate = activate
    def operation(texts):
        assert unet.forward == nh.forward
        if multi is not None:
            assert nh.original == multi._call
        assert unet.attn.forward is not unet.attn._old_forward
        assert negative.active
        if failure: raise RuntimeError('cancelled')
        return 'ok'
    if failure:
        with pytest.raises(RuntimeError, match='cancelled'):
            run_pipeline(engine, pipe, ["<'a'> img"], operation, check=lambda: None, tensor_factory=lambda x:x,
                         runtime_hook=nh, extra_runtime_hooks=(negative,), pre_runtime_hooks=(multi,) if multi else ())
    else:
        assert run_pipeline(engine, pipe, ["<'a'> img"], operation, check=lambda: None, tensor_factory=lambda x:x,
                            runtime_hook=nh, extra_runtime_hooks=(negative,), pre_runtime_hooks=(multi,) if multi else ()) == 'ok'
    negative.remove(); nh.remove()
    if multi is not None: multi.remove()
    assert hasattr(unet, '_hf_hook') and hasattr(unet.attn, '_hf_hook')
    assert not negative.active


@pytest.mark.parametrize(('offload', 'photo', 'multidiffusion', 'text', 'blocked'), [
    (True, True, True, "<'a'> img", True),
    (True, True, True, 'a img', False),
    (False, True, True, "<'a'> img", False),
    (True, False, True, "<'a'> img", False),
    (True, True, False, "<'a'> img", False),
])
def test_offload_photo_merge_multidiffusion_fails_before_wrapper_install(offload, photo, multidiffusion, text, blocked):
    from sdfx.embedding_merge_pipeline import validate_offload_multidiffusion
    from sdfx.protocol import EngineError
    if blocked:
        with pytest.raises(EngineError, match='MultiDiffusion.*offload'):
            validate_offload_multidiffusion(offload, photo, multidiffusion, (text,))
    else:
        validate_offload_multidiffusion(offload, photo, multidiffusion, (text,))


@pytest.mark.parametrize('hooks', [{}, {'model_keyword': False}, {'model_keyword': {'enabled': False}}])
def test_offload_guard_preserves_pre_keyword_workflow_when_keyword_disabled(hooks):
    from sdfx.embedding_merge_pipeline import validate_keyword_offload_multidiffusion
    validate_keyword_offload_multidiffusion(hooks, True, True, True, ("<'a'> img",))


def test_offload_guard_still_rejects_keyword_enabled_unsupported_combination():
    from sdfx.embedding_merge_pipeline import validate_keyword_offload_multidiffusion
    from sdfx.protocol import EngineError
    with pytest.raises(EngineError, match='MultiDiffusion.*offload'):
        validate_keyword_offload_multidiffusion({'model_keyword': {}}, True, True, True, ("<'a'> img",))


def test_multidiffusion_restores_accelerate_forward_after_remove():
    torch = pytest.importorskip('torch')
    from accelerate.hooks import ModelHook, add_hook_to_module
    from sdfx.unet_hooks import MultiDiffusionWrapper
    model = torch.nn.Linear(1, 1)
    add_hook_to_module(model, ModelHook())
    previous = model.forward
    wrapper = MultiDiffusionWrapper(model, {}).install()
    wrapper.remove()
    assert model.forward is previous
    assert hasattr(model, '_hf_hook')


@pytest.mark.parametrize('fail', [False, True])
def test_offload_rebind_keeps_real_multidiffusion_tiling_and_cleanup(fail):
    torch = pytest.importorskip('torch')
    from accelerate.hooks import ModelHook, add_hook_to_module, remove_hook_from_module
    from sdfx.unet_hooks import MultiDiffusionWrapper
    class Net(torch.nn.Module):
        def __init__(self): super().__init__(); self.calls = 0
        def forward(self, sample, timestep, encoder_hidden_states, return_dict=False):
            self.calls += 1
            return (sample + 1,)
    engine, pipe, _ = setup(); engine._offload = True
    unet = engine.pipe.unet = pipe.unet = Net()
    add_hook_to_module(unet, ModelHook())
    multi = MultiDiffusionWrapper(unet, {'tile_height': 64, 'tile_width': 64,
                                         'tile_overlap': 16, 'method': 'MultiDiffusion'}).install()
    def access(operation):
        remove_hook_from_module(unet)
        try: return operation()
        finally: add_hook_to_module(unet, ModelHook())
    engine._with_embedding_encoder_access = access
    def activate(_): remove_hook_from_module(unet); add_hook_to_module(unet, ModelHook())
    engine._activate = activate
    def operation(texts):
        value = unet(torch.zeros(1, 4, 16, 16), 0, torch.zeros(1, 1, 1), return_dict=False)[0]
        assert unet.calls > 1, 'actual tiled calls must survive reattachment'
        torch.testing.assert_close(value, torch.ones_like(value))
        if fail: raise RuntimeError('cancelled')
        return 'ok'
    if fail:
        with pytest.raises(RuntimeError, match='cancelled'):
            run_pipeline(engine, pipe, ["<'a'> img"], operation, check=lambda: None,
                         tensor_factory=lambda x:x, pre_runtime_hooks=(multi,))
    else:
        assert run_pipeline(engine, pipe, ["<'a'> img"], operation, check=lambda: None,
                            tensor_factory=lambda x:x, pre_runtime_hooks=(multi,)) == 'ok'
    multi.remove()
    assert hasattr(unet, '_hf_hook') and hasattr(unet, '_old_forward')


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
    from sdfx.conditioning_highres_fix import conditioning_scope
    scope = {'__package__': 'sdfx', 'torch': SimpleNamespace(inference_mode=nullcontext), 'nullcontext': nullcontext,
             'conditioning_scope': conditioning_scope, 'conditioning_weight': None, 'latent_source': None,
             'recovery': nullcontext(), 'prepare_noise': lambda *args: nullcontext(),
             'pipe': Pipeline(), 'noise_factory': None, 'kw': raw, 'photo_capture': None, 'nh': None, 'negative_runtime': None, 'forward_wrappers': [],
             'self': SimpleNamespace(pipe=object(), device='cuda'), 'xplan': SimpleNamespace(photomaker={'enabled': True}),
             'ctx': SimpleNamespace(check_interrupt=lambda: None)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), '<actual-call>', 'exec'), scope)
    assert seen == [("<'a'> img", "<'a'*0.5>")]
    assert scope['out'] == {'prompt': 'MERGED img', 'negative_prompt': 'NEGATIVE', 'steps': 5}
    assert raw['prompt'] == "<'a'> img"
    assert parameter.device == 'cpu'


def test_actual_pipeline_call_merges_heterogeneous_photo_prompts_before_batch_bridge(monkeypatch):
    import ast
    from contextlib import nullcontext
    from pathlib import Path
    from sdfx import embedding_merge_pipeline, photomaker_batch
    tree = ast.parse((Path(__file__).parents[1] / 'sdfx/engine.py').read_text(encoding='utf-8'))
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == '_run_unscoped')
    block = next(node for node in ast.walk(method) if isinstance(node, ast.With)
                 and any(isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Attribute)
                         and item.context_expr.func.attr == 'inference_mode' for item in node.items))
    seen = []
    def lifecycle(engine, pipe, prompts, operation, **kwargs):
        seen.append(tuple(prompts))
        return operation(('RED img', 'BLUE img', 'BAD', 'WORSE'))
    monkeypatch.setattr(embedding_merge_pipeline, 'run_pipeline', lifecycle)
    original_bridge = photomaker_batch.heterogeneous_prompts
    monkeypatch.setattr(photomaker_batch, 'heterogeneous_prompts',
        lambda pipe, prompts, **kw: original_bridge(pipe, prompts, concat=lambda rows: sum(rows, []), **kw))
    parameter = SimpleNamespace(device='cpu')
    class Tokenizer:
        def encode(self, value, **kw): return value.split()
        def decode(self, value, **kw): return ' '.join(value)
        def convert_tokens_to_ids(self, value): return value
    class Encoder:
        def parameters(self): return iter([parameter])
        def to(self, device): parameter.device = device
        def forward(self, images, embeds, mask): return embeds
    class Pipeline:
        trigger_word = 'img'
        def __init__(self): self.tokenizer, self.id_encoder = Tokenizer(), Encoder()
        def encode_prompt_with_trigger_word(self, **kw):
            return ([kw['prompt']], None, ['pooled'], None, ['mask'])
        def encode_prompt(self, **kw): return kw['prompt']
        def __call__(self, **kw):
            values = self.encode_prompt_with_trigger_word(prompt=kw['prompt'], num_images_per_prompt=1)
            return {'prompts': values[0], 'negative': kw['negative_prompt'], 'device': parameter.device}
    raw = {'prompt': ["<'red'> img", "<'blue'> img"], 'negative_prompt': ["<'bad'>", "<'worse'>"]}
    pipe = Pipeline()
    from sdfx.conditioning_highres_fix import conditioning_scope
    scope = {'__package__': 'sdfx', 'torch': SimpleNamespace(inference_mode=nullcontext), 'nullcontext': nullcontext,
             'conditioning_scope': conditioning_scope, 'conditioning_weight': None, 'latent_source': None,
             'recovery': nullcontext(), 'prepare_noise': lambda *args: nullcontext(), 'pipe': pipe,
             'noise_factory': None, 'kw': raw, 'photo_capture': None, 'nh': None, 'negative_runtime': None, 'forward_wrappers': [],
             'self': SimpleNamespace(pipe=object(), device='cuda'), 'xplan': SimpleNamespace(photomaker={'enabled': True}),
             'ctx': SimpleNamespace(check_interrupt=lambda: None)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), '<actual-call>', 'exec'), scope)
    assert seen == [("<'red'> img", "<'blue'> img", "<'bad'>", "<'worse'>")]
    assert scope['out'] == {'prompts': ['RED img', 'BLUE img'], 'negative': ['BAD', 'WORSE'], 'device': 'cuda'}
    assert raw['prompt'][0] == "<'red'> img" and parameter.device == 'cpu'
    assert 'encode_prompt_with_trigger_word' not in pipe.__dict__


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
