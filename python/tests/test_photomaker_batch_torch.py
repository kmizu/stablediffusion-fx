import pytest

torch = pytest.importorskip('torch')
from sdfx.photomaker_batch import heterogeneous_prompts


def test_trigger_and_identity_fusion_keep_per_image_order_and_tensor_shape():
    class Tokenizer:
        def encode(self, text, **kwargs): return text.split()
        def decode(self, ids, **kwargs): return ' '.join(ids)
        def convert_tokens_to_ids(self, token): return token
    class IdEncoder(torch.nn.Module):
        def forward(self, pixels, embeds, mask):
            assert pixels.shape == (1, 1, 3, 2, 2) and embeds.shape == (1, 2, 1) and mask.shape == (1, 2)
            return embeds + mask.unsqueeze(-1)
    class Pipeline:
        trigger_word = 'img'
        def __init__(self): self.tokenizer = Tokenizer(); self.id_encoder = IdEncoder()
        def encode_prompt_with_trigger_word(self, **kwargs):
            value = 1. if kwargs['prompt'].startswith('red') else 2.
            return (torch.full((1, 2, 1), value), None, torch.full((1, 1), value), None,
                    torch.tensor([[value, 0.]]))
        def encode_prompt(self, **kwargs):
            assert kwargs['prompt'] == ['red person', 'blue person']
            return torch.tensor([[1.], [2.]])
    pipe = Pipeline(); prompts = ['red person img', 'blue person img']
    with heterogeneous_prompts(pipe, prompts):
        positive, _, pooled, _, mask = pipe.encode_prompt_with_trigger_word(prompt=prompts, num_images_per_prompt=1)
        assert positive.shape == (2, 2, 1) and pooled[:, 0].tolist() == [1., 2.]
        assert pipe.encode_prompt(prompt='ignored', negative_prompt='bad').shape == (2, 1)
        result = pipe.id_encoder(torch.zeros((1, 1, 3, 2, 2)), positive, mask)
        assert result[:, :, 0].tolist() == [[2., 1.], [4., 2.]]
    assert 'forward' not in pipe.id_encoder.__dict__


def test_routed_photo_encodings_keep_cfg_halves_and_identity_fusion_separate():
    from contextlib import contextmanager
    state = []
    @contextmanager
    def route(index, positive):
        state.append((index, positive))
        try:
            yield
        finally:
            state.pop()
    class Tokenizer:
        def encode(self, text, **kwargs): return text.split()
        def decode(self, ids, **kwargs): return ' '.join(ids)
        def convert_tokens_to_ids(self, token): return token
    class IdEncoder(torch.nn.Module):
        def forward(self, pixels, embeds, mask): return embeds + mask.unsqueeze(-1)
    class Pipeline:
        trigger_word = 'img'
        def __init__(self): self.tokenizer = Tokenizer(); self.id_encoder = IdEncoder()
        def values(self):
            index, positive = state[-1]
            value = 10 * index + (1 if positive else 2)
            return tuple(torch.full((1, 1, 1), float(value + offset)) for offset in (0, 100, 200, 300))
        def encode_prompt(self, **kwargs): return self.values()
        def encode_prompt_with_trigger_word(self, **kwargs):
            return (*self.values(), torch.ones((1, 1)))
    pipe = Pipeline()
    with heterogeneous_prompts(pipe, ['red img', 'blue img'], route=route):
        positive, negative, pooled, neg_pooled, mask = pipe.encode_prompt_with_trigger_word(num_images_per_prompt=1)
        assert positive[:, 0, 0].tolist() == [1., 11.]
        assert negative[:, 0, 0].tolist() == [102., 112.]
        assert pooled[:, 0, 0].tolist() == [201., 211.]
        assert neg_pooled[:, 0, 0].tolist() == [302., 312.]
        assert pipe.id_encoder(torch.zeros(1), positive, mask)[:, 0, 0].tolist() == [2., 12.]
        assert pipe.encode_prompt(prompt='unused', negative_prompt=['bad', 'worse'])[1][:, 0, 0].tolist() == [102., 112.]
    assert not state


def test_capture_feeds_real_neutral_unet_before_and_after_identity_merge():
    from contextlib import contextmanager
    from types import SimpleNamespace
    from sdfx.photomaker_batch import PhotoBranchCapture, photo_branch_transform
    from sdfx.neutral_prompt import parse
    from sdfx.neutral_runtime import Plan, RuntimeHook
    import diffusers
    state = []
    def route(branch):
        @contextmanager
        def enter(index, positive):
            state.append((branch, index, positive))
            try: yield
            finally: state.pop()
        return enter
    class Tokenizer:
        def encode(self, text, **kw): return text.split()
        def decode(self, ids, **kw): return ' '.join(ids)
        def convert_tokens_to_ids(self, token): return token
    class IdEncoder(torch.nn.Module):
        def forward(self, pixels, embeds, mask): return embeds + mask.unsqueeze(-1) * 10
    class Pipeline:
        trigger_word = 'img'
        def __init__(self): self.tokenizer, self.id_encoder = Tokenizer(), IdEncoder()
        def values(self):
            branch, index, positive = state[-1]
            value = float(1 + 2 * branch + index) if positive else 0.
            return (torch.full((1, 4, 1), value), torch.zeros(1, 4, 1), None, None)
        def encode_prompt(self, **kw): return self.values()
        def encode_prompt_with_trigger_word(self, **kw): return (*self.values(), torch.ones(1, 4))
    seen = []
    class UNet(torch.nn.Module):
        def forward(self, sample, timestep, encoder_hidden_states=None, **kw):
            seen.append(encoder_hidden_states.clone())
            return (encoder_hidden_states.reshape_as(sample),)
    specs = [dict(prompts=[f'{name} img'], route=route(i), metadata={}) for i, name in enumerate(('red', 'blue'))]
    plan = Plan(parse('red AND blue'), [{}, {}])
    capture = PhotoBranchCapture(specs); capture.bind(plan)
    pipe = Pipeline()
    with heterogeneous_prompts(pipe, specs[0]['prompts'], route=specs[0]['route'], capture=capture):
        trigger = pipe.encode_prompt_with_trigger_word(prompt='red img', num_images_per_prompt=1)
        pipe.encode_prompt(prompt='red', negative_prompt='bad')
        pipe.id_encoder(torch.zeros(1), trigger[0], trigger[4])
        scheduler = diffusers.DDIMScheduler(clip_sample=False); scheduler.set_timesteps(3)
        unet = UNet()
        hook = RuntimeHook(unet, scheduler, plan, 3, 0, branch_transform=photo_branch_transform(0)).install()
        try:
            unet(torch.zeros(2, 4, 1, 1), scheduler.timesteps[0])
            hook.callback(None, 0, None, {})
            unet(torch.zeros(2, 4, 1, 1), scheduler.timesteps[1])
        finally:
            hook.remove()
    assert [float(t[-1, 0, 0]) for t in seen] == [1., 3., 11., 13.]
    assert not state and 'forward' not in pipe.id_encoder.__dict__


@pytest.mark.parametrize('cancel,scheduled,cfg,merge,allow_merge,signed', [(False, False, 3, False, False, False),
    (True, False, 3, False, False, False), (False, True, 3, False, False, False), (False, False, .5, False, False, False),
    (False, False, 3, True, False, False), (False, False, 3, True, True, False), (False, False, 3, False, False, True)])
def test_engine_run_wires_composable_photo_capture_and_cleans_up(monkeypatch, cancel, scheduled, cfg, merge, allow_merge, signed):
    from types import SimpleNamespace
    import diffusers
    from sdfx import engine as engine_module, extensions, neutral_runtime, photomaker_batch
    from sdfx.clora.runtime import Job
    from sdfx.engine import Engine
    from sdfx.neutral_prompt import parse
    from sdfx.neutral_runtime import Plan
    from sdfx.params import GenerationParams
    class Tokenizer:
        def encode(self, text, **kw): return text.split()
        def decode(self, ids, **kw): return ' '.join(ids)
        def convert_tokens_to_ids(self, token): return token
    class IdEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
        def forward(self, images, embeds, mask): return embeds + 10 * mask.unsqueeze(-1)
    seen = []
    class UNet(torch.nn.Module):
        config = SimpleNamespace(time_cond_proj_dim=None)
        def forward(self, sample, timestep, encoder_hidden_states=None, **kw):
            seen.append(encoder_hidden_states.clone())
            if cancel and len(seen) == 3:
                raise RuntimeError('cancel during identity pass')
            return (encoder_hidden_states[:, :4, :1].reshape_as(sample),)
    class Pipe:
        trigger_word = 'img'
        def __init__(self): self.tokenizer, self.id_encoder, self.unet = Tokenizer(), IdEncoder(), UNet()
        def values(self, text):
            value = next(value for colour, value in (('red', 1.), ('blue', 3.), ('green', 5.), ('gold', 7.)) if text.startswith(colour))
            return (torch.full((1, 4, 1), value), torch.zeros(1, 4, 1), None, None)
        def encode_prompt(self, **kw):
            assert kw['do_classifier_free_guidance'] is True
            return self.values(kw['prompt'])
        def encode_prompt_with_trigger_word(self, **kw):
            assert kw['do_classifier_free_guidance'] is True
            return (*self.values(kw['prompt']), torch.ones(1, 4))
        def __call__(self, **kw):
            trigger = self.encode_prompt_with_trigger_word(prompt=kw['prompt'], num_images_per_prompt=1)
            self.encode_prompt(prompt=kw['prompt'], negative_prompt=kw['negative_prompt'])
            self.id_encoder(torch.zeros(1), trigger[0], trigger[4])
            first = self.unet(torch.zeros(2 if cfg > 1 else 1, 4, 1, 1), self.scheduler.timesteps[0])[0]
            if cfg <= 1: assert float(first[0, 0, 0, 0]) == pytest.approx(4.)
            kw['callback_on_step_end'](self, 0, self.scheduler.timesteps[0], {})
            self.unet(torch.zeros(2 if cfg > 1 else 1, 4, 1, 1), self.scheduler.timesteps[1])
            return SimpleNamespace(images=['done'])
    job = Job.__new__(Job); job.state = None; job.config = {}
    branch = lambda text: {'_clora_branch': text, '_clora_prompt': text, '_clora_negative': 'bad',
                           'prompt_embeds': torch.ones(1, 4, 1), 'negative_prompt_embeds': torch.zeros(1, 4, 1)}
    if scheduled:
        def timeline(text):
            value = branch(text)
            result = type('Scheduled', (dict,), {})(value); result.schedule = [(1, value), (2, value)]
            return result
        branches = [timeline('[red:green:0.5] person img'), timeline('[blue:gold:0.5] person img')]
    else:
        branches = [branch('red (hat:-1) person img' if signed else 'red person img'),
                    branch('blue (hair:-2) person img' if signed else 'blue person img')]
    if signed:
        for value in branches: value['_negpip'] = True
    if allow_merge:
        for index, branch_value in enumerate(branches):
            colour = ('red', 'blue')[index]
            branch_value['_photo_raw_branch'] = f"{colour} <'person'> img"
            branch_value['_photo_raw_prompt'] = f"{colour} <'person'> img"
            branch_value['_photo_raw_negative'] = 'bad'
    plan = Plan(parse('red person img AND blue person img'), branches)
    pipe, engine = Pipe(), Engine.__new__(Engine)
    engine.sdxl = True; engine._composable_job = job; engine.pipe = pipe; engine.device = 'cpu'; engine.cuda = False; engine.dtype = torch.float32
    scheduler = diffusers.DDIMScheduler(clip_sample=False); scheduler.set_timesteps(2)
    engine._scheduler = lambda *a: (scheduler, None)
    engine._generators = lambda *a: None
    engine._callback = lambda *a: (lambda pipe, i, t, tensors: tensors)
    engine._activate = lambda *a: None
    monkeypatch.setattr(neutral_runtime, 'validate', lambda *a, **kw: None)  # production guard remains enabled
    monkeypatch.setattr(photomaker_batch, 'validate_pipeline', lambda pipe: None)
    if allow_merge:
        from sdfx import embedding_merge_pipeline
        monkeypatch.setattr(photomaker_batch, 'reject_unverified_merge_composable', lambda texts: None)
        def merge_scope(engine, pipe, texts, operation, **kwargs):
            assert any("<'person'>" in value for value in texts)
            return operation(tuple(value.replace("<'person'>", 'person') for value in texts))
        monkeypatch.setattr(embedding_merge_pipeline, 'run_pipeline', merge_scope)
    monkeypatch.setattr(engine_module, 'load_rgb', lambda path: path)
    monkeypatch.setattr(extensions, 'pre_run', lambda *a, **kw: extensions.RunPlan('txt2img', photomaker={'start_merge_step': 0}))
    monkeypatch.setattr(extensions, 'special_pipe', lambda *a: pipe)
    monkeypatch.setattr(extensions, 'install', lambda *a, **kw: None)
    if signed:
        from sdfx import negpip
        monkeypatch.setattr(negpip, 'encode_photo_target', lambda pipe, text, weight, **kw: torch.tensor([[weight]]))
        monkeypatch.setattr(negpip, 'Rebindable', lambda pipe: SimpleNamespace(install=lambda: SimpleNamespace(remove=lambda: None)))
    gp = GenerationParams(prompt="red <'person'> img" if merge else 'red person img', negative_prompt='bad', steps=2,
                          hooks={'photomaker': {'images': ['id'], 'trigger_word': 'img'}, 'composable_lora': {},
                                 **({'negpip': {}} if signed else {})})
    ctx = SimpleNamespace(check_interrupt=lambda: None, warn=lambda *a: None)
    def invoke():
        return engine._run('txt2img', gp, None, ctx, sampler='DDIM', schedule=None, cfg_scale=cfg,
            embeds={'_neutral': plan}, steps=2, w=64, h=64, seeds=[1], pass_='base', image_idx=0,
            images_total=1, output_type='latent')
    if merge and not allow_merge:
        from sdfx.protocol import EngineError, UNSUPPORTED
        with pytest.raises(EngineError, match='Embedding Merge with Composable PhotoMaker') as error: invoke()
        assert error.value.code == UNSUPPORTED and not seen
    elif cancel:
        with pytest.raises(RuntimeError, match='cancel during identity pass'): invoke()
        assert [float(t[-1, 0, 0]) for t in seen] == [1., 3., 11.]
    else:
        result, _ = invoke()
        assert result == ['done']
        assert [float(t[-1, 0, 0]) for t in seen] == ([1., 3., 15., 17.] if scheduled else [1., 3., 11., 13.])
        if signed:
            assert all(t.shape[-1] == 2 and t[-1, -1, -1].item() == -1 for t in seen)
    assert job.state is None and 'forward' not in pipe.unet.__dict__ and 'forward' not in pipe.id_encoder.__dict__
def test_signed_photo_conditioning_marks_identity_after_fusion_and_pads_per_image():
    from sdfx.photomaker_batch import signed_photo_conditioning
    import torch
    text = {'prompt_embeds': torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]),
            'negative_prompt_embeds': torch.tensor([[[9., 10.]], [[11., 12.]]]),
            'pooled_prompt_embeds': torch.tensor([[13.], [14.]])}
    identity = dict(text, prompt_embeds=text['prompt_embeds'] + 100)
    positive_terms = [[torch.tensor([[20., 21.]])], [torch.tensor([[22., 23.], [24., 25.]])]]
    negative_terms = [[], [torch.tensor([[30., 31.]])]]
    plain, fused = signed_photo_conditioning(text, identity, positive_terms, negative_terms)
    assert plain['_negpip'] and fused['_negpip']
    assert plain['prompt_embeds'].shape == fused['prompt_embeds'].shape == (2, 4, 3)
    assert plain['negative_prompt_embeds'].shape == fused['negative_prompt_embeds'].shape == (2, 4, 3)
    assert plain['prompt_embeds'][0, :, -1].tolist() == [1, 1, -1, 0]
    assert plain['prompt_embeds'][1, :, -1].tolist() == [1, 1, -1, -1]
    assert plain['negative_prompt_embeds'][0, :, -1].tolist() == [1, 0, 0, 0]
    assert plain['negative_prompt_embeds'][1, :, -1].tolist() == [1, -1, 0, 0]
    assert torch.equal(fused['prompt_embeds'][:, :2, :-1], identity['prompt_embeds'])
    assert torch.equal(fused['pooled_prompt_embeds'], text['pooled_prompt_embeds'])
    assert '_negpip' not in text


def test_photo_capture_publishes_signed_conditions_after_identity_fusion():
    from contextlib import nullcontext
    from types import SimpleNamespace
    from sdfx.photomaker_batch import PhotoBranchCapture, photo_branch_transform
    import torch
    raw = torch.ones(1, 2, 2)
    negative = torch.zeros(1, 1, 2)
    spec = {'prompts': ['woman img'], 'route': lambda *a: nullcontext(), 'metadata': {},
            'signed_positive_terms': [[torch.full((1, 2), 3.)]], 'signed_negative_terms': [[]]}
    capture = PhotoBranchCapture([spec]); plan = SimpleNamespace(branches=[{}]); capture.bind(plan)
    capture.plain = [(raw, negative, torch.ones(1, 1), torch.ones(1, 1))]
    capture.trigger = [(raw, negative, torch.ones(1, 1), torch.ones(1, 1), torch.ones(1, 2))]
    capture.identity = [raw * 5]
    capture.publish()
    route = photo_branch_transform(0)
    text, identity = route(0, plan.branches[0], 1), route(0, plan.branches[0], 2)
    assert text['_negpip'] and identity['_negpip']
    assert text['prompt_embeds'][0, :, -1].tolist() == [1, 1, -1]
    assert torch.equal(identity['prompt_embeds'][0, :2, :-1], torch.full((2, 2), 5.))


def test_signed_photo_schedule_keeps_each_variant_and_image_terms_separate():
    from contextlib import nullcontext
    from types import SimpleNamespace
    from sdfx.photomaker_batch import PhotoBranchCapture, photo_branch_transform
    from sdfx.neutral_runtime import at
    import torch
    base = torch.ones(2, 2, 2)
    negative = torch.zeros(2, 1, 2)
    def spec(first, second):
        return {'prompts': ['woman img', 'man img'], 'route': lambda *a: nullcontext(), 'metadata': {},
                'signed_positive_terms': [[torch.full((1, 2), first)], [torch.full((1, 2), second)]],
                'signed_negative_terms': [[], []]}
    capture = PhotoBranchCapture([spec(3., 4.), spec(5., 6.)], groups=[[(1, 0), (3, 1)]])
    plan = SimpleNamespace(branches=[{}]); capture.bind(plan)
    capture.plain = [(base, negative, None, None)] * 2
    capture.trigger = [(base, negative, None, None, torch.ones(2, 2))] * 2
    capture.identity = [base * 10, base * 20]
    capture.publish()
    route = photo_branch_transform(0)
    first = route(0, at(plan.branches[0], 1), 1)
    later = route(0, at(plan.branches[0], 2), 2)
    assert first['prompt_embeds'][:, -1, 0].tolist() == [3., 4.]
    assert later['prompt_embeds'][:, -1, 0].tolist() == [5., 6.]
    assert torch.equal(later['prompt_embeds'][:, :2, 0], torch.full((2, 2), 20.))
    assert (later['prompt_embeds'][:, -1, -1] == -1).all()


def test_fused_signed_photo_attention_matches_independent_value_negation():
    from types import SimpleNamespace
    from diffusers.models.attention_processor import Attention
    from sdfx.negpip import install
    from sdfx.photomaker_batch import signed_photo_conditioning
    import torch
    torch.manual_seed(17)
    attention = Attention(query_dim=4, cross_attention_dim=4, heads=2, dim_head=2, bias=True)
    pipe = SimpleNamespace(unet=torch.nn.ModuleList([attention]))
    text = {'prompt_embeds': torch.randn(2, 2, 4), 'negative_prompt_embeds': torch.randn(2, 1, 4)}
    identity = dict(text, prompt_embeds=text['prompt_embeds'] + .25)
    extra = [[torch.randn(1, 4)], [torch.randn(2, 4)]]
    _, fused = signed_photo_conditioning(text, identity, extra, [[], []])
    marked = fused['prompt_embeds']; context, signs = marked[..., :-1], marked[..., -1]
    query = torch.randn(2, 3, 4)
    q = attention.to_q(query).view(2, 3, 2, 2).transpose(1, 2)
    k = attention.to_k(context).view(2, 4, 2, 2).transpose(1, 2)
    v = (attention.to_v(context) * signs[..., None]).view(2, 4, 2, 2).transpose(1, 2)
    mask = torch.zeros(2, 1, 1, 4).masked_fill(signs[:, None, None, :] == 0, -torch.inf)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    expected = attention.to_out[1](attention.to_out[0](expected.transpose(1, 2).reshape(2, 3, 4)))
    remove = install(pipe)
    try: torch.testing.assert_close(attention(query, encoder_hidden_states=marked), expected)
    finally: remove()


def test_signed_photo_terms_encode_per_image_under_matching_composable_routes_and_restore():
    from contextlib import contextmanager
    from sdfx.photomaker_batch import PhotoBranchCapture
    import torch
    events = []
    @contextmanager
    def route(index, positive):
        events.append(('enter', index, positive))
        try: yield
        finally: events.append(('exit', index, positive))
    spec = {'prompts': ['woman img, (hat:-2)', 'man img, (glasses:-0.5)'],
            'negatives': ['(noise:-1)', 'plain'], 'route': route, 'metadata': {}}
    capture = PhotoBranchCapture([spec])
    def encode(text, weight):
        events.append(('encode', text, weight))
        return torch.ones(1, 3) * weight
    before = (list(spec['prompts']), list(spec['negatives']))
    with capture.signed_terms(encode):
        assert '(hat:-2)' not in spec['prompts'][0] and '(noise:-1)' not in spec['negatives'][0]
        assert spec['signed_positive_terms'][0][0][0, 0] == 2
        assert spec['signed_positive_terms'][1][0][0, 0] == .5
        assert spec['signed_negative_terms'][0][0][0, 0] == 1
        assert spec['signed_negative_terms'][1] == []
    assert (spec['prompts'], spec['negatives']) == before
    assert 'signed_positive_terms' not in spec and 'signed_negative_terms' not in spec
    assert events == [('enter', 0, True), ('encode', 'hat', 2.), ('exit', 0, True),
                      ('enter', 1, True), ('encode', 'glasses', .5), ('exit', 1, True),
                      ('enter', 0, False), ('encode', 'noise', 1.), ('exit', 0, False)]


def test_signed_photo_scope_restores_all_specs_when_encoder_fails():
    from contextlib import nullcontext
    from sdfx.photomaker_batch import PhotoBranchCapture
    specs = [{'prompts': ['woman img, (hat:-1)'], 'negatives': ['plain'],
              'route': lambda *args: nullcontext(), 'metadata': {}},
             {'prompts': ['man img, (hat:-2)'], 'negatives': ['plain'],
              'route': lambda *args: nullcontext(), 'metadata': {}}]
    calls = []
    def encode(text, weight):
        calls.append(weight)
        if len(calls) == 2: raise RuntimeError('cancelled')
        return torch.ones(1, 2)
    before = [(list(spec['prompts']), list(spec['negatives'])) for spec in specs]
    with pytest.raises(RuntimeError, match='cancelled'):
        with PhotoBranchCapture(specs).signed_terms(encode): pass
    assert [(spec['prompts'], spec['negatives']) for spec in specs] == before
    assert all('signed_positive_terms' not in spec and 'signed_negative_terms' not in spec for spec in specs)
