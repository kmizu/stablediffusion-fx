import pytest

from sdfx.photomaker_batch import heterogeneous_prompts, validate_pipeline


class Tokenizer:
    def encode(self, value, **kwargs): return value.split()
    def decode(self, values, **kwargs): return ' '.join(values)
    def convert_tokens_to_ids(self, value): return value


class IdEncoder:
    def forward(self, images, embeds, mask):
        assert images == 'same-id' and len(embeds) == len(mask) == 1
        return [embeds[0] + ':' + mask[0]]


class Pipe:
    trigger_word = 'img'
    def __init__(self): self.tokenizer, self.id_encoder = Tokenizer(), IdEncoder(); self.calls = []
    def encode_prompt_with_trigger_word(self, **kwargs):
        self.calls.append(kwargs['prompt'])
        return ([kwargs['prompt']], None, ['pooled:' + kwargs['prompt']], None, ['mask:' + kwargs['prompt']])
    def encode_prompt(self, **kwargs):
        assert kwargs['negative_prompt'] == ['bad', 'bad']
        return kwargs['prompt']


def cat(values): return [entry for value in values for entry in value]


def test_heterogeneous_conditioning_and_cleanup_after_failure():
    pipe = Pipe(); prompts = ['red person img', 'blue person img']
    with pytest.raises(RuntimeError, match='cancel'):
        with heterogeneous_prompts(pipe, prompts, concat=cat):
            outputs = pipe.encode_prompt_with_trigger_word(prompt=prompts, num_images_per_prompt=1)
            assert outputs[0] == prompts and outputs[2] == ['pooled:' + value for value in prompts]
            assert pipe.tokenizer.encode(prompts) == prompts[0].split()
            assert pipe.encode_prompt(prompt='ignored', negative_prompt='bad') == ['red person', 'blue person']
            assert pipe.id_encoder.forward('same-id', outputs[0], outputs[4]) == [
                'red person img:mask:red person img', 'blue person img:mask:blue person img']
            raise RuntimeError('cancel')
    assert pipe.calls == prompts
    for obj, name in ((pipe, 'encode_prompt'), (pipe, 'encode_prompt_with_trigger_word'),
                      (pipe.tokenizer, 'encode'), (pipe.id_encoder, 'forward')):
        assert name not in obj.__dict__


def test_bridge_rejects_multiple_images_per_prompt():
    pipe = Pipe()
    with heterogeneous_prompts(pipe, ['red img', 'blue img'], concat=cat):
        with pytest.raises(ValueError, match='one image per prompt'):
            pipe.encode_prompt_with_trigger_word(prompt=['red img', 'blue img'], num_images_per_prompt=2)


def test_bridge_rejects_unrecognized_pipeline():
    with pytest.raises(ValueError, match='PhotoMaker pipeline'):
        validate_pipeline(Pipe())


def test_bridge_rolls_back_partial_setup_when_patch_fails():
    class FrozenTokenizer(Tokenizer):
        def __setattr__(self, name, value): raise RuntimeError('cannot patch tokenizer')
    pipe = Pipe(); pipe.tokenizer = FrozenTokenizer()
    with pytest.raises(RuntimeError, match='cannot patch'):
        with heterogeneous_prompts(pipe, ['red img', 'blue img'], concat=cat):
            pytest.fail('setup should not succeed')
    assert 'encode_prompt' not in pipe.__dict__
    assert 'encode_prompt_with_trigger_word' not in pipe.__dict__
    assert 'forward' not in pipe.id_encoder.__dict__


def test_bridge_rejects_misaligned_identity_batch():
    pipe = Pipe()
    with heterogeneous_prompts(pipe, ['red img', 'blue img'], concat=cat):
        with pytest.raises(ValueError, match='batch dimensions'):
            pipe.id_encoder.forward('same-id', ['only-one'], ['mask-a', 'mask-b'])


def test_branch_routing_separates_positive_and_negative_for_each_photo_prompt():
    from contextlib import contextmanager
    active = []
    seen = []
    @contextmanager
    def route(index, positive):
        active.append((index, positive))
        try:
            yield
        finally:
            active.pop()
    class RoutedPipe(Pipe):
        def encode_prompt_with_trigger_word(self, **kwargs):
            index, positive = active[-1]; seen.append(('trigger', index, positive))
            return ([f'p{index}:{positive}'], [f'n{index}:{positive}'],
                    [f'pp{index}:{positive}'], [f'np{index}:{positive}'], [f'm{index}:{positive}'])
        def encode_prompt(self, **kwargs):
            index, positive = active[-1]; seen.append(('plain', index, positive))
            return ([f'p{index}:{positive}'], [f'n{index}:{positive}'],
                    [f'pp{index}:{positive}'], [f'np{index}:{positive}'])
    pipe = RoutedPipe()
    with heterogeneous_prompts(pipe, ['red img', 'blue img'], concat=cat, route=route):
        trigger = pipe.encode_prompt_with_trigger_word(prompt=['red img', 'blue img'], num_images_per_prompt=1)
        plain = pipe.encode_prompt(prompt='ignored', negative_prompt=['bad1', 'bad2'])
    assert trigger == (['p0:True', 'p1:True'], ['n0:False', 'n1:False'],
                       ['pp0:True', 'pp1:True'], ['np0:False', 'np1:False'], ['m0:True', 'm1:True'])
    assert plain == trigger[:4]
    assert seen == [(phase, index, positive) for phase in ('trigger', 'plain') for index in (0, 1)
                    for positive in (True, False)]
    assert not active


def test_routed_single_prompt_and_cancellation_restore_pipeline_and_scope():
    from contextlib import contextmanager
    active = []
    @contextmanager
    def route(index, positive):
        active.append((index, positive))
        try:
            yield
        finally:
            active.pop()
    class CancelPipe(Pipe):
        def encode_prompt_with_trigger_word(self, **kwargs):
            if not active[-1][1]:
                raise RuntimeError('cancel')
            return (['p'], ['n'], ['pp'], ['np'], ['mask'])
    pipe = CancelPipe()
    with pytest.raises(RuntimeError, match='cancel'):
        with heterogeneous_prompts(pipe, ['person img'], concat=cat, route=route):
            pipe.encode_prompt_with_trigger_word(prompt='person img', num_images_per_prompt=1)
    assert not active
    assert 'encode_prompt_with_trigger_word' not in pipe.__dict__
    assert 'forward' not in pipe.id_encoder.__dict__


def test_composable_route_uses_each_images_branch_and_restores_job_state():
    from sdfx.photomaker_batch import composable_text_route
    from sdfx.clora.runtime import Job
    job = Job.__new__(Job)
    job.state = 'previous'
    branch = {'_clora_branch': ['red <lora:a:1>', 'blue <lora:b:2>'],
              '_clora_prompt': ['red <lora:a:1>', 'blue <lora:b:2>'],
              '_clora_negative': ['bad', 'worse']}
    route = composable_text_route(job, branch, count=2, total=20)
    with route(1, False):
        assert job.state == ('blue <lora:b:2>', [False], -1, 20, 'text_encoder', 'base',
                             'blue <lora:b:2>', 'worse')
    assert job.state == 'previous'
    with pytest.raises(ValueError, match='routing'):
        composable_text_route(job, dict(branch, _clora_negative=['bad']), count=2, total=20)


def test_merge_stage_uses_official_zero_based_boundary_without_mutating_branches():
    from sdfx.photomaker_batch import select_merge_stage
    text = {'prompt_embeds': 'plain', 'pooled_prompt_embeds': 'plain pooled',
            'negative_prompt_embeds': 'negative', 'negative_pooled_prompt_embeds': 'negative pooled',
            '_clora_branch': ['red', 'blue']}
    identity = {'prompt_embeds': 'identity', 'pooled_prompt_embeds': 'identity pooled',
                'negative_prompt_embeds': 'negative', 'negative_pooled_prompt_embeds': 'negative pooled',
                '_clora_branch': ['red', 'blue']}
    assert select_merge_stage(text, identity, step=11, start_merge_step=10) == text
    assert select_merge_stage(text, identity, step=12, start_merge_step=10) == identity
    with pytest.raises(ValueError, match='routing'):
        select_merge_stage(text, dict(identity, _clora_branch=['wrong']), step=12, start_merge_step=10)


def test_official_call_order_captures_per_branch_text_identity_and_routing():
    from contextlib import contextmanager
    from sdfx.photomaker_batch import PhotoBranchCapture
    active = []
    def route(branch):
        @contextmanager
        def enter(index, positive):
            active.append((branch, index, positive))
            try: yield
            finally: active.pop()
        return enter
    class CapturingPipe(Pipe):
        def encode_prompt_with_trigger_word(self, **kw):
            b, i, sign = active[-1]
            return ([f'id:{b}:{i}:{sign}'], [f'n:{b}:{i}:{sign}'],
                    [f'pool:{b}:{i}:{sign}'], [f'np:{b}:{i}:{sign}'], [f'mask:{b}:{i}'])
        def encode_prompt(self, **kw):
            b, i, sign = active[-1]
            assert kw.get('negative_prompt_embeds') is None
            assert kw.get('negative_pooled_prompt_embeds') is None
            return ([f'text:{b}:{i}:{sign}'], [f'n:{b}:{i}:{sign}'],
                    [f'tpool:{b}:{i}:{sign}'], [f'np:{b}:{i}:{sign}'])
    specs = [dict(prompts=[f'{colour} person img', f'{colour} cat img'], route=route(b),
                  metadata={'_clora_branch': [f'{colour}1', f'{colour}2'],
                            '_clora_prompt': [f'{colour}1', f'{colour}2'], '_clora_negative': ['bad', 'bad']})
             for b, colour in enumerate(('red', 'blue'))]
    from types import SimpleNamespace
    pipe = CapturingPipe(); capture = PhotoBranchCapture(specs)
    plan = SimpleNamespace(branches=[{}, {}])
    capture.bind(plan)
    with heterogeneous_prompts(pipe, specs[0]['prompts'], concat=cat, route=specs[0]['route'], capture=capture):
        trig = pipe.encode_prompt_with_trigger_word(prompt=specs[0]['prompts'], num_images_per_prompt=1)
        pipe.encode_prompt(prompt='ignored', negative_prompt='bad', negative_prompt_embeds='primary negative',
                           negative_pooled_prompt_embeds='primary pooled')
        primary = pipe.id_encoder.forward('same-id', trig[0], trig[4])
        branches = capture.branches()
        assert plan.branches == branches
    assert primary == ['id:0:0:True:mask:0:0', 'id:0:1:True:mask:0:1']
    assert len(branches) == 2 and branches[1]['prompt_embeds'] == ['text:1:0:True', 'text:1:1:True']
    assert branches[1]['negative_prompt_embeds'] == ['n:1:0:False', 'n:1:1:False']
    assert branches[1]['_photomaker_id']['prompt_embeds'] == ['id:1:0:True:mask:1:0', 'id:1:1:True:mask:1:1']
    assert branches[1]['_photomaker_id']['_clora_branch'] == ['blue1', 'blue2']
    assert not active


def test_prepare_capture_binds_composable_branch_metadata_and_strips_network_tags():
    from types import SimpleNamespace
    from sdfx.clora.runtime import Job
    from sdfx.photomaker_batch import prepare_composable_capture
    job = Job.__new__(Job); job.state = None
    branch = {'_clora_branch': ['red person img <lora:a:1>', 'blue person img <lora:b:1>'],
              '_clora_prompt': ['red person img <lora:a:1>', 'blue person img <lora:b:1>'],
              '_clora_negative': ['bad', 'worse']}
    plan = SimpleNamespace(branches=[branch])
    capture = prepare_composable_capture(plan, job, count=2, total=20)
    assert capture.specs[0]['prompts'] == ['red person img ', 'blue person img ']
    with capture.specs[0]['route'](1, True):
        assert job.state[0] == branch['_clora_branch'][1]
    assert job.state is None and capture.plan is plan
    static_branch = type('Static', (dict,), {'schedule': [(20, branch)]})(branch)
    assert prepare_composable_capture(SimpleNamespace(branches=[static_branch]), job, count=2, total=20).plan is not None
    alternate = dict(branch, _clora_branch=['green person img', 'gold person img'])
    branch_with_schedule = type('Scheduled', (dict,), {'schedule': [(1, branch), (20, alternate)]})(branch)
    scheduled = prepare_composable_capture(SimpleNamespace(branches=[branch_with_schedule]), job, count=2, total=20)
    assert [spec['prompts'] for spec in scheduled.specs] == [
        ['red person img ', 'blue person img '], ['green person img', 'gold person img']]
    assert scheduled.groups == [[(1, 0), (20, 1)]]


def test_capture_rejects_missing_cfg_half_before_publishing_plan():
    from types import SimpleNamespace
    from sdfx.photomaker_batch import PhotoBranchCapture
    capture = PhotoBranchCapture([{'prompts': ['person img'], 'route': lambda *args: None, 'metadata': {}}])
    plan = SimpleNamespace(branches=[{'original': True}]); capture.bind(plan)
    capture.trigger = [(['identity'], None, None, None, ['mask'])]
    capture.plain = [(['text'], None, None, None)]
    capture.identity = [['identity']]
    with pytest.raises(ValueError, match='guidance'):
        capture.publish()
    assert plan.branches == [{'original': True}]


def test_unverified_merge_composable_guard_is_narrow_and_fails_closed():
    from sdfx.photomaker_batch import reject_unverified_merge_composable
    reject_unverified_merge_composable(['ordinary ' * 10000, 'bad'])
    with pytest.raises(ValueError, match='Embedding Merge with Composable PhotoMaker'):
        reject_unverified_merge_composable(['ordinary', "negative <'style'>"])


def test_engine_configuration_prepares_batch_and_rejects_unsupported_conditions():
    from types import SimpleNamespace
    from sdfx.clora.runtime import Job
    from sdfx.photomaker_batch import configure_composable_photo
    job = Job.__new__(Job); job.state = None
    branch = {'_clora_branch': ['red person img <lora:a:1>', 'blue person img <lora:b:1>'],
              '_clora_prompt': ['red person img <lora:a:1>', 'blue person img <lora:b:1>'],
              '_clora_negative': ['bad', 'bad']}
    plan = SimpleNamespace(branches=[branch]); kw = {'prompt': 'old', 'negative_prompt': 'bad'}
    capture = configure_composable_photo(plan, job, kw, count=2, total=20, cfg=7, trigger='img')
    assert kw['prompt'] == ['red person img ', 'blue person img ']
    assert kw['negative_prompt'] == ['bad', 'bad'] and kw['num_images_per_prompt'] == 1
    assert capture.plan is plan
    assert configure_composable_photo(plan, job, {}, count=2, total=20, cfg=.5, trigger='img').plan is plan
    branch['_clora_branch'][1] = 'blue person <lora:b:1>'
    with pytest.raises(ValueError, match='trigger'):
        configure_composable_photo(plan, job, {}, count=2, total=20, cfg=7, trigger='img')


def test_capture_retains_schedule_variants_before_selecting_photo_merge_stage():
    from types import SimpleNamespace
    from sdfx.clora.runtime import Job
    from sdfx.neutral_runtime import at
    from sdfx.photomaker_batch import prepare_composable_capture, photo_branch_transform
    job = Job.__new__(Job); job.state = None
    def item(text):
        return {'_clora_branch': text, '_clora_prompt': text, '_clora_negative': 'bad'}
    initial, later = item('red img'), item('blue img')
    branch = type('Scheduled', (dict,), {'schedule': [(1, initial), (3, later)]})(initial)
    plan = SimpleNamespace(branches=[branch])
    capture = prepare_composable_capture(plan, job, count=1, total=3)
    capture.trigger = [(['id red'], ['negative red'], ['pool red'], ['np red'], ['mask']),
                       (['id blue'], ['negative blue'], ['pool blue'], ['np blue'], ['mask'])]
    capture.plain = [(['plain red'], ['negative red'], ['tp red'], ['np red']),
                     (['plain blue'], ['negative blue'], ['tp blue'], ['np blue'])]
    capture.identity = [['fused red'], ['fused blue']]
    capture.publish()
    branch = plan.branches[0]
    select = photo_branch_transform(0)
    assert select(0, at(branch, 1), 1)['prompt_embeds'] == ['plain red']
    assert select(0, at(branch, 2), 2)['prompt_embeds'] == ['fused blue']
    assert select(0, at(branch, 2), 2)['_clora_branch'] == 'blue img'


def test_capture_expands_scheduled_text_and_negative_at_each_timeline_endpoint():
    from types import SimpleNamespace
    from sdfx.clora.runtime import Job
    from sdfx.photomaker_batch import prepare_composable_capture
    job = Job.__new__(Job); job.state = None
    original = {'_clora_branch': '[red:blue:0.5] person img <lora:a:1>',
                '_clora_prompt': '[red:blue:0.5] person img <lora:a:1>',
                '_clora_negative': '[bad:worse:0.5]'}
    branch = type('Scheduled', (dict,), {'schedule': [(2, original), (4, original)]})(original)
    capture = prepare_composable_capture(SimpleNamespace(branches=[branch]), job, count=1, total=4, schedule_steps=4)
    assert [s['prompts'] for s in capture.specs] == [['red person img '], ['blue person img ']]
    assert [s['negatives'] for s in capture.specs] == [['bad'], ['worse']]
    assert capture.groups == [[(2, 0), (4, 1)]]


def test_photo_schedule_expansion_does_not_parse_edits_inside_merge_expression():
    from types import SimpleNamespace
    from sdfx.clora.runtime import Job
    from sdfx.photomaker_batch import prepare_composable_capture
    job = Job.__new__(Job); job.state = None
    source = "[red:blue:0.5] <'a[b:c:0.5]'> person img"
    value = {'_clora_branch': source, '_clora_prompt': source, '_clora_negative': 'bad'}
    branch = type('Scheduled', (dict,), {'schedule': [(2, value), (4, value)]})(value)
    capture = prepare_composable_capture(SimpleNamespace(branches=[branch]), job, count=1, total=4, schedule_steps=4)
    assert [s['prompts'] for s in capture.specs] == [["red <'a[b:c:0.5]'> person img"],
                                                     ["blue <'a[b:c:0.5]'> person img"]]
