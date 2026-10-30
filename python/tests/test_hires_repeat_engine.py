"""Generation orchestration with stub model calls: source reuse, ordering and files."""
import io
from types import SimpleNamespace
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('diffusers')
from PIL import Image
from sdfx.engine import Engine
from sdfx.protocol import Emitter, JobContext, EngineError, INTERRUPTED


@pytest.fixture
def harness(monkeypatch, tmp_path):
    engine = Engine()
    engine.model = dict(name='fixture', hash='fixture')
    engine.kind = 'sd15'
    engine.vae_info = None
    engine.loaded_embedding_files = {}
    engine._intermediate_recorder = None
    calls = []
    monkeypatch.setattr(engine, 'ensure_model', lambda *a: calls.append('load'))
    monkeypatch.setattr(engine, '_apply_loras', lambda *a: {})
    monkeypatch.setattr(engine, '_load_embeddings', lambda *a: None)
    monkeypatch.setattr(engine, '_encode_scheduled', lambda *a, **kw: {})
    monkeypatch.setattr(engine, '_adjust_embeds', lambda gp, embeds, ctx: embeds)
    monkeypatch.setattr(engine, '_controlnets', lambda *a: ([], [], {}))
    monkeypatch.setattr(engine, '_noise', lambda *a: torch.zeros(2, 4, 8, 8))
    monkeypatch.setattr(engine, '_pag_config', lambda *a: None)
    def run(task, gp, req, ctx, **kw):
        calls.append(dict(task=task, **kw))
        return [Image.new('RGB', (kw['w'], kw['h']), (s % 256, 0, 0)) for s in kw['seeds']], SimpleNamespace(schedule='Normal')
    monkeypatch.setattr(engine, '_run', run)
    request = dict(params=dict(seed=10, subseed=20, steps=2, width=64, height=64,
        batch_size=2, n_iter=2, hires=dict(enabled=True, scale=2, upscaler='Lanczos'),
        hooks=dict(hires_fix_tweaks=dict(batch_count=3, seed_enabled=True, seed=100))),
        output_dir=str(tmp_path), preview_method='none')
    return engine, request, calls


def context():
    return JobContext(Emitter(io.BytesIO()), 'repeat-test')


def test_base_once_cached_hires_source_seed_order_and_metadata(harness):
    engine, request, calls = harness
    result = engine.txt2img(request, context())
    runs = [x for x in calls if isinstance(x, dict)]
    assert [x['pass_'] for x in runs] == ['base', 'hires', 'hires', 'hires'] * 2
    assert runs[1]['image'] is runs[2]['image'] is runs[3]['image']
    assert runs[5]['image'] is runs[6]['image'] is runs[7]['image']
    assert [list(x['seeds']) for x in runs if x['pass_'] == 'hires'] == [
        [100, 101], [101, 102], [102, 103], [102, 103], [103, 104], [104, 105]]
    assert [x['seed'] for x in result['images']] == [10, 11] * 3 + [12, 13] * 3
    assert [x['hires_seed'] for x in result['images']] == [100, 101, 101, 102, 102, 103, 102, 103, 103, 104, 104, 105]
    assert len(set(x['path'] for x in result['images'])) == 12
    for x in result['images']:
        with Image.open(x['path']) as im:
            assert im.size == (128, 128)
            assert im.getpixel((0, 0))[0] == x['hires_seed']


def test_excessive_output_rejected_before_loading(harness):
    engine, request, calls = harness
    request['params'].update(batch_size=16, n_iter=100)
    with pytest.raises(EngineError):
        engine.txt2img(request, context())
    assert calls == []


def test_completed_repeat_files_survive_interruption(harness, monkeypatch, tmp_path):
    engine, request, calls = harness
    original = engine._run
    def interrupted(*a, **kw):
        if kw['pass_'] == 'hires' and kw['image_idx'] == 2:
            raise EngineError(INTERRUPTED, 'fixture interruption')
        return original(*a, **kw)
    monkeypatch.setattr(engine, '_run', interrupted)
    result = engine.txt2img(request, context())
    assert result['interrupted'] is True
    assert len(result['images']) == 2
    assert len(list(tmp_path.glob('*.png'))) == 2
    assert [x['pass_'] for x in calls if isinstance(x, dict)] == ['base', 'hires']


@pytest.mark.parametrize('cancel_at', ['detector', 'inpaint'])
def test_completed_adetailer_item_survives_cancellation_of_next_item(harness, monkeypatch, tmp_path, cancel_at):
    from sdfx import adetailer
    engine, request, calls = harness
    request['params']['hooks']['adetailer'] = dict(units=[dict(model='opencv_face')])
    seen = []
    def masks(image, *args):
        seen.append(image)
        if len(seen) == 2 and cancel_at == 'detector':
            raise EngineError(INTERRUPTED, 'cancel during second detector')
        return [Image.new('L', image.size, 255)]
    monkeypatch.setattr(adetailer, 'unit_masks', masks)
    original = engine._run
    def run(*args, **kwargs):
        if cancel_at == 'inpaint' and kwargs['pass_'] == 'adetailer' and kwargs['image_idx'] == 1:
            raise EngineError(INTERRUPTED, 'cancel during second inpaint')
        return original(*args, **kwargs)
    monkeypatch.setattr(engine, '_run', run)
    result = engine.txt2img(request, context())
    assert result['interrupted'] is True
    assert len(result['images']) == 1
    image = result['images'][0]
    assert image['seed'] == 10 and image['hires_seed'] == 100
    assert len(list(tmp_path.glob('*.png'))) == 1
    assert Image.open(image['path']).size == (128, 128)
    runs = [x for x in calls if isinstance(x, dict) and x['pass_'] == 'adetailer']
    assert [(x['image_idx'], x['seeds']) for x in runs] == [(0, [100])]


def test_per_item_adetailer_keeps_seed_and_progress_order_across_batches(harness, monkeypatch):
    from sdfx import adetailer
    engine, request, calls = harness
    request['params']['hooks']['adetailer'] = dict(units=[dict(model='opencv_face')])
    monkeypatch.setattr(adetailer, 'unit_masks', lambda image, *args: [Image.new('L', image.size, 255)])
    result = engine.txt2img(request, context())
    runs = [x for x in calls if isinstance(x, dict) and x['pass_'] == 'adetailer']
    assert len(result['images']) == len(runs) == 12
    assert [x['image_idx'] for x in runs] == list(range(12))
    assert [x['seeds'][0] for x in runs] == [im['hires_seed'] for im in result['images']]


@pytest.mark.parametrize('code,fail_first', [('interrupted', True), ('oom', False)])
def test_empty_interruption_and_non_interruption_errors_still_raise(harness, monkeypatch, code, fail_first):
    engine, request, calls = harness
    original = engine._run
    def failing(*a, **kw):
        if kw['pass_'] == 'hires' and (fail_first or kw['image_idx'] == 2):
            raise EngineError(code, 'fixture')
        return original(*a, **kw)
    monkeypatch.setattr(engine, '_run', failing)
    with pytest.raises(EngineError) as error:
        engine.txt2img(request, context())
    assert error.value.code == code


def test_latent_source_stays_latent_and_is_upscaled_once_per_base(harness, monkeypatch):
    engine, request, calls = harness
    request['params']['hires']['upscaler'] = 'Latent'
    original = engine._run
    def run(*a, **kw):
        result = original(*a, **kw)
        if kw['pass_'] == 'base':
            assert kw['output_type'] == 'latent'
            return torch.ones(2, 4, 8, 8), result[1]
        assert isinstance(kw['image'], torch.Tensor)
        assert kw['image'].shape == (2, 4, 16, 16)
        return result
    monkeypatch.setattr(engine, '_run', run)
    result = engine.txt2img(request, context())
    assert len(result['images']) == 12
    runs = [x for x in calls if isinstance(x, dict)]
    assert runs[1]['image'] is runs[2]['image'] is runs[3]['image']


def test_disabled_and_default_tweaks_preserve_ordinary_pixels(harness):
    engine, request, calls = harness
    request['params']['hooks'] = {}
    ordinary = engine.txt2img(request, context())
    request['params']['hooks'] = dict(hires_fix_tweaks={})
    enabled = engine.txt2img(request, context())
    assert len(ordinary['images']) == len(enabled['images']) == 4
    for a, b in zip(ordinary['images'], enabled['images']):
        assert Image.open(a['path']).tobytes() == Image.open(b['path']).tobytes()
        assert a['seed'] == b['seed']
    assert all(x.get('hires_noise') is None for x in calls if isinstance(x, dict))


def test_variation_repeats_keep_primary_and_pass_real_noise_config(harness):
    engine, request, calls = harness
    request['params']['hooks']['hires_fix_tweaks'].update(extras=True, subseed=200,
        strength=.5, resize_width=64, resize_height=64)
    result = engine.txt2img(request, context())
    runs = [x for x in calls if isinstance(x, dict) and x['pass_'] == 'hires']
    assert all(tuple(x['seeds']) == (100, 100) for x in runs)
    assert runs[0]['hires_noise'] == ((200, 201), .5, 64, 64)
    assert runs[-1]['hires_noise'] == ((204, 205), .5, 64, 64)
    assert all(x['hires_subseed_strength'] == .5 for x in result['images'])


def test_resolved_empty_hires_prompts_do_not_fall_back_to_base(harness, monkeypatch):
    engine, request, calls = harness
    request['params'].update(prompt='base', negative_prompt='base negative')
    request['params']['hooks']['hires_fix_tweaks']['prompts_resolved'] = True
    encoded = []
    def encode(pos, neg, *a, **kw):
        encoded.append((pos, neg))
        return {}
    monkeypatch.setattr(engine, '_encode_scheduled', encode)
    engine.txt2img(request, context())
    assert encoded == [('base', 'base negative'), ('', '')]


def test_pass_loras_apply_during_encoding_sampling_and_restore_after_interrupt(harness, monkeypatch, tmp_path):
    engine, request, calls = harness
    for name in ('base', 'hr'):
        (tmp_path / name).write_bytes(b'fixture')
    request['lora_paths'] = {name: str(tmp_path / name) for name in ('base', 'hr')}
    request['params']['loras'] = [dict(name='base')]
    request['hires_loras'] = [dict(name='hr')]
    active, encoded, sampled = [], [], []
    def apply(gp, *a):
        active[:] = [x.name for x in gp.loras]
        return {x.name: 'hash' for x in gp.loras}
    monkeypatch.setattr(engine, '_apply_loras', apply)
    def encode(*a, **kw):
        encoded.append(tuple(active))
        return {}
    monkeypatch.setattr(engine, '_encode_scheduled', encode)
    original = engine._run
    def run(*a, **kw):
        sampled.append((kw['pass_'], tuple(active)))
        if kw['pass_'] == 'hires' and kw['image_idx'] == 2:
            raise EngineError(INTERRUPTED, 'fixture')
        return original(*a, **kw)
    monkeypatch.setattr(engine, '_run', run)
    result = engine.txt2img(request, context())
    assert encoded == [('base',), ('hr',)]
    assert sampled == [('base', ('base',)), ('hires', ('hr',)), ('hires', ('hr',))]
    assert active == ['base']
    assert result['lora_hashes'] == {'base': 'hash', 'hr': 'hash'}
