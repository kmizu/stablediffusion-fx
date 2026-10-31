"""Real generation control flow, with expensive model operations substituted."""
from types import SimpleNamespace
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('diffusers')
from PIL import Image
from test_hires_repeat_engine import harness, context
from sdfx.protocol import EngineError, INTERRUPTED


def setup_keyword(harness, tmp_path, monkeypatch):
    engine, request, calls = harness
    checkpoint = tmp_path / 'model.safetensors'; checkpoint.write_bytes(b'model')
    engine.model['path'] = str(checkpoint)
    directory = str(tmp_path / 'config')
    engine.model_keyword({'action': 'save', 'kind': 'checkpoint', 'model_path': str(checkpoint),
        'mapping_dir': directory, 'keywords': 'red|blue'}, None)
    request['params']['prompt'] = 'cat'
    request['params']['hooks']['model_keyword'] = {'_mapping_dir': directory, 'model_choice': 'iterate'}
    seen = []
    def encode(pos, neg, gp, bs, steps, *args, **kwargs):
        seen.append((pos, bs, args))
        value = 1. if pos.startswith('red') else 2. if pos.startswith('blue') else 3.
        return {'prompt_embeds': torch.full((bs, 1, 1), value),
                'negative_prompt_embeds': torch.zeros((bs, 1, 1))}
    monkeypatch.setattr(engine, '_encode_scheduled', encode)
    monkeypatch.setattr(engine, '_encode_plain', lambda *args: {'prompt_embeds': torch.zeros((1, 1, 1))})
    return engine, request, calls, seen


def test_generation_batches_distinct_prompts_and_preserves_hires_fallback(harness, tmp_path, monkeypatch):
    engine, request, calls, seen = setup_keyword(harness, tmp_path, monkeypatch)
    result = engine.txt2img(request, context())
    base = [call for call in calls if isinstance(call, dict) and call['pass_'] == 'base']
    assert [call['embeds']['prompt_embeds'][:, 0, 0].tolist() for call in base] == [[1., 2.], [1., 2.]]
    assert [image['model_keyword_source_prompt'] for image in result['images'][:2]] == ['red cat', 'blue cat']
    assert any(prompt == 'cat' and bs == 2 and args for prompt, bs, args in seen)


def test_keyword_adetailer_receives_per_image_prompt_and_preserves_completed_item(harness, tmp_path, monkeypatch):
    engine, request, calls, seen = setup_keyword(harness, tmp_path, monkeypatch)
    request['params']['hires'] = {'enabled': False}
    request['params']['hooks'].pop('hires_fix_tweaks')
    request['params']['hooks']['adetailer'] = {'units': [{'model': 'opencv_face'}]}
    prompts = []
    def detail(gp, req, ctx, cfg, imgs, seeds, idx, total, pag):
        prompts.append((gp.prompt, seeds[0]))
        if len(prompts) == 2: raise EngineError(INTERRUPTED, 'cancel second')
        return imgs
    monkeypatch.setattr(engine, '_adetailer', detail)
    result = engine.txt2img(request, context())
    assert prompts == [('red cat', 10), ('blue cat', 11)]
    assert result['interrupted'] and len(result['images']) == 1


def test_composable_job_is_not_recreated_between_keyword_batches(harness, tmp_path, monkeypatch):
    from contextlib import nullcontext
    engine, request, calls, seen = setup_keyword(harness, tmp_path, monkeypatch)
    request['params']['hires'] = {'enabled': False}
    request['params']['hooks'].pop('hires_fix_tweaks')
    request['params']['hooks']['composable_lora'] = {}
    applied = []
    class Job:
        config = {}
        def using(self, *args, **kwargs): return nullcontext()
        def finish(self, *args): pass
        def remove(self): pass
    def apply(gp, req, ctx):
        applied.append([ref.name for ref in gp.loras]); engine._composable_job = Job(); return {}
    monkeypatch.setattr(engine, '_apply_loras', apply)
    result = engine.txt2img(request, context())
    assert len(applied) == 1
    assert [image['effective_prompt'] for image in result['images']] == ['red cat', 'blue cat', 'red cat', 'blue cat']


def test_img2imgalt_receives_single_inversion_and_per_image_sampling(harness, tmp_path, monkeypatch):
    engine, request, calls, seen = setup_keyword(harness, tmp_path, monkeypatch)
    source = tmp_path / 'source.png'; Image.new('RGB', (64, 64), 'red').save(source)
    request['params']['img2img'] = {'init_image': str(source), 'denoising_strength': .5}
    request['params']['hires'] = {'enabled': False}
    request['params']['hooks'].pop('hires_fix_tweaks')
    request['params']['hooks']['img2imgalt'] = {}
    captured = []
    def alternate(gp, req, ctx, alt, image, embeds, seeds, idx, total, w, h, cn, pag, inverse_embeds=None):
        captured.append((embeds['prompt_embeds'][:, 0, 0].tolist(), inverse_embeds['prompt_embeds'].shape[0]))
        return [Image.new('RGB', (w, h)) for _ in seeds], SimpleNamespace(schedule='Normal')
    monkeypatch.setattr(engine, '_img2imgalt', alternate)
    result = engine.img2img(request, context())
    assert captured == [([1., 2.], 1), ([1., 2.], 1)]
    assert len(result['images']) == 4


def test_keyword_composable_img2img_does_not_finalize_adapter_as_inpaint_job(harness, tmp_path, monkeypatch):
    from contextlib import nullcontext
    engine, request, calls, _ = setup_keyword(harness, tmp_path, monkeypatch)
    source = tmp_path / 'source.png'; Image.new('RGB', (64, 64), 'red').save(source)
    request['params']['img2img'] = {'init_image': str(source), 'denoising_strength': .5}
    request['params']['hires'] = {'enabled': False}
    request['params']['hooks'].pop('hires_fix_tweaks')
    request['params']['hooks'].update({'composable_lora': {}, 'img2imgalt': {}})
    class AdapterJob:
        config = {}
        def using(self, *args, **kwargs): return nullcontext()
        def finish(self, *args): pass
        def remove(self): pass
    def apply(gp, req, ctx):
        engine._composable_job = AdapterJob()
        return {}
    monkeypatch.setattr(engine, '_apply_loras', apply)
    monkeypatch.setattr(engine, '_img2imgalt', lambda gp, req, ctx, alt, image, embeds, seeds, idx, total, w, h, cn, pag, inverse_embeds=None:
                        ([Image.new('RGB', (w, h)) for _ in seeds], SimpleNamespace(schedule='Normal')))
    result = engine.img2img(request, context())
    assert [image['effective_prompt'] for image in result['images']] == ['red cat', 'blue cat', 'red cat', 'blue cat']


def test_photomaker_single_image_pass_receives_effective_prompt(harness, tmp_path, monkeypatch):
    engine, request, calls, seen = setup_keyword(harness, tmp_path, monkeypatch)
    request['params']['batch_size'] = 1
    request['params']['hires'] = {'enabled': False}
    request['params']['hooks'].pop('hires_fix_tweaks')
    request['params']['hooks']['photomaker'] = {}
    prompts = []
    original = engine._run
    def run(task, gp, *args, **kwargs):
        prompts.append(gp.prompt)
        return original(task, gp, *args, **kwargs)
    monkeypatch.setattr(engine, '_run', run)
    engine.txt2img(request, context())
    assert prompts == ['red cat', 'blue cat']


def test_photomaker_receives_distinct_keyword_prompts_in_one_batch(harness, tmp_path, monkeypatch):
    engine, request, calls, seen = setup_keyword(harness, tmp_path, monkeypatch)
    request['params']['hires'] = {'enabled': False}
    request['params']['hooks'].pop('hires_fix_tweaks')
    request['params']['hooks']['photomaker'] = {}
    batches = []
    original = engine._run
    def run(task, gp, *args, **kwargs):
        batches.append((kwargs.get('photomaker_prompts'), kwargs['seeds']))
        return original(task, gp, *args, **kwargs)
    monkeypatch.setattr(engine, '_run', run)
    engine.txt2img(request, context())
    assert len(batches) == 2 and all(batch[0] == ['red cat', 'blue cat'] for batch in batches)
    assert all(len(batch[1]) == 2 for batch in batches)


def test_keyword_inserted_embedding_is_in_texts_for_embedding_loader(harness, tmp_path, monkeypatch):
    engine, request, calls, seen = setup_keyword(harness, tmp_path, monkeypatch)
    request['params']['hires'] = {'enabled': False}
    request['params']['hooks'].pop('hires_fix_tweaks')
    request['params']['hooks']['model_keyword']['embedding'] = 'keyword_embedding'
    captured = []
    monkeypatch.setattr(engine, '_load_embeddings', lambda req, texts, ctx: captured.extend(texts))
    result = engine.txt2img(request, context())
    assert any('keyword_embedding red cat' in value for value in captured)
    assert any('keyword_embedding blue cat' in value for value in captured)
    assert all('keyword_embedding' in item['effective_prompt'] for item in result['images'])


def test_img2img_hires_receives_keyword_preview_and_keeps_base_prompts(harness, tmp_path, monkeypatch):
    from sdfx import img2img_hires_fix
    engine, request, calls, seen = setup_keyword(harness, tmp_path, monkeypatch)
    request['params']['hires'] = {'enabled': False}
    request['params']['hooks'].pop('hires_fix_tweaks')
    request['params']['hooks']['img2img_hires_fix'] = {'upscaler': 'Lanczos', 'ratio': 2}
    prompts = []
    def refine(engine, gp, req, ctx, cfg, image, seed, index, total, *args):
        prompts.append(gp.prompt); return image
    monkeypatch.setattr(img2img_hires_fix, 'refine', refine)
    result = engine.txt2img(request, context())
    assert prompts == ['red cat'] * 4
    assert [image['model_keyword_refinement_prompt'] for image in result['images']] == prompts
    assert [image['effective_prompt'] for image in result['images']] == ['red cat', 'blue cat', 'red cat', 'blue cat']


def test_keyword_dynamic_rank_changes_between_generation_batches(harness, tmp_path, monkeypatch):
    engine, request, calls, seen = setup_keyword(harness, tmp_path, monkeypatch)
    checkpoint = engine.model['path']; directory = str(tmp_path / 'config')
    adapter = tmp_path / 'A.safetensors'; adapter.write_bytes(b'fixture')
    engine.model_keyword({'action': 'save', 'kind': 'checkpoint', 'model_path': checkpoint,
        'mapping_dir': directory, 'keywords': 'red <lora:A:1:1:1>|blue <lora:A:1:1:2>'}, None)
    request['params']['batch_size'] = 1
    request['params']['hires'] = {'enabled': False}
    request['params']['hooks'].pop('hires_fix_tweaks')
    request['lora_paths'] = {'A': str(adapter)}
    applied = []
    monkeypatch.setattr(engine, '_apply_loras', lambda gp, req, ctx: applied.append([(r.name, r.dyn) for r in gp.loras]) or {})
    result = engine.txt2img(request, context())
    assert applied[-2:] == [[('A', 1)], [('A', 2)]]
    assert [image['effective_prompt'] for image in result['images']] == [
        '<lora:A:1:1:1>red  cat', '<lora:A:1:1:2>blue  cat']
