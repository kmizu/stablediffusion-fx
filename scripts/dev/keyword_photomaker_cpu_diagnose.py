"""CPU-only differential of pinned official v1 trigger encoder and merge rows.

Uses cached real tokenizer vocabularies and tiny randomly initialized CLIP models;
no checkpoint, ID encoder, denoising, GPU, downloads or quality claims.
"""
import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.loaders import TextualInversionLoaderMixin
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from sdfx.embedding_merge_pipeline import run_pipeline


source = Path(os.environ['SDFX_PHOTOMAKER_SOURCE']) / 'photomaker' / 'pipeline.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'PhotoMakerStableDiffusionXLPipeline')
method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'encode_prompt_with_trigger_word')
method.returns = None
for arg in method.args.posonlyargs + method.args.args + method.args.kwonlyargs:
    arg.annotation = None
module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
namespace = {'torch': torch, 'TextualInversionLoaderMixin': TextualInversionLoaderMixin,
             'USE_PEFT_BACKEND': True, 'StableDiffusionXLLoraLoaderMixin': type('Unused', (), {})}
exec(compile(module, str(source), 'exec'), namespace)


class Pipeline(TextualInversionLoaderMixin):
    hf_device_map = None
    components = {}
    _execution_device = torch.device('cpu')
    config = SimpleNamespace(force_zeros_for_empty_prompt=False)
    trigger_word = 'img'
    num_tokens = 1
    encode_prompt_with_trigger_word = namespace['encode_prompt_with_trigger_word']
    encode_prompt = StableDiffusionXLPipeline.encode_prompt


def encoder(tokenizer, projected):
    config = CLIPTextConfig(vocab_size=len(tokenizer), hidden_size=8, intermediate_size=16,
                            num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=77,
                            eos_token_id=2 if os.environ.get('SDFX_TEST_CLIP_LEGACY_EOS') == '1' else tokenizer.eos_token_id,
                            projection_dim=8)
    return (CLIPTextModelWithProjection if projected else CLIPTextModel)(config).eval()


torch.manual_seed(173)
pipe = Pipeline()
pipe.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14', local_files_only=True)
pipe.tokenizer_2 = CLIPTokenizer.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0',
                                                 subfolder='tokenizer_2', local_files_only=True)
pipe.text_encoder = encoder(pipe.tokenizer, False)
pipe.text_encoder_2 = encoder(pipe.tokenizer_2, True)
for tokenizer in (pipe.tokenizer, pipe.tokenizer_2):
    tokenizer.add_tokens(['img'], special_tokens=True)
engine = SimpleNamespace(pipe=pipe, device=torch.device('cpu'), embedding_token_owners={},
                         _with_embedding_encoder_access=lambda operation: operation(),
                         _activate=lambda pipe: None, _free=lambda: None)


def evaluate(texts):
    with torch.no_grad():
        return pipe.encode_prompt_with_trigger_word(prompt=texts[0], negative_prompt=texts[1],
            num_id_images=1, num_images_per_prompt=1, do_classifier_free_guidance=True)


def experimental_corrected_pool(operation, texts):
    model = pipe.text_encoder_2.text_model
    previous = model.eos_token_id
    try:
        if os.environ.get('SDFX_TEST_PHOTO_EOS_FIX') == '1' and previous == 2:
            model.eos_token_id = pipe.tokenizer_2.eos_token_id
        return operation(texts)
    finally:
        model.eos_token_id = previous


baseline = evaluate(('red portrait of a woman img', 'lowres'))
actual = run_pipeline(engine, pipe, ("red portrait of a <'woman'> img", "<'lowres'>"),
                      lambda texts: experimental_corrected_pool(evaluate, texts), check=lambda: None)
for name, left, right in zip(('positive', 'negative', 'positive_pooled', 'negative_pooled', 'identity_mask'), baseline, actual):
    print(json.dumps({'name': name, 'shape': list(left.shape), 'equal': bool(torch.equal(left, right)),
                      'max_abs': float((left.float() - right.float()).abs().max())}))


def text_only(texts):
    ids = pipe.tokenizer.encode(texts[0], add_special_tokens=False)
    ids.remove(pipe.tokenizer.convert_tokens_to_ids('img'))
    decoded = pipe.tokenizer.decode(ids, add_special_tokens=False)
    with torch.no_grad():
        return pipe.encode_prompt(prompt=decoded, negative_prompt=texts[1], num_images_per_prompt=1,
                                  do_classifier_free_guidance=True)


base_plain = text_only(('red portrait of a woman img', 'lowres'))
merged_plain = run_pipeline(engine, pipe, ("red portrait of a <'woman'> img", "<'lowres'>"),
                            lambda texts: experimental_corrected_pool(text_only, texts), check=lambda: None)
for name, left, right in zip(('plain_positive', 'plain_negative', 'plain_positive_pooled', 'plain_negative_pooled'), base_plain, merged_plain):
    print(json.dumps({'name': name, 'shape': list(left.shape), 'equal': bool(torch.equal(left, right)),
                      'max_abs': float((left.float() - right.float()).abs().max())}))


# Reproduce #85's two-image, two-branch and two-schedule conditioning route
# through the actual pinned trigger encoder and installed SDXL text encoder.
# Fake identity fusion is deterministic; no trained ID model or UNet is loaded.
from sdfx.clora.runtime import Job
from sdfx.photomaker_batch import prepare_composable_capture, heterogeneous_prompts
from sdfx.neutral_runtime import Plan
from sdfx.neutral_prompt import parse

class Identity(torch.nn.Module):
    def forward(self, images, embeds, mask):
        return embeds + mask.unsqueeze(-1).to(embeds.dtype)

pipe.id_encoder = Identity()
job = Job.__new__(Job); job.state = None
def capture_batch(merge):
    def metadata(first, second):
        return {'_clora_branch': [first, second], '_clora_prompt': [first, second],
                '_clora_negative': ["<'lowres'>" if merge else 'lowres'] * 2}
    woman = "<'woman'>" if merge else 'woman'
    items = [metadata(f'{colour} [portrait:photo:0.5] of a {woman} img <lora:base:0.8>',
                      f'{other} [portrait:photo:0.5] of a {woman} img <lora:base:0.8>')
             for colour, other in (('red', 'blue'),)]
    second = metadata(f'a {woman} img, upper body', f'a {woman} img, upper body')
    class Scheduled(dict): pass
    branches = []
    for item in (items[0], second):
        branch = Scheduled(item); branch.schedule = [(4, item), (8, item)]; branches.append(branch)
    plan = Plan(parse('one AND two'), branches)
    capture = prepare_composable_capture(plan, job, count=2, total=8, schedule_steps=8)
    original = [(list(spec['prompts']), list(spec['negatives'])) for spec in capture.specs]
    positive, negative = original[0]
    texts = positive + negative + [value for ps, ns in original for value in ps + ns]
    def invoke(resolved):
        index = len(positive) + len(negative)
        for spec, (ps, ns) in zip(capture.specs, original):
            spec['prompts'] = list(resolved[index:index + len(ps)]); index += len(ps)
            spec['negatives'] = list(resolved[index:index + len(ns)]); index += len(ns)
        first = capture.specs[0]
        with heterogeneous_prompts(pipe, first['prompts'], route=first['route'], capture=capture):
            trigger = pipe.encode_prompt_with_trigger_word(prompt=first['prompts'],
                negative_prompt=list(resolved[len(positive):len(positive) + len(negative)]),
                num_id_images=1, num_images_per_prompt=1, do_classifier_free_guidance=True)
            pipe.encode_prompt(prompt=first['prompts'], negative_prompt=list(resolved[len(positive):len(positive) + len(negative)]),
                negative_prompt_embeds=trigger[1], negative_pooled_prompt_embeds=trigger[3],
                num_images_per_prompt=1, do_classifier_free_guidance=True)
            pipe.id_encoder(torch.zeros(1), trigger[0], trigger[4])
            return plan.branches
    with torch.no_grad():
        return run_pipeline(engine, pipe, texts, lambda resolved: experimental_corrected_pool(invoke, resolved) if merge else invoke(resolved), check=lambda: None)

torch.set_num_threads(2)
before, after = capture_batch(False), capture_batch(True)
for branch_index, (left, right) in enumerate(zip(before, after)):
    for (end, a), (_, b) in zip(left.schedule, right.schedule):
        for stage in ('text', 'identity'):
            av, bv = (a, b) if stage == 'text' else (a['_photomaker_id'], b['_photomaker_id'])
            for key in ('prompt_embeds', 'negative_prompt_embeds', 'pooled_prompt_embeds', 'negative_pooled_prompt_embeds'):
                equal = torch.equal(av[key], bv[key]); diff = float((av[key].float() - bv[key].float()).abs().max())
                print(json.dumps({'branch': branch_index, 'end': end, 'stage': stage, 'key': key,
                                  'equal': bool(equal), 'max_abs': diff}))

# Exercise the real Composable forward-hook machinery on the two tiny text
# encoders using a synthetic, deterministic delta and per-branch tag weights.
job.ctx = SimpleNamespace(check_interrupt=lambda: None)
job.config, job.trace, job.adapters = {}, {}, {}
job._weight = lambda adapter, patch, layer, text, negative, step, total, phase, full, neg: (
    0.8 if '<lora:base:' in text and not negative else 0.0)
adapter = SimpleNamespace(name='base')
patch = SimpleNamespace(delta=lambda x: x.new_ones((*x.shape[:-1], 16)))
handles = []
for component, encoder in (('text_encoder', pipe.text_encoder), ('text_encoder_2', pipe.text_encoder_2)):
    target = encoder.text_model.encoder.layers[0].mlp.fc1
    handles.append(target.register_forward_hook(job._hook(component, 'synthetic', adapter, patch)))
try:
    with_hooks_before, with_hooks_after = capture_batch(False), capture_batch(True)
finally:
    for handle in handles:
        handle.remove()
for branch_index, (left, right) in enumerate(zip(with_hooks_before, with_hooks_after)):
    for (end, a), (_, b) in zip(left.schedule, right.schedule):
        for stage in ('text', 'identity'):
            av, bv = (a, b) if stage == 'text' else (a['_photomaker_id'], b['_photomaker_id'])
            for key in ('prompt_embeds', 'negative_prompt_embeds', 'pooled_prompt_embeds', 'negative_pooled_prompt_embeds'):
                print(json.dumps({'hooked': True, 'branch': branch_index, 'end': end, 'stage': stage, 'key': key,
                                  'equal': bool(torch.equal(av[key], bv[key])),
                                  'max_abs': float((av[key].float() - bv[key].float()).abs().max())}))
