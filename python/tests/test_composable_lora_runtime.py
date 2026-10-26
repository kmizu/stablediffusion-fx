from types import SimpleNamespace
import pytest
torch = pytest.importorskip('torch')


def fixture_job(tmp_path, prompt=None, hires=None, **config):
    from safetensors.torch import save_file
    from sdfx.clora.runtime import Job
    from sdfx.params import GenerationParams, GenerationRequest, LoraRef, HiresParams
    pipe = SimpleNamespace(unet=torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False)),
        text_encoder=torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False)), text_encoder_2=None)
    for model in (pipe.unet, pipe.text_encoder):
        model[0].weight.data.copy_(torch.eye(2))
    path = tmp_path / 'a.safetensors'
    save_file({f'{prefix}_0.{key}': torch.eye(2) for prefix in ('lora_unet', 'lora_te')
               for key in ('lora_down.weight', 'lora_up.weight')}, str(path))
    gp = GenerationParams(prompt=prompt or '<lora:A:1> red AND blue', loras=[LoraRef('A')],
                          hooks={'composable_lora': config}, hires=hires or HiresParams())
    req = GenerationRequest(params=gp, lora_paths={'A': str(path)})
    ctx = SimpleNamespace(check_interrupt=lambda: None, warn=lambda _: None)
    return pipe, Job(pipe, gp, req, ctx)


def test_real_module_hooks_isolate_cfg_rows_and_restore_exact_weights(tmp_path):
    pipe, job = fixture_job(tmp_path)
    original = pipe.unet[0].weight.detach().clone()
    x = torch.ones(2, 2)
    with job.using('<lora:A:1> red', [False, True], 1, 10):
        torch.testing.assert_close(pipe.unet(x), torch.tensor([[1., 1.], [2., 2.]]))
    with job.using('blue', [False, True], 1, 10):
        torch.testing.assert_close(pipe.unet(x), x)
    job.remove()
    torch.testing.assert_close(pipe.unet(x), x, rtol=0, atol=0)
    torch.testing.assert_close(pipe.unet[0].weight, original, rtol=0, atol=0)
    assert not pipe.unet[0]._forward_hooks


def test_negative_controls_and_single_block_exception_are_independent(tmp_path):
    pipe, job = fixture_job(tmp_path, uc_diffusion_model=True)
    with job.using('<lora:A:1> red', [False, True], 1, 10):
        torch.testing.assert_close(pipe.unet(torch.ones(2, 2)), torch.full((2, 2), 2.))
    with job.using('<lora:A:1> red', [False], -1, 10, phase='text_encoder'):
        torch.testing.assert_close(pipe.text_encoder(torch.ones(1, 2)), torch.ones(1, 2))
    job.remove()


def test_exception_does_not_leave_active_branch_or_hooks(tmp_path):
    pipe, job = fixture_job(tmp_path)
    with pytest.raises(RuntimeError):
        with job.using('<lora:A:1>', [True], 1, 10):
            raise RuntimeError('cancelled')
    torch.testing.assert_close(pipe.unet(torch.ones(1, 2)), torch.ones(1, 2))
    job.remove()


def test_native_engine_encoding_retains_branch_scopes_and_each_negative_encoder(tmp_path):
    from sdfx.engine import Engine
    pipe, job = fixture_job(tmp_path)
    engine = Engine.__new__(Engine)
    engine._composable_job = job
    engine.dtype = torch.float32
    def encode(p, n, gp, bs):
        return {'prompt_embeds': pipe.text_encoder(torch.ones(bs, 2)).unsqueeze(1),
                'negative_prompt_embeds': pipe.text_encoder(torch.ones(bs, 2)).unsqueeze(1)}
    engine._encode = encode
    result = engine._encode_scheduled(job.gp.prompt, '', job.gp, 1, 10)
    plan = result['_neutral']
    assert len(plan.branches) == 2
    assert plan.branches[0]['_clora_branch'] == '<lora:A:1> red '
    torch.testing.assert_close(plan.branches[0]['prompt_embeds'], torch.full((1, 1, 2), 2.))
    torch.testing.assert_close(plan.branches[1]['prompt_embeds'], torch.ones(1, 1, 2))
    assert all(torch.equal(b['negative_prompt_embeds'], torch.ones(1, 1, 2)) for b in plan.branches)
    job.remove()


def test_prediction_composition_uses_real_adapter_scope_for_each_branch(tmp_path):
    from sdfx.neutral_runtime import RuntimeHook, Plan
    from sdfx.neutral_prompt import parse
    from sdfx.clora.runtime import BranchHook
    from diffusers import DDIMScheduler
    pipe, job = fixture_job(tmp_path)
    pipe.unet.forward = lambda sample, timestep, **kwargs: (pipe.unet[0](sample),)
    branches = [dict(prompt_embeds=torch.ones(1, 1, 2), negative_prompt_embeds=torch.ones(1, 1, 2),
                     _clora_branch=text) for text in ('<lora:A:.5> red', '<lora:A:1.5> blue')]
    scheduler = DDIMScheduler(clip_sample=False)
    scheduler.set_timesteps(4)
    hook = RuntimeHook(pipe.unet, scheduler, Plan(parse('red AND blue'), branches), 6, 0,
                       branch_hooks=[BranchHook(job, scheduler, 4)]).install()
    try:
        result = pipe.unet(torch.ones(2, 2), scheduler.timesteps[0])[0]
        torch.testing.assert_close(result, torch.tensor([[1., 1.], [3., 3.]]))
        assert job.state is None
    finally:
        hook.remove()
        job.remove()


def test_plot_contains_actual_zero_and_active_weights_and_visible_image(tmp_path):
    from PIL import Image
    pipe, job = fixture_job(tmp_path, plot_weights=True)
    for step in range(1, 4):
        with job.using('<lora:A:1> red', [False, True], step, 3):
            pipe.unet(torch.ones(2, 2))
    result = {'images': []}
    job.finish(result, str(tmp_path))
    records = result['composable_lora_trace']['records']
    assert len(records) == 6
    assert {r['mean_weight'] for r in records} == {0, 1}
    assert {r['step'] for r in records} == {1, 2, 3}
    with Image.open(result['images'][0]['path']) as image:
        assert image.size == (1200, 680)
    job.remove()


@pytest.mark.parametrize('exclude, expected', [(False, [[2., 2.], [2., 2.]]),
                                               (True, [[1., 1.], [2., 2.]])])
def test_single_prompt_preserves_source_negative_exception(tmp_path, exclude, expected):
    pipe, job = fixture_job(tmp_path, prompt='<lora:A:1> red', single_no_uc=exclude)
    with job.using('<lora:A:1> red', [False, True], 1, 10):
        torch.testing.assert_close(pipe.unet(torch.ones(2, 2)), torch.tensor(expected))
    job.remove()


def test_hires_weights_and_negative_single_exception_belong_to_the_current_pass(tmp_path):
    from sdfx.params import HiresParams
    pipe, job = fixture_job(tmp_path, prompt='<lora:A:1> red',
                           hires=HiresParams(enabled=True, prompt='<lora:A:.5> blue AND plain'))
    with job.using('<lora:A:1> red', [False, True], 1, 10, pass_name='base'):
        torch.testing.assert_close(pipe.unet(torch.ones(2, 2)), torch.full((2, 2), 2.))
    with job.using('<lora:A:.5> blue', [False, True], 1, 5, pass_name='hires'):
        torch.testing.assert_close(pipe.unet(torch.ones(2, 2)), torch.tensor([[1., 1.], [1.5, 1.5]]))
    job.remove()


def test_advanced_command_receives_documented_native_context(tmp_path, monkeypatch):
    monkeypatch.setenv('SDFX_ALLOW_CODE', '1')
    pipe, job = fixture_job(tmp_path, with_step=True)
    expression = ("0.25 if layer_name == 'unet.0' and lora.name == 'A' and "
                  "lora_module.module.weight.ndim == 2 and lora_type == 'lora' and lora_name == 'A' and "
                  "lora_count == 1 and block_lora_count == 1 and not is_negative and current_prompt and "
                  "sd_processing.steps == 20 and step == 15 and steps == 20 and life == 0.5 and weight == 1 else 0")
    source = '[<lora:A:1> #cmd\\(' + expression + '\\):10]'
    with job.using(source, [False, True], 15, 20):
        torch.testing.assert_close(pipe.unet(torch.ones(2, 2)), torch.tensor([[1., 1.], [1.25, 1.25]]))
    job.remove()


@pytest.mark.parametrize('hires_total, actual', [(None, 20), (4, 4)])
def test_global_hires_total_is_also_supplied_to_prepare_step_commands(tmp_path, monkeypatch, hires_total, actual):
    from sdfx.engine import Engine
    from sdfx.params import HiresParams
    monkeypatch.setenv('SDFX_ALLOW_CODE', '1')
    source = '[<lora:A:1> #cmd\\(enable_prepare_step=True\nsteps/100\\)] red'
    pipe, job = fixture_job(tmp_path, prompt=source, with_step=True, hires_global_step=True,
                           hires=HiresParams(enabled=True, second_pass_steps=4))
    engine = Engine.__new__(Engine)
    engine._composable_job, engine.dtype = job, torch.float32
    engine._encode = lambda p, n, gp, bs: {
        'prompt_embeds': pipe.text_encoder(torch.ones(bs, 2)).unsqueeze(1),
        'negative_prompt_embeds': pipe.text_encoder(torch.ones(bs, 2)).unsqueeze(1)}
    result = engine._encode_scheduled(source, '', job.gp, 1, 20, hires_total, gradient_total=actual)
    torch.testing.assert_close(result['_neutral'].branches[0]['prompt_embeds'], torch.full((1, 1, 2), 1.24))
    job.remove()


def test_plot_trace_persists_outside_request_temp_directory(tmp_path):
    import json
    from pathlib import Path
    temp = tmp_path / '.tmp'
    persistent = tmp_path / 'outputs'
    temp.mkdir()
    pipe, job = fixture_job(tmp_path, plot_weights=True, output_dir=str(persistent))
    result = {'images': []}
    try:
        job.finish(result, str(temp))
        trace = Path(result['composable_lora_trace']['path'])
        assert trace.is_relative_to(persistent)
        assert trace.is_file()
        assert json.loads(trace.read_text())['records'] == []
        assert not list(temp.glob('*.json'))
        for image in result['images']:
            Path(image['path']).unlink()
        assert trace.is_file()
    finally:
        job.remove()
