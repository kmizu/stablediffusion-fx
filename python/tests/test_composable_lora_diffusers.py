"""Real native UNet/SDXL pooled-conditioning math, small random CPU models only."""
import pytest
torch = pytest.importorskip('torch')
diffusers = pytest.importorskip('diffusers')
from types import SimpleNamespace


@pytest.mark.parametrize('sdxl', [False, True])
def test_real_diffusers_unet_matches_independent_scoped_predictions(tmp_path, sdxl):
    from safetensors.torch import save_file
    from sdfx.clora.runtime import Job, BranchHook
    from sdfx.neutral_runtime import RuntimeHook, Plan
    from sdfx.neutral_prompt import parse
    from sdfx.params import GenerationParams, GenerationRequest, LoraRef
    torch.manual_seed(29)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    job, hook = None, None
    try:
        xl_config = dict(addition_embed_type='text_time', addition_time_embed_dim=4,
                         projection_class_embeddings_input_dim=36) if sdxl else {}
        model = diffusers.UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
            down_block_types=('CrossAttnDownBlock2D',), up_block_types=('CrossAttnUpBlock2D',),
            block_out_channels=(32,), layers_per_block=1, cross_attention_dim=8,
            attention_head_dim=4, norm_num_groups=8, **xl_config).eval()
        layer = 'down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q'
        file = tmp_path / 'adapter.safetensors'
        prefix = 'lora_unet_' + layer.replace('.', '_')
        save_file({prefix + '.lora_down.weight': torch.randn(2, 32) * .1,
                   prefix + '.lora_up.weight': torch.randn(32, 2) * .1}, str(file))
        pipe = SimpleNamespace(unet=model, text_encoder=None, text_encoder_2=None)
        gp = GenerationParams(prompt='red <lora:A:.3> AND blue <lora:A:.9>', loras=[LoraRef('A')],
                              hooks={'composable_lora': {}})
        req = GenerationRequest(params=gp, lora_paths={'A': str(file)})
        job = Job(pipe, gp, req, SimpleNamespace(check_interrupt=lambda: None))
        samples = torch.randn(4, 4, 8, 8)  # batch two, CFG negative then positive
        pe, ne = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
        branches = [dict(prompt_embeds=pe, negative_prompt_embeds=ne, _clora_branch=text)
                    for text in ('red <lora:A:.3>', 'blue <lora:A:.9>')]
        kwargs = {}
        if sdxl:
            negative_pooled = torch.randn(2, 12)
            for branch in branches:
                branch.update(pooled_prompt_embeds=torch.randn(2, 12), negative_pooled_prompt_embeds=negative_pooled)
            kwargs['added_cond_kwargs'] = dict(text_embeds=torch.cat([negative_pooled, branches[0]['pooled_prompt_embeds']]),
                                               time_ids=torch.tensor([[64., 64., 0., 0., 64., 64.]]).repeat(4, 1))
        scheduler = diffusers.DDIMScheduler(clip_sample=False)
        scheduler.set_timesteps(4)
        timestep = scheduler.timesteps[0]
        independent = []
        with torch.no_grad():
            for branch in branches:
                native = dict(encoder_hidden_states=torch.cat([ne, pe]), **kwargs)
                if sdxl:
                    native['added_cond_kwargs'] = dict(kwargs['added_cond_kwargs'],
                        text_embeds=torch.cat([negative_pooled, branch['pooled_prompt_embeds']]))
                with job.using(branch['_clora_branch'], [False, True], 1, 4):
                    independent.append(model(samples, timestep, **native).sample)
            unconditional = independent[0][:2]
            expected = unconditional + sum(p[2:] - unconditional for p in independent)
            hook = RuntimeHook(model, scheduler, Plan(parse('red AND blue'), branches), 6, 0,
                               branch_hooks=[BranchHook(job, scheduler, 4)]).install()
            result = model(samples, timestep, **kwargs).sample
        torch.testing.assert_close(result[:2], unconditional, rtol=0, atol=0)
        torch.testing.assert_close(result[2:], expected, rtol=1e-4, atol=1e-5)
        assert job.state is None
    finally:
        if hook is not None:
            hook.remove()
        if job is not None:
            job.remove()
        torch.set_num_threads(threads)
