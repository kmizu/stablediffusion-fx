import torch
from sdfx.model_keyword_conditioning import collate_plain, collate_schedules, collate_neutral


def test_composable_merge_provenance_collates_mixed_expression_and_plain_items():
    def item(source):
        return {'prompt_embeds': torch.ones(1, 1, 1), 'negative_prompt_embeds': torch.zeros(1, 1, 1),
                '_clora_branch': source, '_clora_prompt': source, '_clora_negative': 'bad'}
    first = item('__nui_em_1__'); first['_photo_raw_branch'] = "<'a'>"
    first['_photo_raw_prompt'] = "<'a'>"; first['_photo_raw_negative'] = 'bad'
    result = collate_plain([first, item('plain')], torch.zeros(1, 1, 1))
    assert result['_clora_branch'] == ['__nui_em_1__', 'plain']
    assert result['_photo_raw_branch'] == ["<'a'>", 'plain']
    assert result['_photo_raw_negative'] == ['bad', 'bad']


def test_collates_distinct_image_conditions_with_empty_token_padding():
    def image(length, value):
        return {'prompt_embeds': torch.full((1, length, 2), value),
                'negative_prompt_embeds': torch.full((1, length, 2), -value),
                'pooled_prompt_embeds': torch.full((1, 3), value),
                'negative_pooled_prompt_embeds': torch.full((1, 3), -value)}
    output = collate_plain([image(2, 1.), image(4, 2.)], torch.zeros((1, 2, 2)))
    assert output['prompt_embeds'].shape == (2, 4, 2)
    assert torch.equal(output['prompt_embeds'][0, :2], torch.ones((2, 2)))
    assert torch.equal(output['prompt_embeds'][0, 2:], torch.zeros((2, 2)))
    assert torch.equal(output['pooled_prompt_embeds'][:, 0], torch.tensor([1., 2.]))


def test_negpip_sign_channels_are_preserved_across_items():
    marked = torch.tensor([[[1., 1.], [2., -1.]]])
    ordinary = torch.tensor([[[3.]]])
    items = [{'prompt_embeds': marked, 'negative_prompt_embeds': marked, '_negpip': True},
             {'prompt_embeds': ordinary, 'negative_prompt_embeds': ordinary}]
    result = collate_plain(items, torch.zeros((1, 1, 1)))
    assert result['_negpip'] is True
    assert torch.equal(result['prompt_embeds'][0, :, -1], torch.tensor([1., -1.]))
    assert torch.equal(result['prompt_embeds'][1, :, -1], torch.tensor([1., 0.]))


def test_different_step_boundaries_produce_one_shape_stable_batch_schedule():
    def image(length, value):
        return {'prompt_embeds': torch.full((1, length, 2), value),
                'negative_prompt_embeds': torch.full((1, length, 2), -value)}
    timelines = [[(2, image(2, 1.)), (5, image(4, 2.))], [(3, image(2, 3.)), (5, image(2, 4.))]]
    merged = collate_schedules(timelines, torch.zeros((1, 2, 2)))
    assert [end for end, _ in merged] == [2, 3, 5]
    assert all(item['prompt_embeds'].shape == (2, 4, 2) for _, item in merged)
    assert [item['prompt_embeds'][:, 0, 0].tolist() for _, item in merged] == [[1., 3.], [2., 3.], [2., 4.]]


def test_negpip_appearing_later_keeps_marker_width_and_padding_at_every_step():
    plain = {'prompt_embeds': torch.ones((1, 1, 1)), 'negative_prompt_embeds': torch.ones((1, 1, 1))}
    marked = {'prompt_embeds': torch.tensor([[[2., 1.], [3., -1.]]]),
              'negative_prompt_embeds': torch.tensor([[[2., 1.], [3., -1.]]]), '_negpip': True}
    merged = collate_schedules([[(1, plain), (2, marked)], [(2, plain)]], torch.zeros((1, 1, 1)))
    assert all(item['_negpip'] and item['prompt_embeds'].shape == (2, 2, 2) for _, item in merged)
    assert torch.equal(merged[0][1]['prompt_embeds'][:, :, -1], torch.tensor([[1., 0.], [1., 0.]]))
    assert torch.equal(merged[1][1]['prompt_embeds'][:, :, -1], torch.tensor([[1., -1.], [1., 0.]]))


def test_composable_hook_uses_each_images_branch_weight_within_one_batch():
    from types import SimpleNamespace
    from sdfx.clora.runtime import Job
    job = object.__new__(Job)
    job.state = (['1', '2'], [False, True], 1, 10, 'unet', 'base', ['a', 'b'], ['n', 'n'])
    job.ctx = SimpleNamespace(check_interrupt=lambda: None)
    job.config, job.trace = {}, {}
    job._weight = lambda adapter, patch, layer, text, negative, step, total, phase, full, neg: float(text) * (-1 if negative else 1)
    adapter = SimpleNamespace(name='A')
    patch = SimpleNamespace(delta=lambda value: torch.ones_like(value))
    hook = job._hook('unet', 'layer', adapter, patch)
    result = hook(None, (torch.zeros((4, 1)),), torch.zeros((4, 1)))
    assert torch.equal(result[:, 0], torch.tensor([-1., -2., 1., 2.]))


def test_neutral_runtime_composes_each_image_with_its_own_branch_tree():
    from diffusers import DDIMScheduler
    from sdfx.neutral_prompt import parse
    from sdfx.neutral_runtime import Plan, RuntimeHook
    class UNet(torch.nn.Module):
        def forward(self, sample, timestep, encoder_hidden_states=None, **kwargs):
            return (encoder_hidden_states[:, :1, :1].reshape(-1, 1, 1, 1),)
    unet = UNet()
    scheduler = DDIMScheduler(clip_sample=False); scheduler.set_timesteps(4)
    negative = torch.zeros((2, 1, 1))
    branches = [dict(prompt_embeds=torch.tensor([[[1.]], [[2.]]]), negative_prompt_embeds=negative),
                dict(prompt_embeds=torch.tensor([[[100.]], [[4.]]]), negative_prompt_embeds=negative)]
    plan = Plan(parse('one'), branches, image_trees=[parse('one'), parse('one AND two')])
    hook = RuntimeHook(unet, scheduler, plan, cfg=3, phi=0).install()
    try:
        result = unet(torch.zeros((4, 1, 1, 1)), scheduler.timesteps[0])[0]
        torch.testing.assert_close(result[:, 0, 0, 0], torch.tensor([0., 0., 1., 6.]))
    finally:
        hook.remove()


def test_collated_composable_branches_retain_each_images_routing_metadata():
    def item(branch, prompt):
        return {'prompt_embeds': torch.ones((1, 1, 1)), 'negative_prompt_embeds': torch.zeros((1, 1, 1)),
                '_clora_branch': branch, '_clora_prompt': prompt, '_clora_negative': 'negative'}
    result = collate_plain([item('A', 'first'), item('B', 'second')], torch.zeros((1, 1, 1)))
    assert result['_clora_branch'] == ['A', 'B']
    assert result['_clora_prompt'] == ['first', 'second']
    assert result['_clora_negative'] == ['negative', 'negative']


def test_scheduled_composable_metadata_follows_each_image_at_each_step():
    def item(branch):
        return {'prompt_embeds': torch.ones((1, 1, 1)), 'negative_prompt_embeds': torch.zeros((1, 1, 1)),
                '_clora_branch': branch, '_clora_prompt': 'prompt-' + branch, '_clora_negative': 'negative'}
    merged = collate_schedules([[(1, item('A')), (3, item('B'))], [(2, item('X')), (3, item('Y'))]],
                               torch.zeros((1, 1, 1)))
    assert [value['_clora_branch'] for _, value in merged] == [['A', 'X'], ['B', 'X'], ['B', 'Y']]
    assert [value['_clora_prompt'] for _, value in merged] == [
        ['prompt-A', 'prompt-X'], ['prompt-B', 'prompt-X'], ['prompt-B', 'prompt-Y']]


def test_collate_neutral_keeps_each_images_tree_and_branch_order():
    from sdfx.neutral_runtime import Plan
    from sdfx.neutral_prompt import parse
    def item(value):
        return {'prompt_embeds': torch.tensor([[[value]]]), 'negative_prompt_embeds': torch.zeros((1, 1, 1))}
    one, two = parse('one'), parse('one AND two')
    first = item(1.)
    second = dict(item(2.), _neutral=Plan(two, [item(2.), item(4.)]))
    class Scheduled(dict):
        pass
    def make_schedule(variants):
        result = Scheduled(variants[0][1]); result.schedule = variants; return result
    result = collate_neutral([first, second], [one, two], torch.zeros((1, 1, 1)), 3, make_schedule)
    plan = result['_neutral']
    assert plan.image_trees == [one, two] and len(plan.branches) == 2
    assert plan.branches[0]['prompt_embeds'][:, 0, 0].tolist() == [1., 2.]
    assert plan.branches[1]['prompt_embeds'][:, 0, 0].tolist() == [1., 4.]


def test_collate_neutral_keeps_per_image_composable_routing():
    from sdfx.neutral_runtime import Plan
    from sdfx.neutral_prompt import parse
    def item(value, branch):
        return {'prompt_embeds': torch.tensor([[[value]]]), 'negative_prompt_embeds': torch.zeros((1, 1, 1)),
                '_clora_branch': branch, '_clora_prompt': 'prompt-' + branch, '_clora_negative': ''}
    one, two = parse('one'), parse('one AND two')
    first = dict(item(1., 'a'), _neutral=Plan(one, [item(1., 'a')]))
    second = dict(item(2., 'b'), _neutral=Plan(two, [item(2., 'b'), item(4., 'c')]))
    result = collate_neutral([first, second], [one, two], torch.zeros((1, 1, 1)), 3, lambda variants: variants)
    plan = result['_neutral']
    assert plan.branches[0]['_clora_branch'] == ['a', 'b']
    assert plan.branches[1]['_clora_branch'] == ['a', 'c']


def test_padding_uses_each_images_own_text_encoder_empty_condition():
    def item(length, value):
        tensor = torch.full((1, length, 1), value)
        return {'prompt_embeds': tensor, 'negative_prompt_embeds': tensor}
    result = collate_plain([item(1, 1.), item(2, 2.)],
                           [torch.full((1, 1, 1), 5.), torch.full((1, 1, 1), 9.)])
    assert result['prompt_embeds'][:, :, 0].tolist() == [[1., 5.], [2., 2.]]
