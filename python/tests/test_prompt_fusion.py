import pytest

torch = pytest.importorskip('torch')


def test_full_upstream_language_is_available():
    from sdfx.prompt_fusion import builder
    assert builder('$x = red\n$x flower', 10).get_prompt_database() == ['red flower']
    assert builder('$f($x) = $x flower\n$f(red)', 10).get_prompt_database() == ['red flower']
    assert builder('[a:b:c: , 4, :catmull]', 10).get_prompt_database() == ['a', 'b', 'c']
    assert builder('[a|b:0.5]', 10).get_prompt_database() == ['a', 'b', 'a']
    assert '(a:1.0)' in builder('(a:1,2)', 3).get_prompt_database()[0]


def test_mean_does_not_mutate_cached_conditioning():
    from sdfx.prompt_fusion import sample
    def encode(text):
        return {'prompt_embeds': torch.full((1, 77, 2), {'a': 2., 'b': 6.}.get(text, 0.))}
    result = sample('[a:b::mean]', 4, encode)
    for _, d in result:
        torch.testing.assert_close(d['prompt_embeds'], torch.full((1, 77, 2), 4.))


def test_linear_zero_based_clock_and_negative_and_pooled():
    from sdfx.prompt_fusion import sample
    def encode(text):
        value = {'a': 0., 'b': 8.}.get(text, 0.)
        return {'prompt_embeds': torch.full((1, 77, 2), value),
                'pooled_prompt_embeds': torch.full((1, 3), value)}
    result = sample('[a:b: , 3]', 5, encode)
    assert [float(d['prompt_embeds'][0, 0, 0]) for _, d in result] == [0, 2, 4, 6, 8]
    assert [float(d['pooled_prompt_embeds'][0, 0]) for _, d in result] == [0, 2, 4, 6, 8]


def test_slerp_zero_origin_is_finite():
    from sdfx.prompt_fusion import sample
    result = sample('[a:b: , ]', 4, lambda _: {'prompt_embeds': torch.zeros(1, 77, 2)},
                    scale=1)
    assert all(torch.isfinite(d['prompt_embeds']).all() for _, d in result)


def test_engine_routes_fusion_and_preserves_disabled_baseline():
    from sdfx.engine import Engine
    from sdfx.params import GenerationParams
    engine = Engine.__new__(Engine)
    engine.dtype = torch.float32
    engine._encode = lambda p, n, gp, bs: {
        'prompt_embeds': torch.full((bs, 77, 2), {'a': 0., 'b': 8.}.get(p, 3.)),
        'negative_prompt_embeds': torch.full((bs, 77, 2), 5.)}
    gp = GenerationParams(hooks={'prompt_fusion': {'enabled': True}})
    out = engine._encode_scheduled('[a:b: , 3]', '', gp, 2, 5)
    assert len(out.schedule) == 5
    assert [float(d['prompt_embeds'][1, 0, 0]) for _, d in out.schedule] == [0, 2, 4, 6, 8]
    assert all(d['negative_prompt_embeds'].shape == (2, 77, 2) for _, d in out.schedule)
    assert engine._encode_scheduled('a', '', gp, 1, 5)['prompt_embeds'][0, 0, 0] == 0


@pytest.mark.parametrize('curve', ['linear', 'bezier', 'catmull'])
def test_curves_keep_first_last_endpoints_and_finite_intermediates(curve):
    from sdfx.prompt_fusion import sample
    values = {'a': 0., 'b': 3., 'c': 8.}
    out = sample(f'[a:b:c: , 2, 5:{curve}]', 8,
                 lambda s: {'prompt_embeds': torch.full((1, 77, 2), values.get(s, 0.))})
    assert out[0][1]['prompt_embeds'][0, 0, 0] == 0
    assert out[-1][1]['prompt_embeds'][0, 0, 0] == 8


def test_attention_ramp_uses_context_editing_interval():
    from sdfx.prompt_fusion import sample, leaf_texts
    from sdfx import prompt
    def encode(text):
        result = []
        for end, rendered in enumerate(leaf_texts(text, 6), 1):
            value = sum(w for t, w in prompt.parse_prompt_attention(rendered) if t.strip())
            result.append((end, {'prompt_embeds': torch.full((1, 77, 2), value)}))
        return result
    result = sample('[(a:1,3)::3]', 6, encode)
    assert [float(d['prompt_embeds'][0, 0, 0]) for _, d in result] == pytest.approx([1, 5/3, 7/3, 3, 0, 0])


def test_long_conditioning_pads_with_empty_chunks_and_does_not_mutate_inputs():
    from sdfx.prompt_fusion import sample
    values = {'a': torch.ones(1, 77, 2), 'b': torch.full((1, 154, 2), 3.), '': torch.full((1, 77, 2), 2.)}
    result = sample('[a:b::mean]', 3, lambda s: {'prompt_embeds': values[s]})
    assert result[0][1]['prompt_embeds'].shape == (1, 154, 2)
    assert result[0][1]['prompt_embeds'][0, 0, 0] == 2
    assert result[0][1]['prompt_embeds'][0, 100, 0] == 2.5
    assert torch.equal(values['a'], torch.ones(1, 77, 2))


def test_alternation_speed_and_weighted_mean():
    from sdfx.prompt_fusion import sample
    encode = lambda s: {'prompt_embeds': torch.full((1, 77, 2), {'a': 0., 'b': 8.}.get(s, 0.))}
    out = sample('[a|b:0.5]', 5, encode)
    assert [float(d['prompt_embeds'][0, 0, 0]) for _, d in out] == [0, 4, 8, 4, 0]
    out = sample('[a:b:1,3:mean]', 3, encode)
    assert all(d['prompt_embeds'][0, 0, 0] == 6 for _, d in out)


def test_engine_negative_branch_curves_and_pooled_are_batched_independently():
    from sdfx.engine import Engine
    from sdfx.params import GenerationParams
    engine = Engine.__new__(Engine)
    engine.dtype = torch.float32
    def encode(p, n, gp, bs):
        v = lambda t: {'a': 0., 'b': 8.}.get(t, 1.)
        return {'prompt_embeds': torch.full((bs, 77, 2), v(p)),
                'negative_prompt_embeds': torch.full((bs, 77, 2), v(n)),
                'pooled_prompt_embeds': torch.full((bs, 3), v(p)),
                'negative_pooled_prompt_embeds': torch.full((bs, 3), v(n))}
    engine._encode = encode
    gp = GenerationParams(hooks={'prompt_fusion': {}})
    out = engine._encode_scheduled('[a:b: , 3]', '[b:a: , 3]', gp, 2, 5)
    assert [float(d['negative_pooled_prompt_embeds'][1, 0]) for _, d in out.schedule] == [8, 6, 4, 2, 0]


def test_exponential_leaf_expansion_fails_before_large_allocation():
    from sdfx.prompt_fusion import builder
    with pytest.raises(ValueError, match='512'):
        builder('[a:b::mean]' * 10, 5)


@pytest.mark.parametrize('override', [True, False])
def test_alternative_does_not_silently_freeze_fusion_inversion(override):
    from sdfx.engine import Engine
    from sdfx.params import GenerationParams
    from sdfx.protocol import EngineError, UNSUPPORTED
    engine = Engine.__new__(Engine)
    gp = GenerationParams(prompt='' if override else '[a:b::mean]', hooks={'prompt_fusion': {}})
    with pytest.raises(EngineError, match='inversion') as error:
        engine._img2imgalt(gp, None, None, {'override_prompt': '[a:b::mean]'} if override else {},
                          None, {}, [1], 0, 1, 64, 64, None, None)
    assert error.value.code == UNSUPPORTED


@pytest.mark.parametrize('expression', ['[a:b: , $n]', '(a:1,$n)'])
@pytest.mark.parametrize('number', ['1e309', '-1e309', 'nan', 'inf', '9' * 400])
def test_nonfinite_numeric_variables_are_engine_parameter_errors(expression, number):
    from sdfx.engine import Engine
    from sdfx.params import GenerationParams
    from sdfx.protocol import EngineError, INVALID_PARAMS
    engine = Engine.__new__(Engine)
    engine.dtype = torch.float32
    engine._encode = lambda p, n, gp, bs: {
        'prompt_embeds': torch.ones(bs, 77, 2),
        'negative_prompt_embeds': torch.zeros(bs, 77, 2)}
    gp = GenerationParams(hooks={'prompt_fusion': {}})
    with pytest.raises(EngineError, match='finite') as error:
        engine._encode_scheduled(f'$n = {number}\n{expression}', '', gp, 1, 5)
    assert error.value.code == INVALID_PARAMS
