from sdfx.prompt_fusion import active


def test_feature_is_advertised_without_torch():
    from sdfx.hooks import KNOWN_HOOKS
    from sdfx.dryrun import FakeEngine
    assert 'prompt_fusion' in KNOWN_HOOKS
    assert 'prompt_fusion' in FakeEngine().features()


def test_ordinary_and_escaped_prompts_keep_native_encoding_route():
    for text in ('a, (red flower:1.2)', '[a:b:.5]', '[a|b]', r'\$name', 'a flower',
                 'caption: red, blue [contrast]'):
        assert not active(text)
    for text in ('$a = red\n$a', '(a:1,2)', '[a:b::mean]', '[a|b:.5]', '[a:b: , ]',
                 '[space station : kitchen mixer :: mean]', '[a:b:\n , 10]', '(a:\n 1,2)'):
        assert active(text)
