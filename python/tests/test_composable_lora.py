import math
import pytest


def controllers(text):
    from sdfx.clora.controllers import parse_step_rendering_syntax
    return parse_step_rendering_syntax(text)


def weight(text, name='A', step=1, total=20):
    from sdfx.clora.controllers import check_lora_weight
    return check_lora_weight(controllers(text)[0], 'lora:' + name, step, total, {})


@pytest.mark.parametrize('text, values', [
    ('[<lora:A:1>::10]', [1, 1, 1, 0, 0]),
    ('[<lora:A:1>:10]', [0, 0, 1, 1, 1]),
    ('[[<lora:A:1>::15]:10]', [0, 0, 1, 1, 0]),
    ('[<lora:A:1> #increment:10]', [0, 0, 0, .5, 1]),
    ('[<lora:A:1> #decrease:10]', [0, 0, 1, .5, 0]),
])
def test_source_steps_and_controllers(text, values):
    assert [weight(text, step=s) for s in (1, 9, 10, 15, 20)] == values


def test_branch_duplicate_and_fractional_semantics():
    from sdfx.clora.controllers import check_lora_weight
    branches = controllers('red <lora:A:.5> AND blue <lora:A:-.8> <lyco:B:.3>')
    assert len(branches) == 2
    assert check_lora_weight(branches[0], 'lora:A', 1, 20, {}) == .5
    assert check_lora_weight(branches[1], 'lora:A', 1, 20, {}) == -.8
    assert weight('[<lora:A:1>:0.5]', step=9) == 0
    assert weight('[<lora:A:1>:0.5]', step=10) == 1
    assert weight('<lora:A:.5> <lora:A:-.8>') == -.8


def test_math_commands_escaped_hash_prepare_and_finite_errors(monkeypatch):
    monkeypatch.delenv('SDFX_ALLOW_CODE', raising=False)
    assert weight(r'[<lora:A:2> #cmd\(sin\(life\)\):10]', step=15) == pytest.approx(2)
    assert weight(r'[<lora:A:2> \u0023cmd\(warmup\(0.5\)\):10]', step=12) == pytest.approx(.8)
    assert weight('[<lora:A:2> #increment:10]', step=-1) == 2
    with pytest.raises(ValueError, match='finite'):
        weight(r'[<lora:A:1> #cmd\(1e309\):10]', step=15)


def test_advanced_command_uses_existing_custom_code_opt_in(monkeypatch):
    from sdfx.protocol import EngineError
    code = '[<lora:A:2> #cmd\\(\ndef f\\(\\)\\:\n    return life\nf\\(\\)\n\\):10]'
    monkeypatch.delenv('SDFX_ALLOW_CODE', raising=False)
    with pytest.raises(EngineError, match='Allow custom code'):
        controllers(code)
    monkeypatch.setenv('SDFX_ALLOW_CODE', '1')
    assert weight(code, step=15) == 1


def test_single_step_window_is_defined_and_finite():
    assert math.isfinite(weight('[[<lora:A:1> #increment::10]:10]', step=10))


def test_prepare_step_command_can_opt_in_to_text_encoder_weight(monkeypatch):
    monkeypatch.setenv('SDFX_ALLOW_CODE', '1')
    source = '[<lora:A:2> #cmd\\(enable_prepare_step=True\n0.25\\)]'
    assert weight(source, step=-1) == .5


def test_alternation_uses_the_source_global_step_modulo_rule():
    source = '[<lora:A:1>|<lora:B:.5>]'
    assert [weight(source, 'A', s) for s in range(1, 5)] == [0, 1, 0, 1]
    assert [weight(source, 'B', s) for s in range(1, 5)] == [.5, 0, .5, 0]


def test_feature_is_available_to_torch_free_clients():
    from sdfx.hooks import KNOWN_HOOKS
    from sdfx.dryrun import FakeEngine
    assert 'composable_lora' in KNOWN_HOOKS
    assert 'composable_lora' in FakeEngine().features()


def test_documented_cube_root_helper_accepts_negative_real_inputs():
    assert weight(r'[<lora:A:1> #cmd\(cbrt\(-8\)\):10]', step=15) == pytest.approx(-2.)


@pytest.mark.parametrize('config', [{}, True, 'true', {'enabled': True}, {'enabled': 'true'}])
def test_active_composable_positive_and_does_not_warn_plain_text(tmp_path, config):
    import io
    from sdfx.base import BaseEngine
    from sdfx.protocol import Emitter, JobContext
    context = JobContext(Emitter(io.BytesIO()), 'parse')
    BaseEngine().parse_generation({'output_dir': str(tmp_path), 'params': {
        'prompt': 'red <lora:A:.5> AND blue <lora:A:1>',
        'hooks': {'composable_lora': config}}}, 'txt2img', context)
    assert not any('composable diffusion (AND)' in warning for warning in context.warnings)


@pytest.mark.parametrize('hooks', [{}, {'composable_lora': False}, {'composable_lora': 'false'},
                                   {'composable_lora': {'enabled': False}},
                                   {'composable_lora': {'enabled': 'false'}}])
def test_disabled_composable_positive_and_keeps_plain_text_warning(tmp_path, hooks):
    import io
    from sdfx.base import BaseEngine
    from sdfx.protocol import Emitter, JobContext
    context = JobContext(Emitter(io.BytesIO()), 'parse')
    BaseEngine().parse_generation({'output_dir': str(tmp_path), 'params': {
        'prompt': 'red AND blue', 'hooks': hooks}}, 'txt2img', context)
    assert sum('composable diffusion (AND)' in warning for warning in context.warnings) == 1


def test_active_composable_negative_and_keeps_unsupported_warning(tmp_path):
    import io
    from sdfx.base import BaseEngine
    from sdfx.protocol import Emitter, JobContext
    context = JobContext(Emitter(io.BytesIO()), 'parse')
    BaseEngine().parse_generation({'output_dir': str(tmp_path), 'params': {
        'prompt': 'red AND blue', 'negative_prompt': 'bad AND worse',
        'hooks': {'composable_lora': {}}}}, 'txt2img', context)
    assert sum('composable diffusion (AND)' in warning for warning in context.warnings) == 1
