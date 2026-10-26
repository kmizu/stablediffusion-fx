import importlib.util
from pathlib import Path
import sys


def script():
    path = Path(__file__).resolve().parents[2] / 'scripts/dev/composable_lora_check.py'
    spec = importlib.util.spec_from_file_location('composable_lora_acceptance', path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_acceptance_includes_source_controllers_formats_and_actual_trace_assertions():
    harness = script()
    cases = {c['name']: c for c in harness.build_cases(True, True)}
    assert {'separate-branches', 'shared-adapter-plot', 'start', 'stop', 'switch', 'nested-window',
            'increment', 'decrease', 'math-command', 'python-function', 'escaped-hash', 'alternate',
            'locon', 'lokr', 'batch', 'heun', 'hires-global', 'hires-restart', 'img2img', 'fusion', 'neutral'} <= set(cases)
    for name in ('enabled-no-adapter', 'disabled-no-adapter', 'restored'):
        assert cases[name]['equal_to'] == 'baseline'
    assert cases['and-zero-weights']['equal_to'] == 'and-no-adapter'
    assert cases['shared-adapter-plot']['params']['hooks']['composable_lora']['plot_weights']
    assert cases['hires-global']['params']['hooks']['composable_lora']['hires_global_step']
    assert 'python-function' not in {c['name'] for c in harness.build_cases(False, False)}


def test_advanced_execution_is_explicit_and_guards_cover_compatibility_boundaries():
    harness = script()
    guards = {c['name']: c for c in harness.build_guards('input.png')}
    assert {'python-opt-in', 'photomaker', 'alternative', 'negpip-signed', 'native-lbw'} <= set(guards)
    assert 'python-opt-in' not in {c['name'] for c in harness.build_guards('input.png', True)}
