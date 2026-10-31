import importlib.util
from pathlib import Path


def test_diagnostic_reports_conditioning_and_restore_mismatches():
    path = Path(__file__).resolve().parents[2] / 'scripts/dev/keyword_photo_trace_analyze.py'
    spec = importlib.util.spec_from_file_location('keyword_photo_trace_analyze', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    def photo(sequence, digest):
        return {'sequence': sequence, 'event': 'photo_branches', 'data': [
            {'branch': 0, 'end': 2, 'stage': 'text', 'key': 'prompt_embeds', 'sha256': digest}]}
    events = [photo(1, 'a'), {'sequence': 2, 'event': 'embedding_before', 'data': ['old']},
              photo(3, 'b'), {'sequence': 4, 'event': 'embedding_after', 'data': ['changed']},
              photo(5, 'a'), photo(6, 'b')]
    report = module.analyze(events)
    assert report['comparisons']['plain_vs_merge']['mismatched'] == 1
    assert report['comparisons']['plain_repeat']['mismatched'] == 0
    assert report['comparisons']['merge_repeat']['mismatched'] == 0
    assert not report['embedding_restores'][0]['equal']
    assert not report['unmatched_captures']


def test_unet_fingerprints_report_first_changed_input_or_output():
    path = Path(__file__).resolve().parents[2] / 'scripts/dev/keyword_photo_trace_analyze.py'
    spec = importlib.util.spec_from_file_location('keyword_photo_trace_analyze_unet', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    events = []
    for index, output in enumerate(('plain', 'merge', 'plain', 'merge')):
        events.append({'sequence': index * 2 + 1, 'event': 'photo_branches', 'data': []})
        events.append({'sequence': index * 2 + 2, 'event': 'neutral_unet',
                       'data': {'raw_step': 1, 'input': {'sha256': 'same'}, 'output': {'sha256': output}}})
    report = module.analyze(events)['unet_comparisons']
    assert report['plain_vs_merge']['mismatched'] == 1
    assert report['plain_vs_merge']['details'][0]['key'] == 'output'
    assert report['plain_repeat']['mismatched'] == report['merge_repeat']['mismatched'] == 0


def test_missing_unet_calls_are_not_reported_as_a_match():
    path = Path(__file__).resolve().parents[2] / 'scripts/dev/keyword_photo_trace_analyze.py'
    spec = importlib.util.spec_from_file_location('keyword_photo_trace_analyze_missing', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    events = []
    for index, calls in enumerate((2, 0, 2, 0)):
        events.append({'sequence': index * 10, 'event': 'photo_branches', 'data': []})
        for step in range(calls):
            events.append({'sequence': index * 10 + step + 1, 'event': 'neutral_unet',
                           'data': {'raw_step': step + 1, 'input': {'sha256': 'same'}, 'output': {'sha256': 'same'}}})
    report = module.analyze(events)['unet_comparisons']
    assert report['plain_vs_merge']['mismatched'] == 2
    assert not report['plain_vs_merge']['complete']
    assert not report['merge_repeat']['complete']
    assert report['merge_repeat']['mismatched'] > 0
    assert report['plain_repeat']['complete']
