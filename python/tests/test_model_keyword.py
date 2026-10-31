import hashlib
import random
import subprocess
import sys
from pathlib import Path
import pytest

from sdfx.model_keyword import legacy_hash, parse_rows, find_match, render_prompt, choose, save_custom, delete_custom, plan_prompts, MappingIndex, inspect_model, merge_schedules, mapping_action, plan_for_model, IterationState, plan_with_preview, compose_per_item, transpose_branches, civitai_keywords, lookup_civitai, KeywordPlanner, keyword_choices, batch_extra_networks, assemble_neutral, validate_batch_plan, parse_lora_activation


def test_legacy_hash_uses_only_64k_at_one_megabyte(tmp_path):
    path = tmp_path / 'sample.safetensors'
    block = bytes(range(256)) * 256
    path.write_bytes(b'x' * 0x100000 + block + b'y')
    assert legacy_hash(path) == hashlib.sha256(block).hexdigest()[:8]


def test_csv_collision_similarity_and_later_tie():
    rows = parse_rows('ab12, "red, blue|green",similar.safetensors\nab12,custom,similar.safetensors\n', 1)
    assert rows[0].keywords == 'red, blue|green'
    assert find_match(rows, 'AB12', 'similar.safetensors').keywords == 'custom'
    assert find_match(rows, 'ab12', 'unrelated') is not None
    assert find_match(rows, 'absent', 'similar.safetensors') is None


def test_per_image_iteration_and_seeded_random_choice():
    assert [choose('a|b|c', 'iterate', i, random.Random(1)) for i in range(5)] == ['a', 'b', 'c', 'a', 'b']
    assert choose('a|b', 'keyword1, keyword2', 0, random.Random(1)) == 'a, b'
    assert choose('a|b', 'a, b', 0, random.Random(1)) == 'a, b'
    assert choose('a|b', 'b', 0, random.Random(1)) == 'b'
    assert choose('a|b', 'stale', 0, random.Random(1)) == 'a'
    assert choose('a|b', 'random', 0, random.Random(2)) == choose('a|b', 'random', 0, random.Random(2))


def test_placement_order_and_lora_tag():
    assert render_prompt('portrait', 'model', 'embedding', 'lora trigger', 'model keyword first',
                         'keyword, prompt', 'nested/style.safetensors', 0.7) == '<lora:style:0.7>, model, lora trigger, embedding, portrait'
    assert render_prompt('portrait', 'model', 'embedding', None, 'textual inversion first',
                         'prompt keyword', None, 0.7) == 'portrait embedding model'


def test_custom_csv_save_update_and_delete_preserves_other_rows(tmp_path):
    path = tmp_path / 'custom.csv'
    path.write_text('# Keep this comment\nabc, old, old.safetensors\n', encoding='utf8')
    save_custom(path, 'ABC', 'blue, flower|red', '日本語.safetensors')
    save_custom(path, 'ABC', 'new|value', '日本語.safetensors')
    text = path.read_text(encoding='utf8')
    assert '# Keep this comment' in text
    rows = parse_rows(text, 1)
    assert len(rows) == 2
    assert find_match(rows, 'abc', '日本語.safetensors').keywords == 'new|value'
    assert delete_custom(path, 'abc', '日本語.safetensors')
    assert not delete_custom(path, 'abc', '日本語.safetensors')
    assert find_match(parse_rows(path.read_text(encoding='utf8'), 1), 'abc', 'old.safetensors').keywords == 'old'


def test_batch_prompts_track_model_and_lora_iteration_independently():
    prompts = plan_prompts(['one', 'two', 'three'], 'red|blue', 'iterate', 'x|y|z', '< iterate >',
                           'ti', 'model keyword first', 'keyword, prompt', 'style.safetensors', 0.5,
                           0, 1, random.Random(4))
    assert prompts == ['<lora:style:0.5>, red, y, ti, one', '<lora:style:0.5>, blue, z, ti, two',
                       '<lora:style:0.5>, red, x, ti, three']


def test_index_refreshes_custom_rows_and_preserves_source(tmp_path):
    default = tmp_path / 'default.csv'
    user = tmp_path / 'user.csv'
    default.write_text('abc,base,model.safetensors\n', encoding='utf8')
    user.write_text('abc,custom,model.safetensors\n', encoding='utf8')
    index = MappingIndex(default, user)
    match = index.lookup('ABC', 'model.safetensors')
    assert (match.keywords, match.source) == ('custom', 1)
    delete_custom(user, 'abc', 'model.safetensors')
    match = index.lookup('abc', 'model.safetensors')
    assert (match.keywords, match.source) == ('base', 0)


def test_bundled_checkpoint_database_is_the_pinned_upstream_bytes(tmp_path):
    data = Path(__file__).resolve().parents[1] / 'sdfx' / 'data' / 'model_keyword'
    expected = {'model-keyword.txt': '9e35fc373c397a1c288e8f6d10cd12f244b20ab84b831b410f261eabf8dcacdf'}
    for filename, digest in expected.items():
        assert hashlib.sha256((data / filename).read_bytes()).hexdigest() == digest
    entry = MappingIndex(data / 'model-keyword.txt', tmp_path / 'absent.csv').lookup('00762d5b', 'anything')
    assert entry.keywords == 'realistic|cosplay' and entry.source == 0


def test_bundled_lora_catalog_works_offline_and_local_catalog_takes_precedence(tmp_path):
    from sdfx.base import BaseEngine
    from sdfx.model_keyword import UPSTREAM_LORA_SHA256
    data = Path(__file__).resolve().parents[1] / 'sdfx' / 'data' / 'model_keyword' / 'lora-keyword.txt'
    assert hashlib.sha256(data.read_bytes()).hexdigest() == UPSTREAM_LORA_SHA256
    engine = BaseEngine()
    index = engine._model_keyword_index('lora', str(tmp_path))
    assert index.paths[0] == data
    (tmp_path / 'lora-keyword.txt').write_text('00003273,custom,example.safetensors\n', encoding='utf8')
    assert engine._model_keyword_index('lora', str(tmp_path)).paths[0] == tmp_path / 'lora-keyword.txt'


def test_no_mapping_and_no_selected_lora_keyword_do_not_inject_words():
    rng = random.Random(3)
    assert plan_prompts(['portrait'], None, 'iterate', None, 'None', None,
                        'model keyword first', 'keyword, prompt', None, 0.7, 0, 0, rng) == ['portrait']
    assert plan_prompts(['portrait'], None, 'iterate', 'cat|dog', 'None', None,
                        'model keyword first', 'keyword, prompt', 'style.safetensors', 0.7, 0, 0, rng) == [
                            '<lora:style:0.7>, portrait']


@pytest.mark.parametrize(('placement', 'expected'), [
    ('keyword prompt', 'key portrait'), ('prompt keyword', 'portrait key'),
    ('keyword, prompt', 'key, portrait'), ('prompt, keyword', 'portrait, key'),
])
def test_all_keyword_placement_modes(placement, expected):
    assert render_prompt('portrait', 'key', None, None, 'model keyword first', placement, None, 1) == expected


def test_inspect_model_reports_legacy_hash_match_and_source(tmp_path):
    model = tmp_path / 'example.safetensors'
    model.write_bytes(b'z' * 42)
    digest = hashlib.sha256(b'').hexdigest()[:8]
    default = tmp_path / 'default.csv'
    user = tmp_path / 'user.csv'
    default.write_text(f'{digest}, red|blue, {model.name}\n', encoding='utf8')
    index = MappingIndex(default, user)
    assert inspect_model(model, index) == {'filename': model.name, 'hash': digest,
                                            'keywords': ['red', 'blue'], 'source': 'default'}
    save_custom(user, digest, 'yellow', model.name)
    assert inspect_model(model, index)['source'] == 'custom'
    assert inspect_model(model, index)['keywords'] == ['yellow']


def test_image_schedules_are_merged_at_every_boundary():
    first = [(2, 'a'), (5, 'b')]
    second = [(3, 'x'), (5, 'y')]
    assert merge_schedules([first, second], tuple) == [(2, ('a', 'x')), (3, ('b', 'x')), (5, ('b', 'y'))]


def test_schedule_rejects_empty_or_nonmonotonic_inputs():
    with pytest.raises(ValueError):
        merge_schedules([[]], tuple)
    with pytest.raises(ValueError):
        merge_schedules([[(2, 'x'), (1, 'y')]], tuple)


def test_mapping_actions_save_check_delete_and_reject_bad_input(tmp_path):
    model = tmp_path / 'model.safetensors'
    model.write_bytes(b'model')
    data = tmp_path / 'default.csv'
    data.write_text('', encoding='utf8')
    user = tmp_path / 'user.csv'
    index = MappingIndex(data, user)
    assert mapping_action('check', model, index)['source'] == 'no match'
    assert mapping_action('save', model, index, 'first|second')['keywords'] == ['first', 'second']
    assert mapping_action('delete', model, index)['deleted'] is True
    assert mapping_action('check', model, index)['keywords'] == []
    with pytest.raises(ValueError):
        mapping_action('save', model, index, ' ')
    with pytest.raises(ValueError):
        mapping_action('unknown', model, index)


def test_concurrent_processes_do_not_lose_mapping_updates(tmp_path):
    path = tmp_path / 'user.csv'
    code = ('import sys; from sdfx.model_keyword import save_custom; '
            'p, prefix = sys.argv[1:]; input(); '
            '[(save_custom(p, "abc", str(i), prefix + str(i))) for i in range(40)]')
    processes = [subprocess.Popen([sys.executable, '-c', code, str(path), prefix], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for prefix in ('a', 'b')]
    for process in processes:
        process.stdin.write('\n'); process.stdin.flush()
    for process in processes:
        out, err = process.communicate(timeout=30)
        assert process.returncode == 0, out + err
    assert len(parse_rows(path.read_text(encoding='utf8'), 1)) == 80


def test_concurrent_index_read_and_edit_do_not_fail(tmp_path):
    default, user = tmp_path / 'default.csv', tmp_path / 'user.csv'
    default.write_text('abc, base, model\n', encoding='utf8')
    user.write_text('abc, custom, model\n', encoding='utf8')
    code = ('import sys; from sdfx.model_keyword import save_custom; '
            'p = sys.argv[1]; print("ready", flush=True); input(); '
            '[(save_custom(p, "abc", str(i), "model")) for i in range(100)]')
    process = subprocess.Popen([sys.executable, '-c', code, str(user)], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert process.stdout.readline().strip() == 'ready'
    process.stdin.write('\n'); process.stdin.flush()
    index = MappingIndex(default, user)
    for _ in range(150):
        assert index.lookup('abc', 'model') is not None
    out, err = process.communicate(timeout=30)
    assert process.returncode == 0, out + err


def test_worker_model_mapping_check_save_delete(tmp_path):
    from sdfx.base import BaseEngine
    model = tmp_path / 'model.safetensors'
    model.write_bytes(b'tiny')
    request = {'action': 'save', 'kind': 'checkpoint', 'model_path': str(model),
               'mapping_dir': str(tmp_path / 'config'), 'keywords': 'a|b'}
    engine = BaseEngine()
    assert engine.model_keyword(request, None)['keywords'] == ['a', 'b']
    checked = engine.model_keyword(dict(request, action='check'), None)
    assert checked['source'] == 'custom'
    assert checked['choices'] == ['keyword1, keyword2', 'random', 'iterate', 'a', 'b']
    assert engine.model_keyword(dict(request, action='delete'), None)['deleted'] is True


def test_worker_lora_custom_mapping_overrides_bundled_and_local_catalogs(tmp_path):
    from sdfx.base import BaseEngine
    model = tmp_path / 'style.safetensors'
    model.write_bytes(b'tiny')
    request = {'action': 'save', 'kind': 'lora', 'model_path': str(model),
               'mapping_dir': str(tmp_path / 'config'), 'keywords': 'one|two'}
    engine = BaseEngine()
    saved = engine.model_keyword(request, None)
    assert saved['keywords'] == ['one', 'two'] and saved['default_catalog_available']
    user_catalog = tmp_path / 'config' / 'lora-keyword.txt'
    user_catalog.write_text(f'{legacy_hash(model)},default,{model.name}\n', encoding='utf8')
    assert engine.model_keyword(dict(request, action='delete'), None)['keywords'] == ['default']
    assert engine.model_keyword(dict(request, action='check'), None)['default_catalog_available']


def test_jsonl_mapping_actions_are_serialized_and_report_errors(tmp_path):
    import io
    import json
    from sdfx.base import BaseEngine
    from sdfx.protocol import Emitter
    from sdfx.server import Server
    model = tmp_path / 'model.safetensors'
    model.write_bytes(b'tiny')
    stream = io.BytesIO()
    server = Server(BaseEngine(), Emitter(stream))
    params = {'kind': 'checkpoint', 'model_path': str(model), 'mapping_dir': str(tmp_path / 'config')}
    server.worker.start()
    try:
        for rid, action in [('bad', 'unknown'), ('save', 'save'), ('check', 'check')]:
            server.handle_line(json.dumps({'id': rid, 'type': 'model_keyword',
                'params': dict(params, action=action, keywords='one|two')}))
        server.q.put(None)
        server.worker.join(10)
        assert not server.worker.is_alive()
    finally:
        if server.worker.is_alive():
            server.q.put(None); server.worker.join(5)
    messages = {item['id']: item for item in map(json.loads, stream.getvalue().splitlines())}
    assert messages['bad']['code'] == 'invalid_params'
    assert messages['check']['result']['keywords'] == ['one', 'two']


def test_relative_lora_name_disambiguates_equal_hashes(tmp_path):
    from sdfx.model_keyword import inspect_model
    model = tmp_path / 'style.safetensors'
    model.write_bytes(b'fixture')
    digest = legacy_hash(model)
    default, user = tmp_path / 'default.csv', tmp_path / 'user.csv'
    default.write_text(f'{digest}, first, a/style.safetensors\n{digest}, second, b/style.safetensors\n', encoding='utf8')
    index = MappingIndex(default, user)
    assert inspect_model(model, index, 'a/style.safetensors')['keywords'] == ['first']
    assert inspect_model(model, index, 'b/style.safetensors')['keywords'] == ['second']


def test_batch_planning_uses_actual_checkpoint_and_lora_mappings(tmp_path):
    checkpoint = tmp_path / 'model.safetensors'; checkpoint.write_bytes(b'one')
    lora = tmp_path / 'style.safetensors'; lora.write_bytes(b'two')
    model_data, lora_data = tmp_path / 'models.csv', tmp_path / 'loras.csv'
    digest = legacy_hash(checkpoint)
    model_data.write_text(f'{digest},red|blue,{checkpoint.name}\n', encoding='utf8')
    lora_data.write_text(f'{digest},cat|dog,{lora.name}\n', encoding='utf8')
    result = plan_for_model(['one', 'two', 'three'], checkpoint, MappingIndex(model_data, tmp_path / 'm-user.csv'),
        lora, MappingIndex(lora_data, tmp_path / 'l-user.csv'),
        {'model_choice': 'iterate', 'lora_choice': '< iterate >', 'lora_name': lora.name,
         'placement': 'keyword, prompt', 'order': 'model keyword first', 'multiplier': 0.5}, random.Random(2))
    assert result == ['<lora:style:0.5>, red, cat, one', '<lora:style:0.5>, blue, dog, two',
                      '<lora:style:0.5>, red, cat, three']


def test_iteration_counters_advance_only_for_iterating_multi_keyword_choices():
    state = IterationState()
    assert state.reserve(3, 'a|b', 'iterate', 'c|d|e', '< iterate >') == (0, 0)
    assert state.reserve(2, 'a', 'iterate', 'c|d', 'None') == (3, 3)
    assert state.reserve(1, 'a|b', 'iterate', 'c|d', '< iterate >') == (3, 3)
    assert state.reserve(1, 'a|b', 'keyword1', None, 'None') == (4, 4)


def test_preview_does_not_advance_iteration_but_does_consume_random_draws():
    rng = random.Random(7)
    preview, images = plan_with_preview('preview', ['first', 'second'], 'red|blue', 'iterate',
        'cat|dog', '< random >', None, 'model keyword first', 'keyword, prompt',
        'style.safetensors', 0.5, 0, 0, rng)
    assert preview == '<lora:style:0.5>, red, dog, preview'
    assert images == ['<lora:style:0.5>, red, cat, first', '<lora:style:0.5>, blue, dog, second']


def test_distinct_per_image_branch_trees_use_only_their_own_branches():
    import numpy as np
    # Two images with one and two branches respectively; unused second branch for image 0 is deliberately huge.
    trees = [(1.,), (2., -1.)]
    branches = [np.array([[3.], [5.]]), np.array([[1000.], [7.]])]
    def weighted(weights, values):
        return sum(weight * value for weight, value in zip(weights, values))
    result = compose_per_item(trees, branches, len, weighted, lambda xs: np.concatenate(xs, axis=0))
    assert result.tolist() == [[3.], [3.]]


def test_unequal_branch_lists_are_transposed_without_losing_image_order():
    assert transpose_branches([['a'], ['x', 'y'], ['m', 'n', 'o']]) == [
        ['a', 'x', 'm'], ['a', 'y', 'n'], ['a', 'x', 'o']]
    with pytest.raises(ValueError):
        transpose_branches([['a'], []])


def test_public_civitai_result_requires_full_file_hash_and_uses_only_trained_words():
    sha = 'a' * 64
    response = {'id': 123, 'trainedWords': ['one', 'two'], 'description': 'unrelated long text',
                'files': [{'hashes': {'AutoV1': 'DEADBEEF', 'SHA256': sha.upper()}}]}
    assert civitai_keywords(response, sha) == ['one', 'two']
    with pytest.raises(ValueError):
        civitai_keywords(response, 'b' * 64)
    with pytest.raises(ValueError):
        civitai_keywords(response, 'deadbeef')
    with pytest.raises(ValueError):
        civitai_keywords(dict(response, trainedWords=['valid', {'not': 'text'}]), sha)


def test_online_lookup_requires_explicit_consent_and_matches_local_full_hash(tmp_path):
    model = tmp_path / 'local.safetensors'
    model.write_bytes(b'model file')
    seen = []
    def fetch(sha):
        seen.append(sha)
        return {'files': [{'hashes': {'SHA256': sha}}], 'trainedWords': ['blue']}
    with pytest.raises(ValueError):
        lookup_civitai(model, False, fetch)
    assert not seen
    assert lookup_civitai(model, True, fetch) == {'keywords': ['blue'], 'source': 'Civitai live', 'sha256': hashlib.sha256(b'model file').hexdigest()}
    assert len(seen) == 1


def test_online_lookup_rejects_changed_model_and_obeys_cancellation(tmp_path):
    model = tmp_path / 'model.safetensors'; model.write_bytes(b'model')
    def mutating_fetch(sha):
        model.write_bytes(b'changed')
        return {'files': [{'hashes': {'SHA256': sha}}], 'trainedWords': ['a']}
    with pytest.raises(ValueError, match='changed'):
        lookup_civitai(model, True, mutating_fetch)
    called = []
    def interrupt():
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError, match='cancelled'):
        lookup_civitai(model, True, lambda sha: called.append(sha), interrupt)
    assert called == []


def test_legacy_hash_rejects_model_replaced_during_read(tmp_path, monkeypatch):
    import sdfx.model_keyword as mod
    path = tmp_path / 'model.safetensors'; path.write_bytes(b'original')
    original = mod.hashlib.sha256
    def mutate(data):
        path.write_bytes(b'changed contents')
        return original(data)
    monkeypatch.setattr(mod.hashlib, 'sha256', mutate)
    with pytest.raises(ValueError, match='changed'):
        legacy_hash(path)


def test_runtime_planner_keeps_independent_iteration_across_jobs(tmp_path):
    checkpoint = tmp_path / 'model.safetensors'; checkpoint.write_bytes(b'one')
    model_data = tmp_path / 'models.csv'
    model_data.write_text(f'{legacy_hash(checkpoint)},red|blue,{checkpoint.name}\n', encoding='utf8')
    index = MappingIndex(model_data, tmp_path / 'user.csv')
    planner = KeywordPlanner(index, None, IterationState())
    opts = {'model_choice': 'iterate', 'placement': 'keyword, prompt'}
    first = planner.plan('preview', ['one', 'two', 'three'], checkpoint, None, opts, random.Random(7))
    assert first['preview'] == 'red, preview'
    assert first['prompts'] == ['red, one', 'blue, two', 'red, three']
    assert first['checkpoint'] == {'filename': checkpoint.name, 'legacy_hash': legacy_hash(checkpoint), 'mapping_source': 'default'}
    assert first['lora'] is None
    second = planner.plan('preview', ['four'], checkpoint, None, opts, random.Random(7))
    assert second['preview'] == 'blue, preview' and second['prompts'] == ['blue, four']


def test_runtime_planner_treats_ui_none_embedding_as_no_embedding(tmp_path):
    checkpoint = tmp_path / 'model.safetensors'; checkpoint.write_bytes(b'model')
    data = tmp_path / 'default.csv'; data.write_text('', encoding='utf8')
    planner = KeywordPlanner(MappingIndex(data, tmp_path / 'user.csv'), None, IterationState())
    result = planner.plan('preview', ['portrait'], checkpoint, None, {'embedding': 'None'}, random.Random(1))
    assert result['prompts'] == ['portrait']


@pytest.mark.parametrize('bad_options', [
    {'placement': 'bad'}, {'order': 'bad'}, {'multiplier': float('nan')}, {'multiplier': 2.1},
])
def test_runtime_planner_rejects_invalid_options_before_reserving_iteration(tmp_path, bad_options):
    checkpoint = tmp_path / 'model.safetensors'; checkpoint.write_bytes(b'one')
    model_data = tmp_path / 'models.csv'
    model_data.write_text(f'{legacy_hash(checkpoint)},red|blue,{checkpoint.name}\n', encoding='utf8')
    state = IterationState()
    planner = KeywordPlanner(MappingIndex(model_data, tmp_path / 'user.csv'), None, state)
    with pytest.raises(ValueError):
        planner.plan('preview', ['one'], checkpoint, None,
                     dict({'model_choice': 'iterate'}, **bad_options), random.Random(7))
    assert state.reserve(0, None, 'None', None, 'None') == (0, 0)


def test_worker_planning_reuses_iteration_state_and_reports_catalog_availability(tmp_path):
    from sdfx.base import BaseEngine
    checkpoint = tmp_path / 'model.safetensors'; checkpoint.write_bytes(b'model')
    mapping_dir = tmp_path / 'config'
    engine = BaseEngine()
    base = {'kind': 'checkpoint', 'model_path': str(checkpoint), 'mapping_dir': str(mapping_dir)}
    engine.model_keyword(dict(base, action='save', keywords='red|blue'), None)
    request = {'checkpoint': str(checkpoint), 'mapping_dir': str(mapping_dir),
               'preview': 'preview', 'prompts': ['one', 'two', 'three'],
               'options': {'model_choice': 'iterate', 'placement': 'keyword, prompt'}, 'choice_seed': 7}
    first = engine.model_keyword_plan(request, None)
    assert first['prompts'] == ['red, one', 'blue, two', 'red, three']
    assert first['preview'] == 'red, preview'
    second = engine.model_keyword_plan(dict(request, prompts=['four']), None)
    assert second['prompts'] == ['blue, four']


def test_jsonl_plan_is_queued_after_mapping_edit(tmp_path):
    import io
    import json
    from sdfx.base import BaseEngine
    from sdfx.protocol import Emitter
    from sdfx.server import Server
    checkpoint = tmp_path / 'model.safetensors'; checkpoint.write_bytes(b'model')
    directory = str(tmp_path / 'config')
    stream = io.BytesIO(); server = Server(BaseEngine(), Emitter(stream)); server.worker.start()
    try:
        edit = {'kind': 'checkpoint', 'model_path': str(checkpoint), 'mapping_dir': directory,
                'action': 'save', 'keywords': 'red|blue'}
        plan = {'checkpoint': str(checkpoint), 'mapping_dir': directory, 'preview': 'p', 'prompts': ['a', 'b'],
                'options': {'model_choice': 'iterate', 'placement': 'keyword, prompt'}, 'choice_seed': 7}
        server.handle_line(json.dumps({'id': 'edit', 'type': 'model_keyword', 'params': edit}))
        server.handle_line(json.dumps({'id': 'plan', 'type': 'model_keyword_plan', 'params': plan}))
        server.q.put(None); server.worker.join(10)
        assert not server.worker.is_alive()
    finally:
        if server.worker.is_alive():
            server.q.put(None); server.worker.join(5)
    responses = {m['id']: m for m in map(json.loads, stream.getvalue().splitlines())}
    assert responses['plan']['result']['prompts'] == ['red, a', 'blue, b']


def test_worker_planner_resolves_user_lora_catalog_and_falls_back_to_bundled(tmp_path):
    from sdfx.base import BaseEngine
    checkpoint = tmp_path / 'checkpoint.safetensors'; checkpoint.write_bytes(b'checkpoint')
    lora = tmp_path / 'style.safetensors'; lora.write_bytes(b'lora')
    mapping = tmp_path / 'config'; mapping.mkdir()
    (mapping / 'lora-keyword.txt').write_text(f'{legacy_hash(lora)},cat|dog,{lora.name}\n', encoding='utf8')
    engine = BaseEngine()
    request = {'checkpoint': str(checkpoint), 'lora': str(lora), 'mapping_dir': str(mapping),
               'preview': 'preview', 'prompts': ['one', 'two'], 'choice_seed': 7,
               'options': {'lora_name': lora.name, 'lora_choice': '< iterate >', 'placement': 'keyword, prompt'}}
    result = engine.model_keyword_plan(request, None)
    assert result['lora_catalog_available'] is True
    assert result['prompts'] == ['<lora:style:0.7>, cat, one', '<lora:style:0.7>, dog, two']
    (mapping / 'lora-keyword.txt').unlink()
    assert engine.model_keyword_plan(request, None)['lora_catalog_available'] is True


def test_dynamic_keyword_choices_follow_original_checkpoint_and_lora_modes():
    assert keyword_choices('checkpoint', []) == ['keyword1, keyword2', 'random', 'iterate', 'keyword1', 'keyword2']
    assert keyword_choices('checkpoint', ['a', 'b']) == ['keyword1, keyword2', 'random', 'iterate', 'a', 'b']
    assert keyword_choices('lora', ['a']) == ['None', 'a']
    assert keyword_choices('lora', ['a', 'b']) == ['None', 'a, b', 'a', 'b', '< iterate >', '< random >']
    with pytest.raises(ValueError):
        keyword_choices('unknown', [])


def test_reforge_extra_networks_use_first_prompt_per_batch_without_rewriting_whitespace():
    prompts = ['before  <lora:first:0.8> , after', 'before <lora:second:0.4> after', 'broken <lora:unfinished']
    cleaned, first = batch_extra_networks(prompts)
    assert cleaned == ['before   , after', 'before  after', 'broken <lora:unfinished']
    assert first == {'lora': [['first', '0.8']]}
    assert batch_extra_networks([]) == ([], None)
    assert batch_extra_networks(['plain', 'later <lora:ignored:1>']) == (['plain', 'later '], {})


def test_worker_plan_returns_batch_first_adapter_activation_and_effective_text(tmp_path):
    from sdfx.base import BaseEngine
    checkpoint = tmp_path / 'model.safetensors'; checkpoint.write_bytes(b'model')
    directory = str(tmp_path / 'config')
    engine = BaseEngine()
    engine.model_keyword({'action': 'save', 'kind': 'checkpoint', 'model_path': str(checkpoint),
                          'mapping_dir': directory, 'keywords': '<lora:A:0.5>|<lora:B:0.8>'}, None)
    plan = engine.model_keyword_plan({'checkpoint': str(checkpoint), 'mapping_dir': directory,
        'preview': 'p', 'prompts': ['one', 'two', 'three', 'four'], 'batch_size': 2, 'choice_seed': 3,
        'options': {'model_choice': 'iterate'}}, None)
    assert plan['prompts'] == ['<lora:A:0.5> one', '<lora:B:0.8> two', '<lora:A:0.5> three', '<lora:B:0.8> four']
    assert [batch['networks'] for batch in plan['batches']] == [{'lora': [['A', '0.5']]}, {'lora': [['A', '0.5']]}]
    assert plan['batches'][0]['clean_prompts'] == [' one', ' two']
    assert plan['batches'][0]['adapters'] == [{'kind': 'lora', 'name': 'A', 'te': 0.5, 'unet': 0.5, 'dyn': None}]


def test_invalid_batch_does_not_advance_worker_iteration(tmp_path):
    from sdfx.base import BaseEngine
    from sdfx.protocol import EngineError
    checkpoint = tmp_path / 'model.safetensors'; checkpoint.write_bytes(b'model')
    directory = str(tmp_path / 'config'); engine = BaseEngine()
    engine.model_keyword({'action': 'save', 'kind': 'checkpoint', 'model_path': str(checkpoint),
                          'mapping_dir': directory, 'keywords': 'red|blue'}, None)
    request = {'checkpoint': str(checkpoint), 'mapping_dir': directory, 'preview': 'p',
               'prompts': ['one', 'two', 'three'], 'batch_size': 2,
               'options': {'model_choice': 'iterate'}}
    with pytest.raises(EngineError):
        engine.model_keyword_plan(request, None)
    with pytest.raises(EngineError):
        engine.model_keyword_plan(dict(request, prompts=[]), None)
    request['batch_size'] = 1
    assert engine.model_keyword_plan(request, None)['prompts'][0] == 'red one'


def test_assemble_neutral_aligns_trees_and_fills_unused_branches():
    from types import SimpleNamespace
    one = {'value': 'A'}
    two = {'value': 'B', '_neutral': SimpleNamespace(tree='tree-b', phi=0.2, branches=['B0', 'B1'])}
    trees, branches, phi = assemble_neutral([one, two], ['tree-a', 'unused'], lambda values: tuple(values))
    assert trees == ['tree-a', 'tree-b']
    assert branches == [(one, 'B0'), (one, 'B1')]
    assert phi == 0.2


def test_assemble_neutral_rejects_inconsistent_per_item_phi():
    from types import SimpleNamespace
    plans = [dict(_neutral=SimpleNamespace(tree='a', phi=0.1, branches=['a'])),
             dict(_neutral=SimpleNamespace(tree='b', phi=0.2, branches=['b']))]
    with pytest.raises(ValueError):
        assemble_neutral(plans, ['a', 'b'], tuple)


def test_worker_planning_obeys_pre_dispatch_interruption_without_advancing(tmp_path):
    from sdfx.base import BaseEngine
    checkpoint = tmp_path / 'model.safetensors'; checkpoint.write_bytes(b'model')
    directory = str(tmp_path / 'config'); engine = BaseEngine()
    engine.model_keyword({'action': 'save', 'kind': 'checkpoint', 'model_path': str(checkpoint),
                          'mapping_dir': directory, 'keywords': 'red|blue'}, None)
    request = {'checkpoint': str(checkpoint), 'mapping_dir': directory, 'preview': 'p', 'prompts': ['a'],
               'options': {'model_choice': 'iterate'}}
    class Cancelled:
        def check_interrupt(self):
            raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError, match='cancelled'):
        engine.model_keyword_plan(request, Cancelled())
    assert engine.model_keyword_plan(request, None)['prompts'] == ['red a']


def test_batch_plan_validation_rejects_tampered_first_prompt_networks():
    plan = {'preview': 'preview', 'prompts': ['<lora:A:1> one', '<lora:B:1> two'],
            'batches': [{'clean_prompts': [' one', ' two'], 'networks': {'lora': [['A', '1']]},
                         'adapters': [{'kind': 'lora', 'name': 'A', 'te': 1.0, 'unet': 1.0, 'dyn': None}]}]}
    assert validate_batch_plan(plan, batch_size=2, n_iter=1) is None
    plan['batches'][0]['networks'] = {'lora': [['B', '1']]}
    with pytest.raises(ValueError, match='batch'):
        validate_batch_plan(plan, batch_size=2, n_iter=1)


def test_reforge_lora_activation_preserves_named_and_positional_weights_and_dyn():
    assert parse_lora_activation(['style']) == {'name': 'style', 'te': 1.0, 'unet': 1.0, 'dyn': None}
    assert parse_lora_activation(['style', '0.5', '0.7', '12', 'te=0.2', 'dyn=16']) == {
        'name': 'style', 'te': 0.2, 'unet': 0.7, 'dyn': 16}
    assert parse_lora_activation(['style', '0.5', 'te=0.25'])['unet'] == 0.25
    with pytest.raises(ValueError):
        parse_lora_activation(['style', 'NaN'])




def test_composable_adapter_weights_respect_each_image_and_guidance_group():
    from sdfx.clora.runtime import branch_weights
    def weight(text, negative, full, negative_prompt):
        return float(text) * (-1 if negative else 1) + len(full) * 10 + len(negative_prompt)
    values = branch_weights([False, True], ['1', '2'], ['a', 'bb'], ['x', 'yy'], weight)
    assert values == [10., 20., 12., 24.]
    assert branch_weights([False, True], '1', 'a', 'x', weight) == [10., 12.]
    with pytest.raises(ValueError):
        branch_weights([True], ['1', '2'], ['a'], ['x', 'y'], weight)


def test_worker_cancel_after_selection_does_not_consume_iteration(tmp_path):
    from sdfx.base import BaseEngine
    checkpoint = tmp_path / 'model.safetensors'; checkpoint.write_bytes(b'model')
    directory = str(tmp_path / 'config'); engine = BaseEngine()
    engine.model_keyword({'action': 'save', 'kind': 'checkpoint', 'model_path': str(checkpoint),
                          'mapping_dir': directory, 'keywords': 'red|blue'}, None)
    request = {'checkpoint': str(checkpoint), 'mapping_dir': directory, 'preview': 'p', 'prompts': ['a'],
               'options': {'model_choice': 'iterate'}}
    class CancelAfterPlan:
        calls = 0
        def check_interrupt(self):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError, match='cancelled'):
        engine.model_keyword_plan(request, CancelAfterPlan())
    assert engine.model_keyword_plan(request, None)['prompts'] == ['red a']


def test_iteration_transaction_rollback_does_not_rewind_other_reservations():
    import threading
    state = IterationState()
    entered = threading.Event(); proceed = threading.Event(); result = []
    def cancelled():
        with pytest.raises(RuntimeError):
            with state.transaction():
                state.reserve(1, 'a|b', 'iterate', None, 'None')
                entered.set()
                assert proceed.wait(5)
                raise RuntimeError('cancelled')
    def successful():
        assert entered.wait(5)
        with state.transaction():
            result.append(state.reserve(1, 'a|b', 'iterate', None, 'None'))
    first = threading.Thread(target=cancelled); second = threading.Thread(target=successful)
    first.start(); second.start(); assert entered.wait(5); proceed.set()
    first.join(5); second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert result == [(0, 0)]
    assert state.reserve(1, 'a|b', 'iterate', None, 'None') == (1, 0)


def test_batch_plan_rejects_unsupported_first_prompt_network_instead_of_dropping_it():
    plan = {'preview': '', 'prompts': ['<hypernet:style:1> cat'],
            'batches': [{'clean_prompts': [' cat'], 'networks': {'hypernet': [['style', '1']]}, 'adapters': []}]}
    with pytest.raises(ValueError, match='unsupported.*hypernet'):
        validate_batch_plan(plan, batch_size=1, n_iter=1)


def test_neutral_collation_builds_batch_branch_schedules_without_torch(monkeypatch):
    from sdfx.model_keyword_conditioning import collate_neutral
    from sdfx.neutral_prompt import parse
    from sdfx.neutral_runtime import Plan
    import sdfx.model_keyword_conditioning as conditioning
    one, two = parse('one'), parse('one AND two')
    a, b, c = {'name': 'a'}, {'name': 'b'}, {'name': 'c'}
    items = [a, {'name': 'b', '_neutral': Plan(two, [b, c])}]
    monkeypatch.setattr(conditioning, 'collate_schedules',
                        lambda timelines, empty: [(3, {'members': [t[0][1]['name'] for t in timelines]})])
    output = collate_neutral(items, [one, two], None, 3, lambda variants: variants)
    assert output['_neutral'].image_trees == [one, two]
    assert [branch['members'] for branch in output['_neutral'].branches] == [['a', 'b'], ['a', 'c']]


def test_per_image_empty_padding_routing_without_torch(monkeypatch):
    import sys
    from types import SimpleNamespace
    from sdfx.model_keyword_conditioning import collate_plain
    class Tensor:
        def __init__(self, rows):
            self.rows = rows; self.shape = (len(rows), len(rows[0]), 1)
            self.device = 'cpu'; self.dtype = 'float'
        def abs(self): return self
        def any(self): return any(value != 0 for row in self.rows for value in row)
        def to(self, *args): return self
        def repeat(self, batch, tokens, dim): return Tensor([self.rows[0] * tokens for _ in range(batch)])
    def cat(tensors, dim):
        if dim == 0: return Tensor([row for tensor in tensors for row in tensor.rows])
        return Tensor([sum((tensor.rows[i] for tensor in tensors), []) for i in range(tensors[0].shape[0])])
    fake = SimpleNamespace(cat=cat, zeros_like=lambda t: Tensor([[0] * t.shape[1] for _ in range(t.shape[0])]))
    monkeypatch.setitem(sys.modules, 'torch', fake)
    def item(values):
        t = Tensor([values]); return {'prompt_embeds': t, 'negative_prompt_embeds': t}
    result = collate_plain([item([1]), item([2, 2])], [Tensor([[5]]), Tensor([[9]])])
    assert result['prompt_embeds'].rows == [[1, 5], [2, 2]]


def test_generation_batch_adapters_use_first_prompt_and_keep_manual_refs(tmp_path):
    from sdfx.model_keyword import generation_batch_loras
    from sdfx.params import LoraRef
    path = tmp_path / 'style.safetensors'; path.write_bytes(b'x')
    manual = LoraRef(name='manual', weight=.2)
    batch = {'adapters': [{'kind': 'lora', 'name': 'style', 'te': .4, 'unet': .7, 'dyn': None}]}
    refs = generation_batch_loras(batch, [manual], {'style': str(path)})
    assert refs == [manual, LoraRef(name='style', weight=.4, unet=.7)]
    aliases = {'STYLE': str(path)}
    assert generation_batch_loras(batch, [], aliases)[0].name == 'style'
    assert aliases['style'] == str(path)
    assert generation_batch_loras({'adapters': [dict(batch['adapters'][0], dyn=4)]}, [], {'style': str(path)})[0].dyn == 4
    with pytest.raises(ValueError, match='style'):
        generation_batch_loras(batch, [], {})


def test_keyword_batch_preserves_lbw_and_step_window_in_lora_tags(tmp_path):
    from sdfx.model_keyword import generation_batch_loras
    path = tmp_path / 'style.safetensors'; path.write_bytes(b'x')
    tags = ['style', '0.7', '0.6', '2', 'lbw=ALL', 'start=0.2', 'stop=0.8']
    adapter = parse_lora_activation(tags)
    ref = generation_batch_loras({'adapters': [adapter]}, [], {'style': str(path)})[0]
    assert (ref.dyn, ref.lbw, ref.start, ref.stop) == (2, 'ALL', .2, .8)
    ref = generation_batch_loras({'adapters': [parse_lora_activation(
        ['style', 'step=3-8', 'start=2', 'LBW=INS'])]}, [], {'style': str(path)})[0]
    assert (ref.lbw, ref.start, ref.stop) == ('INS', 2., 8.)


def test_batch_conditioning_encodes_each_prompt_and_preserves_schedule(monkeypatch):
    import sdfx.model_keyword_conditioning as conditioning
    seen = []
    monkeypatch.setattr(conditioning, 'collate_schedules', lambda timelines, empty: [
        (2, {'values': [timeline[0][1]['value'] for timeline in timelines]}),
        (4, {'values': [timeline[-1][1]['value'] for timeline in timelines]})])
    class Scheduled(dict):
        def __init__(self, variants):
            super().__init__(variants[0][1]); self.schedule = variants
    def encode(text):
        seen.append(text)
        if text == 'first': return Scheduled([(2, {'value': 1}), (4, {'value': 2})])
        return {'value': 3}
    result = conditioning.encode_batch(['first', 'second'], encode, None, 4, Scheduled)
    assert seen == ['first', 'second']
    assert result.schedule[-1][1]['values'] == [2, 3]


def test_dryrun_generation_emits_each_effective_prompt(tmp_path):
    import io
    from sdfx.dryrun import FakeEngine
    from sdfx.protocol import Emitter, JobContext
    checkpoint = tmp_path / 'model.safetensors'; checkpoint.write_bytes(b'model')
    directory = str(tmp_path / 'config'); engine = FakeEngine()
    engine.model = {'name': 'model', 'path': str(checkpoint), 'kind': 'sd15', 'hash': '1234'}
    engine.model_keyword({'action': 'save', 'kind': 'checkpoint', 'model_path': str(checkpoint),
                          'mapping_dir': directory, 'keywords': 'red|blue'}, None)
    ctx = JobContext(Emitter(io.BytesIO()), 'test')
    result = engine.txt2img({'output_dir': str(tmp_path), 'params': {'seed': 7, 'width': 64, 'height': 64,
        'steps': 1, 'batch_size': 2, 'n_iter': 2, 'prompt': 'cat',
        'hooks': {'model_keyword': {'_mapping_dir': directory, 'model_choice': 'iterate'}}}}, ctx)
    assert [image['effective_prompt'] for image in result['images']] == ['red cat', 'blue cat', 'red cat', 'blue cat']
    assert [image['model_keyword_source_prompt'] for image in result['images']] == ['red cat', 'blue cat', 'red cat', 'blue cat']


def test_batch_effective_prompts_record_activated_first_image_adapters():
    from sdfx.model_keyword import replay_prompts_for_batch
    prompts = ['red <lora:A:0.5> cat', 'blue <lora:B:1> dog']
    assert replay_prompts_for_batch(prompts) == ['<lora:A:0.5>red  cat', '<lora:A:0.5>blue  dog']


def test_multiple_adapter_replay_preserves_exact_conditioned_spacing():
    from sdfx.model_keyword import batch_extra_networks, replay_prompts_for_batch
    original = ['<lora:A:0.7> red <lora:A:0.7:0.6:2:lbw=ALL> cat',
                '<lora:B:1> blue dog']
    replay = replay_prompts_for_batch(original)
    assert batch_extra_networks(replay)[0] == batch_extra_networks(original)[0]


def test_composable_signed_negative_keeps_marker_and_aligns_both_contexts():
    from sdfx.clora.runtime import merge_signed_contexts
    calls = []
    def align(variants):
        calls.append(variants)
        return [(step, dict(value, _negpip=True, aligned=True)) for step, value in variants]
    positive = {'prompt_embeds': 'positive', 'negative_prompt_embeds': 'unused'}
    negative = {'prompt_embeds': 'unused', 'negative_prompt_embeds': 'negative', '_negpip': True}
    result = merge_signed_contexts(positive, negative, align)
    assert len(calls) == 1 and result['_negpip'] is True
    assert result['prompt_embeds'] == 'positive' and result['negative_prompt_embeds'] == 'negative'
    assert result['aligned'] is True


def test_composable_inversion_hook_requires_routed_branches():
    from types import SimpleNamespace
    from sdfx.clora.runtime import inversion_branch_hook
    job = SimpleNamespace(config={})
    gp = SimpleNamespace(hires=SimpleNamespace(enabled=False), img2img=object())
    scheduler = object()
    hook = inversion_branch_hook(job, SimpleNamespace(branches=[{'_clora_branch': 'red'}]), scheduler, gp, 12)
    assert hook.job is job and hook.scheduler is scheduler and hook.total == 12 and hook.pass_name == 'inversion'
    assert inversion_branch_hook(job, SimpleNamespace(branches=[{}]), scheduler, gp, 12) is None
    assert inversion_branch_hook(None, SimpleNamespace(branches=[]), scheduler, gp, 12) is None


def test_keyword_composable_inversion_override_requires_resolved_adapter(tmp_path):
    from sdfx.base import BaseEngine
    from sdfx.params import GenerationParams, GenerationRequest
    from sdfx.protocol import EngineError
    checkpoint = tmp_path / 'model.safetensors'; checkpoint.write_bytes(b'model')
    adapter = tmp_path / 'inversion.safetensors'; adapter.write_bytes(b'adapter')
    engine = BaseEngine(); engine.model = {'path': str(checkpoint)}
    gp = GenerationParams(prompt='cat', hooks={
        'model_keyword': {'_mapping_dir': str(tmp_path / 'config')},
        'composable_lora': {}, 'img2imgalt': {'override_prompt': '<lora:inversion:0.6:0.7:2> dog'}})
    request = GenerationRequest(params=gp, lora_paths={}, manual_loras=[])
    with pytest.raises(EngineError, match='inversion'):
        engine.generation_keyword_plan(gp, request, 7, None)
    request.lora_paths['inversion'] = str(adapter)
    plan, _ = engine.generation_keyword_plan(gp, request, 7, None)
    assert plan['prompts'] == ['cat']


def test_generation_plan_activates_only_first_image_network_per_batch(tmp_path):
    from sdfx.base import BaseEngine
    from sdfx.params import GenerationParams, GenerationRequest, LoraRef
    checkpoint = tmp_path / 'model.safetensors'; checkpoint.write_bytes(b'model')
    a = tmp_path / 'A.safetensors'; a.write_bytes(b'a')
    b = tmp_path / 'B.safetensors'; b.write_bytes(b'b')
    directory = str(tmp_path / 'config'); engine = BaseEngine()
    engine.model = {'path': str(checkpoint)}
    engine.model_keyword({'action': 'save', 'kind': 'checkpoint', 'model_path': str(checkpoint),
        'mapping_dir': directory, 'keywords': 'red <lora:A:0.5>|blue <lora:B:0.7>'}, None)
    gp = GenerationParams(prompt='cat', batch_size=2, n_iter=1, loras=[LoraRef(name='B', weight=.2)],
        hooks={'model_keyword': {'_mapping_dir': directory, 'model_choice': 'iterate'}})
    req = GenerationRequest(params=gp, lora_paths={'A': str(a), 'B': str(b)})
    plan, refs = engine.generation_keyword_plan(gp, req, 7, None)
    assert [ref.name for ref in refs[0]] == ['B', 'A']
    assert plan['batches'][0]['clean_prompts'] == ['red  cat', 'blue  cat']
    assert plan['replay_prompts'] == ['<lora:A:0.5>red  cat', '<lora:A:0.5>blue  cat']


def test_adetailer_receives_each_effective_prompt_and_seed():
    from sdfx.model_keyword import postprocess_keyword_items
    seen = []
    result = postprocess_keyword_items(['image-1', 'image-2'], ['red cat', 'blue cat'], [7, 8],
        lambda image, prompt, seed, index: seen.append((image, prompt, seed, index)) or image + '!')
    assert result == ['image-1!', 'image-2!']
    assert seen == [('image-1', 'red cat', 7, 0), ('image-2', 'blue cat', 8, 1)]


def test_composable_plan_resolves_adapters_from_every_image(tmp_path):
    from sdfx.model_keyword import generation_composable_loras
    paths = {}
    for name in ('A', 'B'):
        path = tmp_path / (name + '.safetensors'); path.write_bytes(b'x'); paths[name] = str(path)
    refs = generation_composable_loras(['<lora:A:0.5> cat', '<lora:B:0.7> dog'], [], paths)
    assert [item.name for item in refs] == ['A', 'B']


def test_composable_padding_uses_branch_and_guidance_specific_empty(monkeypatch):
    import sys
    from types import SimpleNamespace
    from sdfx.model_keyword_conditioning import collate_plain
    class Tensor:
        def __init__(self, rows):
            self.rows = rows; self.shape = (len(rows), len(rows[0]), 1); self.device = 'cpu'; self.dtype = 'float'
        def abs(self): return self
        def any(self): return any(x != 0 for row in self.rows for x in row)
        def to(self, *args): return self
        def repeat(self, batch, tokens, dim): return Tensor([self.rows[0] * tokens for _ in range(batch)])
    def cat(tensors, dim):
        if dim == 0: return Tensor([row for tensor in tensors for row in tensor.rows])
        return Tensor([sum((tensor.rows[i] for tensor in tensors), []) for i in range(tensors[0].shape[0])])
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cat=cat, zeros_like=lambda t: Tensor([[0] * t.shape[1]])))
    def item(values, branch):
        t = Tensor([values]); return {'prompt_embeds': t, 'negative_prompt_embeds': t, '_clora_branch': branch}
    def empty(index, item, key):
        return Tensor([[5 if key == 'prompt_embeds' else 7]])
    result = collate_plain([item([1], 'a'), item([2, 2], 'b')], empty)
    assert result['prompt_embeds'].rows == [[1, 5], [2, 2]]
    assert result['negative_prompt_embeds'].rows == [[1, 7], [2, 2]]


def test_photomaker_plans_different_or_identical_batch_prompts(tmp_path):
    from sdfx.base import BaseEngine
    from sdfx.params import GenerationParams, GenerationRequest
    checkpoint = tmp_path / 'model.safetensors'; checkpoint.write_bytes(b'model')
    directory = str(tmp_path / 'config'); engine = BaseEngine(); engine.model = {'path': str(checkpoint)}
    engine.model_keyword({'action': 'save', 'kind': 'checkpoint', 'model_path': str(checkpoint),
        'mapping_dir': directory, 'keywords': 'red|blue'}, None)
    options = {'_mapping_dir': directory, 'model_choice': 'iterate'}
    gp = GenerationParams(prompt='a person img', batch_size=2, hooks={'model_keyword': options, 'photomaker': {}})
    plan, _ = engine.generation_keyword_plan(gp, GenerationRequest(params=gp), 1, None)
    assert plan['replay_prompts'] == ['red a person img', 'blue a person img']
    options['model_choice'] = 'red'
    plan, _ = engine.generation_keyword_plan(gp, GenerationRequest(params=gp), 1, None)
    assert plan['replay_prompts'] == ['red a person img', 'red a person img']


def test_generation_validation_failure_does_not_consume_iteration(tmp_path):
    from sdfx.base import BaseEngine
    from sdfx.params import GenerationParams, GenerationRequest
    from sdfx.protocol import EngineError
    checkpoint = tmp_path / 'model.safetensors'; checkpoint.write_bytes(b'model')
    model = tmp_path / 'A.safetensors'; model.write_bytes(b'a')
    directory = str(tmp_path / 'config'); engine = BaseEngine(); engine.model = {'path': str(checkpoint)}
    engine.model_keyword({'action': 'save', 'kind': 'checkpoint', 'model_path': str(checkpoint),
        'mapping_dir': directory, 'keywords': 'red <lora:missing:1>|blue'}, None)
    gp = GenerationParams(prompt='cat', hooks={'model_keyword': {'_mapping_dir': directory, 'model_choice': 'iterate'}})
    req = GenerationRequest(params=gp, lora_paths={'A': str(model)})
    with pytest.raises(EngineError, match='missing'):
        engine.generation_keyword_plan(gp, req, 1, None)
    with pytest.raises(EngineError, match='missing'):
        engine.generation_keyword_plan(gp, req, 1, None)


def test_single_image_photomaker_keyword_plan_is_allowed(tmp_path):
    from sdfx.base import BaseEngine
    from sdfx.params import GenerationParams, GenerationRequest
    checkpoint = tmp_path / 'model.safetensors'; checkpoint.write_bytes(b'model')
    directory = str(tmp_path / 'config'); engine = BaseEngine(); engine.model = {'path': str(checkpoint)}
    gp = GenerationParams(prompt='a person img', hooks={'model_keyword': {'_mapping_dir': directory}, 'photomaker': {}})
    plan, _ = engine.generation_keyword_plan(gp, GenerationRequest(params=gp), 1, None)
    assert plan['replay_prompts'] == ['a person img']


def test_lora_choice_from_previous_model_does_not_insert_first_keyword(tmp_path):
    from sdfx.base import BaseEngine
    checkpoint = tmp_path / 'model.safetensors'; checkpoint.write_bytes(b'model')
    lora = tmp_path / 'chosen.safetensors'; lora.write_bytes(b'lora')
    directory = str(tmp_path / 'config'); engine = BaseEngine()
    engine.model_keyword({'action': 'save', 'kind': 'lora', 'model_path': str(lora),
                          'mapping_dir': directory, 'keywords': 'red|blue'}, None)
    plan = engine.model_keyword_plan({'checkpoint': str(checkpoint), 'lora': str(lora), 'mapping_dir': directory,
        'preview': 'cat', 'prompts': ['cat'], 'options': {'lora_choice': 'old-model-word'}}, None)
    assert plan['prompts'] == ['<lora:chosen:0.7> cat']


def test_custom_edit_preserves_unrelated_bytes_and_retains_previous_copy(tmp_path):
    path = tmp_path / 'custom-mappings.txt'
    old = b'# handwritten comment\r\n"old",  "two\nlines", other\r\nabc, red, model\r\n'
    path.write_bytes(old)
    save_custom(path, 'abc', 'blue', 'model')
    assert path.read_bytes().startswith(b'# handwritten comment\r\n"old",  "two\nlines", other\r\n')
    backups = list(tmp_path.glob('custom-mappings.txt.backup-*'))
    assert len(backups) == 1 and backups[0].read_bytes() == old


def test_img2img_hires_combination_uses_keyword_preview_for_inherited_prompt(tmp_path):
    from sdfx.base import BaseEngine
    from sdfx.params import GenerationParams, GenerationRequest
    checkpoint = tmp_path / 'model.safetensors'; checkpoint.write_bytes(b'model')
    directory = str(tmp_path / 'config'); engine = BaseEngine(); engine.model = {'path': str(checkpoint)}
    engine.model_keyword({'action': 'save', 'kind': 'checkpoint', 'model_path': str(checkpoint),
                          'mapping_dir': directory, 'keywords': 'red|blue'}, None)
    gp = GenerationParams(prompt='cat', batch_size=2,
        hooks={'model_keyword': {'_mapping_dir': directory, 'model_choice': 'iterate'}, 'img2img_hires_fix': {}})
    plan, _ = engine.generation_keyword_plan(gp, GenerationRequest(params=gp), 1, None)
    assert plan['preview'] == 'red cat'
    assert plan['prompts'] == ['red cat', 'blue cat']
    assert plan['refinement_prompt'] == 'red cat'


def test_dynamic_rank_transform_keeps_original_alpha_scaling_without_torch():
    from sdfx.dynamic_lora import truncate_state_dict
    class Tensor:
        def __init__(self, shape, value=None): self.shape, self.value = shape, value
        def __getitem__(self, key):
            if not isinstance(key, tuple): key = (key,)
            shape = list(self.shape)
            for axis, item in enumerate(key):
                if isinstance(item, slice): shape[axis] = len(range(shape[axis])[item])
            return Tensor(tuple(shape), self.value)
        def contiguous(self): return self
        def __mul__(self, number): return Tensor(self.shape, self.value * number)
    source = {'layer.lora_down.weight': Tensor((4, 3)), 'layer.lora_up.weight': Tensor((2, 4)),
              'layer.alpha': Tensor((), 8.)}
    result = truncate_state_dict(source, 2)
    assert result['layer.lora_down.weight'].shape == (2, 3)
    assert result['layer.lora_up.weight'].shape == (2, 2)
    assert result['layer.alpha'].value == 4.
    assert source['layer.lora_down.weight'].shape == (4, 3)


def test_dynamic_rank_rejects_dora_and_unknown_factor_names():
    from sdfx.dynamic_lora import truncate_state_dict
    with pytest.raises(ValueError, match='DoRA'):
        truncate_state_dict({'layer.dora_scale': object()}, 2)
    with pytest.raises(ValueError, match='tensor name'):
        truncate_state_dict({'layer.lora_A.other.weight': object()}, 2)


def test_composable_plan_keeps_per_image_rank_variants(tmp_path):
    from sdfx.model_keyword import generation_composable_loras
    path = tmp_path / 'A.safetensors'; path.write_bytes(b'x')
    refs = generation_composable_loras(['<lora:A:1:1:2> cat', '<lora:A:1:1:3> dog'], [], {'A': str(path)})
    assert [(ref.name, ref.dyn) for ref in refs] == [('A', 2), ('A', 3)]
    same = generation_composable_loras(['<lora:A:1:1:2> <lora:A:1:1:3>'], [], {'A': str(path)})
    assert [(ref.name, ref.dyn) for ref in same] == [('A', 2), ('A', 3)]


def test_dynamic_rank_accepts_native_peft_dora_only_when_explicitly_enabled():
    from sdfx.dynamic_lora import truncate_state_dict
    class Tensor:
        shape = (2, 2)
    source = {'unet.block.lora_magnitude_vector.weight': object(),
              'unet.block.lora_A.weight': Tensor(), 'unet.block.lora_B.weight': Tensor()}
    with pytest.raises(ValueError, match='DoRA'):
        truncate_state_dict(source, 2)
    assert truncate_state_dict(source, 2, allow_peft_dora=True) == source
    with pytest.raises(ValueError, match='DoRA'):
        truncate_state_dict({'lora_unet_x.dora_scale': object()}, 2, allow_peft_dora=True)


def test_dynamic_rank_accepts_kohya_dora_only_in_explicit_mode():
    from sdfx.dynamic_lora import truncate_state_dict
    class Tensor:
        shape = (2, 2)
    source = {'lora_unet_x.dora_scale': object(),
              'lora_unet_x.lora_down.weight': Tensor(), 'lora_unet_x.lora_up.weight': Tensor()}
    with pytest.raises(ValueError, match='DoRA'):
        truncate_state_dict(source, 2)
    assert truncate_state_dict(source, 2, allow_kohya_dora=True) == source


def test_kohya_dora_loader_preserves_magnitude_and_alphas_without_torch(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    from sdfx.dynamic_lora import load_kohya_dora
    diffusers, loaders, conversion = ModuleType('diffusers'), ModuleType('diffusers.loaders'), ModuleType('diffusers.loaders.lora_conversion_utils')
    conversion._maybe_map_sgm_blocks_to_diffusers = lambda data, config: data
    class Tensor:
        def numel(self): return 2
        def reshape(self, shape): assert shape == (2,); return self
    conversion._convert_non_diffusers_lora_to_diffusers = lambda data: ({'unet.x.lora_magnitude_vector.weight': Tensor()}, {'alpha': 2.})
    for name, module in [('diffusers', diffusers), ('diffusers.loaders', loaders),
                         ('diffusers.loaders.lora_conversion_utils', conversion)]:
        monkeypatch.setitem(sys.modules, name, module)
    calls = []
    layer = SimpleNamespace(weight=SimpleNamespace(ndim=2, shape=(2, 3)))
    pipe = SimpleNamespace(unet=SimpleNamespace(config={}, get_submodule=lambda _: layer), text_encoder=object(), text_encoder_2=object(), lora_scale=1.)
    pipe.load_lora_into_unet = lambda data, **kw: calls.append(('unet', data, kw))
    pipe.load_lora_into_text_encoder = lambda data, **kw: calls.append((kw['prefix'], data, kw))
    load_kohya_dora(pipe, {'lora_unet_x.dora_scale': 7}, 'adapter')
    assert [call[0] for call in calls] == ['unet', 'text_encoder', 'text_encoder_2']
    assert all(call[2]['network_alphas'] == {'alpha': 2.} and call[2]['adapter_name'] == 'adapter' for call in calls)


def test_dora_weight_hooks_use_per_component_multipliers_and_cleanup():
    from sdfx.dynamic_lora import install_dora_weight_hooks
    class Handle:
        def __init__(self, owner): self.owner = owner
        def remove(self): self.owner.hook = None
    class Dora:
        def __init__(self): self.hook = None
        def register_forward_hook(self, fn): self.hook = fn; return Handle(self)
    class Model:
        def __init__(self, leaf): self.leaf = leaf
        def named_modules(self): return [('layer', type('Layer', (), {'lora_magnitude_vector': {'a': self.leaf}})())]
    unet, text = Dora(), Dora()
    weights = {'a': {'unet': .25, 'text_encoder': .75}}
    handles = install_dora_weight_hooks([('unet', Model(unet)), ('text_encoder', Model(text))], weights)
    assert unet.hook(None, (), 8.) == 2.
    assert text.hook(None, (), 8.) == 6.
    weights['a']['unet'] = 0.
    assert unet.hook(None, (), 8.) == 0. and text.hook(None, (), 8.) == 6.
    for handle in handles: handle.remove()
    assert unet.hook is None and text.hook is None
    with pytest.raises(ValueError, match='missing'):
        install_dora_weight_hooks([('unet', Model(unet))],
                                  {'a': {'unet': .5}, 'missing': {'unet': 1.}})
    assert unet.hook is None


def test_dora_block_capture_preserves_distinct_layer_multipliers():
    from sdfx.dynamic_lora import capture_dora_block_weights, install_dora_weight_hooks
    class Handle:
        def __init__(self, owner): self.owner = owner
        def remove(self): self.owner.hook = None
    class Dora:
        def register_forward_hook(self, fn): self.hook = fn; return Handle(self)
    class Layer:
        def __init__(self, scale):
            self.lora_magnitude_vector = {'a': Dora()}
            self.lora_alpha, self.r, self.scaling = {'a': 4.}, {'a': 2}, {'a': scale * 2}
    first, second, text = Layer(.25), Layer(.5), Layer(.75)
    class Model:
        def __init__(self, layers): self.layers = layers
        def named_modules(self): return self.layers
    components = [('unet', Model([('one', first), ('two', second)])),
                  ('text_encoder', Model([('text', text)]))]
    captured = capture_dora_block_weights(components, ['a'])
    handles = install_dora_weight_hooks(components, captured)
    assert first.lora_magnitude_vector['a'].hook(None, (), 8.) == 2.
    assert second.lora_magnitude_vector['a'].hook(None, (), 8.) == 4.
    assert text.lora_magnitude_vector['a'].hook(None, (), 8.) == 6.
    captured['a'][('unet', 'one')]['unet'] = 0.
    assert first.lora_magnitude_vector['a'].hook(None, (), 8.) == 0.
    assert second.lora_magnitude_vector['a'].hook(None, (), 8.) == 4.
    for handle in handles: handle.remove()


def test_upstream_catalog_install_requires_consent_and_validates_without_overwriting(tmp_path):
    import hashlib, io
    from sdfx.model_keyword import install_upstream_lora_catalog, UPSTREAM_LORA_URL
    contents = b'12345678, keyword, example.safetensors\n'
    index = MappingIndex(tmp_path / 'lora-keyword.txt', tmp_path / 'lora-keyword-user.txt', allow_missing_default=True)
    assert index.lookup('12345678', 'example.safetensors') is None
    calls = []
    def fetch(url):
        calls.append(url)
        return io.BytesIO(contents)
    options = {'open_url': fetch, 'expected_sha256': hashlib.sha256(contents).hexdigest(), 'expected_size': len(contents)}
    with pytest.raises(ValueError, match='confirmation'):
        install_upstream_lora_catalog(tmp_path, confirmed=False, **options)
    assert calls == [] and not (tmp_path / 'lora-keyword.txt').exists()
    with pytest.raises(ValueError, match='hash and size'):
        install_upstream_lora_catalog(tmp_path, confirmed=True, **dict(options, expected_sha256='0' * 64))
    assert not (tmp_path / 'lora-keyword.txt').exists()
    path = Path(install_upstream_lora_catalog(tmp_path, confirmed=True, **options))
    assert path.read_bytes() == contents and calls[-1] == UPSTREAM_LORA_URL
    assert index.lookup('12345678', 'example.safetensors').keywords == 'keyword'
    with pytest.raises(FileExistsError): install_upstream_lora_catalog(tmp_path, confirmed=True, **options)
    assert path.read_bytes() == contents and len(calls) == 2


def test_upstream_catalog_install_worker_action_requires_explicit_true(tmp_path, monkeypatch):
    from sdfx.base import BaseEngine
    from sdfx.protocol import EngineError
    from sdfx import model_keyword
    calls = []
    def install(directory, *, confirmed):
        calls.append((directory, confirmed))
        if confirmed is not True:
            raise ValueError('explicit confirmation is required')
        return str(Path(directory) / 'lora-keyword.txt')
    monkeypatch.setattr(model_keyword, 'install_upstream_lora_catalog', install)
    engine = BaseEngine()
    params = {'action': 'install_upstream_lora_catalog', 'mapping_dir': str(tmp_path)}
    with pytest.raises(EngineError, match='confirmation'): engine.model_keyword(params, None)
    params['confirmed'] = True
    assert engine.model_keyword(params, None)['path'].endswith('lora-keyword.txt')
    assert calls == [(str(tmp_path), None), (str(tmp_path), True)]


def test_catalog_download_rejects_truncated_and_oversized_streams_without_residue(tmp_path):
    import hashlib, io
    from sdfx.model_keyword import install_upstream_lora_catalog
    expected = b'0123456789'
    options = {'expected_sha256': hashlib.sha256(expected).hexdigest(), 'expected_size': len(expected)}
    for data in (expected[:-1], expected + b'x'):
        with pytest.raises(ValueError):
            install_upstream_lora_catalog(tmp_path, confirmed=True, open_url=lambda _: io.BytesIO(data), **options)
        assert not list(tmp_path.iterdir())


def test_catalog_download_never_replaces_file_created_during_transfer(tmp_path):
    import hashlib, io
    from sdfx.model_keyword import install_upstream_lora_catalog
    contents = b'12345678, keyword, model.safetensors\n'
    destination = tmp_path / 'lora-keyword.txt'
    def fetch(_):
        destination.write_bytes(b'external catalog')
        return io.BytesIO(contents)
    with pytest.raises(FileExistsError):
        install_upstream_lora_catalog(tmp_path, confirmed=True, open_url=fetch,
            expected_sha256=hashlib.sha256(contents).hexdigest(), expected_size=len(contents))
    assert destination.read_bytes() == b'external catalog'
    assert list(tmp_path.iterdir()) == [destination]
