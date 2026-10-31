"""Compare optional PhotoMaker diagnostic fingerprints without loading torch."""
import json
import sys
from pathlib import Path


def analyze(events):
    branches = [event for event in events if event['event'] == 'photo_branches']
    def by_slot(event):
        return {(v['branch'], v['end'], v['stage'], v['key']): v for v in event['data']}
    def compare(left, right):
        a, b = by_slot(left), by_slot(right)
        keys = set(a) | set(b)
        mismatches = []
        for key in sorted(keys, key=str):
            if key not in a or key not in b or a[key]['sha256'] != b[key]['sha256']:
                mismatches.append({'slot': list(key), 'left': a.get(key), 'right': b.get(key)})
        return {'compared': len(keys), 'mismatched': len(mismatches), 'details': mismatches}
    comparisons = {}
    if len(branches) >= 4:
        for name, i, j in (('plain_vs_merge', 0, 1), ('plain_repeat', 0, 2), ('merge_repeat', 1, 3)):
            comparisons[name] = compare(branches[i], branches[j])
    unet_groups = []
    for event in events:
        if event['event'] == 'photo_branches':
            unet_groups.append([])
        elif event['event'] == 'neutral_unet' and unet_groups:
            unet_groups[-1].append(event['data'])
    unet_comparisons = {}
    if len(unet_groups) >= 4:
        for name, i, j in (('plain_vs_merge', 0, 1), ('plain_repeat', 0, 2), ('merge_repeat', 1, 3)):
            left, right = unet_groups[i], unet_groups[j]
            differences = []
            for index, (a, b) in enumerate(zip(left, right)):
                for key in ('input', 'output'):
                    if a[key]['sha256'] != b[key]['sha256']:
                        differences.append({'index': index, 'key': key, 'left': a[key], 'right': b[key],
                                            'left_raw_step': a['raw_step'], 'right_raw_step': b['raw_step']})
            missing = abs(len(left) - len(right)) or int(not left and not right)
            unet_comparisons[name] = {'left_calls': len(left), 'right_calls': len(right),
                                      'complete': bool(left) and bool(right) and not missing,
                                      'mismatched': len(differences) + missing, 'details': differences}
    pending, restores = [], []
    for event in events:
        if event['event'] == 'embedding_before':
            pending.append(event)
        elif event['event'] == 'embedding_after':
            if not pending:
                restores.append({'after': event['sequence'], 'error': 'unmatched restore'})
            else:
                before = pending.pop()
                restores.append({'before': before['sequence'], 'after': event['sequence'],
                                 'equal': before['data'] == event['data']})
    return {'photo_branch_events': len(branches), 'comparisons': comparisons, 'unet_comparisons': unet_comparisons,
            'embedding_restores': restores, 'unmatched_captures': [event['sequence'] for event in pending]}


if __name__ == '__main__':
    source, target = map(Path, sys.argv[1:3])
    events = [json.loads(line) for line in source.read_text(encoding='utf8').splitlines()]
    result = analyze(events)
    with target.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'photo_branch_events': result['photo_branch_events'],
                      'comparisons': {k: {x: v[x] for x in ('compared', 'mismatched')}
                                      for k, v in result['comparisons'].items()},
                      'embedding_restores': len(result['embedding_restores']),
                      'unmatched_captures': len(result['unmatched_captures'])}))
