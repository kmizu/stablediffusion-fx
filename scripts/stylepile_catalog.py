"""Audit pinned GPL-3.0 StylePile data without executing its WebUI script.

Usage: python scripts/stylepile_catalog.py ARCHIVE TREE_JSON OUTPUT_DIRECTORY
The archive and recursive Git tree must have been fetched at REVISION.
"""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

REVISION = '85b549a10ff6c5b1604e980cd2107253b088e832'
SOURCE = 'https://github.com/some9000/StylePile'
FIELDS = ('ElementConcept', 'ResultNames', 'ResultTypeBefore', 'ResultTypePositives',
          'ResultTypeNegatives', 'Preset', 'PresetNegatives', 'AlwaysBad')


def audit(archive, tree_path, destination):
    archive, destination = Path(archive), Path(destination)
    tree = json.loads(Path(tree_path).read_text(encoding='utf-8-sig'))
    if tree.get('truncated'):
        raise ValueError('Incomplete source tree cannot establish asset coverage')
    entries = {item['path']: item for item in tree['tree'] if item['type'] == 'blob' and
               (item['path'].startswith('StylePile/') or item['path'] in ('LICENSE', 'scripts/StylePile.py'))}
    files, manifest = {}, []
    with zipfile.ZipFile(archive) as source:
        prefix = f'StylePile-{REVISION}/'
        for name, entry in sorted(entries.items()):
            if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
                raise ValueError('Unsafe source path')
            data = source.read(prefix + name)
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode('ascii') + b'\0' + data).hexdigest()
            if blob != entry['sha'] or len(data) != entry['size']:
                raise ValueError(f'Source blob mismatch: {name}')
            files[name] = data
            manifest.append(dict(path=name, bytes=len(data), git_blob=blob,
                                 sha256=hashlib.sha256(data).hexdigest(),
                                 source=f'{SOURCE}/blob/{REVISION}/{name}'))
    script = ast.parse(files['scripts/StylePile.py'].decode('utf-8-sig'))
    literals = {}
    for node in script.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in FIELDS:
                    literals[target.id] = ast.literal_eval(node.value)
    if set(literals) != set(FIELDS):
        raise ValueError('Pinned source catalog fields missing')
    choices = {}
    for name in files:
        path = PurePosixPath(name)
        if path.suffix.lower() == '.jpg':
            choices.setdefault(path.parent.name, []).append(dict(name=path.stem, preview=name))
    inspiration = {PurePosixPath(name).stem: data.decode('utf-8-sig').splitlines()
                   for name, data in files.items() if name.startswith('StylePile/Inspiration/') and name.endswith('.txt')}
    catalog = dict(source=SOURCE, revision=REVISION, license='GPL-3.0', literals=literals,
                   choices=choices, inspiration=inspiration)
    # All source blobs are verified and decoded before any output is published.
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    for name, value in [('catalog.json', catalog), ('manifest.json', dict(revision=REVISION,
        source=SOURCE, archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), files=manifest))]:
        (destination / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n',
                                       encoding='utf-8', newline='\n')
    return dict(files=len(files), choices={key: len(value) for key, value in choices.items()},
                inspiration={key: len(value) for key, value in inspiration.items()})


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    print(json.dumps(audit(*sys.argv[1:]), indent=2))
