"""Exercise the actual pinned, opt-in upstream catalog download in a new directory.

Run only when network use is coordinated. The downloaded third-party-derived
catalog remains in the supplied directory; this does not establish rights to
redistribute it. The directory must not exist before this check.
"""
import argparse
import hashlib
import json
from pathlib import Path

from sdfx.model_keyword import (MappingIndex, UPSTREAM_LORA_BYTES, UPSTREAM_LORA_SHA256,
                                UPSTREAM_LORA_URL, install_upstream_lora_catalog, parse_rows)


def main(output):
    directory = Path(output).absolute()
    directory.mkdir(parents=True, exist_ok=False)
    path = Path(install_upstream_lora_catalog(str(directory), confirmed=True))
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    if path.stat().st_size != UPSTREAM_LORA_BYTES or digest.hexdigest() != UPSTREAM_LORA_SHA256:
        raise AssertionError('installed bytes do not match the pinned upstream catalog')
    entries = parse_rows(path.read_text(encoding='utf8'), 0)
    if not entries:
        raise AssertionError('catalog contains no readable entries')
    sample = entries[0]
    index = MappingIndex(path, directory / 'lora-keyword-user.txt', allow_missing_default=True)
    found = index.lookup(sample.model_hash, sample.filename)
    if found is None or found.model_hash != sample.model_hash:
        raise AssertionError('installed catalog was not searchable')
    try:
        install_upstream_lora_catalog(str(directory), confirmed=True)
    except FileExistsError:
        pass
    else:
        raise AssertionError('second installation unexpectedly replaced the catalog')
    report = {'source': UPSTREAM_LORA_URL, 'size': path.stat().st_size, 'sha256': digest.hexdigest(),
              'entry_count': len(entries), 'sample_hash': sample.model_hash,
              'searchable': True, 'repeat_install_rejected': True}
    (directory / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('new_output_directory')
    main(parser.parse_args().new_output_directory)
