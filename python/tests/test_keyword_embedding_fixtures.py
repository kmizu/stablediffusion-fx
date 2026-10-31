import importlib.util
import json
from pathlib import Path

import pytest

file = Path(__file__).resolve().parents[2] / 'scripts/dev/make_keyword_embedding_fixtures.py'
spec = importlib.util.spec_from_file_location('make_keyword_embedding_fixtures', file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_synthetic_fixture_manifest_and_existing_directory_protection(tmp_path):
    destination = tmp_path / 'new'
    result = module.create(destination)
    assert result['synthetic_keyword_sd15']['dimensions'] == {'clip_l': 768}
    assert result['synthetic_keyword_sdxl']['dimensions'] == {'clip_l': 768, 'clip_g': 1280}
    assert all(value['synthetic'] for value in result.values())
    assert json.loads((destination / 'synthetic-embeddings.json').read_text(encoding='utf8')) == result
    before = {p.name: p.read_bytes() for p in destination.iterdir()}
    with pytest.raises(FileExistsError): module.create(destination)
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == before
