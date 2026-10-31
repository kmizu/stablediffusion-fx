import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

file = Path(__file__).resolve().parents[2] / 'scripts/dev/check_keyword_catalog_download.py'
spec = importlib.util.spec_from_file_location('check_keyword_catalog_download', file)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def test_check_records_search_and_will_not_reuse_output_directory(tmp_path, monkeypatch):
    data = b'12345678, keyword, model.safetensors\n'
    def install(directory, *, confirmed):
        assert confirmed
        path = Path(directory) / 'lora-keyword.txt'
        with path.open('xb') as out: out.write(data)
        return str(path)
    monkeypatch.setattr(check, 'install_upstream_lora_catalog', install)
    monkeypatch.setattr(check, 'UPSTREAM_LORA_BYTES', len(data))
    monkeypatch.setattr(check, 'UPSTREAM_LORA_SHA256', hashlib.sha256(data).hexdigest())
    destination = tmp_path / 'new'
    check.main(destination)
    report = json.loads((destination / 'verification.json').read_text(encoding='utf8'))
    assert report['entry_count'] == 1 and report['searchable'] and report['repeat_install_rejected']
    with pytest.raises(FileExistsError): check.main(destination)
    assert (destination / 'lora-keyword.txt').read_bytes() == data
