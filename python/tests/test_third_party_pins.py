import importlib.util
from pathlib import Path
import hashlib
import pytest

spec = importlib.util.spec_from_file_location('third_party', Path(__file__).resolve().parents[2] / 'scripts/third_party.py')
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def test_pin_accepts_only_line_ending_variants():
    archived = b'Copyright example\r\nPermission granted\r\n'
    expected = hashlib.sha256(archived).hexdigest()
    assert collector.verify_license_bytes(archived, expected) is None
    assert collector.verify_license_bytes(archived.replace(b'\r\n', b'\n'), expected) is None
    lf_digest = hashlib.sha256(archived.replace(b'\r\n', b'\n')).hexdigest()
    assert collector.verify_license_bytes(archived, lf_digest) is None
    with pytest.raises(ValueError):
        collector.verify_license_bytes(b'Copyright changed\nPermission granted\n', expected)
    with pytest.raises(ValueError):
        collector.verify_license_bytes(archived.rstrip(), expected)
