"""TEST-ONLY inverse interoperability check against an unchanged external pyAesCrypt implementation.

Usage: python script.py REFERENCE_CRYPTO_PY ISOLATED_REFERENCE_PACKAGES NATIVE_OUTPUT_DIRECTORY
Reference pin13a188f8799306ad9034167bf0bc0489319db274, pyAesCrypt/crypto.py, Apache-2.0.
No reference code or Python crypto library is imported by the product.
"""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys


def main():
    reference, packages, outputs = map(Path, sys.argv[1:])
    expected_source = "a6e0140577c4163e7db480188d55d600563c6ac86a55ed665a6447bf64eaf0c6"
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == expected_source, "Reference source differs from pinned upstream bytes"
    sys.path.insert(0, str(packages.resolve()))
    spec = importlib.util.spec_from_file_location("external_pyaescrypt_reference", reference)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = []
    for plain in sorted(outputs.glob("*.plain")):
        encoded = plain.with_suffix(".aes")
        decoded = io.BytesIO()
        module.decryptStream(io.BytesIO(encoded.read_bytes()), decoded, "桜-päss")
        assert decoded.getvalue() == plain.read_bytes(), plain.name
        cases.append(dict(name=plain.stem, bytes=len(decoded.getvalue()),
                          sha256=hashlib.sha256(decoded.getvalue()).hexdigest()))
    assert len(cases) == 8, cases
    result = dict(status="passed", external_reference_pin="13a188f8799306ad9034167bf0bc0489319db274",
                  external_reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(), cases=cases)
    (outputs / "reference-decrypt-report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
