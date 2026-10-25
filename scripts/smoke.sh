#!/usr/bin/env bash
# Engine smoke test (dry-run protocol round trip) + Python tests.
set -e
cd "$(dirname "$0")/.."
sbt -batch "engine/runMain sdfx.engine.tools.Smoke"
cd python
if [ -x .venv-test/bin/python ]; then PY=.venv-test/bin/python
else
  T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
  python3 -m venv "$T/v"; PY="$T/v/bin/python"
  "$PY" -m pip install -q pytest pillow numpy
fi
"$PY" -m pytest -q
