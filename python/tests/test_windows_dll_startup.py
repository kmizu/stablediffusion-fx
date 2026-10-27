"""The engine must not inherit jpackage's JVM DLL search directory."""

import ctypes
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DLL search semantics")
def test_entrypoint_clears_inherited_directory_before_server_import(tmp_path):
    package = tmp_path / "sdfx"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    source = Path(__file__).resolve().parents[1] / "sdfx" / "__main__.py"
    (package / "__main__.py").write_bytes(source.read_bytes())
    (package / "server.py").write_text(
        "import ctypes, json\n"
        "buffer = ctypes.create_unicode_buffer(32768)\n"
        "ctypes.windll.kernel32.GetDllDirectoryW(len(buffer), buffer)\n"
        "directory_at_import = buffer.value\n"
        "def main(*, dry_run, delay_ms):\n"
        "    print(json.dumps({'directory': directory_at_import, 'dry_run': dry_run}))\n"
        "    return 0\n",
        encoding="utf-8",
    )
    inherited = tmp_path / "runtime" / "bin"
    inherited.mkdir(parents=True)
    wrapper = (
        "import ctypes, subprocess, sys\n"
        "assert ctypes.windll.kernel32.SetDllDirectoryW(sys.argv[1])\n"
        "child = subprocess.run([sys.executable, '-m', 'sdfx', '--dry-run'], "
        "capture_output=True, text=True, cwd=sys.argv[2])\n"
        "print(child.stdout, end='')\n"
        "print(child.stderr, end='', file=sys.stderr)\n"
        "sys.exit(child.returncode)\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", wrapper, str(inherited), str(tmp_path)],
        env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"directory": "", "dry_run": True}


@pytest.mark.skipif(sys.platform != "win32", reason="Windows API error handling")
def test_failed_reset_stops_before_server_import(monkeypatch):
    entry = importlib.import_module("sdfx.__main__")
    called = []
    fake_server = types.ModuleType("sdfx.server")
    fake_server.main = lambda **kwargs: called.append(kwargs)
    monkeypatch.setitem(sys.modules, "sdfx.server", fake_server)

    class FailingCall:
        def __call__(self, directory):
            assert directory is None
            ctypes.set_last_error(5)
            return 0

    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: types.SimpleNamespace(SetDllDirectoryW=FailingCall()))
    with pytest.raises(OSError):
        entry.main(["--dry-run"])
    assert called == []


def test_non_windows_startup_does_not_change_dll_search(monkeypatch):
    entry = importlib.import_module("sdfx.__main__")
    fake_server = types.ModuleType("sdfx.server")
    fake_server.main = lambda **kwargs: 37
    monkeypatch.setitem(sys.modules, "sdfx.server", fake_server)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: pytest.fail("Windows API used"), raising=False)
    assert entry.main(["--dry-run"]) == 37
