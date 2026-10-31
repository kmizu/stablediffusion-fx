import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('case,fail', [('signed', False), ('signed', True), ('multidiffusion', False), ('multidiffusion', True)])
def test_guarded_diagnostic_restores_exact_sources_and_keeps_backup(tmp_path, monkeypatch, case, fail):
    root = Path(__file__).resolve().parents[2]
    files = ['python/sdfx/clora/runtime.py', 'python/sdfx/engine.py', 'python/sdfx/embedding_merge_pipeline.py']
    original = {}
    for name in files:
        source = root / name
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes()); original[name] = target.read_bytes()
    (tmp_path / 'scripts/dev/keyword_photo_trace').mkdir(parents=True)
    temp = tmp_path / 'temp'; temp.mkdir()
    for name in ('sdfx-photomaker-v1-source', 'sdfx-photomaker-genuine-target-v2'):
        (temp / name).mkdir()
    weights = tmp_path / 'weights.bin'; weights.write_bytes(b'test')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('TEMP', str(temp))
    monkeypatch.setenv('SDFX_PHOTOMAKER_MODEL', str(weights))
    monkeypatch.setenv('SDFX_KEYWORD_LORA', str(weights))
    monkeypatch.setattr(sys, 'argv', ['guarded', case, sys.executable, str(tmp_path / 'out')])
    def run(*args, **kwargs):
        assert (tmp_path / files[0]).read_bytes() != original[files[0]]
        assert (tmp_path / files[1]).read_bytes() != original[files[1]]
        if case == 'multidiffusion': assert (tmp_path / files[2]).read_bytes() != original[files[2]]
        if fail: raise RuntimeError('simulated launch failure')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, 'run', run)
    if fail:
        with pytest.raises(RuntimeError, match='simulated launch failure'):
            runpy.run_path(str(root / 'scripts/dev/run_guarded_photo_diagnostic.py'), run_name='__main__')
    else:
        runpy.run_path(str(root / 'scripts/dev/run_guarded_photo_diagnostic.py'), run_name='__main__')
    assert all((tmp_path / name).read_bytes() == data for name, data in original.items())
    assert (tmp_path / 'out.source-backup.zip').is_file()
