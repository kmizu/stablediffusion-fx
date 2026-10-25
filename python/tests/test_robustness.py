"""Stdout isolation, non-ASCII paths, worker survival, interrupt races, dry-run validation parity."""
import io
import json
import os
import subprocess
import sys
import textwrap
import time
import types

import pytest

from test_server_dryrun import ROOT, Engine, gen_params, png_size, write_ckpt


def test_stdout_pollution_guard(tmp_path):
    """After setup_stdio, prints / raw fd writes / child processes / warnings never reach protocol stdout."""
    script = textwrap.dedent("""
        import os, subprocess, sys, warnings
        from sdfx.server import setup_stdio
        em = setup_stdio()
        print("python print")
        sys.stdout.write("sys.stdout write\\n")
        os.write(1, b"raw fd write\\n")
        subprocess.run([sys.executable, "-c", "print('child process print')"])
        warnings.warn("a library warning")
        em.emit({"id": None, "type": "hello", "ok": True})
    """)
    env = dict(os.environ, PYTHONPATH=ROOT)
    p = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True, timeout=60)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    lines = p.stdout.decode("utf-8").splitlines()
    assert lines == ['{"id":null,"type":"hello","ok":true}']
    err = p.stderr.decode(errors="replace")
    for s in ("python print", "sys.stdout write", "raw fd write", "child process print", "a library warning"):
        assert s in err


def test_non_ascii_paths(tmp_path):
    base = tmp_path / "\u753b\u50cf\u751f\u6210" / "\u30e2\u30c7\u30eb"   # Japanese directory names
    (base / "\u30b5\u30d6").mkdir(parents=True)
    ck = write_ckpt(base / "\u30b5\u30d6" / "\u5c0f\u3055\u3044.safetensors")
    out = tmp_path / "\u51fa\u529b \u30d5\u30a9\u30eb\u30c0"
    e = Engine()
    try:
        e.next()
        res = e.collect(e.send("list_models", {"dirs": [str(base)]}))[-1]
        assert res["result"]["models"][0]["name"] == "\u30b5\u30d6/\u5c0f\u3055\u3044"
        assert res["result"]["models"][0]["path"] == ck and res["result"]["models"][0]["kind"] == "sd15"
        res = e.collect(e.send("load_model", {"path": ck}))[-1]
        assert res["type"] == "result" and res["result"]["name"] == "\u5c0f\u3055\u3044"
        res = e.collect(e.send("txt2img", gen_params(out, prompt="\u732b, (\u6843\u306e\u82b1:1.2)")))[-1]
        assert res["type"] == "result", res
        path = res["result"]["images"][0]["path"]
        assert os.path.dirname(path) == str(out) and png_size(path) == (128, 96)
        assert e.close() == 0
    finally:
        if e.p.poll() is None:
            e.p.kill()


def test_worker_survives_unencodable_text(tmp_path):
    e = Engine()
    try:
        e.next()
        # a lone surrogate in a path: the error message echoes it; it must be emitted (escaped), not crash the worker
        rid = e.send("load_model", {"path": "C:/models/\ud800bad.safetensors"})
        res = e.collect(rid)[-1]
        assert res["type"] == "error"
        res = e.collect(e.send("memory"))[-1]
        assert res["type"] == "result"
        res = e.collect(e.send("list_files", {"kind": "lora", "dirs": [str(tmp_path)]}))[-1]
        assert res["result"] == {"files": []}
        assert e.close() == 0
    finally:
        if e.p.poll() is None:
            e.p.kill()


def test_emitter_never_raises():
    from sdfx.protocol import Emitter
    buf = io.BytesIO()
    em = Emitter(buf)
    em.emit({"id": "x", "type": "result", "result": {"s": "a\ud800b"}})
    em.emit({"id": "y", "type": "result", "result": {"o": object()}})
    lines = [json.loads(l) for l in buf.getvalue().splitlines()]
    assert lines[0]["result"]["s"] == "a\ud800b" and lines[1]["id"] == "y"


def test_interrupt_right_after_submit_cancels_queued(tmp_path):
    e = Engine("--dry-run-delay-ms", "30")
    try:
        e.next()
        r1 = e.send("txt2img", gen_params(tmp_path, steps=30))
        r2 = e.send("txt2img", gen_params(tmp_path, steps=30))
        e.send("interrupt", {})          # interrupts r1 (running or about to start)
        e.send("interrupt", {"target_id": r2})  # cancels r2 while still queued
        finals = {}
        while len(finals) < 2:
            m = e.next(20)
            if m.get("id") in (r1, r2) and m["type"] in ("result", "error"):
                finals[m["id"]] = m
        assert finals[r1]["code"] == "interrupted" and finals[r2]["code"] == "interrupted"
        # an interrupt for an unknown id is ignored and does not affect later jobs
        e.collect(e.send("interrupt", {"target_id": "nope"}))
        res = e.collect(e.send("txt2img", gen_params(tmp_path, steps=2), rid="nope"))[-1]
        assert res["type"] == "result"
        assert e.close() == 0
    finally:
        if e.p.poll() is None:
            e.p.kill()


@pytest.fixture
def eng():
    e = Engine()
    e.next()
    yield e
    if e.p.poll() is None:
        e.p.kill()


def test_dryrun_validation_parity(eng, tmp_path):
    from PIL import Image
    src = tmp_path / "s.png"
    Image.new("RGB", (32, 32)).save(src)
    req = gen_params(tmp_path, loras=[{"name": "foo", "weight": 1.0}])
    req["lora_paths"] = {"foo": str(tmp_path / "missing.safetensors")}
    res = eng.collect(eng.send("txt2img", req))[-1]
    assert res["code"] == "invalid_params" and "LoRA file not found" in res["message"]
    req = gen_params(tmp_path, controlnet=[{"enabled": True, "model": "cn", "image": str(src), "preprocessor": "none"}])
    req["controlnet_paths"] = {"cn": str(tmp_path / "cn_missing.safetensors")}
    res = eng.collect(eng.send("txt2img", req))[-1]
    assert res["code"] == "invalid_params" and "ControlNet model not found" in res["message"]
    req = gen_params(tmp_path, controlnet=[{"enabled": True, "model": "cn", "image": None, "preprocessor": "none"}])
    req["controlnet_paths"] = {"cn": str(src)}
    res = eng.collect(eng.send("txt2img", req))[-1]
    assert res["code"] == "invalid_params" and "has no image" in res["message"]
    hires = {"enabled": True, "upscaler": "4x-UltraSharp"}
    res = eng.collect(eng.send("txt2img", gen_params(tmp_path, hires=hires)))[-1]
    assert res["code"] == "invalid_params" and "unknown upscaler" in res["message"]
    req = gen_params(tmp_path, hires=hires)
    req["upscaler_paths"] = {"4x-UltraSharp": str(tmp_path / "nope.pth")}
    res = eng.collect(eng.send("txt2img", req))[-1]
    assert res["code"] == "invalid_params" and "upscaler model not found" in res["message"]
    res = eng.collect(eng.send("upscale", {"image_path": str(src), "upscaler": "X", "scale": 2, "output_dir": str(tmp_path),
                                           "upscaler_paths": {"X": str(tmp_path / "x.pth")}}))[-1]
    assert res["code"] == "invalid_params"
    # prompt-syntax warnings are reported by the dry run too
    res = eng.collect(eng.send("txt2img", gen_params(tmp_path / "o", prompt="cat AND dog", steps=1)))[-1]
    assert res["type"] == "result" and any("composable diffusion" in w for w in res["result"]["warnings"])
    res = eng.collect(eng.send("txt2img", gen_params(tmp_path / "o", prompt="[cat:dog:0.5]", steps=1)))[-1]
    assert res["type"] == "result" and not any("prompt editing" in w for w in res["result"]["warnings"])


def test_build_engine_degrades_on_non_import_errors(monkeypatch):
    from sdfx import server
    from sdfx.unavailable import UnavailableEngine

    class Broken(types.ModuleType):
        def __getattr__(self, name):
            raise OSError("[WinError 126] fbgemm.dll could not be loaded")

    monkeypatch.setitem(sys.modules, "sdfx.engine", Broken("sdfx.engine"))
    eng = server.build_engine(False)
    assert isinstance(eng, UnavailableEngine) and "WinError 126" in eng.reason
    assert eng.hello()["torch"] is None
