import io
import json
import threading

from sdfx import protocol as P


def test_envelopes():
    assert P.result_msg("r1", {"a": 1}) == {"id": "r1", "type": "result", "result": {"a": 1}}
    e = P.error_msg("r1", "oom", "boom", "tb")
    assert e == {"id": "r1", "type": "error", "code": "oom", "message": "boom", "detail": "tb"}
    assert P.error_msg("r1", "weird", "x")["code"] == "internal"
    p = P.progress_msg("r1", 5, 20, "hires", 1.23456, 0, 2)
    assert p == {"id": "r1", "type": "event", "event": "progress", "step": 5, "total": 20, "pass": "hires",
                 "eta_s": 1.235, "image": 0, "images": 2, "batch_size": 1}
    assert P.progress_msg("r1", 1, 20, image=4, images=20, batch_size=4)["batch_size"] == 4
    assert P.preview_msg("r1", "abc", 3) == {"id": "r1", "type": "event", "event": "preview", "jpeg_b64": "abc", "step": 3}
    assert P.log_msg(None, "warn", "m") == {"id": None, "type": "event", "event": "log", "level": "warn", "message": "m"}
    assert P.log_msg(None, "bogus", "m")["level"] == "info"


def test_encode_one_line_utf8_and_nan():
    b = P.encode({"x": "日本語\nline", "f": float("nan")})
    assert b.endswith(b"\n") and b.count(b"\n") == 1
    d = json.loads(b.decode("utf-8"))
    assert d == {"x": "日本語\nline", "f": None}


def test_emitter_thread_safe_lines():
    buf = io.BytesIO()
    em = P.Emitter(buf)

    def w(i):
        for j in range(200):
            em.emit({"i": i, "j": j, "pad": "x" * 100})

    ts = [threading.Thread(target=w, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    lines = buf.getvalue().decode().splitlines()
    assert len(lines) == 800
    assert all(json.loads(line)["pad"] == "x" * 100 for line in lines)


def test_job_context_interrupt():
    em = P.Emitter(io.BytesIO())
    ev = threading.Event()
    ctx = P.JobContext(em, "r", ev)
    ctx.check_interrupt()
    ev.set()
    try:
        ctx.check_interrupt()
        raise AssertionError("expected interrupt")
    except P.EngineError as e:
        assert e.code == "interrupted"


def test_warn_dedup():
    buf = io.BytesIO()
    ctx = P.JobContext(P.Emitter(buf), "r")
    ctx.warn("a")
    ctx.warn("a")
    assert ctx.warnings == ["a"]
    assert len(buf.getvalue().splitlines()) == 1
