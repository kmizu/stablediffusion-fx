"""stdin/stdout JSON Lines server: a reader thread (this one) and a single worker thread.

`ping`, `interrupt` and `shutdown` are answered directly on the reader thread; everything else is queued
and executed in order by the worker.
"""
from __future__ import annotations

import collections
import json
import logging
import os
import queue
import sys
import threading
import traceback
from typing import Optional

from .params import ParamError
from .protocol import (INTERNAL, INTERRUPTED, INVALID_PARAMS, OOM, UNSUPPORTED, Emitter, EngineError, JobContext,
                       error_msg, result_msg)

log = logging.getLogger("sdfx.server")

QUEUED_TYPES = ("list_models", "list_files", "list_samplers", "load_model", "set_vae", "txt2img", "img2img",
                "preprocess", "upscale", "unload", "memory", "restore_faces", "caption", "run_code", "img2vid",
                "novel_view")


def classify_exception(e: BaseException) -> tuple[str, str, str]:
    detail = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    if isinstance(e, EngineError):
        return e.code, e.message, e.detail or (detail if e.code == INTERNAL else "")
    if isinstance(e, (ParamError, FileNotFoundError)):
        return INVALID_PARAMS, str(e), detail
    name = type(e).__name__
    msg = str(e)
    if name == "OutOfMemoryError" or "out of memory" in msg.lower() or isinstance(e, MemoryError):
        return OOM, f"out of memory: {msg.splitlines()[0] if msg else name}", detail
    return INTERNAL, f"{name}: {msg}", detail


class Server:
    def __init__(self, engine, emitter: Emitter):
        self.engine = engine
        self.emitter = emitter
        self.q: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self.interrupt = threading.Event()
        self.lock = threading.Lock()
        self.current_id: Optional[str] = None
        self.cancelled: set = set()
        self.queued: "collections.deque[str]" = collections.deque()  # ids waiting in the queue, in order
        self.closing = False
        self._hello: Optional[dict] = None
        self.worker = threading.Thread(target=self._work, name="sdfx-worker", daemon=True)

    # -- reader side --------------------------------------------------------
    def handle_line(self, line: str) -> bool:
        """Process one input line. Returns False when the server should exit (shutdown)."""
        line = line.strip()
        if not line:
            return True
        try:
            msg = json.loads(line)
        except ValueError as e:
            self.emitter.emit(error_msg(None, INVALID_PARAMS, f"malformed JSON: {e}"))
            return True
        if not isinstance(msg, dict):
            self.emitter.emit(error_msg(None, INVALID_PARAMS, "request must be a JSON object"))
            return True
        rid = msg.get("id")
        if rid is not None and not isinstance(rid, str):
            rid = str(rid)
        typ = msg.get("type")
        params = msg.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            self.emitter.emit(error_msg(rid, INVALID_PARAMS, "params must be an object"))
            return True

        if typ == "ping":
            self.emitter.emit(result_msg(rid, self.hello()))
        elif typ == "interrupt":
            target = params.get("target_id")
            target = None if target is None else str(target)
            with self.lock:
                if self.current_id is not None and (target is None or target == self.current_id):
                    self.interrupt.set()
                elif target is None and self.queued:
                    # nothing running yet (e.g. interrupt right after submit): cancel the next queued job
                    self.cancelled.add(self.queued[0])
                elif target is not None and target in self.queued:
                    self.cancelled.add(target)
                # unknown / finished ids are ignored (nothing is remembered)
            self.emitter.emit(result_msg(rid, {}))
        elif typ == "shutdown":
            self.interrupt.set()
            self.emitter.emit(result_msg(rid, {}))
            return False
        elif typ in QUEUED_TYPES:
            with self.lock:
                if rid is not None:
                    self.queued.append(rid)
            self.q.put((rid, typ, params))
        else:
            self.emitter.emit(error_msg(rid, UNSUPPORTED, f"unknown request type '{typ}'"))
        return True

    # -- worker side --------------------------------------------------------
    def hello(self) -> dict:
        if self._hello is None:
            self._hello = self.engine.hello()
        return dict(self._hello)

    def _work(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            try:
                self._handle(item)
            except BaseException as e:  # noqa: BLE001 - the worker must never die
                log.error("worker failure: %s", traceback.format_exc())
                try:
                    self.emitter.emit(error_msg(item[0], INTERNAL, f"internal error: {type(e).__name__}"))
                except Exception:
                    pass

    def _handle(self, item):
        rid, typ, params = item
        with self.lock:
            if rid is not None:
                try:
                    self.queued.remove(rid)
                except ValueError:
                    pass
            if (rid is not None and rid in self.cancelled) or self.closing:
                self.cancelled.discard(rid)
                self.emitter.emit(error_msg(rid, INTERRUPTED, "interrupted before start"))
                return
            self.current_id = rid
            self.interrupt.clear()
        ctx = JobContext(self.emitter, rid, self.interrupt)
        try:
            handler = getattr(self.engine, typ)
            res = handler(params, ctx)
            self.emitter.emit(result_msg(rid, res if res is not None else {}))
        except BaseException as e:  # noqa: BLE001 - every failure must become an error envelope
            code, message, detail = classify_exception(e)
            if code == OOM:
                try:
                    self.engine.on_oom()
                except Exception:
                    pass
            if code == INTERNAL:
                log.error("request %s (%s) failed:\n%s", rid, typ, detail)
            self.emitter.emit(error_msg(rid, code, message, detail if code != INTERRUPTED else ""))
        finally:
            with self.lock:
                self.current_id = None
                self.interrupt.clear()
                self.cancelled.discard(rid)

    def run(self, stdin) -> int:
        self.emitter.emit(dict({"id": None, "type": "hello"}, **self.hello()))
        self.worker.start()
        for raw in stdin:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            if not self.handle_line(raw):
                return 0  # shutdown: caller exits immediately
            if self.emitter.broken:
                self.interrupt.set()
                return 0
        # EOF (the app closed our stdin): stop the running job, drop queued ones, exit
        with self.lock:
            self.closing = True
            if self.current_id is not None:
                self.interrupt.set()
        self.q.put(None)
        self.worker.join(timeout=60)
        return 0


def setup_stdio() -> Emitter:
    """Reserve the real stdout for the protocol; route fd 1 and sys.stdout to stderr."""
    sys.stdout.flush()
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    out = os.fdopen(proto_fd, "wb", buffering=0)
    sys.stdout = sys.stderr
    return Emitter(out)


def build_engine(dry_run: bool, delay_ms: int = 0):
    if dry_run:
        from .dryrun import FakeEngine
        return FakeEngine(delay_ms)
    try:
        from .engine import Engine
        return Engine()
    except Exception as e:  # ImportError, but also OSError (DLL load), RuntimeError (CUDA init), AttributeError ...
        log.error("real engine unavailable; starting degraded:\n%s", traceback.format_exc())
        from .unavailable import UnavailableEngine
        return UnavailableEngine(f"{type(e).__name__}: {e}")


def main(dry_run: bool = False, delay_ms: int = 0) -> int:
    emitter = setup_stdio()
    engine = build_engine(dry_run, delay_ms)
    server = Server(engine, emitter)
    code = server.run(sys.stdin.buffer)
    try:
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(code)
