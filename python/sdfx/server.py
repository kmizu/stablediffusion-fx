"""stdin/stdout JSON Lines server: a reader thread (this one) and a single worker thread.

Control and status requests are answered directly on the reader thread; model work is queued.
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
                       error_msg, result_msg, log_msg)

log = logging.getLogger("sdfx.server")

QUEUED_TYPES = ("list_models", "list_files", "list_samplers", "load_model", "set_vae", "txt2img", "img2img",
                "preprocess", "upscale", "unload", "memory", "restore_faces", "caption", "run_code", "img2vid",
                "novel_view", "merge_models", "convert_model", "image_tool", "tokenize", "system_info",
                "system_benchmark", "system_submission_preview", "system_submit", "vae_tools", "temporal_lab")


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
        self.current_owner: Optional[str] = None
        self.current_type: Optional[str] = None
        self.current_ctx: Optional[JobContext] = None
        self.cancelled: set = set()
        self.queued: "collections.deque[str]" = collections.deque()  # ids waiting in the queue, in order
        self.queued_owners: dict[str, Optional[str]] = {}
        self.closing = False
        # Process-global runtime protection, independent of checkpoint and job state.
        self.prevent_interruption = False
        self.protection_revision = 0
        self._hello: Optional[dict] = None
        self.worker = threading.Thread(target=self._work, name="sdfx-worker", daemon=True)

    # -- reader side --------------------------------------------------------
    def _protection_locked(self):
        """Caller holds self.lock, shared with cancellation state transitions."""
        return dict(enabled=self.prevent_interruption, revision=self.protection_revision)

    def _interruption_decision_locked(self):
        decision = dict(accepted=not self.prevent_interruption, blocked=self.prevent_interruption,
                        affected=False, protection=self._protection_locked())
        if self.prevent_interruption:
            decision['message'] = 'Interruptions are blocked; turn off Prevent interruption to continue'
        return decision

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
        owner = msg.get('owner_id')
        if 'owner_id' in msg and (not isinstance(owner, str) or not 1 <= len(owner) <= 256):
            self.emitter.emit(error_msg(rid, INVALID_PARAMS, 'owner_id must be a nonempty string of at most 256 characters'))
            return True

        if typ == "interruption_protection":
            if 'enabled' in params and not isinstance(params['enabled'], bool):
                self.emitter.emit(error_msg(rid, INVALID_PARAMS, 'enabled must be boolean'))
                return True
            with self.lock:
                if 'enabled' in params and params['enabled'] != self.prevent_interruption:
                    self.prevent_interruption = params['enabled']
                    self.protection_revision += 1
                data = self._protection_locked()
            self.emitter.emit(result_msg(rid, data))
        elif typ == "restart_authorization":
            with self.lock:
                data = self._interruption_decision_locked()
            if data['blocked']:
                self.emitter.emit(log_msg(rid, 'warn', data['message']))
            self.emitter.emit(result_msg(rid, data))
        elif typ == "system_status":
            try:
                data = self.engine.system_status(params, None)
                with self.lock:
                    data['state'] = dict(request_id=self.current_id, request_type=self.current_type, queued=len(self.queued), interrupted=self.interrupt.is_set(),
                        progress=dict(self.current_ctx.latest_progress) if self.current_ctx else {})
                self.emitter.emit(result_msg(rid, data))
            except Exception as e:
                code, message, detail = classify_exception(e)
                self.emitter.emit(error_msg(rid, code, message))
        elif typ == "ping":
            self.emitter.emit(result_msg(rid, self.hello()))
        elif typ == "interrupt":
            target = params.get("target_id")
            target = None if target is None else str(target)
            expected_owner = params.get('owner_id')
            if 'owner_id' in params and (not isinstance(expected_owner, str) or not 1 <= len(expected_owner) <= 256):
                self.emitter.emit(error_msg(rid, INVALID_PARAMS, 'owner_id must be a nonempty string of at most 256 characters'))
                return True
            with self.lock:
                data = self._interruption_decision_locked()
                data['active_id'] = None
                if data['accepted']:
                    # Job ownership is checked at the same reader admission point as policy.
                    # A late cancel for A must never mutate B, even if its request ID was
                    # independently observed after the native queue changed jobs.
                    if (self.current_id is not None and (target is None or target == self.current_id)
                            and (expected_owner is None or expected_owner == self.current_owner)):
                        self.interrupt.set()
                        data['affected'] = True
                        data['active_id'] = self.current_id
                    else:
                        queued_target = next((item for item in self.queued
                            if (target is None or target == item)
                            and (expected_owner is None or self.queued_owners.get(item) == expected_owner)), None)
                        if queued_target is not None:
                            self.cancelled.add(queued_target)
                            data['affected'] = True
                # unknown / finished ids are ignored (nothing is remembered)
            if data['blocked']:
                self.emitter.emit(log_msg(rid, 'warn', data['message']))
            self.emitter.emit(result_msg(rid, data))
        elif typ == "shutdown":
            self.interrupt.set()
            self.emitter.emit(result_msg(rid, {}))
            return False
        elif typ in QUEUED_TYPES:
            with self.lock:
                if rid is not None:
                    self.queued.append(rid)
                    self.queued_owners[rid] = owner
            # Ownership is transport metadata, never injected into model parameters/results.
            self.q.put((rid, typ, params, owner))
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
        rid, typ, params = item[:3]
        owner = item[3] if len(item) > 3 else None  # direct/internal legacy callers
        with self.lock:
            if rid is not None:
                self.queued_owners.pop(rid, None)
                try:
                    self.queued.remove(rid)
                except ValueError:
                    pass
            if (rid is not None and rid in self.cancelled) or self.closing:
                self.cancelled.discard(rid)
                self.emitter.emit(error_msg(rid, INTERRUPTED, "interrupted before start"))
                return
            self.current_id = rid
            self.current_owner = owner
            self.current_type = typ
            self.interrupt.clear()
        ctx = JobContext(self.emitter, rid, self.interrupt)
        with self.lock:
            self.current_ctx = ctx
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
                self.current_owner = None
                self.current_type = None
                self.current_ctx = None
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
