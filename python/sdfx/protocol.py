"""JSON Lines envelope helpers. Every line written goes through a single locked Emitter."""
from __future__ import annotations

import json
import math
import threading
import time
from typing import Any, Optional

from . import ENGINE_VERSION, PROTOCOL_VERSION  # noqa: F401  (re-exported)

INVALID_PARAMS = "invalid_params"
MODEL_LOAD_FAILED = "model_load_failed"
NO_MODEL = "no_model"
OOM = "oom"
INTERRUPTED = "interrupted"
UNSUPPORTED = "unsupported"
INTERNAL = "internal"

ERROR_CODES = (INVALID_PARAMS, MODEL_LOAD_FAILED, NO_MODEL, OOM, INTERRUPTED, UNSUPPORTED, INTERNAL)


class EngineError(Exception):
    """An error that maps directly to a protocol error envelope."""

    def __init__(self, code: str, message: str, detail: str = ""):
        super().__init__(f"[{code}] {message}")
        if code not in ERROR_CODES:
            code = INTERNAL
        self.code = code
        self.message = message
        self.detail = detail


def _clean(v: Any) -> Any:
    """Make values JSON-safe (NaN/inf -> None, tuples -> lists)."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def _default(o):
    try:
        return float(o) if hasattr(o, "__float__") else str(o)
    except Exception:
        return str(o)


def encode(obj: dict) -> bytes:
    """One UTF-8 JSON line. Never raises: unencodable text (lone surrogates) falls back to \\u escapes."""
    try:
        return (json.dumps(_clean(obj), ensure_ascii=False, separators=(",", ":"), allow_nan=False,
                           default=_default) + "\n").encode("utf-8")
    except (UnicodeEncodeError, ValueError, TypeError):
        return (json.dumps(_clean(obj), ensure_ascii=True, separators=(",", ":"), default=_default) + "\n").encode("ascii")


def result_msg(req_id: Optional[str], result: dict) -> dict:
    return {"id": req_id, "type": "result", "result": result}


def error_msg(req_id: Optional[str], code: str, message: str, detail: str = "") -> dict:
    if code not in ERROR_CODES:
        code = INTERNAL
    return {"id": req_id, "type": "error", "code": code, "message": message, "detail": detail or ""}


def event_msg(req_id: Optional[str], event: str, **fields: Any) -> dict:
    d = {"id": req_id, "type": "event", "event": event}
    d.update(fields)
    return d


def progress_msg(req_id, step: int, total: int, pass_: str = "base", eta_s: float = 0.0, image: int = 0, images: int = 1) -> dict:
    return event_msg(req_id, "progress", step=int(step), total=int(total), **{"pass": pass_}, eta_s=round(float(eta_s), 3),
                     image=int(image), images=int(images))


def preview_msg(req_id, jpeg_b64: str, step: int) -> dict:
    return event_msg(req_id, "preview", jpeg_b64=jpeg_b64, step=int(step))


def log_msg(req_id, level: str, message: str) -> dict:
    if level not in ("info", "warn", "error"):
        level = "info"
    return event_msg(req_id, "log", level=level, message=message)


class Emitter:
    """Writes one JSON object per line to a binary stream, flushing every line."""

    def __init__(self, stream):
        self._stream = stream
        self._lock = threading.Lock()
        self.broken = False

    def emit(self, obj: dict) -> None:
        try:
            data = encode(obj)
        except Exception as e:  # last resort: never let an emit kill a thread
            data = encode({"id": obj.get("id") if isinstance(obj, dict) else None, "type": "error", "code": INTERNAL,
                           "message": f"unserialisable message ({type(e).__name__})", "detail": ""})
        with self._lock:
            if self.broken:
                return
            try:
                view = memoryview(data)
                while view:
                    n = self._stream.write(view)
                    if n is None:  # buffered stream wrote everything
                        break
                    view = view[n:]
                self._stream.flush()
            except (BrokenPipeError, OSError, ValueError):
                self.broken = True


class JobContext:
    """Per-request handle passed to engine methods: emits events and checks for interruption."""

    def __init__(self, emitter: Emitter, req_id: Optional[str], interrupt_event: Optional[threading.Event] = None):
        self.emitter = emitter
        self.req_id = req_id
        self.interrupt_event = interrupt_event or threading.Event()
        self.warnings: list[str] = []
        self._pass_start = time.monotonic()
        self._last_step = 0
        self.latest_progress: dict = {}

    # -- interruption ------------------------------------------------------
    @property
    def interrupted(self) -> bool:
        return self.interrupt_event.is_set() or self.emitter.broken

    def check_interrupt(self) -> None:
        if self.interrupted:
            raise EngineError(INTERRUPTED, "interrupted")

    # -- events ------------------------------------------------------------
    def start_pass(self) -> None:
        self._pass_start = time.monotonic()

    def progress(self, step: int, total: int, pass_: str = "base", image: int = 0, images: int = 1, eta_s: Optional[float] = None) -> None:
        if eta_s is None:
            elapsed = time.monotonic() - self._pass_start
            eta_s = (elapsed / step) * max(0, total - step) if step > 0 else 0.0
        self.latest_progress = progress_msg(self.req_id, step, total, pass_, eta_s, image, images)
        self.emitter.emit(self.latest_progress)

    def preview(self, jpeg_b64: str, step: int) -> None:
        self.emitter.emit(preview_msg(self.req_id, jpeg_b64, step))

    def log(self, level: str, message: str) -> None:
        self.emitter.emit(log_msg(self.req_id, level, message))

    def warn(self, message: str) -> None:
        """Record a warning for the result and also emit it as a log event."""
        if message not in self.warnings:
            self.warnings.append(message)
            self.log("warn", message)
