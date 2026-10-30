"""Stdlib-only fake of the sdfx engine for the Scala engine tests.

Options: --hello-delay SECONDS, --no-hello, --crash-on-start, --crash-on-load, --fail-list KIND, --tag TEXT (echoed in hello engine_version).
Extra request types: echo {delay_ms, value}, fail {code, message}, crash, get_state, stdout_noise.
txt2img/img2img honour params.extra: force_error (error code), step_delay (seconds per step), ignore_interrupt ("true").
"""
import base64
import json
import os
import queue
import struct
import sys
import threading
import time
import zlib

args = sys.argv[1:]


def opt(name, default=None):
    if name in args:
        i = args.index(name)
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            return args[i + 1]
        return True
    return default


out_lock = threading.Lock()
stdout = open(sys.stdout.fileno(), "w", encoding="utf-8", newline="\n", buffering=1, closefd=False)


def send(obj):
    line = json.dumps(obj, ensure_ascii=False)
    with out_lock:
        stdout.write(line + "\n")
        stdout.flush()


def result(rid, res):
    send({"id": rid, "type": "result", "result": res})


def error(rid, code, message, detail=""):
    send({"id": rid, "type": "error", "code": code, "message": message, "detail": detail})


def event(rid, name, **kw):
    d = {"id": rid, "type": "event", "event": name}
    d.update(kw)
    send(d)


def png_bytes(w, h, rgb):
    raw = bytearray()
    row = bytes(rgb) * w
    for _ in range(h):
        raw.append(0)
        raw.extend(row)

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b"")


state = {
    "loaded": None,
    "vae": None,
    "last_generation": None,
    "last_extra": None,
    "requests": [],
    "load_count": 0,
}
interrupted = threading.Event()
work = queue.Queue()
work_lock = threading.Lock()
active = {"id": None, "owner": None}
queued_owners = {}
cancelled = set()


def model_info():
    p = state["loaded"]
    if p is None:
        return None
    name = os.path.splitext(os.path.basename(p))[0]
    vae = state["vae"]
    return {
        "name": name,
        "path": p,
        "kind": "sd15",
        "hash": "abcdef1234567890",
        "loaded_ms": 5,
        "vae_name": os.path.basename(vae) if vae else None,
        "vae_hash": "feed0000" if vae else None,
    }


def generate(rid, kind, req):
    params = req["params"]
    state["last_generation"] = {"type": kind, "request": req}
    if state["loaded"] is None and not params.get("extra", {}).get("allow_no_model"):
        error(rid, "no_model", "no model loaded")
        return
    extra = params.get("extra", {}) or {}
    if extra.get("force_error"):
        error(rid, extra["force_error"], "forced " + extra["force_error"])
        return
    step_delay = float(extra.get("step_delay", "0"))
    ignore = extra.get("ignore_interrupt") == "true"
    steps = int(params.get("steps", 1))
    n = int(params.get("batch_size", 1)) * int(params.get("n_iter", 1))
    seed = int(params["seed"])
    sub = int(params.get("subseed", -1))
    w, h = int(params["width"]), int(params["height"])
    out_dir = req["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    images = []
    for i in range(n):
        for s in range(1, steps + 1):
            if interrupted.is_set() and not ignore:
                break
            time.sleep(step_delay)
            event(rid, "progress", step=s, total=steps, image=i, images=n, eta_s=0.1)
            if s == steps:
                event(rid, "preview", step=s, jpeg_b64=base64.b64encode(b"fakejpeg").decode())
        if interrupted.is_set() and not ignore:
            break
        path = os.path.join(out_dir, "tmp-%s-%d.png" % (rid, i))
        with open(path, "wb") as f:
            f.write(png_bytes(w, h, ((seed + i) % 256, 80, 160)))
        image_result = {"path": path, "seed": seed + i, "subseed": sub + i if sub >= 0 else -1, "width": w, "height": h}
        if extra.get("hires_metadata") == "true":
            image_result.update(hires_seed=100 + i, hires_subseed=200 + i, hires_subseed_strength=0.4,
                hires_repeat=1, hires_seed_resize_from_w=512, hires_seed_resize_from_h=768)
        if extra.get("hires_metadata_large") == "true":
            image_result.update(hires_seed="9007199254740993", hires_subseed="9223372036854775807")
        # Synthetic typed provenance for Generator transport tests, not hardware/timing simulation.
        hardware = req.get("hardware_info", {})
        if hardware.get("enabled"):
            image_result["hardware_info"] = {"hardware_info": "Fixture GPU, Fixture CPU, 32GB RAM",
                "time_taken": f"{i + 1}.0 sec.", "remark": hardware.get("remark") or None}
        images.append(image_result)
    if interrupted.is_set() and not ignore and not images:
        error(rid, "interrupted", "interrupted")
        return
    trace = None
    composable = params.get("hooks", {}).get("composable_lora", {})
    if composable.get("plot_weights"):
        trace_dir = os.path.join(composable.get("output_dir", out_dir), "intermediates", "composable-lora")
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.join(trace_dir, str(rid) + ".json")
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump({"records": []}, f)
        trace = {"path": trace_path}
    m = model_info() or {}
    result(rid, {
        "images": images,
        "interrupted": extra.get("partial_interrupted") == "true",
        "composable_lora_trace": trace,
        "timing_s": 0.01,
        "model_name": m.get("name"),
        "model_hash": m.get("hash", "")[:10] or None,
        "vae_name": m.get("vae_name"),
        "vae_hash": m.get("vae_hash"),
        "lora_hashes": {k: "1234abcd" for k in req.get("lora_paths", {})},
        "scheduler": "Karras",
        "warnings": [],
    })


def handle_work(rid, typ, params):
    try:
        if typ in ("txt2img", "img2img"):
            generate(rid, typ, params)
        elif typ == "load_model":
            if opt("--crash-on-load"):
                sys.stderr.write("crash during load" + chr(10))
                sys.stderr.flush()
                os._exit(4)
            time.sleep(float(params.get("delay", 0)) if isinstance(params.get("delay"), (int, float)) else 0)
            if "missing" in params["path"]:
                error(rid, "model_load_failed", "cannot load " + params["path"])
                return
            state["loaded"] = params["path"]
            state["vae"] = params.get("vae_path")
            state["load_count"] += 1
            result(rid, model_info())
        elif typ == "system_status":
            count = state.get("hardware_status_requests", 0)
            state["hardware_status_requests"] = count + 1
            if count == 0:
                time.sleep(float(opt("--hardware-status-delay-once", "0")))
            result(rid, {"pid": os.getpid(), "hardware_info": "Fixture GPU, Fixture CPU, 32GB RAM"})
        elif typ == "set_vae":
            if state["loaded"] is None:
                error(rid, "no_model", "no model loaded")
                return
            state["vae"] = params.get("path")
            result(rid, model_info())
        elif typ == "upscale":
            out = os.path.join(params["output_dir"], "up-%s.png" % rid)
            os.makedirs(params["output_dir"], exist_ok=True)
            with open(out, "wb") as f:
                f.write(png_bytes(16, 16, (1, 2, 3)))
            output = {"path": out}
            hardware = params.get("hardware_info", {})
            if hardware.get("enabled"):
                output["hardware_info"] = {"hardware_info": "Fixture GPU, Fixture CPU, 32GB RAM",
                    "time_taken": "7.5 sec.", "remark": hardware.get("remark") or None}
            result(rid, output)
        elif typ == "preprocess":
            out = os.path.join(params["output_dir"], "pre-%s.png" % rid)
            os.makedirs(params["output_dir"], exist_ok=True)
            with open(out, "wb") as f:
                f.write(png_bytes(16, 16, (9, 9, 9)))
            result(rid, {"path": out})
        elif typ == "restore_faces":
            out = os.path.join(params["output_dir"], "faces-%s.png" % rid)
            os.makedirs(params["output_dir"], exist_ok=True)
            with open(out, "wb") as f:
                f.write(png_bytes(16, 16, (5, 5, 5)))
            state["last_extra"] = {"type": typ, "params": params}
            result(rid, {"path": out})
        elif typ == "caption":
            state["last_extra"] = {"type": typ, "params": params}
            result(rid, {"caption": "fake caption"})
        elif typ == "unload":
            state["loaded"] = None
            state["vae"] = None
            result(rid, {})
        else:
            error(rid, "unsupported", "unknown type " + typ)
    except Exception as e:  # noqa
        error(rid, "internal", str(e), repr(e))


def worker():
    while True:
        rid, typ, params, owner = work.get()
        with work_lock:
            queued_owners.pop(rid, None)
            skip = rid in cancelled
            cancelled.discard(rid)
            if not skip:
                interrupted.clear()
                active.update(id=rid, owner=owner)
        if skip:
            error(rid, "interrupted", "interrupted before start")
            continue
        try:
            handle_work(rid, typ, params)
        finally:
            with work_lock:
                active.update(id=None, owner=None)
                interrupted.clear()


def list_dir(dirs, exts):
    files = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _, names in os.walk(d):
            for n in sorted(names):
                if os.path.splitext(n)[1].lower() in exts:
                    p = os.path.join(root, n)
                    rel = os.path.relpath(p, d).replace("\\", "/")
                    files.append({"name": os.path.splitext(rel)[0], "path": p, "size_mb": 0.0})
    return files


def main():
    protection = {"enabled": False, "revision": 0}
    if opt("--crash-on-start"):
        sys.stderr.write("boom at start\n")
        sys.exit(2)
    delay = float(opt("--hello-delay", "0"))
    if delay:
        time.sleep(delay)
    sys.stderr.write("fake engine starting\n")
    sys.stderr.flush()
    if not opt("--no-hello"):
        send({"id": None, "type": "hello", "protocol": 1, "engine_version": "fake-" + str(opt("--tag", "0")),
              "python": sys.version.split()[0], "torch": None, "diffusers": None, "cuda": False, "device": "cpu",
              "vram_mb": 0, "dry_run": True, "features": ["fake"]})
    threading.Thread(target=worker, daemon=True).start()
    stdin = open(sys.stdin.fileno(), "r", encoding="utf-8", closefd=False)
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        rid, typ, params = msg.get("id"), msg.get("type"), msg.get("params") or {}
        state["requests"].append(typ)
        if typ == "ping":
            result(rid, {"pong": True})
        elif typ == "shutdown":
            result(rid, {})
            shutdown_gate = opt('--shutdown-gate')
            if shutdown_gate:
                with open(str(shutdown_gate) + '.entered', 'w', encoding='utf-8') as marker:
                    marker.write('waiting for isolated test shutdown gate')
                while os.path.exists(shutdown_gate):
                    time.sleep(.01)
            stdout.flush()
            os._exit(0)
        elif typ == "crash":
            sys.stderr.write("crashing on purpose\n")
            sys.stderr.flush()
            os._exit(3)
        elif typ == "interruption_protection":
            if "enabled" in params and not isinstance(params["enabled"], bool):
                error(rid, "invalid_params", "enabled must be boolean")
                continue
            if "enabled" in params and params["enabled"] != protection["enabled"]:
                protection["enabled"] = params["enabled"]
                protection["revision"] += 1
            result(rid, dict(protection))
        elif typ in ("interrupt", "restart_authorization"):
            blocked = protection["enabled"]
            affected = False
            active_id = None
            if typ == "interrupt" and not blocked:
                target, owner = params.get("target_id"), params.get("owner_id")
                with work_lock:
                    if (active["id"] is not None and (target is None or target == active["id"])
                            and (owner is None or owner == active["owner"])):
                        interrupted.set()
                        affected = True
                        active_id = active["id"]
                    else:
                        queued = next((key for key, value in queued_owners.items()
                                       if (target is None or target == key) and (owner is None or owner == value)), None)
                        if queued is not None:
                            cancelled.add(queued)
                            affected = True
            result(rid, {"accepted": not blocked, "blocked": blocked, "affected": affected, "active_id": active_id,
                         "protection": dict(protection), "message": "Interruptions are blocked" if blocked else None})
        elif typ == "memory":
            result(rid, {"allocated_mb": 1.5, "reserved_mb": 2.0, "total_mb": 8192.0, "ram_used_mb": 100.0})
        elif typ == "echo":
            def later(rid=rid, params=params):
                time.sleep(params.get("delay_ms", 0) / 1000.0)
                result(rid, {"value": params.get("value")})
            threading.Thread(target=later, daemon=True).start()
        elif typ == "fail":
            error(rid, params.get("code", "oom"), params.get("message", "failed"), "trace here")
        elif typ == "stdout_noise":
            with out_lock:
                stdout.write("this is not json\n")
                stdout.flush()
            result(rid, {})
        elif typ == "get_state":
            result(rid, {"loaded": state["loaded"], "vae": state["vae"], "last_generation": state["last_generation"],
                         "load_count": state["load_count"], "pid": os.getpid(), "last_extra": state["last_extra"],
                         "requests": list(state["requests"])})
        elif typ == "list_models":
            result(rid, {"models": [dict(f, kind="sd15") for f in list_dir(params.get("dirs", []), {".safetensors", ".ckpt"})]})
        elif typ == "list_files" and opt("--fail-list") == params.get("kind"):
            error(rid, "internal", "listing failed on purpose")
        elif typ == "list_files":
            result(rid, {"files": list_dir(params.get("dirs", []), {".safetensors", ".pt", ".pth", ".ckpt", ".bin"})})
        elif typ == "list_samplers":
            result(rid, {"samplers": [{"name": "Euler a", "supported": True, "default_scheduler": "Automatic"},
                                      {"name": "DPM++ 2M", "supported": True, "default_scheduler": "Karras"},
                                      {"name": "Weird", "supported": False, "default_scheduler": "Automatic"}],
                         "schedulers": ["Automatic", "Karras"],
                         "upscalers": ["Latent", "Lanczos"],
                         "preprocessors": ["none", "canny"]})
        else:
            owner = msg.get("owner_id")
            with work_lock:
                queued_owners[rid] = owner
            work.put((rid, typ, params, owner))


if __name__ == "__main__":
    main()
