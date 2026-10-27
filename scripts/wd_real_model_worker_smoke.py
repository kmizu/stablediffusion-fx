"""One real WD v3 + SD img2img worker smoke; explicitly local assets only.

Runs the actual CLI/worker with no monkeypatch. Saves outputs and a JSON report
under a new directory in --output. Set PYTHONPATH to this checkout's python/ and
use the desired Python environment. HF offline flags do not sandbox networking.
One 256x256, two-step image is an operation check, not a quality evaluation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local pinned WD ONNX")
    parser.add_argument("--tags", required=True, help="local pinned WD tag CSV")
    parser.add_argument("--checkpoint", required=True, help="local SD checkpoint")
    parser.add_argument("--family", required=True, choices=("sd15", "sdxl"))
    parser.add_argument("--image", required=True, help="local original input image")
    parser.add_argument("--output", required=True, help="existing directory for new result folder")
    parser.add_argument("--vram-mode", choices=("gpu", "model_offload", "sequential_offload"), default="gpu")
    parser.add_argument("--timeout", type=float, default=600.0, help="per response deadline in seconds")
    args = parser.parse_args()
    if not 0 < args.timeout <= 3600:
        parser.error("timeout must be in (0, 3600]")
    model, tags, checkpoint, image, output = [Path(p).resolve(strict=True) for p in
                                               (args.model, args.tags, args.checkpoint, args.image, args.output)]
    if not all(p.is_file() for p in (model, tags, checkpoint, image)) or not output.is_dir():
        raise ValueError("asset paths must be files and output must be a directory")
    run = Path(tempfile.mkdtemp(prefix=f"wd-real-{args.family}-", dir=output))
    images = run / "inputs"; images.mkdir()
    destination = run / "output"; destination.mkdir()
    temp = run / "wd-temp"; temp.mkdir()
    copied = images / ("input" + image.suffix.lower())
    report = {"status": "failed", "family": args.family, "run": str(run),
              "inputs": {"wd_model": str(model), "wd_tags": str(tags),
                         "checkpoint": str(checkpoint), "image": str(image)},
              "settings": {"width": 256, "height": 256, "steps": 2, "seed": 12345,
                           "sampler": "Euler", "denoising_strength": .75, "vram_mode": args.vram_mode},
              "limits": ["single 256x256 two-step img2img result, not a quality evaluation",
                         "WD v3 CPU only; no full batch extension parity",
                         "PNG metadata is checked here; native PNG Info and Restore require a separate check",
                         "HF offline flags are library controls, not an OS network sandbox"],
              "failure": ""}
    process = None
    started = time.monotonic()
    try:
        source_paths = {"wd_model": model, "wd_tags": tags, "checkpoint": checkpoint, "image": image}
        before = {key: digest(path) for key, path in source_paths.items()}
        report["source_sha256_before"] = before
        shutil.copyfile(image, copied)
        if digest(copied) != before["image"]:
            raise RuntimeError("copied input differs from original")
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        command = [sys.executable, "-m", "sdfx", "--wd-v3-model", str(model),
                   "--wd-v3-tags", str(tags), "--wd-v3-temp", str(temp)]
        with (run / "worker-stderr.log").open("wb") as stderr:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                       text=True, encoding="utf-8", bufsize=1, env=env)
            lines = queue.Queue()
            def pump():
                try:
                    for line in process.stdout: lines.put(line)
                finally: lines.put(None)
            threading.Thread(target=pump, daemon=True).start()
            def read(deadline, stage):
                try: line = lines.get(timeout=max(0, deadline - time.monotonic()))
                except queue.Empty as exc: raise TimeoutError(f"{stage} exceeded {args.timeout}s") from exc
                if line is None: raise RuntimeError(f"worker exited before {stage}: {process.poll()}")
                return json.loads(line)
            def request(rid, kind, params, owner="wd-smoke"):
                deadline = time.monotonic() + args.timeout
                process.stdin.write(json.dumps({"id": rid, "type": kind, "params": params, "owner_id": owner}) + "\n")
                process.stdin.flush()
                while True:
                    row = read(deadline, kind)
                    if row.get("id") == rid and row.get("type") in ("result", "error"):
                        if row["type"] == "error": raise RuntimeError(f"{kind} failed: {row}")
                        return row["result"]
            hello = read(time.monotonic() + args.timeout, "startup")
            report["hello"] = hello
            if "batch_interrogate_wd_v3_partial" not in hello.get("features", []):
                raise RuntimeError("configured WD partial capability unavailable")
            if not hello.get("cuda"):
                raise RuntimeError("real SD GPU smoke requested but CUDA is unavailable")
            report["memory_before_load"] = request("memory-before", "memory", {})
            loaded = request("load", "load_model", {"path": str(checkpoint), "dtype": "fp16", "vram_mode": args.vram_mode})
            report["loaded_model"] = loaded
            token = loaded.get("model_token")
            if loaded.get("kind") != args.family or not isinstance(token, str) or not token:
                raise RuntimeError("loaded checkpoint family or token mismatch")
            report["memory_after_load"] = request("memory-loaded", "memory", {})
            prompt = "a small blue ceramic teapot, studio photograph"
            generation = {"params": {"prompt": prompt, "negative_prompt": "text, watermark",
                          "seed": 12345, "steps": 2, "width": 256, "height": 256,
                          "sampler_name": "Euler", "scheduler": "Automatic", "cfg_scale": 5.0,
                          "img2img": {"denoising_strength": .75}}}
            prep = request("prepare", "batch_interrogate_prepare", {"images_root": str(images),
                "generation": generation, "output_parent": str(destination), "output_name": "result",
                "checkpoint_token": token})
            report["prepared"] = prep
            secret = prep.get("token")
            if not isinstance(secret, str) or len(secret) != 32 or prep.get("image_count") != 1:
                raise RuntimeError("unexpected preparation result")
            outcome = request("execute", "batch_interrogate_execute", {"token": secret})
            report["result"] = outcome
            rows = outcome.get("rows")
            if not outcome.get("partial") or not isinstance(rows, list) or len(rows) != 1 or rows[0].get("status") != "complete":
                raise RuntimeError("WD batch did not produce exactly one complete result")
            row = rows[0]
            if not isinstance(row.get("positive"), str) or not row["positive"].endswith(prompt) or row["positive"] == prompt:
                raise RuntimeError("effective prompt did not include observable WD tags")
            if row.get("negative") != "text, watermark" or not row.get("interrogators"):
                raise RuntimeError("effective prompt or WD provenance missing")
            outputs = row.get("result", {}).get("images", [])
            if len(outputs) != 1 or outputs[0].get("seed") != 12345:
                raise RuntimeError("generation output count or seed mismatch")
            generated = Path(outputs[0]["path"]).resolve(strict=True)
            published = (destination / "result").resolve(strict=True)
            if not generated.is_relative_to(published) or not generated.is_file():
                raise RuntimeError("output escaped owned result directory")
            from PIL import Image
            with Image.open(generated) as result_image:
                if result_image.size != (256, 256): raise RuntimeError("unexpected output dimensions")
                report["png_text"] = {str(k): str(v) for k, v in result_image.info.items() if isinstance(v, str)}
                persisted = result_image.info.get("parameters", "")
                if not persisted.startswith(row["positive"] + "\n") or row["negative"] not in persisted:
                    raise RuntimeError("effective prompts missing from PNG metadata")
                if before["wd_model"] not in persisted or "Seed: 12345" not in persisted:
                    raise RuntimeError("WD identity or actual seed missing from PNG metadata")
            report["generated"] = {"path": str(generated), "sha256": digest(generated),
                                   "effective_positive": row["positive"], "effective_negative": row["negative"],
                                   "prompt_record_location": "PNG parameters / worker result / JSON report", "provenance": row["interrogators"]}
            report["memory_after_generation"] = request("memory-generated", "memory", {})
            request("unload", "unload", {})
            report["memory_after_unload"] = request("memory-unloaded", "memory", {})
            process.stdin.close()
            process.wait(timeout=30)
            report["worker_exit"] = process.returncode
            if process.returncode != 0: raise RuntimeError(f"worker exit {process.returncode}")
        after = {key: digest(path) for key, path in source_paths.items()}
        report["source_sha256_after"] = after
        if before != after: raise RuntimeError("a source file changed")
        if list(temp.iterdir()) or list(destination.glob(".sdfx-temporal-*")):
            raise RuntimeError("owned staging was retained")
        report["status"] = "passed"
    except BaseException as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        if process is not None and process.poll() is None:
            try:
                if process.stdin and not process.stdin.closed: process.stdin.close()
                process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait(timeout=10)
            report["worker_exit"] = process.returncode
        if "source_sha256_before" in report and "source_sha256_after" not in report:
            try:
                report["source_sha256_after"] = {key: digest(path) for key, path in source_paths.items()}
                report["sources_unchanged_after_failure"] = report["source_sha256_after"] == report["source_sha256_before"]
            except Exception as exc:
                report["source_recheck_failure"] = f"{type(exc).__name__}: {exc}"
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report_path = run / "report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"status": report["status"], "report": str(report_path), "failure": report["failure"]}, ensure_ascii=False), flush=True)
    if report["status"] != "passed": raise SystemExit(1)


if __name__ == "__main__":
    main()
