"""External official-SDK acceptance. Run in an isolated venv with mcp==2.3.0.

Each transport starts its own server process and isolated app home. Real-model
runs require explicit --checkpoint and --models; --dry-run is labeled in receipt.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import hashlib
import base64
import struct
import zlib
import subprocess
import time

from mcp import Client, StdioServerParameters
from PIL import Image


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def exercise(client, args, receipt):
    receipt["protocol_version"] = client.protocol_version
    receipt["server_info"] = client.server_info.model_dump(mode="json")
    listed = await client.list_tools()
    names = {tool.name for tool in listed.tools}
    receipt["tools"] = [tool.model_dump(mode="json") for tool in listed.tools]
    assert {"txt2img", "img2img", "get_job", "cancel_job", "list_models"} <= names
    receipt["calls"] = []
    receipt["progress_events"] = []
    receipt["image_blocks"] = []

    async def on_progress(progress, total, message):
        receipt["progress_events"].append({"progress": progress, "total": total, "message": message})

    async def call(name, values, allow_cancelled=False, progress=False, expect_image=None):
        kwargs = {"read_timeout_seconds": 900}
        if progress:
            kwargs["progress_callback"] = on_progress
        result = await client.call_tool(name, values, **kwargs)
        encoded = result.model_dump(mode="json")
        receipt["calls"].append({"tool": name, "arguments": values, "result": encoded})
        data = encoded.get("structuredContent", encoded.get("structured_content")) or {}
        assert (not encoded.get("isError", encoded.get("is_error", False)) or
                (allow_cancelled and data.get("status") == "cancelled")), encoded
        if expect_image is not None:
            blocks = [block for block in encoded.get("content", [])
                      if block.get("type") == "image" and block.get("mimeType", block.get("mime_type")) == expect_image]
            assert blocks, f"{name} did not return an {expect_image} image content block"
            decoded = base64.b64decode(blocks[0].get("data", ""), validate=True)
            assert decoded, f"{name} returned an empty image content block"
            receipt["image_blocks"].append({"mime_type": expect_image, "decoded_bytes": len(decoded)})
        return data

    await call("list_models", {})
    await call("list_samplers", {})
    await call("list_scripts", {})
    if args.require_extensions:
        assert {"list_always_on", "inspect_generation_metadata"} <= names
        await call("list_always_on", {"tab": "txt2img"})
    # Wait for the engine with the ordinary API instead of assuming process readiness.
    await call("set_model", {"name": args.checkpoint})
    params = dict(prompt="a blue ceramic teapot on a wooden table, soft daylight",
                  negative_prompt="blurry, distorted", width=args.side, height=args.side,
                  steps=args.steps, seed=173, return_images=False, sampler_name="Euler")
    generated = await call("txt2img", params, progress=True)
    assert generated.get("images"), generated
    assert receipt["progress_events"], "official SDK received no progress notifications"
    assert receipt["progress_events"][-1]["progress"] == 100, receipt["progress_events"]
    image = generated["images"][0]["path"]
    assert Path(image).is_file(), image
    await call("get_image", {"path": image, "max_side": 256}, expect_image="image/png")
    if args.require_extensions:
        metadata = await call("inspect_generation_metadata", {"image": image})
        restored = dict(metadata["generation_args"], return_images=False)
        if metadata["always_on"]:
            restored["always_on"] = metadata["always_on"]
        if "model_keyword" not in metadata["always_on"]:
            restored["model_keyword"] = False
        replayed = await call("txt2img", restored)
        receipt["replay_pairs"] = [[image, replayed["images"][0]["path"]]]
        extension_params = dict(params, always_on={"freeu": {"enabled": True, "b1": 1.15, "b2": 1.2}})
        enhanced = await call("txt2img", extension_params)
        assert "FreeU Stages" in enhanced["images"][0]["infotext"]
        ext_metadata = await call("inspect_generation_metadata", {"image": enhanced["images"][0]["path"]})
        assert ext_metadata["always_on"].get("freeu", {}).get("enabled") is True
        replay_args = dict(ext_metadata["generation_args"], always_on=ext_metadata["always_on"], return_images=False)
        if "model_keyword" not in replay_args["always_on"]:
            replay_args["model_keyword"] = False
        ext_replay = await call("txt2img", replay_args)
        receipt["replay_pairs"].append([enhanced["images"][0]["path"], ext_replay["images"][0]["path"]])
        for first, second in receipt["replay_pairs"]:
            with Image.open(first) as a, Image.open(second) as b:
                assert a.size == b.size and a.convert("RGB").tobytes() == b.convert("RGB").tobytes(), (first, second)
    edited = await call("img2img", dict(params, init_image=image, denoising_strength=0.4))
    assert edited.get("images"), edited
    if args.require_extensions:
        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        side = args.side
        rows = (b"\0" + bytes([0]) * (side // 2) + bytes([255]) * (side - side // 2)) * side
        mask = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
        inpainted = await call("img2img", dict(params, init_image=image, mask=base64.b64encode(mask).decode(),
                                               denoising_strength=0.4, inpaint_full_res=True))
        assert inpainted.get("images"), inpainted
    queued = await call("txt2img_async", params)
    job = queued["job_id"]
    for _ in range(120):
        status = await call("get_job", {"job_id": job, "wait_seconds": 5, "return_images": False})
        if status.get("status") in {"done", "failed", "cancelled"}:
            break
    assert status.get("status") == "done", status
    long_params = dict(params, steps=100)
    if args.dry_run:
        long_params["extra"] = {"step_delay": "0.03"}
    pending = await call("txt2img_async", long_params)
    cancelled = await call("cancel_job", {"job_id": pending["job_id"]})
    receipt["cancel_response"] = cancelled
    for _ in range(60):
        state = await call("get_job", {"job_id": pending["job_id"], "wait_seconds": 2,
                                       "return_images": False}, allow_cancelled=True)
        if state.get("status") in {"done", "failed", "cancelled"}:
            break
    assert state.get("status") == "cancelled", state
    # A cancelled job must leave the service able to generate again.
    recovery = await call("txt2img", params)
    assert recovery.get("images"), recovery
    receipt["success"] = True


async def run(args, receipt):
    app_home = args.output / "app-home"
    (app_home / "config").mkdir(parents=True, exist_ok=True)
    (app_home / "python").mkdir(exist_ok=True)
    if args.dry_run:
        models = app_home / "models" / "checkpoints"
        models.mkdir(parents=True, exist_ok=True)
        (models / (args.checkpoint + ".safetensors")).write_bytes(b"dry-run fixture")
    else:
        models = args.models.resolve()
    settings = dict(python_exe=str(args.python.resolve()), output_dir=str(args.output / "images"),
                    model_dirs={"checkpoints": [str(models)]}, last_model=args.checkpoint,
                    vram_mode="model_offload", dtype="fp16", preview_every=2)
    (app_home / "config" / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(args.source.resolve() / "python"))
    if args.dry_run:
        env["SDFX_DRY_RUN_DELAY_MS"] = "30"
    command = [str(args.java), "-Xlog:disable", "-Xlog:all=warning:stderr", "-Dfile.encoding=UTF-8",
               "-Djava.awt.headless=true", "-cp", str(args.jar.resolve()), "sdfx.mcp.McpMain",
               "--home", str(app_home), "--drain-seconds", "5"]
    if args.launcher:
        command = [str(args.launcher.resolve()), "--mcp", "--home", str(app_home), "--drain-seconds", "5"]
    if args.dry_run:
        command.append("--dry-run")
    receipt.update(transport=args.transport, dry_run=args.dry_run,
                   sdk=importlib.metadata.version("mcp"), command=command, settings=settings,
                   jar_sha256=hashlib.sha256(args.jar.read_bytes()).hexdigest())
    if args.transport == "stdio":
        async with Client(StdioServerParameters(command=command[0], args=command[1:],
                                                env=env, cwd=str(args.source.resolve())),
                          read_timeout_seconds=900) as client:
            await exercise(client, args, receipt)
    else:
        port = free_port()
        command += ["--http", str(port)]
        with (args.output / "server-stderr.log").open("w", encoding="utf-8") as err:
            server = subprocess.Popen(command, env=env, cwd=args.source, stdout=err, stderr=err)
            try:
                deadline = time.monotonic() + 60
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f"HTTP server exited {server.returncode}")
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                            break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise TimeoutError("HTTP server did not start")
                        await asyncio.sleep(0.1)
                async with Client(f"http://127.0.0.1:{port}/mcp", read_timeout_seconds=900) as client:
                    await exercise(client, args, receipt)
            finally:
                # Only this harness's directly created server is terminated.
                server.terminate()
                server.wait(timeout=30)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["stdio", "http"], required=True)
    p.add_argument("--java", type=Path)
    p.add_argument("--launcher", type=Path)
    p.add_argument("--jar", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--require-extensions", action="store_true")
    p.add_argument("--models", type=Path)
    p.add_argument("--checkpoint", default="fixture")
    p.add_argument("--side", type=int, default=512)
    p.add_argument("--steps", type=int, default=12)
    args = p.parse_args()
    if not args.java and not args.launcher:
        p.error("--java or --launcher is required")
    if not args.dry_run and not args.models:
        p.error("real runs require --models")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {"success": False}
    try:
        asyncio.run(run(args, receipt))
    finally:
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")


if __name__ == "__main__":
    main()
