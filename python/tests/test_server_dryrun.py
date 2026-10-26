"""End-to-end tests of `python -m sdfx --dry-run` over real pipes."""
import json
import os
import queue
import struct
import subprocess
import sys
import threading
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Engine:
    def __init__(self, *extra, env=None):
        e = dict(os.environ)
        e["PYTHONPATH"] = ROOT + os.pathsep + e.get("PYTHONPATH", "")
        e.update(env or {})
        self.p = subprocess.Popen([sys.executable, "-m", "sdfx", "--dry-run", *extra], cwd=ROOT,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=e)
        self.q: "queue.Queue" = queue.Queue()
        self.raw: list[bytes] = []
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._drain_err, daemon=True).start()
        self.n = 0

    def _read(self):
        for line in self.p.stdout:
            self.raw.append(line)
            self.q.put(json.loads(line.decode("utf-8")))
        self.q.put(None)

    def _drain_err(self):
        for _ in self.p.stderr:
            pass

    def next(self, timeout=30):
        m = self.q.get(timeout=timeout)
        assert m is not None, "engine closed stdout"
        return m

    def send(self, typ, params=None, rid=None):
        self.n += 1
        rid = rid or f"r{self.n}"
        msg = {"id": rid, "type": typ}
        if params is not None:
            msg["params"] = params
        self.p.stdin.write((json.dumps(msg) + "\n").encode())
        self.p.stdin.flush()
        return rid

    def collect(self, rid, timeout=60):
        """Messages for rid until its result/error (inclusive)."""
        out = []
        end = time.time() + timeout
        while True:
            m = self.next(max(0.1, end - time.time()))
            if m.get("id") != rid:
                continue
            out.append(m)
            if m["type"] in ("result", "error"):
                return out

    def call(self, typ, params=None):
        rid = self.send(typ, params)
        msgs = self.collect(rid)
        return msgs[-1], msgs[:-1]

    def close(self):
        try:
            self.p.stdin.close()
        except Exception:
            pass
        return self.p.wait(timeout=30)


@pytest.fixture
def eng():
    e = Engine()
    yield e
    if e.p.poll() is None:
        e.p.kill()


def write_ckpt(path):
    hdr = {"model.diffusion_model.input_blocks.0.0.weight": {"dtype": "F16", "shape": [320, 4, 3, 3], "data_offsets": [0, 2]},
           "cond_stage_model.transformer.text_model.x": {"dtype": "F16", "shape": [1], "data_offsets": [2, 4]}}
    b = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(b)) + b + b"\0" * 4)
    return str(path)


def gen_params(out, **kw):
    p = {"prompt": "a cat", "negative_prompt": "", "seed": 1000, "subseed": 5, "subseed_strength": 0.0,
         "sampler_name": "DPM++ 2M", "scheduler": "Automatic", "steps": 4, "cfg_scale": 7.0, "width": 128, "height": 96,
         "batch_size": 1, "n_iter": 1, "clip_skip": 1, "eta": None,
         "hires": {"enabled": False, "scale": 2.0, "upscaler": "Latent", "second_pass_steps": 0, "resize_x": 0,
                   "resize_y": 0, "denoising_strength": 0.55, "sampler_name": None, "scheduler": None, "prompt": "",
                   "negative_prompt": "", "cfg_scale": None},
         "img2img": None, "loras": [], "controlnet": [], "hooks": {}, "styles": [], "checkpoint": None, "vae": None,
         "extra": {}}
    p.update(kw)
    return {"params": p, "output_dir": str(out), "lora_paths": {}, "embedding_paths": [], "controlnet_paths": {},
            "upscaler_paths": {}, "preview_every": 2, "preview_method": "approx"}


def png_size(path):
    with open(path, "rb") as f:
        h = f.read(24)
    assert h[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", h[16:24])


def test_hello_ping_and_stdout_clean(eng):
    hello = eng.next()
    assert hello["id"] is None and hello["type"] == "hello"
    assert hello["protocol"] == 1 and hello["engine_version"] == "0.1.0" and hello["dry_run"] is True
    for k in ("python", "torch", "diffusers", "cuda", "device", "vram_mb", "features"):
        assert k in hello
    assert hello["torch"] is None and hello["diffusers"] is None
    res, _ = eng.call("ping")
    assert res["type"] == "result" and res["result"]["protocol"] == 1 and res["result"]["dry_run"] is True
    res, _ = eng.call("list_samplers")
    r = res["result"]
    assert {"samplers", "schedulers", "upscalers", "preprocessors"} <= set(r)
    assert "Latent (nearest-exact)" in r["upscalers"] and "Karras" in r["schedulers"]
    assert eng.close() == 0
    assert all(line.endswith(b"\n") for line in eng.raw)


def test_unknown_type_and_bad_json(eng):
    eng.next()
    res, _ = eng.call("frobnicate")
    assert res["type"] == "error" and res["code"] == "unsupported"
    eng.p.stdin.write(b"{not json\n")
    eng.p.stdin.flush()
    m = eng.next()
    assert m["type"] == "error" and m["id"] is None and m["code"] == "invalid_params"
    assert eng.close() == 0


def test_models_and_txt2img_batch(eng, tmp_path):
    eng.next()
    mdir = tmp_path / "models"
    (mdir / "sub").mkdir(parents=True)
    ck = write_ckpt(mdir / "sub" / "tiny.safetensors")
    res, _ = eng.call("list_models", {"dirs": [str(mdir)]})
    assert res["result"]["models"][0]["name"] == "sub/tiny" and res["result"]["models"][0]["kind"] == "sd15"
    res, _ = eng.call("list_files", {"kind": "lora", "dirs": [str(mdir)]})
    assert res["result"]["files"][0]["name"] == "sub/tiny"
    res, _ = eng.call("load_model", {"path": ck, "dtype": "fp16", "vram_mode": "gpu", "vae_path": None, "vae_tiling": False})
    lm = res["result"]
    assert lm["name"] == "tiny" and lm["kind"] == "sd15" and len(lm["hash"]) == 10 and lm["vae_name"] is None
    res, _ = eng.call("set_vae", {"path": None})
    assert res["type"] == "result" and res["result"]["vae_hash"] is None

    out = tmp_path / "out"
    res, events = eng.call("txt2img", gen_params(out, batch_size=2, steps=4))
    assert res["type"] == "result", res
    r = res["result"]
    assert [im["seed"] for im in r["images"]] == [1000, 1001]
    assert [im["subseed"] for im in r["images"]] == [5, 6]
    for im in r["images"]:
        assert os.path.isfile(im["path"]) and png_size(im["path"]) == (128, 96)
        assert (im["width"], im["height"]) == (128, 96)
    assert r["scheduler"] == "Karras" and r["model_name"] == "tiny" and r["model_hash"] == lm["hash"]
    assert set(r) >= {"images", "timing_s", "model_name", "model_hash", "vae_name", "vae_hash", "lora_hashes", "scheduler", "warnings"}
    prog = [e for e in events if e["event"] == "progress"]
    assert [p["step"] for p in prog] == [1, 2, 3, 4] and all(p["total"] == 4 and p["pass"] == "base" for p in prog)
    assert all(p["images"] == 2 for p in prog)
    previews = [e for e in events if e["event"] == "preview"]
    assert 1 <= len(previews) <= 2 and previews[0]["jpeg_b64"]
    # n_iter continues the seed sequence
    res, _ = eng.call("txt2img", gen_params(out, batch_size=1, n_iter=3, steps=1))
    assert [im["seed"] for im in res["result"]["images"]] == [1000, 1001, 1002]
    assert eng.close() == 0


def test_hires_target_size(eng, tmp_path):
    eng.next()
    hires = {"enabled": True, "scale": 1.5, "upscaler": "Latent", "second_pass_steps": 3, "resize_x": 0, "resize_y": 0,
             "denoising_strength": 0.5, "sampler_name": None, "scheduler": None, "prompt": "", "negative_prompt": "",
             "cfg_scale": None}
    res, events = eng.call("txt2img", gen_params(tmp_path, hires=hires, steps=2))
    assert res["type"] == "result", res
    im = res["result"]["images"][0]
    assert png_size(im["path"]) == (192, 144) and (im["width"], im["height"]) == (192, 144)
    passes = [(e["pass"], e["step"], e["total"]) for e in events if e["event"] == "progress"]
    assert passes[:2] == [("base", 1, 2), ("base", 2, 2)]
    assert [p for p in passes if p[0] == "hires"] == [("hires", 1, 3), ("hires", 2, 3), ("hires", 3, 3)]
    hires["resize_x"] = 300  # 128x96 -> (300, 300*96//128=225) -> floored to 8 like Scala targetSize
    res, _ = eng.call("txt2img", gen_params(tmp_path, hires=hires, steps=1))
    assert png_size(res["result"]["images"][0]["path"]) == (296, 224)
    hires.update(resize_x=200, resize_y=200)  # both sides: exact size (cover + center crop)
    res, _ = eng.call("txt2img", gen_params(tmp_path, hires=hires, steps=1))
    assert png_size(res["result"]["images"][0]["path"]) == (200, 200)


def test_img2img_with_mask(eng, tmp_path):
    from PIL import Image, ImageDraw
    eng.next()
    init = tmp_path / "init.png"
    Image.new("RGB", (200, 150), (0, 0, 255)).save(init)
    mask = Image.new("L", (200, 150), 0)
    ImageDraw.Draw(mask).rectangle((50, 50, 100, 100), fill=255)
    mpath = tmp_path / "mask.png"
    mask.save(mpath)
    i2i = {"init_image": str(init), "mask": str(mpath), "denoising_strength": 0.5, "resize_mode": 0, "mask_blur": 0,
           "inpainting_fill": 1, "inpaint_full_res": False, "inpaint_full_res_padding": 32, "inpainting_mask_invert": False}
    res, events = eng.call("img2img", gen_params(tmp_path / "o", img2img=i2i, steps=10, width=200 - 200 % 8, height=144))
    assert res["type"] == "result", res
    im = res["result"]["images"][0]
    assert png_size(im["path"]) == (200, 144)
    prog = [e for e in events if e["event"] == "progress"]
    assert prog[-1]["total"] == 5  # 10 steps * 0.5 denoise
    with Image.open(im["path"]) as out:
        assert out.convert("RGB").getpixel((5, 5)) == (0, 0, 255)  # unmasked area kept
    # only-masked mode keeps the original size
    i2i.update(inpaint_full_res=True)
    res, _ = eng.call("img2img", gen_params(tmp_path / "o", img2img=i2i, steps=2, width=64, height=64))
    assert png_size(res["result"]["images"][0]["path"]) == (200, 150)
    # img2img without img2img block -> invalid params
    res, _ = eng.call("img2img", gen_params(tmp_path / "o"))
    assert res["type"] == "error" and res["code"] == "invalid_params"


def test_validation_error(eng, tmp_path):
    eng.next()
    res, _ = eng.call("txt2img", gen_params(tmp_path, steps=0, width=100))
    assert res["type"] == "error" and res["code"] == "invalid_params"
    assert "steps must be 1..150" in res["message"] and "width must be a multiple of 8" in res["message"]
    res, _ = eng.call("txt2img", gen_params(tmp_path, sampler_name="Nope"))
    assert res["code"] == "invalid_params"


def test_interrupt_mid_run(tmp_path):
    e = Engine("--dry-run-delay-ms", "50")
    try:
        e.next()
        rid = e.send("txt2img", gen_params(tmp_path, steps=100))
        # wait for a few progress events
        seen = 0
        while seen < 3:
            m = e.next()
            if m.get("id") == rid and m.get("event") == "progress":
                seen += 1
        t0 = time.time()
        irid = e.send("interrupt", {})
        msgs, ir = [], None
        while not msgs or msgs[-1]["type"] not in ("result", "error"):
            m = e.next(10)
            if m.get("id") == irid:
                ir = m
            elif m.get("id") == rid:
                msgs.append(m)
        assert msgs[-1]["type"] == "error" and msgs[-1]["code"] == "interrupted"
        assert time.time() - t0 < 5
        steps = [m["step"] for m in msgs if m.get("event") == "progress"]
        assert not steps or max(steps) < 100
        # The immediate admission response arrives before the interrupted generation's error.
        assert ir == {"id": irid, "type": "result", "result": {
            "accepted": True, "blocked": False, "affected": True, "active_id": rid,
            "protection": {"enabled": False, "revision": 0}}}
        # the flag is cleared for the next generation
        res = e.collect(e.send("txt2img", gen_params(tmp_path, steps=2)))[-1]
        assert res["type"] == "result"
        assert e.close() == 0
    finally:
        if e.p.poll() is None:
            e.p.kill()


def test_shutdown_exit_code(eng):
    eng.next()
    rid = eng.send("shutdown")
    m = eng.next()
    assert m == {"id": rid, "type": "result", "result": {}}
    assert eng.p.wait(timeout=15) == 0


def test_eof_exit_code(eng, tmp_path):
    eng.next()
    res, _ = eng.call("memory")
    assert set(res["result"]) == {"allocated_mb", "reserved_mb", "total_mb", "ram_used_mb"}
    eng.p.stdin.close()
    while eng.q.get(timeout=15) is not None:
        pass
    assert eng.p.wait(timeout=15) == 0


def test_eof_interrupts_running_and_drops_queued(tmp_path):
    e = Engine("--dry-run-delay-ms", "50")
    try:
        e.next()
        r1 = e.send("txt2img", gen_params(tmp_path, steps=150))
        r2 = e.send("txt2img", gen_params(tmp_path, steps=150))
        while True:
            m = e.next()
            if m.get("id") == r1 and m.get("event") == "progress":
                break
        t0 = time.time()
        e.p.stdin.close()
        finals = {}
        while True:
            m = e.q.get(timeout=20)
            if m is None:
                break
            if m["type"] in ("result", "error"):
                finals[m["id"]] = m
        assert e.p.wait(timeout=20) == 0 and time.time() - t0 < 10
        assert finals[r1]["code"] == "interrupted" and finals[r2]["code"] == "interrupted"
    finally:
        if e.p.poll() is None:
            e.p.kill()


def test_upscale_preprocess_unload(eng, tmp_path):
    from PIL import Image
    eng.next()
    src = tmp_path / "s.png"
    Image.new("RGB", (40, 30), (1, 2, 3)).save(src)
    res, _ = eng.call("upscale", {"image_path": str(src), "upscaler": "Lanczos", "scale": 2.0,
                                  "output_dir": str(tmp_path / "u"), "upscaler_paths": {}})
    assert png_size(res["result"]["path"]) == (80, 60)
    res, _ = eng.call("upscale", {"image_path": str(src), "upscaler": "R-ESRGAN 4x+", "scale": 2.0,
                                  "output_dir": str(tmp_path / "u")})
    assert res["code"] == "invalid_params"
    res, _ = eng.call("preprocess", {"preprocessor": "canny", "image_path": str(src), "output_dir": str(tmp_path / "p"),
                                     "args": {}})
    assert os.path.isfile(res["result"]["path"])
    res, _ = eng.call("unload")
    assert res["result"] == {}
