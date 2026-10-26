"""One real-GPU check per engine process: python gpu_one.py <venv python> <out dir> <case>."""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

PY, OUT, CASE = sys.argv[1], sys.argv[2], sys.argv[3]
SD15 = r"D:\stable-diffusion-webui-reForge\models\Stable-diffusion\animekawa_v10.safetensors"
SDXL = r"D:\stable-diffusion-webui-reForge\models\Stable-diffusion\waiIllustriousSDXL_v170.safetensors"
MODELS = os.environ.get("SDFX_MODELS", os.path.join(OUT, "models"))
os.makedirs(MODELS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
FACE = os.path.join(OUT, "face_base.png")


def fetch(url, name):
    p = os.path.join(MODELS, name)
    if not os.path.isfile(p):
        print("downloading", name, flush=True)
        urllib.request.urlretrieve(url, p + ".part")
        os.replace(p + ".part", p)
    return p


def hf(repo, file):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, file, local_dir=os.path.join(MODELS, repo.replace("/", "__")))


env = dict(os.environ, PYTHONUNBUFFERED="1")
if CASE == "custom_code":
    env["SDFX_ALLOW_CODE"] = "1"
proc = subprocess.Popen([PY, "-m", "sdfx"], cwd=r"D:\stable-diffusion-webui-reForge\stablediffusion-fx\python",
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=open(os.path.join(OUT, f"stderr-{CASE}.txt"), "w"), text=True, encoding="utf-8", env=env)
lines, cv, n = [], threading.Condition(), [0]


def reader():
    for ln in proc.stdout:
        with cv:
            lines.append(json.loads(ln))
            cv.notify_all()


threading.Thread(target=reader, daemon=True).start()


def call(typ, params, timeout=5400):
    n[0] += 1
    rid = f"r{n[0]}"
    proc.stdin.write(json.dumps({"id": rid, "type": typ, "params": params}) + "\n")
    proc.stdin.flush()
    t0 = time.time()
    with cv:
        while True:
            for m in lines:
                if m.get("id") == rid and m["type"] in ("result", "error"):
                    lines.remove(m)
                    return m
            cv.wait(1)
            if time.time() - t0 > timeout:
                return {"type": "error", "message": "timeout"}


def gp(**kw):
    p = {"prompt": "portrait photo of a smiling woman, detailed face", "negative_prompt": "lowres", "seed": 7, "subseed": -1,
         "subseed_strength": 0.0, "sampler_name": "DPM++ 2M", "scheduler": "Automatic", "steps": 16, "cfg_scale": 6.0,
         "width": 512, "height": 512, "batch_size": 1, "n_iter": 1, "clip_skip": 1, "eta": None,
         "hires": {"enabled": False, "scale": 2.0, "upscaler": "Latent", "second_pass_steps": 0, "resize_x": 0, "resize_y": 0,
                   "denoising_strength": 0.55, "sampler_name": None, "scheduler": None, "prompt": "", "negative_prompt": "",
                   "cfg_scale": None},
         "img2img": None, "loras": [], "controlnet": [], "hooks": {}, "styles": [], "checkpoint": None, "vae": None, "extra": {}}
    p.update(kw)
    return p


def req(params):
    return {"params": params, "output_dir": OUT, "lora_paths": {}, "embedding_paths": [], "controlnet_paths": {},
            "upscaler_paths": {}, "preview_every": 0, "preview_method": "none"}


def i2i(init, mask=None, denoise=0.75):
    return {"init_image": init, "mask": mask, "denoising_strength": denoise, "resize_mode": 0, "mask_blur": 4,
            "inpainting_fill": 1, "inpaint_full_res": False, "inpaint_full_res_padding": 32, "inpainting_mask_invert": False}


def keep(r, name):
    if r.get("type") == "result":
        res = r["result"]
        imgs = res.get("images") or []
        src = (imgs[0]["path"] if isinstance(imgs[0], dict) else imgs[0]) if imgs else res.get("path")
        if src and os.path.isfile(src):
            dst = os.path.join(OUT, f"{name}{os.path.splitext(src)[1]}")
            os.replace(src, dst)
            return dst
    return None


with cv:
    while not lines:
        cv.wait(1)
    hello = lines.pop(0)


def load(path):
    r = call("load_model", {"path": path, "dtype": "fp16", "vram_mode": "gpu"})
    assert r["type"] == "result", r


def mask_png(size):
    from PIL import Image, ImageDraw
    p = os.path.join(OUT, f"mask{size[0]}x{size[1]}.png")
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rectangle((size[0] // 4, size[1] // 3, 3 * size[0] // 4, 2 * size[1] // 3), fill=255)
    m.save(p)
    return p


t0 = time.time()
r = None
if CASE in ("depth_marigold", "normal_marigold", "recolor_luminance", "tile_resample", "blur_gaussian"):
    r = call("preprocess", {"preprocessor": CASE, "image_path": FACE, "output_dir": OUT, "args": {"down_sampling_rate": 4.0}})
elif CASE == "gfpgan":
    model = fetch("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth", "GFPGANv1.4.pth")
    r = call("restore_faces", {"image_path": FACE, "model": model, "output_dir": OUT})
elif CASE == "face_restore_hook":
    load(SD15)
    model = fetch("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth", "GFPGANv1.4.pth")
    r = call("txt2img", req(gp(hooks={"face_restore": {"model": model, "visibility": 1.0}})))
elif CASE == "blip":
    r = call("caption", {"image_path": FACE, "method": "blip"})
elif CASE == "ldsr":
    r = call("upscale", {"image_path": FACE, "upscaler": "LDSR", "scale": 2.0, "output_dir": OUT, "upscaler_paths": {}})
elif CASE == "controlnet":
    load(SD15)
    cn = hf("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11p_sd15_canny_fp16.safetensors")
    canny = call("preprocess", {"preprocessor": "canny", "image_path": FACE, "output_dir": OUT, "args": {}})
    unit = {"enabled": True, "model": "canny", "image": canny["result"]["path"], "preprocessor": "none", "weight": 1.0,
            "guidance_start": 0.0, "guidance_end": 1.0, "preprocessor_args": {}}
    rq = req(gp(prompt="a robot face, metal, detailed", controlnet=[unit]))
    rq["controlnet_paths"] = {"canny": cn}
    r = call("txt2img", rq)
elif CASE == "codeformer":
    model = fetch("https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth", "codeformer.pth")
    r = call("restore_faces", {"image_path": FACE, "model": model, "output_dir": OUT, "codeformer_weight": 0.5})
elif CASE == "wd14":
    r = call("caption", {"image_path": FACE, "method": "wd14", "threshold": 0.35})
elif CASE == "custom_code":
    r = call("run_code", {"code": "from PIL import Image\nimages = [Image.new('RGB', (64, 64), 'red')]\nresult = type(pipe).__name__",
                          "output_dir": OUT})
elif CASE == "ip_adapter":
    load(SD15)
    ipa = hf("h94/IP-Adapter", "models/ip-adapter_sd15.safetensors")
    r = call("txt2img", req(gp(prompt="a cat", hooks={"ip_adapter": {"model": ipa, "image": FACE, "scale": 0.7}})))
    keep(r, "ip_adapter")
    r2 = call("txt2img", req(gp(prompt="a cat")))
    print("ip_adapter unloaded afterwards:", r2["type"], flush=True)
    keep(r2, "ip_adapter_off")
elif CASE == "controllllite":
    load(SDXL)
    lllite = hf("kohya-ss/controlnet-lllite", "controllllite_v01032064e_sdxl_canny_anime.safetensors")
    base = call("txt2img", req(gp(prompt="1girl, standing, full body, simple background", width=832, height=1216, steps=20)))
    src = keep(base, "lllite_source")
    canny = call("preprocess", {"preprocessor": "canny", "image_path": src, "output_dir": OUT, "args": {}})
    keep(canny, "lllite_canny")
    r = call("txt2img", req(gp(prompt="1girl, standing, full body, red dress, city street", width=832, height=1216, steps=20,
                               hooks={"controllllite": {"model": lllite, "image": os.path.join(OUT, "lllite_canny.png"),
                                                        "strength": 1.0, "start_percent": 0.0, "end_percent": 1.0}})))
elif CASE == "fooocus_inpaint":
    load(SDXL)
    patch = hf("lllyasviel/fooocus_inpaint", "inpaint_v26.fooocus.patch")
    head = hf("lllyasviel/fooocus_inpaint", "fooocus_inpaint_head.pth")
    base = call("txt2img", req(gp(prompt="1girl, upper body, white shirt", width=832, height=1216, steps=20)))
    src = keep(base, "fooocus_source")
    r = call("img2img", req(gp(prompt="1girl, upper body, red scarf", width=832, height=1216, steps=20,
                               img2img=i2i(src, mask_png((832, 1216)), 1.0),
                               hooks={"fooocus_inpaint": {"model": patch, "head": head}})))
elif CASE == "photomaker":
    load(SDXL)
    pm = hf("TencentARC/PhotoMaker", "photomaker-v1.bin")
    r = call("txt2img", req(gp(prompt="a woman img, portrait, upper body", width=1024, height=1024, steps=20,
                               hooks={"photomaker": {"model": pm, "images": [FACE], "trigger_word": "img",
                                                     "start_merge_step": 10}})))
elif CASE == "svd":
    from huggingface_hub import snapshot_download
    model = snapshot_download(os.environ.get("SDFX_SVD_REPO", "stabilityai/stable-video-diffusion-img2vid"),
                              allow_patterns=["*.json", "*.txt", "*fp16.safetensors"],
                              local_dir=os.path.join(MODELS, "svd"))
    load(SD15)
    r = call("img2vid", {"model": model, "image_path": FACE, "output_dir": OUT, "frames": 14, "steps": 20, "width": 1024,
                         "height": 576, "decode_chunk_size": 2, "seed": 1})
elif CASE == "zero123_object":
    load(SD15)
    obj = call("txt2img", req(gp(prompt="a single red sneaker, product photo, plain white background, centered, side view",
                                  negative_prompt="person, text", width=512, height=512, steps=20)))
    src = keep(obj, "z123_object")
    views = []
    for az in (0, 90, 180):
        rv = call("novel_view", {"model": os.path.join(MODELS, "zero123"), "image_path": src, "output_dir": OUT,
                                 "elevation": 0, "azimuth": az, "steps": 50, "cfg_scale": 3.0, "seed": 1})
        views.append(keep(rv, f"z123_az{az}"))
    r = rv
elif CASE == "zero123":
    model = os.path.join(MODELS, "zero123")
    r = call("novel_view", {"model": model, "image_path": FACE, "output_dir": OUT, "elevation": 10, "azimuth": 60,
                            "steps": 30, "seed": 1})
elif CASE == "sched_sd15" or CASE == "sched_sdxl":
    load(SD15 if CASE == "sched_sd15" else SDXL)
    res = {}
    for tag, p in (("cat", "a photo of a cat, studio"), ("dog", "a photo of a dog, studio"),
                   ("edit", "a photo of a [cat:dog:0.5], studio"), ("alt", "a photo of a [cat|dog], studio"),
                   ("hr", "a photo of a [cat:dog:1.5], studio")):
        kw = dict(prompt=p, steps=20, seed=11)
        if CASE == "sched_sdxl":
            kw.update(width=832, height=832)
        if tag == "hr":
            kw["hires"] = dict(gp()["hires"], enabled=True, scale=1.25, second_pass_steps=10, denoising_strength=0.6)
        r = call("txt2img", req(gp(**kw)))
        assert r["type"] == "result", r
        res[tag] = keep(r, f"{CASE}_{tag}")
        print(tag, r["result"]["warnings"], flush=True)
    from PIL import Image, ImageChops, ImageStat
    def d(a, b):
        return round(sum(ImageStat.Stat(ImageChops.difference(Image.open(res[a]).convert("RGB"), Image.open(res[b]).convert("RGB"))).mean) / 3, 2)
    r = {"type": "result", "result": {"result": {"edit-cat": d("edit", "cat"), "edit-dog": d("edit", "dog"), "alt-cat": d("alt", "cat"), "alt-dog": d("alt", "dog")}}}
elif CASE in ("ext_sd15", "ext_sdxl"):
    load(SD15 if CASE == "ext_sd15" else SDXL)
    size = dict(width=832, height=832) if CASE == "ext_sdxl" else {}
    base = dict(prompt="a cozy cabin in a snowy forest, detailed", steps=20, seed=5, **size)
    rb = call("txt2img", req(gp(**base)))
    assert rb["type"] == "result", rb
    b0 = keep(rb, f"{CASE}_base")
    from PIL import Image, ImageChops, ImageStat
    tests = {"vectorscope_cc": {"brightness": 1.5, "saturation": 1.5}, "vectorscope_cc#2": {"r": 2.0, "contrast": 2.0},
             "color_grading": {"r": 0.0}, "color_grading#2": {"r": 0.5, "recenter_strength": 1.0},
             "color_grading#3": {"recenter": False, "normalize": True}} if os.environ.get("ONLY_CG") else {"detail_daemon": {"amount": 0.4}, "vectorscope_cc": {"brightness": 1.5, "saturation": 1.5},
             "latent_mirroring": {"max_step": 0.4}, "tiling": {"tile_x": True, "tile_y": False},
             "anti_burn": {"skip_last": 2, "average": 3}, "resharpen": {"sharpness": 2.0},
             "color_grading": {"r": 0.5, "normalize": True}, "cfg_schedule": {"start_cfg": 12, "end_cfg": 2},
             "cads": {"noise_scale": 0.5}}
    out = {}
    for name, cfgv in tests.items():
        r1 = call("txt2img", req(gp(hooks={name.split("#")[0]: cfgv}, **base)))
        if r1["type"] != "result":
            out[name] = "ERR " + r1.get("message", "")[:80]
            continue
        p = keep(r1, f"{CASE}_{name.replace('#', '_')}")
        diff = sum(ImageStat.Stat(ImageChops.difference(Image.open(b0).convert("RGB"), Image.open(p).convert("RGB"))).mean) / 3
        out[name] = round(diff, 2)
        if r1["result"]["warnings"]:
            out[name] = f"{out[name]} warn={r1['result']['warnings']}"
    print(out, flush=True)
    r = {"type": "result" if all(not str(v).startswith("ERR") for v in out.values()) else "error",
         "result": {"result": out}, "message": str(out)}
elif CASE in ("ad_sd15", "ad_sdxl"):
    load(SD15 if CASE == "ad_sd15" else SDXL)
    size = dict(width=832, height=1216) if CASE == "ad_sdxl" else dict(width=512, height=768)
    base = dict(prompt="photo of a woman sitting on a bench in a park, medium shot", steps=20, seed=3, **size)
    out = {}
    for tag, hooks in (("off", {}), ("face", {"adetailer": {"units": [{"model": "face_yolov8n.pt"}]}}),
                       ("two", {"adetailer": {"units": [{"model": "face_yolov8n.pt", "prompt": "[PROMPT], smiling"},
                                                        {"model": "hand_yolov8n.pt"}]}}),
                       ("cv", {"adetailer": {"units": [{"model": "opencv_face", "denoising_strength": 0.5}]}})):
        t1 = time.time()
        r1 = call("txt2img", req(gp(hooks=hooks, **base)))
        out[tag] = (r1["type"], round(time.time() - t1, 1), (r1.get("result") or {}).get("warnings") or r1.get("message"))
        if r1["type"] == "result":
            keep(r1, f"{CASE}_{tag}")
    print(out, flush=True)
    r = {"type": "result" if all(v[0] == "result" for v in out.values()) else "error", "result": {"result": out},
         "message": str(out)}
else:
    sys.exit(f"unknown case {CASE}")

ok = r is not None and r.get("type") == "result"
detail = (r.get("result", {}).get("caption") or r.get("result", {}).get("warnings") or r.get("result", {}).get("result") or "") \
    if ok else (r or {}).get("message")
saved = keep(r, CASE) if ok and CASE not in ("wd14", "blip") and not CASE.startswith(("sched", "ext_", "ad_")) else None
print(f"{CASE:18s} {'OK ' if ok else 'ERR'} {time.time() - t0:7.1f}s {str(detail)[:400]} {saved or ''}", flush=True)
with open(os.path.join(OUT, "results.tsv"), "a", encoding="utf-8") as fh:
    row = [time.strftime("%H:%M:%S"), CASE, "OK" if ok else "ERR", f"{time.time() - t0:.1f}s",
           str(detail).replace(chr(10), " ")[:120]]
    fh.write(chr(9).join(row) + chr(10))
call("shutdown", {}, timeout=60)
proc.wait(120)
sys.exit(0 if ok else 1)
