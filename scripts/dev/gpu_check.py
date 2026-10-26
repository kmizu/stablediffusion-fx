"""One real-GPU check per engine process: python gpu_one.py <venv python> <out dir> <case>."""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

PY, OUT, CASE = sys.argv[1], sys.argv[2], sys.argv[3]
PY, OUT = os.path.abspath(PY), os.path.abspath(OUT)
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
if CASE in ("neutral_sd15", "neutral_sdxl", "neutral_alt_sd15", "neutral_alt_sdxl"):
    from PIL import Image, ImageChops, ImageStat
    load(SD15 if CASE.endswith("sd15") else SDXL)
    size = dict(width=512, height=768) if CASE.endswith("sd15") else dict(width=832, height=832)
    plain = "portrait of a woman, in a park, detailed"
    common = dict(prompt=plain, steps=16, seed=11, **size)
    enabled = {"neutral_prompt": {}}
    variants = [("baseline", {}), ("noop", dict(hooks=enabled)),
        ("and", dict(prompt=plain + " :0.5 AND red scarf :0.5", hooks=enabled)),
        ("perp", dict(prompt=plain + " AND_PERP black hair :-1", hooks=enabled)),
        ("salt", dict(prompt=plain + " AND_SALT red scarf :0.7", hooks=enabled)),
        ("topk", dict(prompt=plain + " AND_TOPK red scarf :0.7", hooks=enabled)),
        ("nested", dict(prompt=plain + " AND_PERP [black hair AND_SALT red scarf :0.5] :-0.5", hooks=enabled)),
        ("rescale", dict(hooks={"neutral_prompt": {"cfg_rescale": 0.7}})),
        ("schedule", dict(prompt=plain + " AND_TOPK [red scarf:blue scarf:0.5] :0.7", hooks=enabled)),
        ("batch", dict(prompt=plain + " AND_PERP black hair :-1", batch_size=2, hooks=enabled)),
        ("cfg1", dict(prompt=plain + " AND_TOPK red scarf :0.7", cfg_scale=1, hooks=enabled)),
        ("hires", dict(prompt=plain + " AND_TOPK red scarf :0.7", hires=dict(gp()["hires"], enabled=True, scale=1.25, second_pass_steps=8), hooks=enabled)),
        ("img2img", dict(prompt=plain + " AND_TOPK red scarf :0.7", img2img=i2i(os.path.join(OUT, CASE + "_baseline.png"), denoise=0.5), hooks=enabled)),
        ("euler", dict(prompt=plain + " AND_PERP black hair :-0.5", sampler_name="Euler", hooks=enabled)),
        ("heun", dict(prompt=plain + " AND_TOPK [red scarf:blue scarf:0.5] :0.7", sampler_name="Heun", hooks=enabled)),
        ("negpip", dict(prompt=plain + " AND_TOPK (black hair:-1) :0.7", hooks={"neutral_prompt": {}, "negpip": {}})),
        ("gradient", dict(prompt=plain + " AND_TOPK {red scarf::0.1-1.5} :0.7", hooks={"neutral_prompt": {}, "weight_gradient": {}})),
        ("restored", {})]
    if CASE.startswith("neutral_alt_"):
        init = i2i(os.path.join(OUT, CASE + "_baseline.png"), denoise=0.5)
        alt = {"inversion_steps": 8, "decode_cfg_scale": 1}
        def alternative(prompt=plain, phi=0, decode=1, extra=None, count=8):
            hooks = {"neutral_prompt": {"cfg_rescale": phi}, "img2imgalt": dict(alt, decode_cfg_scale=decode, inversion_steps=count)}
            hooks.update(extra or {})
            return dict(prompt=prompt, img2img=init, hooks=hooks)
        variants = [("baseline", {}), ("noop", dict(hooks=enabled)),
            ("alt_baseline", dict(img2img=init, hooks={"img2imgalt": alt})),
            ("alt_noop", alternative()),
            ("alt_perp", alternative(plain + " AND_PERP black hair :-0.5")),
            ("alt_salt", alternative(plain + " AND_SALT red scarf :0.5")),
            ("alt_topk_rescale", alternative(plain + " AND_TOPK red scarf :0.5", phi=0.7)),
            ("alt_and_decode0", alternative(plain + " :0.7 AND red scarf :0.3", decode=0, phi=0.4)),
            ("alt_schedule_decodehalf", alternative(plain + " AND_TOPK [red scarf:blue scarf:0.5] :0.5", decode=0.5)),
            ("alt_negpip", alternative(plain + " AND_TOPK (black hair:-1) :0.5", extra={"negpip": {}})),
            ("alt_default50", alternative(plain + " AND_TOPK red scarf :0.5", count=50)),
            ("restored", {})]
    paths, report, errors = {}, {}, []
    for tag, changes in variants:
        result = call("img2img" if changes.get("img2img") else "txt2img", req(gp(**dict(common, **changes))))
        path = keep(result, CASE + "_" + tag)
        if path:
            paths[tag] = path
            report[tag] = {"path": path, "warnings": result["result"]["warnings"], "images": len(result["result"]["images"])}
            if report[tag]["warnings"]:
                errors.append({tag: report[tag]["warnings"]})
        else:
            errors.append({tag: result})
        print(tag, report.get(tag) or result, flush=True)
    for a, b in [("baseline", "noop"), ("baseline", "restored")] + ([("alt_baseline", "alt_noop")] if CASE.startswith("neutral_alt_") else []):
        if a in paths and b in paths:
            with Image.open(paths[a]) as im1, Image.open(paths[b]) as im2:
                diff = sum(ImageStat.Stat(ImageChops.difference(im1.convert("RGB"), im2.convert("RGB"))).mean) / 3
            report[a + "_vs_" + b] = diff
            if diff:
                errors.append({a + "_vs_" + b: diff})
    with open(os.path.join(OUT, CASE + ".json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"cases": report, "errors": errors}, fh, indent=2)
    r = {"type": "error" if errors else "result", "result": {"result": report}, "message": str(errors)}
elif CASE in ("negpip_sd15", "negpip_sdxl"):
    from PIL import Image, ImageChops, ImageStat
    load(SD15 if CASE.endswith("sd15") else SDXL)
    size = dict(width=512, height=768) if CASE.endswith("sd15") else dict(width=832, height=832)
    plain = "portrait of a woman, black hair, in a park, detailed"
    signed = "portrait of a woman, (black hair:-1), in a park, detailed"
    common = dict(prompt=plain, steps=16, seed=11, **size)
    variants = [
        ("baseline", {}), ("noop", dict(hooks={"negpip": {}})),
        ("ordinary_signed", dict(prompt=signed)),
        ("positive_minus1", dict(prompt=signed, hooks={"negpip": {}})),
        ("positive_minus2", dict(prompt=signed.replace(":-1", ":-2"), hooks={"negpip": {}})),
        ("negative_branch", dict(negative_prompt="lowres, (red scarf:-2)", hooks={"negpip": {}})),
        ("schedule", dict(prompt="portrait of a woman, [(black hair:-2):0.5], detailed", hooks={"negpip": {}})),
        ("cross_zero", dict(prompt="portrait of a woman, {black hair::1--2}, detailed", hooks={"negpip": {}, "weight_gradient": {}})),
        ("batch", dict(prompt=signed, batch_size=2, hooks={"negpip": {}})),
        ("pag", dict(prompt=signed, hooks={"negpip": {}, "pag": {"scale": 2}})),
        ("hires", dict(prompt=signed, hires=dict(gp()["hires"], enabled=True, scale=1.25, second_pass_steps=8), hooks={"negpip": {}})),
        ("cfg1", dict(prompt=signed, cfg_scale=1, hooks={"negpip": {}})),
        ("img2img", dict(prompt=signed, img2img=i2i(os.path.join(OUT, CASE + "_baseline.png"), denoise=0.5), hooks={"negpip": {}})),
        ("alternative", dict(prompt=signed, img2img=i2i(os.path.join(OUT, CASE + "_baseline.png"), denoise=0.5),
                             hooks={"negpip": {}, "img2imgalt": {"inversion_steps": 8}})),
        ("restored", {})]
    paths, report, errors = {}, {}, []
    for tag, changes in variants:
        result = call("img2img" if changes.get("img2img") else "txt2img", req(gp(**dict(common, **changes))))
        path = keep(result, CASE + "_" + tag)
        if path:
            paths[tag] = path
            report[tag] = {"path": path, "warnings": result["result"]["warnings"], "images": len(result["result"]["images"])}
        else:
            errors.append({tag: result})
        print(tag, report.get(tag) or result, flush=True)
    def mae(a, b):
        with Image.open(paths[a]) as im1, Image.open(paths[b]) as im2:
            return sum(ImageStat.Stat(ImageChops.difference(im1.convert("RGB"), im2.convert("RGB"))).mean) / 3
    for a, b, equal in [("baseline", "noop", True), ("baseline", "restored", True),
                         ("ordinary_signed", "positive_minus1", False), ("positive_minus1", "positive_minus2", False)]:
        if a in paths and b in paths:
            diff = mae(a, b)
            report[a + "_vs_" + b] = diff
            if (diff == 0) != equal:
                errors.append({a + "_vs_" + b: diff})
    with open(os.path.join(OUT, CASE + ".json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"cases": report, "errors": errors}, fh, indent=2)
    r = {"type": "error" if errors else "result", "result": {"result": report}, "message": str(errors)}
elif CASE in ("alt_quality_sd15", "alt_quality_sdxl"):
    from PIL import Image, ImageChops, ImageStat
    import numpy as np
    import cv2
    load(SD15 if CASE.endswith("sd15") else SDXL)
    size = dict(width=512, height=768) if CASE.endswith("sd15") else dict(width=832, height=832)
    common = dict(prompt="portrait of a woman wearing (red scarf:0.8), in a park, detailed", steps=16, seed=11, **size)
    baseline = call("txt2img", req(gp(**common)))
    src = keep(baseline, CASE + "_source")
    report, errors = {}, []

    def measure(path):
        with Image.open(path) as im:
            a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
        lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
        return {"lightness_std": float(lab[..., 0].std()), "mean_chroma": float(np.hypot(lab[..., 1], lab[..., 2]).mean()),
                "high_clip": float((a >= 0.98).mean()), "low_clip": float((a <= 0.02).mean())}

    if src:
        report["source"] = dict(path=src, **measure(src))
    else:
        errors.append(baseline)
    variants = [(s, c, d, 0) for s, c, d in (("DPM++ 2M", 6, 1), ("DPM++ 2M", 2, 1), ("DPM++ 2M", 1, 1),
                 ("DPM++ 2M", 6, 6), ("Euler", 6, 1), ("Euler", 2, 1), ("Euler", 1, 1), ("Euler", 2, 2), ("DDIM", 1, 1))]
    variants += [("DPM++ 2M", 6, 0, 0), ("DPM++ 2M", 6, 0.5, 0), ("DPM++ 2M", 6, 1, 1)]
    for sampler, cfg, decode_cfg, randomness in variants:
        tag = sampler.replace("+", "p").replace(" ", "_") + f"_cfg{cfg}_decode{decode_cfg}"
        if randomness:
            tag += f"_random{randomness}"
        result = call("img2img", req(gp(**dict(common, sampler_name=sampler, cfg_scale=cfg,
                      img2img=i2i(src, denoise=1), hooks={"img2imgalt": {"decode_cfg_scale": decode_cfg, "randomness": randomness}}))))
        path = keep(result, CASE + "_" + tag)
        if path:
            report[tag] = dict(path=path, warnings=result["result"]["warnings"], **measure(path))
            with Image.open(src) as a, Image.open(path) as b:
                report[tag]["source_mae"] = sum(ImageStat.Stat(ImageChops.difference(a.convert("RGB"), b.convert("RGB"))).mean) / 3
            if randomness == 1 and report[tag]["source_mae"] != 0:
                errors.append({tag: "randomness=1 did not reproduce the ordinary same-seed baseline"})
        else:
            errors.append({tag: result})
        print(tag, report.get(tag) or result, flush=True)
    with open(os.path.join(OUT, CASE + ".json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"cases": report, "errors": errors}, fh, indent=2)
    r = {"type": "error" if errors else "result", "result": {"result": report}, "message": str(errors)}
elif CASE in ("weight_sd15", "weight_sdxl", "weight_alt_sd15", "weight_alt_sdxl"):
    from PIL import Image, ImageChops, ImageStat
    load(SD15 if CASE.endswith("sd15") else SDXL)
    size = dict(width=512, height=768) if CASE.endswith("sd15") else dict(width=832, height=832)
    prefix, suffix = "portrait of a woman wearing ", ", in a park, detailed"
    plain = prefix + "(red scarf:0.8)" + suffix
    gradient = prefix + "{red scarf::0.2-1.4:e}" + suffix
    common = dict(steps=16, seed=11, **size)
    reference = call("txt2img", req(gp(prompt=plain, **common)))
    source = keep(reference, CASE + "_reference")
    paths, errors, checks = {"reference": source}, [], {}
    if not source:
        errors.append({"reference": reference})
    cases = [
        ("constant", dict(prompt=prefix + "{red scarf::0.8-0.8}" + suffix, hooks={"weight_gradient": {}})),
        ("start", dict(prompt=prefix + "(red scarf:0.2)" + suffix)),
        ("end", dict(prompt=prefix + "(red scarf:1.4)" + suffix)),
        ("exponential", dict(prompt=gradient, hooks={"weight_gradient": {}})),
        ("return", dict(prompt=prefix + "{red scarf::1.4-0.2-1.4:c}" + suffix, hooks={"weight_gradient": {}})),
        ("negative", dict(prompt=plain, negative_prompt="lowres, {red scarf::0.1-1.5:eo}", hooks={"weight_gradient": {}})),
        ("hires", dict(prompt=gradient, hooks={"weight_gradient": {}},
                       hires=dict(gp()["hires"], enabled=True, scale=1.25, second_pass_steps=7, denoising_strength=0.6))),
        ("img2img", dict(prompt=gradient, hooks={"weight_gradient": {}}, img2img=i2i(source, denoise=0.5))),
        ("alternative", dict(prompt=gradient, hooks={"weight_gradient": {}, "img2imgalt": {"inversion_steps": 6}},
                             img2img=i2i(source, denoise=0.4))),
        ("heun_reference", dict(prompt=plain, sampler_name="Heun")),
        ("heun_constant", dict(prompt=prefix + "{red scarf::0.8-0.8}" + suffix,
                               hooks={"weight_gradient": {}}, sampler_name="Heun")),
        ("heun_start", dict(prompt=prefix + "(red scarf:0.2)" + suffix, sampler_name="Heun")),
        ("heun_gradient", dict(prompt=gradient, hooks={"weight_gradient": {}}, sampler_name="Heun")),
        ("restored", dict(prompt=plain)),
    ]
    if CASE.startswith("weight_alt_"):
        cases = [
            ("alternative_start", dict(prompt=prefix + "(red scarf:0.2)" + suffix, hooks={"img2imgalt": {}},
                                       img2img=i2i(source, denoise=0.4))),
            ("alternative", dict(prompt=gradient, hooks={"weight_gradient": {}, "img2imgalt": {}},
                                 img2img=i2i(source, denoise=0.4))),
            ("restored", dict(prompt=plain)),
        ]
    for tag, options in cases:
        result = call("img2img" if options.get("img2img") else "txt2img", req(gp(**dict(common, **options))))
        path = keep(result, CASE + "_" + tag)
        paths[tag] = path
        if not path or result.get("result", {}).get("warnings"):
            errors.append({tag: result})
            continue
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            checks[tag] = {"path": path, "size": list(im.size), "std": ImageStat.Stat(rgb).stddev}
            if max(checks[tag]["std"]) < 2:
                errors.append({tag: "uniform output"})
        print(tag, checks[tag], flush=True)
    differences = {}
    for a, b in (("constant", "reference"), ("restored", "reference"),
                 ("exponential", "start"), ("exponential", "end"),
                 ("heun_constant", "heun_reference"), ("heun_gradient", "heun_start")):
        if paths.get(a) and paths.get(b):
            with Image.open(paths[a]) as x, Image.open(paths[b]) as y:
                diff = sum(ImageStat.Stat(ImageChops.difference(x.convert("RGB"), y.convert("RGB"))).mean) / 3
            differences[f"{a}-{b}"] = diff
            if (a in ("constant", "restored", "heun_constant") and diff != 0) or (a in ("exponential", "heun_gradient") and diff <= 0):
                errors.append({f"{a}-{b}": diff})
    report = {"paths": paths, "checks": checks, "differences": differences, "errors": errors}
    with open(os.path.join(OUT, CASE + ".json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2)
    r = {"type": "error" if errors else "result", "result": {"result": differences}, "message": str(errors)}
elif CASE in ("image_sd15", "image_sdxl"):
    from PIL import Image
    import numpy as np
    load(SD15 if CASE == "image_sd15" else SDXL)
    size = dict(width=512, height=768) if CASE == "image_sd15" else dict(width=832, height=1216)
    r = call("txt2img", req(gp(prompt="a woman wearing a red jacket, standing in a green park, full body, detailed", **size)))
    assert r["type"] == "result", r
    src = keep(r, CASE + "_source")
    assert src, "generation returned no readable source image"
    checks = {}
    for tool, opts in (
        ("pixel_art", {"pixel_size": 8, "colors": 16, "dither": 0.5}),
        ("color_enhance", {"strength": 0.5}),
        ("cas_upscale", {"scale": 2, "sharpness": 0.5}),
        ("person_mask", {"dilate": 8, "invert": True}),
        ("remove_background", {"model": "person_segmentation"}),
        ("remove_background", {"model": "isnet-anime"}),
        ("depth_mask", {"near": 0.5, "far": 1.0}),
    ):
        tag = tool + ("_" + opts["model"] if "model" in opts else "")
        result = call("image_tool", dict(tool=tool, image_path=src, output_dir=OUT, **opts))
        if result["type"] != "result":
            checks[tag] = {"error": result}
            print(tag, result, flush=True)
            continue
        payload = result["result"]
        path = keep(result, CASE + "_" + tag)
        with Image.open(path) as im:
            arr = np.asarray(im)
            checks[tag] = {"path": path, "size": list(im.size), "mode": im.mode,
                           "std": float(arr.std()), "outputs": payload.get("outputs", {})}
            if im.mode == "RGBA":
                alpha = arr[..., 3]
                checks[tag]["alpha_fraction"] = float((alpha > 127).mean())
                if not 0.01 < checks[tag]["alpha_fraction"] < 0.99:
                    checks[tag]["error"] = "cutout alpha is empty or opaque"
            if tool in ("person_mask", "depth_mask"):
                checks[tag]["mask_fraction"] = float((arr[..., 0] > 127).mean())
                if not 0.01 < checks[tag]["mask_fraction"] < 0.99:
                    checks[tag]["error"] = "mask is empty or full"
            expected = (size["width"], size["height"])
            if tool == "cas_upscale":
                expected = tuple(v * 2 for v in expected)
            if im.size != expected:
                checks[tag]["error"] = f"wrong size: {im.size} != {expected}"
        print(tag, checks[tag], flush=True)
        if tool in ("person_mask", "depth_mask"):
            edited = call("img2img", req(gp(prompt="a woman wearing a red jacket, standing on a snowy mountain", steps=20,
                                             img2img=i2i(src, path, 0.8), **size)))
            checks[tag]["inpaint"] = (keep(edited, CASE + "_" + tag + "_inpaint")
                                       if edited["type"] == "result" else edited)
    with open(os.path.join(OUT, CASE + ".json"), "w", encoding="utf-8", newline="\n") as report:
        json.dump(checks, report, indent=2)
    ok = all("error" not in value and ("inpaint" not in value or isinstance(value["inpaint"], str))
             for value in checks.values())
    r = {"type": "result" if ok else "error", "result": {"result": checks}, "message": str(checks)}
elif CASE in ("depth_marigold", "normal_marigold", "recolor_luminance", "tile_resample", "blur_gaussian"):
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
saved = keep(r, CASE) if ok and CASE not in ("wd14", "blip") and not CASE.startswith(("sched", "ext_", "ad_", "image_")) else None
print(f"{CASE:18s} {'OK ' if ok else 'ERR'} {time.time() - t0:7.1f}s {str(detail)[:400]} {saved or ''}", flush=True)
with open(os.path.join(OUT, "results.tsv"), "a", encoding="utf-8") as fh:
    row = [time.strftime("%H:%M:%S"), CASE, "OK" if ok else "ERR", f"{time.time() - t0:.1f}s",
           str(detail).replace(chr(10), " ")[:120]]
    fh.write(chr(9).join(row) + chr(10))
call("shutdown", {}, timeout=60)
proc.wait(120)
sys.exit(0 if ok else 1)
