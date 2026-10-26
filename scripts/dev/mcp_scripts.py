"""Runs every Script/Extras plugin through the real app: java McpMain (stdio) -> engine -> GPU."""
import json
import os
import subprocess
import sys
import threading
import time

HOME = r"D:\stable-diffusion-webui-reForge\stablediffusion-fx"
JAR = os.path.join(HOME, "target", "stablediffusion-fx.jar")
JAVA = sys.argv[1]
ONLY = set(sys.argv[2].split(",")) if len(sys.argv) > 2 and sys.argv[2] else None
OUTS = os.path.join(HOME, "outputs")

p = subprocess.Popen([JAVA, "-Dfile.encoding=UTF-8", "-Dsdfx.home=" + HOME, "-cp", JAR, "sdfx.mcp.McpMain", "--mcp",
                      "--allow-dir", OUTS], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=open(os.path.join(OUTS, "mcp-scripts-stderr.txt"), "w"),
                     text=True, encoding="utf-8")
msgs, cv, n = [], threading.Condition(), [0]


def reader():
    for ln in p.stdout:
        with cv:
            msgs.append(json.loads(ln))
            cv.notify_all()


threading.Thread(target=reader, daemon=True).start()


def rpc(method, params=None, notify=False, timeout=1200):
    if notify:
        p.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method, "params": params or {}}) + "\n")
        p.stdin.flush()
        return None
    n[0] += 1
    i = n[0]
    p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": i, "method": method, "params": params or {}}) + "\n")
    p.stdin.flush()
    t0 = time.time()
    with cv:
        while True:
            for m in msgs:
                if m.get("id") == i:
                    msgs.remove(m)
                    return m
            cv.wait(1)
            if time.time() - t0 > timeout:
                return {"error": {"message": "timeout"}}


def tool(name, args):
    r = rpc("tools/call", {"name": name, "arguments": args})
    if "error" in r:
        return False, r["error"].get("message"), {}
    res = r["result"]
    sc = res.get("structuredContent") or {}
    text = " | ".join(c.get("text", "")[:100] for c in res.get("content", []) if c.get("type") == "text")
    return not res.get("isError", False), text, sc


rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "verify", "version": "1"}})
rpc("notifications/initialized", notify=True)
ok, text, sc = tool("get_status", {})
print("status:", text[:200], flush=True)

base = {"prompt": "a cozy cabin in a snowy forest", "negative_prompt": "lowres", "steps": 12, "width": 512, "height": 512,
        "seed": 11, "sampler_name": "DPM++ 2M"}
if ONLY and ONLY.intersection({"pixel_art", "color_enhance", "fsr_upscale", "person_mask", "remove_background",
                               "depthmap2mask", "inpaint_background"}):
    base.update(prompt="a woman wearing a red jacket, standing in a green park, full body", height=768)
ok, text, sc = tool("txt2img", dict(base, return_images=False))
init = sc["images"][0]["path"]
print("txt2img:", ok, init, flush=True)

i2i = dict(base, init_image=init, denoising_strength=0.5)
cases = [
    ("pixel_art", {"pixel_size": "8", "colors": "16", "dither": "0.5"}, i2i),
    ("color_enhance", {"strength": "0.5"}, i2i),
    ("fsr_upscale", {"scale": "2"}, i2i),
    ("person_mask", {}, i2i),
    ("remove_background", {"model": "isnet-anime", "save_mask": "true"}, i2i),
    ("depthmap2mask", {"save_mask": "true"}, i2i),
    ("inpaint_background", {"save_mask": "true"}, dict(i2i, prompt="a woman wearing a red jacket on a snowy mountain")),
    ("xyz_grid", {"x_type": "CFG Scale", "x_values": "4, 8", "y_type": "Steps", "y_values": "8, 12"}, base),
    ("prompt_matrix", {}, dict(base, prompt="a cozy cabin|snow|sunset")),
    ("prompts_from_file", {"prompts": "a red apple --steps 8\na green pear --seed 5"}, base),
    ("loopback", {"loops": "2"}, i2i),
    ("sd_upscale", {"upscaler": "Lanczos", "scale": "2", "overlap": "64"}, i2i),
    ("outpaint", {"pixels": "64"}, dict(i2i, denoising_strength=0.8)),
    ("outpaint_mk2", {"pixels": "64"}, dict(i2i, denoising_strength=0.8)),
    ("img2img_alt", {}, dict(i2i, denoising_strength=1.0)),
    ("upscale_extras", {"upscaler": "Lanczos", "scale": "2"}, i2i),
    ("face_restore_extras", {"model": "GFPGANv1.4"}, i2i),
    ("caption", {"method": "blip"}, i2i),
    ("flipped_copies", {"vertical": "true"}, i2i),
    ("split_oversized", {}, i2i),
    ("autosized_crop", {}, i2i),
    ("focal_crop", {}, dict(i2i, width=384, height=512)),
    ("svd", {"model": "svd", "frames": "8", "steps": "12", "width": "768", "height": "448"}, i2i),
    ("zero123", {"model": "zero123", "azimuth": "90"}, i2i),
    ("ultimate_sd_upscale", {"upscaler": "Lanczos", "scale": "2", "redraw_mode": "Linear", "seams_fix": "Band pass"},
     dict(i2i, denoising_strength=0.3)),
    ("ultimate_sd_upscale", {"upscaler": "Lanczos", "scale": "2", "redraw_mode": "Chess",
                             "seams_fix": "Half tile offset pass + intersections"}, dict(i2i, denoising_strength=0.3)),
]
summary = []
for sid, values, params in cases:
    if ONLY and sid not in ONLY:
        continue
    t0 = time.time()
    ok, text, sc = tool("run_script", {"script_id": sid, "values": values, "params": params, "return_images": False})
    imgs = len(sc.get("images", []))
    summary.append((sid, ok, imgs))
    print(f"{sid:20s} {'OK ' if ok else 'ERR'} {time.time() - t0:6.1f}s images={imgs} {text[:160]}", flush=True)
p.stdin.close()
p.wait(120)
print("SUMMARY", sum(1 for s in summary if s[1] and s[2] > 0), "/", len(summary))
sys.exit(0 if summary and all(s[1] and s[2] > 0 for s in summary) else 1)
