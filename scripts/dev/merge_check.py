import sys, io, os, time
import numpy as np
from PIL import Image
sys.path.insert(0, r"D:\stable-diffusion-webui-reForge\stablediffusion-fx\python")
from sdfx.engine import Engine
from sdfx.protocol import Emitter, JobContext
out = sys.argv[1]
M = r"D:\stable-diffusion-webui-reForge\models\Stable-diffusion"
A, B = os.path.join(M, "animekawa_v10.safetensors"), os.path.join(M, "aiomonstergirls_v4.safetensors")
e = Engine(); ctx = JobContext(Emitter(io.BytesIO()), "m")
for f in ("merged.safetensors", "merged_blocks.safetensors", "conv.safetensors"):
    if os.path.exists(os.path.join(out, f)): os.remove(os.path.join(out, f))
t = time.time()
r = e.merge_models({"model_a": A, "model_b": B, "alpha": 0.5, "output": os.path.join(out, "merged.safetensors")}, ctx)
print("merge", round(time.time() - t, 1), "s", r["keys"], "keys skipped", r["skipped"], flush=True)
t = time.time()
r2 = e.merge_models({"model_a": A, "model_b": B, "alpha": 0.0, "block_weights": "preset:OUT_ONLY",
                     "output": os.path.join(out, "merged_blocks.safetensors")}, ctx)
print("block merge", round(time.time() - t, 1), "s", flush=True)
r3 = e.convert_model({"model": os.path.join(out, "merged.safetensors"), "output": os.path.join(out, "conv.safetensors"),
                      "precision": "fp16", "ema": "no-ema"}, ctx)
print("convert", r3, flush=True)
base = {"prompt": "1girl, portrait, smile", "negative_prompt": "lowres", "seed": 5, "steps": 16, "width": 512, "height": 512}
imgs = {}
for k, path in (("A", A), ("B", B), ("merged", os.path.join(out, "merged.safetensors")),
                ("blocks", os.path.join(out, "merged_blocks.safetensors")), ("conv", os.path.join(out, "conv.safetensors"))):
    e.load_model({"path": path}, ctx)
    rr = e.txt2img({"params": base, "output_dir": out}, JobContext(Emitter(io.BytesIO()), k))
    dst = os.path.join(out, f"img_{k}.png"); os.replace(rr["images"][0]["path"], dst)
    imgs[k] = np.asarray(Image.open(dst).convert("RGB")).astype(float)
d = lambda a, b: round(float(np.abs(imgs[a] - imgs[b]).mean()), 2)
print({"merged-A": d("merged", "A"), "merged-B": d("merged", "B"), "A-B": d("A", "B"), "conv-merged": d("conv", "merged"),
       "blocks-A": d("blocks", "A")})
c = Image.new("RGB", (256 * 5, 256))
for i, k in enumerate(imgs):
    c.paste(Image.fromarray(imgs[k].astype("uint8")).resize((256, 256)), (256 * i, 0))
c.save(os.path.join(out, "sheet.jpg"), quality=85)
