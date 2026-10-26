import sys, io, os
import numpy as np
from PIL import Image
sys.path.insert(0, r"D:\stable-diffusion-webui-reForge\stablediffusion-fx\python")
from sdfx.engine import Engine
from sdfx.protocol import Emitter, JobContext

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
e = Engine()
e.load_model({"path": r"D:\stable-diffusion-webui-reForge\models\Stable-diffusion\waiIllustriousSDXL_v170.safetensors"},
             JobContext(Emitter(io.BytesIO()), "x"))
lp = r"D:\stable-diffusion-webui-reForge\models\Lora\20_il-shirobana.safetensors"
base = {"prompt": "1girl, smile, cherry blossoms", "negative_prompt": "lowres", "seed": 7, "steps": 12, "cfg_scale": 6.0,
        "width": 832, "height": 1216, "sampler_name": "Euler a"}
cases = {"nolora": None, "w0": {"weight": 0.0}, "plain": {}, "ALL": {"lbw": "ALL"}, "NONE": {"lbw": "NONE"}, "OUTALL": {"lbw": "OUTALL"},
         "INS": {"lbw": "INS"}, "explicit": {"lbw": "1,0,0,1,1,1,1,1,1,1,1,1"}, "stop": {"stop": 0.3},
         "te_only": {"unet": 0.0}, "nolora2": None}
imgs = {}
for k, extra in cases.items():
    p = dict(base, loras=[] if extra is None else [dict({"name": "s", "weight": extra.get("weight", 1.0)}, **{k: v for k, v in extra.items() if k != "weight"})])
    r = e.txt2img({"params": p, "output_dir": out, "lora_paths": {"s": lp}}, JobContext(Emitter(io.BytesIO()), k))
    dst = os.path.join(out, f"lbw_{k}.png")
    os.replace(r["images"][0]["path"], dst)
    imgs[k] = np.asarray(Image.open(dst).convert("RGB")).astype(float)
    print(k, "warnings", r["warnings"], flush=True)
d = lambda a, b: round(float(np.abs(imgs[a] - imgs[b]).mean()), 2)
print({"w0-nolora": d("w0", "nolora"), "NONE-w0": d("NONE", "w0"), "nolora2-nolora": d("nolora2", "nolora"), "ALL-plain": d("ALL", "plain"), "NONE-nolora": d("NONE", "nolora"), "OUTALL-plain": d("OUTALL", "plain"),
       "INS-plain": d("INS", "plain"), "explicit-plain": d("explicit", "plain"), "stop-plain": d("stop", "plain"),
       "stop-nolora": d("stop", "nolora"), "te_only-plain": d("te_only", "plain"), "plain-nolora": d("plain", "nolora")})
ims = [Image.open(os.path.join(out, f"lbw_{k}.png")).convert("RGB").resize((208, 304)) for k in cases]
c = Image.new("RGB", (208 * len(ims), 304))
for i, im in enumerate(ims):
    c.paste(im, (208 * i, 0))
c.save(os.path.join(out, "sheet.jpg"))
