"""Temporary app settings for the jar-based (M) stage: GPU venv, parent checkpoints, downloaded model folders."""
import json
import sys

python_exe, models, out = sys.argv[1], sys.argv[2], sys.argv[3]
json.dump({
    "python_exe": python_exe,
    "model_dirs": {"checkpoints": [r"D:\stable-diffusion-webui-reForge\models\Stable-diffusion"], "vae": ["models/vae"],
                   "lora": ["models/lora"], "controlnet": ["models/controlnet"], "embeddings": ["models/embeddings"],
                   "upscalers": ["models/upscalers"], "face_restore": [models], "svd": [models], "zero123": [models],
                   "photomaker": ["models/photomaker"]},
    "last_model": "animekawa_v10",
}, open(out, "w", encoding="utf-8"))
print("settings written:", out)
