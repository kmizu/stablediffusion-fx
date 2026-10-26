"""Collects the licenses of every project stablediffusion-fx ports code from, re-implements, or depends on.

  python scripts/third_party.py        (needs the GitHub CLI `gh`, logged in)

Writes licenses/<owner>__<repo>.txt (the upstream license text, verbatim) and THIRD_PARTY_NOTICES.md.
Sources: the hand-written BASE list below + the repository of every extension-index entry marked `done`
(docs/extensions-status.json), so new plugins pick up their notice automatically.
"""
import base64
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (repo, how it is used, what)
BASE = [
    ("AUTOMATIC1111/stable-diffusion-webui", "ported", "infotext format, built-in scripts (X/Y/Z, prompt matrix, loopback, "
     "SD upscale, outpainting, img2img alternative, custom code, postprocessing), styles.csv, soft inpainting"),
    ("Panchovix/stable-diffusion-webui-reForge", "ported", "reference implementation; built-in extensions (APG, RescaleCFG, "
     "advanced model sampling, Mahiro, random resolutions), sampler / schedule tables"),
    ("lllyasviel/stable-diffusion-webui-forge", "ported", "forge built-in extensions (FreeU, Dynamic Thresholding, HyperTile, SAG, "
     "Kohya HRFix, MultiDiffusion, Latent Modifier, Never OOM, IP-Adapter, ControlLLLite, PhotoMaker, Fooocus inpaint, SVD, "
     "Z123, preprocessors)"),
    ("comfyanonymous/ComfyUI", "ported", "sigma schedules, APG / Mahiro / RescaleCFG guidance math, Deep Shrink, FreeU"),
    ("crowsonkb/k-diffusion", "ported", "Karras / exponential / polyexponential sigma schedules"),
    ("mcmonkeyprojects/sd-dynamic-thresholding", "ported", "Dynamic Thresholding (CFG-Fix) math"),
    ("tfernd/HyperTile", "ported", "HyperTile attention tiling"),
    ("kohya-ss/sd-scripts", "ported", "ControlNet-LLLite module layout"),
    ("lllyasviel/Fooocus", "ported", "Fooocus inpaint patch / head application"),
    ("threestudio-project/threestudio", "reimplemented", "Zero123 diffusers layout and sampling"),
    ("TencentARC/PhotoMaker", "dependency", "PhotoMaker pipeline (optional Python package)"),
    ("huggingface/diffusers", "dependency", "inference pipelines"),
    ("huggingface/transformers", "dependency", "text / image encoders, BLIP, Marigold support"),
    ("huggingface/peft", "dependency", "LoRA adapters"),
    ("huggingface/accelerate", "dependency", "model offloading"),
    ("huggingface/safetensors", "dependency", "weights I/O"),
    ("chaiNNer-org/spandrel", "dependency", "upscaler / face restoration model loading"),
    ("pytorch/pytorch", "dependency", "tensor runtime"),
    ("opencv/opencv-python", "dependency", "image processing (canny, face detection)"),
    ("python-pillow/Pillow", "dependency", "images"),
    ("microsoft/onnxruntime", "dependency", "WD14 tagger"),
    ("ultralytics/ultralytics", "dependency", "YOLO detectors for ADetailer (optional Python package)"),
    ("danielgatis/rembg", "dependency", "background removal models (optional Python package)"),
    ("openjdk/jfx", "dependency", "JavaFX UI toolkit"),
    ("com-lihaoyi/upickle", "dependency", "JSON"),
    ("scala/scala3", "dependency", "Scala runtime"),
    ("AUTOMATIC1111/stable-diffusion-webui-extensions", "data", "extension index snapshot (docs/extensions-index.json)"),
]


def repo_of(url: str):
    m = re.match(r"https?://github\.com/([^/]+)/([^/#?]+?)(?:\.git)?/?$", url.strip())
    return f"{m.group(1)}/{m.group(2)}" if m else None


def gh_license(repo: str):
    try:
        out = subprocess.run(["gh", "api", f"repos/{repo}/license"], capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        sys.exit("the GitHub CLI `gh` is required")
    if out.returncode != 0:
        return None, None
    j = json.loads(out.stdout)
    text = base64.b64decode(j.get("content", "")).decode("utf-8", "replace")
    return (j.get("license") or {}).get("spdx_id") or "NOASSERTION", text


def main():
    status = json.load(open(os.path.join(ROOT, "docs", "extensions-status.json"), encoding="utf-8"))
    entries = list(BASE)
    seen = {r for r, _, _ in BASE}
    by_repo = {}
    for e in status:
        st = e["status"]
        if not st.startswith("done:"):
            continue
        r = repo_of(e["url"])
        if r:
            by_repo.setdefault(r, []).append((e["name"], st[5:]))
    for r, items in sorted(by_repo.items()):
        if r in seen:
            continue
        seen.add(r)
        what = "; ".join(f"{n} -> `{t}`" for n, t in items)
        entries.append((r, "reimplemented", f"extension rewritten as {what}"))
    os.makedirs(os.path.join(ROOT, "licenses"), exist_ok=True)
    rows = []
    for repo, how, what in entries:
        spdx, text = gh_license(repo)
        fname = repo.replace("/", "__") + ".txt"
        if text:
            open(os.path.join(ROOT, "licenses", fname), "w", encoding="utf-8", newline="\n").write(text)
            lic = f"{spdx} ([text](licenses/{fname}))"
        else:
            lic = "no license file published"
        rows.append(f"| [{repo}](https://github.com/{repo}) | {how} | {lic} | {what} |")
        print(f"{repo:60s} {spdx}")
    md = [
        "# Third-party notices",
        "",
        "stablediffusion-fx is licensed under the GNU Affero General Public License v3.0 (see LICENSE), because it ports",
        "code from AGPL-3.0 (AUTOMATIC1111 web UI, Forge, reForge) and GPL-3.0 (ComfyUI, Fooocus) projects.",
        "",
        "How each project is used:",
        "",
        "- **ported**: code or algorithms were translated from the project; its license applies to those parts.",
        "- **reimplemented**: the feature was written from scratch from the project's documented behaviour; no code was",
        "  copied. The license is listed for attribution.",
        "- **dependency**: installed separately (pip / Maven), not redistributed in this repository.",
        "- **data**: data files included in this repository.",
        "",
        "Projects without a published license file were not copied from; their features were re-implemented from the",
        "description of their behaviour only.",
        "",
        "| Project | Use | License | What |",
        "|---|---|---|---|",
    ] + rows
    open(os.path.join(ROOT, "THIRD_PARTY_NOTICES.md"), "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
    print(len(rows), "projects")


if __name__ == "__main__":
    main()
