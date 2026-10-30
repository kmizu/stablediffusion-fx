"""Collects the licenses of every project stablediffusion-fx ports code from, re-implements, or depends on.

  python scripts/third_party.py        (needs the GitHub CLI `gh`, logged in)
  python scripts/third_party.py --use-archived-licenses
                                       (no network; requires existing notice labels and archived license files)

Writes licenses/<owner>__<repo>.txt (the upstream license text, verbatim) and THIRD_PARTY_NOTICES.md.
Sources: the hand-written BASE list below + the repository of every extension-index entry marked `done`
(docs/extensions-status.json), so new plugins pick up their notice automatically.
"""
import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (repo, how it is used, what)
BASE = [
    ("w-e-w/stable-diffusion-webui-GPU-temperature-protection", "reimplemented", "Temperature pause/wake hysteresis and controls; behavior reference f388d7081945576ca2361c365d16f5edb3250412; independent native/engine implementation"),
    ("openhardwaremonitor/openhardwaremonitor", "dependency", "Optional external running Windows GPU temperature monitor through WMI; not installed or bundled; schema reference 09c16898a4a84c36644e7c07baebf1efd299691e"),
    ("ROCm/rocm_smi_lib", "dependency", "Optional external rocm-smi temperature query on Linux; not installed or bundled"),
    ("SenshiSentou/sd-webui-state-manager", "reimplemented", "Manual named generation settings, per-mode list/inspect/restore/delete, "
     "including shared model settings and inactive script/extension controls -> `core:state-manager`; independent Scala/JavaFX "
     "implementation, no upstream code/assets copied; no history/autosave/thumbnail or upstream storage-format parity; "
     "[source 2af62ec](https://github.com/SenshiSentou/sd-webui-state-manager/tree/2af62ecd5287438bbc309f7e840d34aa5c8e3cb6)"),
    ("DominikDoom/a1111-sd-webui-tagcomplete", "reimplemented", "Local CSV tag/alias completion with keyboard selection "
     "in native prompt editors -> `core:tag-autocomplete`; independent Scala/JavaFX implementation, no upstream code or tag "
     "datasets copied; no wildcard/extra-network/chant/translation parity; "
     "[source 4170882](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete/tree/4170882f90b47be130a0ff9314f663c230b9153d)"),
    ("AUTOMATIC1111/stable-diffusion-webui", "ported", "infotext format, built-in scripts (X/Y/Z, prompt matrix, loopback, "
     "SD upscale, outpainting, img2img alternative, custom code, postprocessing), styles.csv, soft inpainting"),
    ("Panchovix/stable-diffusion-webui-reForge", "ported", "reference implementation; built-in extensions (APG, RescaleCFG, "
     "advanced model sampling, Mahiro, random resolutions), sampler / schedule tables"),
    ("lllyasviel/stable-diffusion-webui-forge", "ported", "forge built-in extensions (FreeU, Dynamic Thresholding, HyperTile, SAG, "
     "Kohya HRFix, MultiDiffusion, Latent Modifier, Never OOM, IP-Adapter, ControlLLLite, PhotoMaker, Fooocus inpaint, SVD, "
     "Z123, preprocessors)"),
    ("comfyanonymous/ComfyUI", "ported", "sigma schedules, APG / Mahiro / RescaleCFG guidance math, Deep Shrink, FreeU"),
    ("AbyszOne/Abysz-LAB-Ext", "ported", "Temporal Lab DFI, deflicker, fuse and frame-extraction controls and "
     "algorithms informed by the pinned implementation; numerical equivalence is not claimed. Native Inter Blur "
     "is active, video sampling uses frame-index timestamps, and output is staged into a new directory; "
     "[source deff0d4](https://github.com/AbyszOne/Abysz-LAB-Ext/tree/deff0d4b4a208f862299b14b93e088f613373153)"),
    ("crowsonkb/k-diffusion", "ported", "Karras / exponential / polyexponential sigma schedules"),
    ("mcmonkeyprojects/sd-dynamic-thresholding", "ported", "Dynamic Thresholding (CFG-Fix) math"),
    ("tfernd/HyperTile", "ported", "HyperTile attention tiling"),
    ("ljleb/sd-webui-neutral-prompt", "ported", "Neutral Prompt parser, nested/signed prompt composition, denoised-x0 operators and CFG rescale"),
    ("ljleb/prompt-fusion-extension", "ported", "Prompt Fusion parser, AST and interpolation algorithms: "
     "upstream lib_prompt_fusion/{ast_nodes,geometries,interpolation_functions,interpolation_tensor,prompt_parser,t_scaler}.py "
     "at [1d86996](https://github.com/ljleb/prompt-fusion-extension/tree/1d86996ac4c3d32efaacb8a115600aa6fde6dab6/lib_prompt_fusion) "
     "adapted into python/sdfx/fusion/; upstream MIT notice also preserved in python/sdfx/fusion/LICENSE"),
    ("AUTOMATIC1111/stable-diffusion-webui-tokenizer", "reimplemented", "CLIP text/ID token inspection -> `tokenizer`; "
     "clean implementation in python/sdfx/tokenizer_tool.py and native Tokenizer tab from documented behaviour "
     "at [ac6d541](https://github.com/AUTOMATIC1111/stable-diffusion-webui-tokenizer/tree/ac6d541c7032e9f9c69c8ead2ed201302b06a4fe); "
     "no upstream code copied"),
    ("AlUlkesh/sd_save_intermediate_images", "reimplemented", "Sampling snapshots, intermediate gallery and video export "
     "-> `save_intermediate`; standalone python/sdfx/save_intermediate.py from upstream behaviour "
     "at [a1d7cec](https://github.com/AlUlkesh/sd_save_intermediate_images/tree/a1d7cecff893280d709767794d5654e8cbc6746f) "
     "(upstream Unlicense)"),
    ("feynlee/latent-upscale", "reimplemented", "Source-resolution VAE encoding, latent interpolation and full-step "
     "img2img scheduling -> `latent_upscale`; "
     "[source b9f75f4](https://github.com/feynlee/latent-upscale/tree/b9f75f44de15170a38b195692cff28246846da7d)"),
    ("AlUlkesh/sd_grid_add_image_number", "reimplemented", "Filename numbers and XYZ axis labels on grid tiles "
     "-> `grid_numbers`; standalone Java2D renderer; "
     "[source 693a8a7](https://github.com/AlUlkesh/sd_grid_add_image_number/tree/693a8a710e1872d3ba9b63a41f69ee705f4ad54c)"),
    ("a2569875/stable-diffusion-webui-composable-lora", "ported", "Composable LoRA/LyCORIS controllers and adapter "
     "algebra -> `composable_lora`; standalone branch routing and native controls; "
     "[source a03d40e](https://github.com/a2569875/stable-diffusion-webui-composable-lora/tree/a03d40eb3a3bc63728f46f1c00229310734c8fef)"),
    ("vladmandic/sd-extension-system-info", "ported", "Fixed benchmark prompts -> native System Info; remaining "
     "diagnostics, benchmark, local history, metadata and explicit legacy submission independently implemented; "
     "[source 006f08f](https://github.com/vladmandic/sd-extension-system-info/tree/006f08f499bbe69c484f0f1cc332bbf0e75526c2). "
     "Optional preinstalled NVIDIA driver diagnostics are queried when available; no driver utility is bundled."),
    ("chrisgoringe/Styles-Editor", "reimplemented", "Native CSV editing, categories, notes, backups and AES-Crypt v2 "
     "-> `core:styles-editor`; [source 025f17f](https://github.com/chrisgoringe/Styles-Editor/tree/025f17f747975a1fc914f67806e09dcb80e6eb69)"),
    ("harukei-tech/sd-webui-extended-style-saver", "reimplemented", "Independent native UUID presets and six-setting "
     "save/apply -> `core:styles-editor`; no upstream code copied; "
     "[source 7750cc6](https://github.com/harukei-tech/sd-webui-extended-style-saver/tree/7750cc6b2dcba91951402a682819bcb4fcc43bea)"),
    ("marcobellaccini/pyAesCrypt", "reimplemented", "AES-Crypt v2 protocol interoperability reference only; generated "
     "test fixture and bidirectional external checks at13a188f8799306ad9034167bf0bc0489319db274. "
     "No reference code or Python crypto dependency is bundled or imported by the product."),
    ("Haoming02/sd-webui-image-comparison", "reimplemented", "Native paired image wipe/opacity/view transforms, "
     "source buttons and metadata differences -> `core:image-compare`; "
     "[source 4354eda](https://github.com/Haoming02/sd-webui-image-comparison/tree/4354edab87ca6634fe1f0b94d3e0dba65f6221f0)"),
    ("light-and-ray/sd-webui-vae-tools", "reimplemented", "Independent native latent encode/decode and failed VAE "
     "decode recovery -> `vae_tools`; no upstream code copied; "
     "[source 47fe6db](https://github.com/light-and-ray/sd-webui-vae-tools/tree/47fe6db4f3a8f12320c2fa040d0ba542d6325425)"),
    ("Zyin055/Config-Presets", "data", "Original 14 txt2img and four img2img starter presets (including each "
     "mode's `None` entry) -> `core:config-presets`; preset behaviour independently implemented with no upstream "
     "code copied; [source e252e60](https://github.com/Zyin055/Config-Presets/tree/e252e60288b1104a45ea12106d92a8d145364fa1)"),
    ("Gerschel/sd_web_ui_preset_utils", "reimplemented", "Shared preset schema and selective apply behaviour "
     "-> `core:preset-utilities`; independently implemented from observed behaviour; no upstream code, prose, "
     "assets, or default JSON copied; [source e301734](https://github.com/Gerschel/sd_web_ui_preset_utils/tree/e301734567f37ac53c6225aec3a5c9858b357e7e) "
     "has no published license file in its tracked tree"),
    ("CurtisDS/stupid-nsfw-card-blur-a1111", "reimplemented", "Native card preview show/blur/hide behaviour -> `core:nsfw-card-blur`; independently reimplemented in Scala/JavaFX, with native URI loading, blur, keyboard and image codec adaptations; no upstream CSS, JavaScript or icons copied; [source b2cdcebf97d1074278fb84e0a6eec3006dab0fe1](https://github.com/CurtisDS/stupid-nsfw-card-blur-a1111/tree/b2cdcebf97d1074278fb84e0a6eec3006dab0fe1)"),
    ("madebyollin/taesd", "dependency", "Optional separately downloaded TAESD/TAESDXL model weights for VAE Tools. "
     "Weights are not bundled; model repositories madebyollin/taesd and madebyollin/taesdxl on Hugging Face "
     "are revision-pinned by python/sdfx/vae_tools.py and download requires explicit enablement."),
    ("light-and-ray/sd-webui-prevent-interruption", "reimplemented", "Runtime-global interruption protection "
     "and explicit user restart admission -> `core:prevent-interrupt`; independent native/JSON Lines implementation, "
     "no upstream code or assets copied; "
     "[source ea22bc0](https://github.com/light-and-ray/sd-webui-prevent-interruption/tree/ea22bc0de11875f3889f09c8e6ed7b879f321d05)"),
    ("light-and-ray/sd-webui-hardware-info-in-metadata", "reimplemented", "Compact hardware, per-image batch timing, "
     "remark, Extras provenance and informational paste notices -> `core:hardware-info`; independently implemented, "
     "no upstream code copied; "
     "[source 784a0dd](https://github.com/light-and-ray/sd-webui-hardware-info-in-metadata/tree/784a0dd8521108815aff007d7a4807ac6f98d251)"),
    ("huchenlei/sd-webui-api-payload-display", "reimplemented", "API payload display behaviour -> `core:api-payload`; "
     "independently reimplemented in Scala/JavaFX from "
     "[source 2d3332d](https://github.com/huchenlei/sd-webui-api-payload-display/tree/2d3332d94ffac0943bd35a1943924ed4650c6e63); "
     "no upstream JavaScript or CSS copied"),
    ("summerstyle/jsonTreeViewer", "reimplemented", "JSON tree interactions referenced by the vendored viewer in "
     "huchenlei/sd-webui-api-payload-display; behaviour independently reimplemented in Scala/JavaFX; "
     "no vendored JavaScript or CSS copied; "
     "[license source 6abfcc1](https://github.com/summerstyle/jsonTreeViewer/blob/6abfcc1202396af12e2806d129bcae0e75b37365/LICENSE)"),
    ("some9000/StylePile", "data", "Pinned GPL-3.0 StylePile catalog, reference images and inspiration texts; "
     "original bytes and license are bundled under plugins/src/main/resources/stylepile, with SHA-256 provenance. "
     "Processing and native UI are independently implemented; no WebUI script is bundled or executed. "
     "[source 85b549a](https://github.com/some9000/StylePile/tree/85b549a10ff6c5b1604e980cd2107253b088e832)"),
    ("vladmandic/sd-extension-steps-animation", "reimplemented", "Native Steps Animation: denoised-step movies, "
     "duration, interpolation, codec choice and partial-export behavior; independently implemented using this app's recorder. "
     "No upstream code or assets copied; no published license file at inspected revision. "
     "[source acac7f1](https://github.com/vladmandic/sd-extension-steps-animation/tree/acac7f166d5613ebfddac4c0a286ba2e5f7f29d1)"),
    ("bluelovers/sd-webui-pnginfo-injection", "reimplemented", "Optional generation-time Hashes metadata -> pnginfo_injection; "
     "independent native and Python implementation of resource provenance; no upstream code, assets or static wildcard maps copied. "
     "No published license file at inspected revision; "
     "[source 93bf780](https://github.com/bluelovers/sd-webui-pnginfo-injection/tree/93bf780426d3571056c6ef96fb7734943f16e711)"),
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
    ("scikit-image/scikit-image", "dependency", "Optional LAB histogram matching for A1111 reference color correction"),
    ("FHPythonUtils/BlendModes", "dependency", "Optional luminosity blending for A1111 reference color correction"),
    ("light-and-ray/sd-webui-color-correction-extras", "reimplemented", "Reference color correction behavior, original-input fallback and independent method comparisons; no wrapper code copied; source 0d86f5a55d5648ca693584a1078761cf8ae26395"),
    ("Alvi-alvarez/sd-Img2img-batch-interrogator", "ported", "Per-image prompt composition, including source-derived emoticon exceptions, MIT; partial WD v3 batch only; [source ed91cec](https://github.com/Alvi-alvarez/sd-Img2img-batch-interrogator/tree/ed91cec663bcddd7ca5514264a61483c7d80f8d2)"),
    ("onnx/onnx", "dependency", "Optional local WD v3 model structure inspection"),
    ("microsoft/onnxruntime", "dependency", "WD14 tagger and optional local WD v3 batch inference"),
    ("ultralytics/ultralytics", "dependency", "YOLO detectors for ADetailer (optional Python package)"),
    ("danielgatis/rembg", "dependency", "background removal models (optional Python package)"),
    ("FFmpeg/FFmpeg", "dependency", "Optional external ffmpeg executable for Save Intermediate and Steps Animation video export; "
     "no FFmpeg binary is bundled or installed by this app. Upstream LICENSE.md explains build-dependent licensing: "
     "LGPL-2.1-or-later by default; --enable-gpl changes the combined build to GPL-2.0-or-later, "
     "--enable-version3 selects version 3, and linked libraries can change the terms. "
     "The selected executable's build and accompanying notices determine its effective license; "
     "--enable-nonfree builds are not redistributable. The collected upstream file is the licensing overview "
     "([inspected source](https://github.com/FFmpeg/FFmpeg/blob/46f23be8b1976ed4c8ec85584daabcf0e57a7b9e/LICENSE.md))."),
    ("openjdk/jfx", "dependency", "JavaFX UI toolkit"),
    ("com-lihaoyi/upickle", "dependency", "JSON"),
    ("scala/scala3", "dependency", "Scala runtime"),
    ("AUTOMATIC1111/stable-diffusion-webui-extensions", "data", "extension index snapshot (docs/extensions-index.json)"),
]

# Keep these source-pinned original LICENSE bytes; do not normalize newlines.
# (commit, license path, SHA-256, SPDX)
PINNED_LICENSES = {
    "w-e-w/stable-diffusion-webui-GPU-temperature-protection": (
        "f388d7081945576ca2361c365d16f5edb3250412", "LICENSE",
        "248a1ab6a57e6f677770a1498efc7a567f22aaae68923ae84c47d8a3db0ee699", "AGPL-3.0"),
    "SenshiSentou/sd-webui-state-manager": (
        "2af62ecd5287438bbc309f7e840d34aa5c8e3cb6", "LICENSE",
        "3fe9f2a7f539d60ed0e8edaab5f9645c0ee144f9df7f3cf29296153fc643bc4e", "MIT"),
    "DominikDoom/a1111-sd-webui-tagcomplete": (
        "4170882f90b47be130a0ff9314f663c230b9153d", "LICENSE",
        "c1532c29734187eddd3948321379fa95ba79191bc3cb41d2787d7dff6e5913b5", "MIT"),
    "AbyszOne/Abysz-LAB-Ext": (
        "deff0d4b4a208f862299b14b93e088f613373153", "LICENSE",
        "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986", "GPL-3.0"),
    "CurtisDS/stupid-nsfw-card-blur-a1111": (
        "b2cdcebf97d1074278fb84e0a6eec3006dab0fe1", "LICENSE.txt",
        "e6b9326517557ffdbf4a4a2a9f10ab20a037e95dd08645412e64a8baf7549742", "MIT"),
    "Zyin055/Config-Presets": (
        "e252e60288b1104a45ea12106d92a8d145364fa1", "LICENSE",
        "c19b98789c10ffe53568c46f4daae8a49024a02715cf2d4f05de04aa05d3a5f9", "MIT"),
    "huchenlei/sd-webui-api-payload-display": (
        "2d3332d94ffac0943bd35a1943924ed4650c6e63", "LICENSE",
        "990b5c560b655fb8623874940f1ac02cd21c3c76c714c997dff0ec3fa8c744fb", "MIT"),
    "summerstyle/jsonTreeViewer": (
        "6abfcc1202396af12e2806d129bcae0e75b37365", "LICENSE",
        "0e6f8814a1368da259cdca5d218c422fcb278b3579651c55079d774799ae7692", "MIT"),
}

# Audited source has no license file. Do not fetch another branch or invent a license text.
# (commit, notice label)
PINNED_UNLICENSED = {
    "Gerschel/sd_web_ui_preset_utils": (
        "e301734567f37ac53c6225aec3a5c9858b357e7e", "no license file published"),
}


def repo_of(url: str):
    m = re.match(r"https?://github\.com/([^/]+)/([^/#?]+?)(?:\.git)?/?$", url.strip())
    return f"{m.group(1)}/{m.group(2)}" if m else None


def verify_license_bytes(raw: bytes, expected: str):
    """Verify a trusted digest, allowing only uniform LF/CRLF transport variants.

    Do not strip whitespace or final newlines; content changes remain failures.
    """
    lf = raw.replace(b"\r\n", b"\n")
    candidates = (raw, lf, lf.replace(b"\n", b"\r\n"))
    if not any(hashlib.sha256(candidate).hexdigest() == expected for candidate in candidates):
        raise ValueError("license content does not match trusted digest")


def gh_license(repo: str):
    if repo in PINNED_LICENSES:
        pin, path, expected, spdx = PINNED_LICENSES[repo]
        out = subprocess.run(["gh", "api", f"repos/{repo}/contents/{path}?ref={pin}"],
                             capture_output=True, text=True, timeout=60, check=True)
        raw = base64.b64decode(json.loads(out.stdout)["content"])
        try:
            verify_license_bytes(raw, expected)
        except ValueError as exc:
            raise RuntimeError(f"pinned license hash mismatch for {repo}@{pin}") from exc
        return spdx, raw
    error = ""
    for _ in range(3):
        try:
            out = subprocess.run(["gh", "api", f"repos/{repo}/license"], capture_output=True, text=True, timeout=60)
        except FileNotFoundError:
            sys.exit("the GitHub CLI `gh` is required")
        except subprocess.TimeoutExpired:
            error = "GitHub request timed out"
            continue
        if out.returncode == 0:
            j = json.loads(out.stdout)
            text = base64.b64decode(j.get("content", "")).decode("utf-8", "replace")
            return (j.get("license") or {}).get("spdx_id") or "NOASSERTION", text
        if "HTTP 404" in out.stderr:
            return None, None
        error = out.stderr.strip() or f"gh exited {out.returncode}"
    # Do not rewrite the notice table with a false 'no license' claim after a network/auth/API failure.
    raise RuntimeError(f"could not retrieve license for {repo}: {error}")


def archived_license_labels():
    """Reuse the existing table's SPDX labels without refetching unrelated projects."""
    path = os.path.join(ROOT, "THIRD_PARTY_NOTICES.md")
    labels = {}
    with open(path, encoding="utf-8") as notice:
        for line in notice:
            match = re.match(r"\| \[([^]]+)\]\(https://github\.com/[^)]+\) \| [^|]+ \| ([^|]+) \|", line)
            if match:
                labels[match.group(1)] = match.group(2).strip()
    return labels


def main(argv=()):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-archived-licenses", action="store_true",
                        help="regenerate notices from checked-in license files and existing SPDX labels; no network")
    args = parser.parse_args(argv)
    archived = archived_license_labels() if args.use_archived_licenses else {}
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
        fname = repo.replace("/", "__") + ".txt"
        license_path = os.path.join(ROOT, "licenses", fname)
        if repo in PINNED_UNLICENSED:
            _, lic = PINNED_UNLICENSED[repo]
        elif args.use_archived_licenses:
            if repo in PINNED_LICENSES:
                _, _, expected, spdx = PINNED_LICENSES[repo]
                with open(license_path, "rb") as source:
                    try:
                        verify_license_bytes(source.read(), expected)
                    except ValueError as exc:
                        raise RuntimeError(f"pinned archived license hash mismatch for {repo}") from exc
                lic = f"{spdx} ([text](licenses/{fname}))"
            elif repo in archived:
                lic = archived[repo]
                if "([text](licenses/" in lic and not os.path.isfile(license_path):
                    raise RuntimeError(f"archived license missing for {repo}")
            else:
                raise RuntimeError(f"no archived license label for {repo}; run full collector")
        else:
            spdx, source_text = gh_license(repo)
            if source_text:
                if isinstance(source_text, bytes):
                    with open(license_path, "wb") as output:
                        output.write(source_text)
                else:
                    with open(license_path, "w", encoding="utf-8", newline="\n") as output:
                        output.write(source_text)
                lic = f"{spdx} ([text](licenses/{fname}))"
            elif os.path.isfile(license_path):
                lic = f"upstream file unavailable; archived license ([text](licenses/{fname}))"
            else:
                lic = "no license file published"
        rows.append(f"| [{repo}](https://github.com/{repo}) | {how} | {lic} | {what} |")
        print(f"{repo:60s} {lic.split(' ', 1)[0]}")
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
        "- **dependency**: installed separately (pip / Maven / external executable), not redistributed in this repository.",
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
    main(sys.argv[1:])
