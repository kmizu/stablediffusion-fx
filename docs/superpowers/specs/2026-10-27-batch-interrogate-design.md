# Img2img batch interrogator: source audit and design

Status: proposal for issue #7; extension index 217 remains `todo:batch_interrogate`. No implementation or model availability is claimed.

## Source and license

Pinned [Img2img Batch Interrogator](https://github.com/Alvi-alvarez/sd-Img2img-batch-interrogator/tree/ed91cec663bcddd7ca5514264a61483c7d80f8d2), commit `ed91cec663bcddd7ca5514264a61483c7d80f8d2`; [README](https://raw.githubusercontent.com/Alvi-alvarez/sd-Img2img-batch-interrogator/ed91cec663bcddd7ca5514264a61483c7d80f8d2/README.md) and [script](https://raw.githubusercontent.com/Alvi-alvarez/sd-Img2img-batch-interrogator/ed91cec663bcddd7ca5514264a61483c7d80f8d2/scripts/sd_tag_batch.py). Original [MIT license](https://raw.githubusercontent.com/Alvi-alvarez/sd-Img2img-batch-interrogator/ed91cec663bcddd7ca5514264a61483c7d80f8d2/LICENSE), copyright 2023 Santiago Alvarez, Git blob `4904e6ea172103ff2cafa67342a1643e0a5d3c42`, 1,073 bytes, requires retaining the copyright and permission notice in copies or substantial portions. Archive exact original bytes and add an accurate source notice on acceptance. The CLIP/Deepbooru/WD backends, model files, tokens, preprocessors and optional external extensions have independent provenance, terms and dependencies; upstream MIT does not license those assets.

## Observed behavior and control inventory

The source shows an Img2img-only, disabled-by-default accordion that interrogates each input before its generation, then composes the result into the positive or negative prompt. A generic caption viewer alone is not equivalent. It has ordered multiple interrogator selection, native CLIP and native Deepbooru; optional CLIP EXT and WD EXT appear when their separate extensions are enabled. Native must expose only actually available, verified adapters and clearly mark unavailable choices; never load A1111 extension code or substitute BLIP for native CLIP, nor WD for Deepbooru, as if equivalent.

- Prompt position: prepend default or append. Reverse mode targets negative rather than positive prompt; output optional weighting (0–1, default .5) wraps the interrogation. Empty result must not add stray syntax.
- CLIP EXT offers ordered multiple models (default `ViT-L-14/openai`), `best`, `fast`, `classic`, `negative` mode (default `best`), unload-after-use (default on) and unload-all. WD EXT offers ordered multiple models (default `wd-v1-4-moat-tagger.v2`), tag threshold .35, remove underscores default on with known emoji exceptions, optional ratings threshold .5, unload-after-use default on and unload-all.
- Filter duplicates from positive prompt, negative prompt and custom comma-separated filter. Attention syntax is removed for comparison. Custom filter can optimize, load and save with explicit overwrite confirmation. Ordered comma-separated find/replace has a visualized pairing and explicit saved overwrite; empty replacements delete. Source pairs only to the shorter list: expose this rule or validate explicitly as a documented deviation.
- Optional punctuation suppression with emoji exceptions, exaggeration (allow duplicate results), debug output and prompt output (default on). Source order is deduplicate unless exaggeration, replacements, positive/negative/custom filters, punctuation, weighting, then prepend/append to chosen prompt. WD tag threshold is strictly `>`; rating threshold is `>=`. Preserve these boundaries; verify escaping and emoji cases against the pinned source without pasting unnecessary data or broad source code.
- Source clears prior prompt contamination and converts images to RGB for interrogators; its reverse-mode cleanup uses a questionable positive-prompt expression. Do not carry contamination or silently reproduce this apparent bug: freeze independent original positive and negative per submitted input, construct each result from those originals, and record this deviation. Preserve the input image/alpha and never overwrite it.

## Current native state and boundaries

`python/sdfx/extras.py:caption_image` supports single-image BLIP and WD14; `Engine.caption` accepts a path and returns a string. BLIP and WD model availability, auto-download and cache behavior require review before reuse, and neither establishes native CLIP/Deepbooru parity. `GenerationPanel` exposes parallel batch size/count; that does not establish ordered multiple input images with per-image prompt transformations. Establish an actual multi-image Img2img source workflow and per-input job snapshots. Preserve existing single-image behavior when disabled, and avoid silently reporting unsupported methods as available.

Expose an Img2img collapsed section with explicit enable, ordered models, position/reverse/weight and context-dependent controls. Show per-image selected input, preview of the exact composed prompt, status/failures and final generation metadata; never modify user prompt fields, source images or input lists in place. Snapshot settings, positive/negative, input identities and backend/model configuration at submission. Order interrogations and outputs deterministically; per-image failure is explicit and does not reuse a previous prompt. Stop/cancel semantics must be visible and scoped to the exact job; no FX blocking or stale callbacks after selection/disposal.

Model acquisition is opt-in and offline-safe: pin revisions/digests and terms, restrict safe deserialization, bound downloads and validate actual backend capability. Model and user data paths need identity/alias checks and no arbitrary writes. Bound image dimensions/bytes, decoded pixels, input count, output text and model caches; provide explicit unload with truthful resource reporting. Avoid automatic CLIP/WD downloads via existing caption helper. Integrate native queue ownership/cancellation/protection, checkpoint/LoRA state and generation metadata; preserve prompt scheduling, seeds and unaffected generation.

## Acceptance

Use deterministic fake interrogator outputs to test ordered models and inputs, prompt independence, negative/reverse, prepend/append/empty, duplicates and attention syntax, all filters, regex punctuation/emoji and underscore exceptions, strict tag vs inclusive rating threshold, replacement order/empty/unpaired lists, weighting, save/load confirmation and disabled baseline. Test invalid or unavailable model, missing dependencies, no implicit network, paths/symlinks, input immutability, bounded memory, failure/cancel/protection and no stale UI after close. Exercise per-input actual Img2img generation with recorded prompts, not just a standalone interrogator.

Run relevant engine/protocol/UI tests and full regressions; actual native UI at normal/min sizes including model choices, options, save confirmation and per-image output. Use a small fixed-seed SD1.5 and SDXL generation with a locally available verified interrogator/model; verify actual per-image prompt differs as expected, image input hashes unchanged and disabled baseline preserved. Demonstrate bounded VRAM allocation/free and cancellation/recovery, not exhaustive models. CLIP EXT/WD EXT/Deepbooru and asset/model licensing remain explicit separate gates if unavailable; keep issue open until source-scope inventory is met. Do not claim model parity from a synthetic backend or close entry 217 on a BLIP-only partial implementation.


## Local WD v3 setup

The partial WD v3 backend requires the optional `onnx==1.21.0` package for
model inspection in addition to the existing `onnxruntime==1.23.2`, NumPy and
Pillow dependencies. Install ONNX explicitly in the Python environment used
by the worker: `python -m pip install onnx==1.21.0`. No model is downloaded
automatically or included in the application; select the locally obtained
model and matching tags CSV in Settings and restart the worker. Only the
reviewed pinned candidate files are admitted.
