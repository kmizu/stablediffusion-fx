# Real-model verification log

## Full pass (2026-10-26 06:32:11 – 07:13:39)

One script run, final code, one log, stages in order. Each stage starts its own engine process(es); the run as a whole
covers every plugin and every test suite. Errors in the log: 0.

| Stage | What runs | Result |
|---|---|---|
| E | engine requests, one fresh engine per item: GFPGAN, CodeFormer, face-restore hook, BLIP, WD14, LDSR, Marigold depth / normal, recolor / tile / blur, custom code, ControlNet canny, IP-Adapter, ControlNet-LLLite, Fooocus inpaint, PhotoMaker v1, SVD, Zero123 | 19 / 19 |
| H15 | every hook plugin on SD1.5 (incl. SAG with DPM++, Revision skip-with-warning) | 20 / 20 |
| HXL | hook plugins on SDXL (incl. Revision, SAG skip-with-warning) | 17 / 17 |
| V | Revision with / without on animagine-xl-3.1 and waiIllustriousSDXL_v170, same seed | 10 / 10 images |
| S | every AlwaysOn plugin through the Scala path (`RealAlwaysOnCheck`: Plugin.process -> JobQueue -> engine -> GPU); random_resolutions -> 576x448 | 18 / 18 |
| M | every Script / Extras plugin through the jar (`McpMain` stdio -> JobQueue -> engine -> GPU) | 17 / 17 |
| T | sbt test (core 200, engine 60, plugins 83, mcp 39, ui 62 = 444), smoke, pytest without torch (197 passed, 10 skipped) and with torch (209 passed) | all passed |

## Earlier per-item runs


Date: 2026-10-26, 02:18–02:55 (re-runs of sag, revision, custom_code, photomaker, fooocus_inpaint, zero123 at 02:40–02:55), NVIDIA RTX 3090, final code (after all fixes), python `.venv-smoke`
(torch 2.9 / diffusers 0.32.2), models: `animekawa_v10` (SD1.5), `waiIllustriousSDXL_v170` (SDXL).

How it was run (all on the same day and code, but **not** one process):

- **E** — engine level: `python -m sdfx`, one fresh engine process per item, driven over the JSON Lines protocol.
  Results appended to `results.tsv` (time, item, OK/ERR, duration).
- **H15 / HXL** — hook plugins: one engine process per model, every hook case as one txt2img/img2img call
  (SD1.5 run 17/17 + 3/3, SDXL run 13/13 + 3/3).
- **S** — Scala path for AlwaysOn plugins (02:55, `mcp/Test/runMain sdfx.mcp.RealAlwaysOnCheck`): each plugin's
  `process` builds the hooks, then JobQueue -> Generator -> Python engine -> GPU on SD1.5; 18/18 OK, output size and
  infotext keys printed per plugin (random_resolutions produced 576x448 from its list). Model-file plugins
  (ip_adapter, controllllite, fooocus_inpaint, photomaker) are covered by E.
- **R** — re-runs at 02:49–02:53 with the final code: depth/normal Marigold, ip_adapter (unloaded afterwards),
  controllllite, photomaker, svd (14 frames, 1024x576 GIF) — all OK.
- **M** — whole app: `java -cp target/stablediffusion-fx.jar sdfx.mcp.McpMain` (stdio MCP) → Scala JobQueue and
  plugins → Python engine → GPU, one `run_script` call per plugin (17/17), outputs checked visually.

Every row below passed. "by design" notes are the documented fallbacks, also observed in the runs.

| Plugin id | Run | Evidence |
|---|---|---|
| xyz_grid | M | 4-cell CFG x Steps grid with labels |
| prompt_matrix | M | 5 images (4 combinations + grid) |
| prompts_from_file | M | 2 images, per-line `--steps` / `--seed` applied |
| loopback | M | 3 images (2 loops + grid) |
| sd_upscale | M | 1024x1024 merged result |
| outpaint | M | expanded canvas inpainted |
| outpaint_mk2 | M | expanded canvas inpainted |
| img2img_alt | M, H15, HXL | DDIM inversion then sampling |
| svd | E 116.3 s, M 44.4 s | 14-frame and 8-frame GIF + first frame PNG |
| zero123 | E 27.4 s, M 12.2 s | Zero123-XL; azimuth 0 = input, 90/180 rotate the object |
| custom_code | E 0.0 s | `run_code` with SDFX_ALLOW_CODE=1 returned `result` and an image; refused over MCP by design |
| upscale_extras | M | 2x Lanczos |
| face_restore_extras | M | GFPGAN v1.4 |
| caption | E (BLIP 5.4 s, WD14 2.1 s), M | "a girl with long white hair and black eyes" / Danbooru tags; .txt written |
| flipped_copies | M | 2 copies |
| split_oversized | M | 1 piece (square input) |
| autosized_crop | M | 64-divisible crop |
| focal_crop | M | 384x512 crop |
| face_restore | E (GFPGAN 2.7 s, CodeFormer 2.8 s, hook 11.4 s) | faces restored |
| reference | H15, HXL | reference-only attention |
| revision | HXL 13.6 s (re-run 02:44 after aligning with forge: positive pooled replaced, negative pooled zeroed) + probe | animagine-xl-3.1: "ignore prompt" reproduces the red sneaker reference, with "a shoe" the shoe turns into the referenced red/white sneaker; Illustrious-based models barely use the pooled vector (weak effect); SD1.5 warns and skips by design |
| ip_adapter | E 31.1 s | ip-adapter_sd15; unloaded on the next call |
| controllllite | E 27.6 s | SDXL canny anime LLLite follows the edge map |
| photomaker | E 25.0 s | PhotoMaker v1 (SDXL) |
| fooocus_inpaint | E 36.6 s | inpaint_v26 patch + head, masked area repainted |
| freeu | H15, HXL | |
| dynamic_thresholding | H15, HXL | CFG 15 with mimic 7 |
| apg | H15, HXL | |
| mahiro | H15, HXL | |
| rescale_cfg | H15, HXL | |
| pag | H15, HXL | |
| sag | H15 (`sag_dpm`) | SD1.5 + DPM++ 2M; Euler a and SDXL warn and skip by design |
| hypertile | H15, HXL | 1024x1024 |
| kohya_hrfix | H15, HXL | 1024² (SD1.5), 1536² (SDXL) |
| multidiffusion | H15 (both methods), HXL | 1024x512 / 1664x832 |
| model_sampling | H15, HXL | eps override |
| latent_modifier | H15, HXL | |
| random_resolutions | S | 576x448 picked from "640x384, 384x640, 576x448"; infotext "Random resolution" |
| soft_inpainting | H15, HXL | differential-diffusion blend + composite |
| neveroom | H15, HXL | sliced attention / tiled VAE |
| Core: ControlNet | E 15.3 s | control_v11p_sd15_canny fp16 |
| Core: LDSR | E 109.0 s | 2x upscale (first run downloads the model) |
| Core: Marigold | E (depth 11.9 s, normal 10.1 s) | |
| Core: recolor / tile / blur preprocessors | E | |

## Image-tool handoff verification (Codex, 2026-10-26)

Tested with RTX 3090, `python/.venv-smoke`, the existing SD1.5 `animekawa_v10` and SDXL
`waiIllustriousSDXL_v170` checkpoints. The source prompt was a woman in a red jacket in a park, seed 7,
16 steps, DPM++ 2M. Source sizes were 512x768 and 832x1216 respectively. This verifies these tools;
it does not establish completion or full upstream feature parity for the remaining extension index.

| Tool / plugin | Real-engine evidence (both models) | App/MCP evidence (SD1.5) |
|---|---|---|
| pixel_art | 16-colour pixelated output, ordered dithering, original dimensions | 1 saved image |
| color_enhance | stronger chroma at strength 0.5; visually compared with source | 1 saved image |
| fsr_upscale | 1024x1536 / 1664x2432 Lanczos + CAS output | 1 saved image |
| person_mask | inverse mask covers 78.89% / 87.73%; person protected in background inpaint | 1 saved image |
| remove_background | person segmentation and rembg isnet-anime; nontrivial RGBA alpha | 2 saved images (cutout + mask) |
| depthmap2mask | DPT nearest-range mask covers 38.49% / 43.87%; masked inpainting succeeds | 2 saved images (edited + mask) |
| inpaint_background | person-mask inpainting changes park to mountains and retains the person | 2 saved images (edited + mask) |

The non-neural tools run on CPU; diffusion, YOLO and DPT used CUDA. rembg uses the providers available in its
ONNX Runtime installation; no claim of CUDA acceleration for rembg. Only the isnet-anime and person-segmentation
background-removal choices were exercised, not every offered rembg model or alpha-matting option.

Reproduce from the repository root (one model per engine process):

```powershell
python scripts/dev/gpu_check.py python/.venv-smoke/Scripts/python.exe outputs/verify-image-tools image_sd15
python scripts/dev/gpu_check.py python/.venv-smoke/Scripts/python.exe outputs/verify-image-tools image_sdxl
```

Each case generates a source, runs seven tool configurations and two inpainting calls, checks sizes and rejects
empty/full masks or alpha, then writes `outputs/verify-image-tools/image_sd15.json` or `image_sdxl.json` and
`results.tsv`. The PNGs and review sheets in that directory were inspected visually. They are local, gitignored
artifacts; rerun the commands to reproduce them in another checkout. Inspect RGBA on a contrasting background:
discarding its alpha when making a thumbnail shows the original RGB background.

The app path used the rebuilt `target/stablediffusion-fx.jar`, Java 21, and temporary settings pointing at the GPU
venv and existing checkpoints, then:

```powershell
python scripts/dev/mcp_scripts.py "C:\Program Files\Java\jdk-21\bin\java.exe" pixel_art,color_enhance,fsr_upscale,person_mask,remove_background,depthmap2mask,inpaint_background
```

Result: 7/7 succeeded. Temporary `config/settings.json` was removed after the run. The script now exits nonzero
on a failed/empty selection instead of relying on its printed summary.

Regression fixes demonstrated with failing tests before the fixes, then passing tests:

- A uint32 generation seed above INT_MAX failed at OpenCV `setRNGSeed`; normalize to a signed 32-bit seed.
- Pixel art's "original size" option cropped 65x49 to 64x48; restore the exact source dimensions.
- `BaseEngine.image_tool` opened an RGB source twice and left the first handle open; decode inside a context manager.
- Background inpainting discarded its auxiliary person-mask path without deleting it; release auxiliary outputs.

Independent review found no critical/important issue. A minor remaining test gap is an explicit EXIF-orientation
fixture for the image-tool loader; the current closure fixture is an ordinary RGB PNG.

Final automated checks for this wave: Scala **474 passed**; GPU Python venv **257 passed**; torch-free venv
**221 passed, 15 skipped**. `ui/assembly` produced the runnable jar. `scripts/extensions_matrix.py` still reports
**83 done / 219 todo / 51 not applicable**; no new target was promoted by this verification-only wave.

## Seamless tile inpainting (Codex, 2026-10-26)

`seamless_tile` is a Scala img2img Script reimplemented from the documented behavior of
`brick2face/seamless-tile-inpainting`: circular half-image offset, central seam mask, transient inpainting, and
optional inverse offset. Controls cover Horizontal, Vertical, Both, mask width, blur, and restoring the layout.
There is no new Python hook or dependency. It does not enable circular padding or claim exact edge equality.

Actual Java 21 jar/MCP runs used the RTX 3090 and both `animekawa_v10` (SD1.5) and `waiIllustriousSDXL_v170` (SDXL).
Each model completed four cases: Both/Horizontal/Vertical restored, plus Both without restoring. Inputs were
512x512, seed 11, 20 steps, DPM++ 2M, mask width 96, blur 4, denoising strength 0.85. Stone-pavement and SD1.5 grass
textures were inspected as repeated 2x2 tiles. These are ordinary checkpoints, not dedicated inpainting models.

| Model / texture | Opposite-edge mean absolute RGB difference (0..255) |
|---|---|
| SDXL stone | input x=43.96 / y=31.46; Horizontal x=4.26; Vertical y=7.30; Both x=7.27 / y=9.88 |
| SD1.5 grass | input x=8.36 / y=5.55; Horizontal x=6.28; Vertical y=4.38; Both x=5.44 / y=6.82 |
| SD1.5 stone | input x=6.57 / y=4.39; Horizontal x=44.20; Vertical y=64.99; Both x=10.07 / y=64.07 |

The last row is a failed improvement, retained as a limitation rather than omitted: this anime checkpoint can
introduce new high-frequency edges in stone textures. Seam quality is model/prompt dependent; the upstream workflow
recommends an inpainting model. Both-direction joins can retain minor seams. Lower boundary difference alone does
not prove a good texture; all repeated previews were inspected as well. No dedicated inpainting checkpoint was
downloaded or validated in this wave.

Local, gitignored evidence: `outputs/verify-seamless/{sd15,sdxl,sd15-grass}.json`, corresponding `.jpg` comparison
sheets, `*-metrics.json` measurements, and individual full-resolution 2x2 PNGs. Reproduction:

```powershell
# Configure the real GPU venv/checkpoints with scripts/dev/write_settings.py, as for the previous app check.
python scripts/dev/mcp_scripts.py "C:\Program Files\Java\jdk-21\bin\java.exe" seamless_tile
# Copy outputs/mcp-script-results.json before starting the next run.
# Set SDFX_CHECKPOINT=waiIllustriousSDXL_v170 for SDXL; SDFX_TEST_PROMPT overrides the texture prompt.
python/.venv-smoke/Scripts/python.exe scripts/dev/tile_review.py outputs/verify-seamless/sd15.json
```

The JavaFX form was captured in a separate dry-run home (`outputs/verify-seamless/ui-home`) and inspected in
`ui.png` / `ui-panel.png`; all four controls and the description fit. Existing user UI state was not changed.

Six Scala tests cover directional offsets, odd-size inverse, central band bounds, registration, batching and actual
seeds, request flags, original arrangement, no-init rejection, cancellation and failure cleanup, and preserving
engine infotext. Missing implementation and a wrong batch seed were observed before fixes. Review then found that
rebuilding infotext discarded model/VAE/LoRA facts and actual variation seeds; a failing regression confirmed it,
and final saving now preserves the engine's existing infotext. The review's cleanup test gaps were also covered.
The final rebuilt jar completed another 4/4 SD1.5 app run; all four saved PNGs retained `Model hash` and the seamless
settings in their `parameters` text (`outputs/verify-seamless/sd15-final.json`).

Final checks: **480 Scala tests passed**, Python GPU venv **261 passed**, torch-free venv **225 passed / 15 skipped**.
`ui/assembly` succeeded after the metadata fix. Coverage is now **84 done / 218 todo / 51 not applicable**;
the 218 entries correspond to 153 distinct unfinished targets.

The required licence refresh exposed an existing collector defect: transient GitHub failures could replace known
MIT/AGPL notices with "no license file published". `scripts/third_party.py` now retries failed requests, distinguishes
actual HTTP 404, aborts a persistent retrieval error before rewriting the table, and keeps archived text linked
when upstream removes a licence file. Four tests passed after first demonstrating the incorrect behavior.
Independent collector review found no important issue; explicit timeout-retry and whole-file unchanged-on-error
fixtures remain minor test gaps. The new tile plugin's upstream AGPL-3.0 text is bundled under `licenses/`.
The final live refresh succeeded for 109 projects; the existing SuperMerger and close-confirmation licences were
retained, and two previously mislabelled MIT notices were restored. No fetched licence text was removed.

## Weight Gradient (Codex, 2026-10-26)

`weight_gradient` is an opt-in AlwaysOn plugin. Pure Python syntax rendering feeds the existing embedding scheduler;
both positive and negative prompts support `{words : first-last step : first-last[-return] weight : curve}`.
The interval and curve are optional (empty interval means the whole pass). Modes `a/e/ei/eo/c/ci/co` reproduce the
upstream README's linear, exponential and circular numeric examples to four decimal places. Fractions floor to
sampling steps; explicit percentages are also accepted. Escaped/malformed tags remain literal; balanced emphasis
and A1111 editing/alternation inside the words compose with the gradient.

Only the README was used: `DingoBite/weight_gradient`, file `readme.md`, blob
`addaac1a522d33fcf896b29c04e84fe129d68cef`. The repository's licence endpoint returned 404. No upstream source code
was read/copied; there is no new dependency.

Gradient clocks restart in each pass and use actual img2img sampling steps. Plain img2img Alternative uses the full
sampling count; masked/inpainting-model requests that fall back to normal img2img use its shorter count. Hires
re-encodes inherited prompts if its sampling count differs. A1111 editing retains its separate hires-offset rules.
Second-order samplers keep the same conditioning for both model evaluations of a logical inference step.
PhotoMaker's separate text-only encoding path rejects gradient tags explicitly instead of silently ignoring them.

Real-model checks used RTX 3090, SD1.5 `animekawa_v10` at 512x768 and SDXL `waiIllustriousSDXL_v170` at 832x832,
seed 11, 16 steps. Fifteen images per model include normal weights, constant tags, fixed endpoints, exponential and
return ramps, negative prompt ramp, hires (7 steps at 1.25x), img2img (strength 0.5), Alternative, Heun references,
and a disabled-hook baseline after switching samplers. Reports/PNGs are local under
`outputs/verify-weight-gradient/final/`, named `weight_sd15*` and `weight_sdxl*`.

| Pixel MAE comparison | SD1.5 | SDXL |
|---|---:|---:|
| Constant tag vs equivalent ordinary weight, DPM++ 2M | 0 | 0 |
| Constant tag vs equivalent ordinary weight, Heun | 0 | 0 |
| Disabled afterward vs original DPM++ 2M baseline | 0 | 0 |
| Exponential ramp vs fixed starting weight | 2.80 | 5.02 |
| Exponential ramp vs fixed ending weight | 27.82 | 61.35 |
| Heun ramp vs fixed starting weight | 3.61 | 6.60 |

All responses completed without engine warnings; comparison sheets were inspected. Early weights can dominate
composition (the SDXL low initial scarf weight produced a park without the person). A changing weight does not
guarantee a particular intermediate subject or preserve identity; the constant controls distinguish normal model
behavior from scheduling. The short six-step inversion Alternative diagnostic showed visibly stronger contrast
on SDXL; its default-inversion comparison is recorded separately rather than treated as an aesthetic success.
The focused default-inversion run (`outputs/verify-weight-gradient/alternative/weight_alt_sdxl.json`) reproduced
the same oversaturation with ordinary fixed weights and the new hook disabled. A side-by-side image was inspected;
this was an existing Alternative-path quality issue, not a claimed improvement by Weight Gradient. The following
repair section records the subsequent amplitude fix and its remaining limits.

The real Scala AppServices path (Plugin.process -> JobQueue -> Generator -> engine -> GPU) completed 1/1 on each
model with the hook active and `Weight Gradient: True` in output infotext, recorded in
`outputs/verify-weight-gradient/app-check.txt`. This was a direct app-service check, not an MCP protocol check.
The rebuilt jar's form/help/compatibility note were inspected via an isolated dry-run home; final UI run exited 0.

Reproduce engine checks:

```powershell
python scripts/dev/gpu_check.py python/.venv-smoke/Scripts/python.exe outputs/verify-weight-gradient/final weight_sd15
python scripts/dev/gpu_check.py python/.venv-smoke/Scripts/python.exe outputs/verify-weight-gradient/final weight_sdxl
```

Independent review found two P2 timing bugs (Alternative count and second-order callbacks). Five regression failures
were observed, then fixed; tests cover real installed Euler, Heun, DPM2, DPM2 ancestral and DPM++ SDE timestep arrays.
Additional tests cover positive/negative schedules, pooled conditioning and CFG/PAG batch layout, inherited hires
steps, feature advertisements and PhotoMaker rejection. Full suites: **481 Scala passed**, **300 Python GPU-venv
passed**, **247 Python torch-free passed / 16 skipped**. `ui/assembly` succeeded. These tests complement, rather than
replace, the real-model image checks.
The refreshed extension ledger reports **85 done / 217 todo (152 targets) / 51 not applicable**. After promotion,
BuiltinsSuite's 18 tests passed, including the check that every done entry names a registered plugin. Live notice
collection completed for 110 projects and added the new reimplementation attribution without losing existing text.

## img2img Alternative amplitude and replay repair (Codex, 2026-10-26)

The quality follow-up used controlled sampler/CFG comparisons and direct latent probes. Reducing CFG alone did
not resolve the observed excessive contrast. DDIMInverseScheduler's generic pixel clipping default was another
real configuration bug (a valid latent 2.5 was reduced to about 0.962 instead of 2.405 after two zero-noise steps),
but it was **not the cause in this loaded model**, whose scheduler already supplied `clip_sample=False`.

The recovered SDXL inverse state had std 0.85673; `_img2imgalt` forced it to 1.0 after every blend. This amplified
the recovered signal and also changed pure seeded random noise when Randomness=1. The repair preserves inverse
amplitude and blends with `((1-r)*inverse + r*random) / sqrt((1-r)^2+r^2)`. It preserves both endpoints instead of
estimating a new global normalization from the batch. No recolouring, CFG cap or sampler-default change was added.

| Same SDXL portrait | Lightness std (LAB L) | Near-white RGB samples | Near-black RGB samples |
|---|---:|---:|---:|
| Source | 24.25 | 1.11% | 1.61% |
| Before: DPM++ 2M, CFG6 / Decode CFG1 | 34.91 | 14.19% | 28.67% |
| After: identical settings | 30.34 | 4.51% | 14.16% |
| After: matched DDIM, CFG1 / Decode CFG1 | 24.64 | 1.06% | 1.71% |

Metrics count RGB samples >=0.98 or <=0.02, not entire clipped pixels. Images and all grid cases were inspected;
the matched reconstruction is close in colour/contrast, not pixel-identical. High CFG, different samplers and
coarse inversion can still change contrast/composition. This remains a DDIM approximation; it does not implement
the parent's legacy Euler/sigma-adjustment inversion option.

Separate fixes make Decode CFG0 and 0.5 meaningful (unconditional/interpolated rather than treated as 1), keep SDXL
pooled conditioning/time-id batch layout correct, and explicitly disable pixel-range clipping for all inverse
configurations. The app now saves `Decode prompt`, `Decode negative prompt` and the effective `Decode steps`,
including zero-means-sampling-steps fallback. Help text explains approximate inversion and randomness endpoints.

Evidence and reproduction (all artifacts local/gitignored under `outputs/`):

```powershell
python scripts/dev/gpu_check.py python/.venv-smoke/Scripts/python.exe outputs/verify-alternative-final alt_quality_sdxl
python scripts/dev/gpu_check.py python/.venv-smoke/Scripts/python.exe outputs/verify-alternative-final alt_quality_sd15
```

Each model completed a source plus 12 comparisons: DPM++/Euler/DDIM, Decode CFG0/.5/1/2/6 and Randomness=1.
Both Randomness=1 outputs had **zero pixel MAE** from ordinary txt2img with the same seed and matching parameters,
with variation strength zero. JSON reports are `verify-alternative-final/alt_quality_{sd15,sdxl}.json`; comparison
images include `comparison.jpg`, `sd15-grid.jpg` and `sdxl-grid.jpg`. Original failing grids are retained under
`verify-alternative-diagnosis`; runtime-only normalization/endpoint probes are under `verify-alternative-latents`.
`scripts/dev/alt_latent_probe.py` explicitly forces normalized/raw variants to keep that comparison reproducible
after the production fix.

The rebuilt Java 21 jar -> MCP -> Scala Script -> Python engine path completed **1/1 per model** with the default
50 decode steps. `verify-alternative-final/app-{sd15,sdxl}.json` records paths; both PNGs were inspected and contain
the new replay keys/effective decode count. Temporary app settings were restored/removed. The updated form was
captured in a separate dry-run home (`ui.png`/`ui-panel.png`) and inspected; the process exited 0.

Analytic tests first showed six clipping/CFG failures and three amplitude/endpoint failures, then **14/14 passed**.
The Scala replay test failed with missing `Decode prompt` before its fix, then BuiltinsSuite **18/18 passed**.
Full Python: **314 passed** in the GPU venv, **247 passed / 17 skipped** without torch. `ui/assembly` succeeded.
Independent review found no important production defect; its stale docstring finding was corrected. No new
dependency, source licence or extension target was added; the ledger remains **85 done / 217 todo / 51 na**.

## NegPiP (Codex, 2026-10-26)

`python/sdfx/negpip.py` and `NegPipPlugin.scala` implement the CLIP path of hako-mikan/sd-webui-negpip (AGPL-3.0;
inspected source blob `33c9b138272bfb855612bdcec5488a72f03d384b`). Negative-weight segments are removed from the
base prompt and independently encoded with their absolute weight. Content/EOS context is appended; attention keys
remain positive, while the value projection output (including bias) is negated. SDXL pooled vectors come from the
cleaned base prompt. Added padding is excluded from softmax. A private sign channel travels with CFG/PAG and
scheduled tensors and is removed before native attention. Native processors/parameter paths stay intact; forward
wrappers and temporary hooks are restored in finally, including inversion in img2img Alternative.

Reproduce each real-model run separately:

```text
python/.venv-smoke/Scripts/python.exe scripts/dev/gpu_check.py python/.venv-smoke/Scripts/python.exe outputs/verify-negpip-sd15 negpip_sd15
python/.venv-smoke/Scripts/python.exe scripts/dev/gpu_check.py python/.venv-smoke/Scripts/python.exe outputs/verify-negpip-sdxl negpip_sdxl
```

Both completed 15 variants (16 images including batch size 2): ordinary baseline, enabled without negative weights,
ordinary signed weighting, NegPiP -1/-2, negative branch, editing onset, Weight Gradient crossing zero, batch, PAG,
hires, CFG1, img2img, Alternative and restored baseline. All had no warnings. JSON reports and inspected grids
are in those output folders. Enabled without negative weights and restored baseline both had **zero pixel MAE**
against the ordinary baseline on both models. NegPiP differed from ordinary signed weighting and from its other
strength. These are behavioral and isolation checks, not proof of a reliable concept classifier or exact upstream
pixel parity. The SD1.5 -1 case visibly lightened black hair; -2 broke the subject. SDXL -1 changed style and -2
removed the subject. This model-dependent strength limit is described in the form; no clamping/recolouring was added.

`RealAlwaysOnCheck ... negpip` completed **1/1 on each model** through AppServices -> plugin -> queue -> generator ->
real Python engine. It asserted hook and replay infotext; saved PNGs were inspected. This is the app-service path,
not an MCP transport check. Logs: `outputs/negpip-app-{sd15,sdxl}.log`. The isolated dry-run UI exited 0 and its
expanded form was inspected (`outputs/verify-negpip-ui/ui-panel.png`).

Tests cover extraction/escaping/BREAK, branch isolation/token cropping/pooled vectors, schedule padding/crossings,
post-projection bias, unequal CFG signs, fused projections, existing masks, no-op identity, modifier metadata and
cleanup after processor exceptions. Missing-module RED preceded implementation; final full Python **323 passed**,
torch-free **250 passed / 23 skipped**. Full Scala **482 passed**, including opt-in/replay and registry checks;
`ui/assembly` succeeded. Independent review found
no important defect; an extra CPU experiment used actual IPAdapterAttnProcessor2_0 with tuple context and verified
finite results plus exact baseline restoration. SD2, actual ControlNet/IP-Adapter weights combined with offload,
and newer upstream LLM architectures were not GPU-verified. PhotoMaker's text-only path rejects negative terms.

No new dependency. The ledger is **86 done / 216 todo (151 targets) / 51 na**; the overall goal remains unfinished.

## Neutral Prompt foundation only (Codex, 2026-10-26)

`python/sdfx/neutral_prompt.py` contains the MIT-adapted parser/expression tree and per-image denoised-x0
composition for AND / AND_PERP / AND_SALT / AND_TOPK, nested groups, signed weights and mean/std CFG rescale.
**It is not connected to generation.** No plugin, feature advertisement, runtime compatibility, UI or real-model
success is claimed. The target remains todo; the plan lists the remaining scheduler/conditioning and image gates.

Initial missing-module RED preceded implementation. Eighteen targeted tests pass: ordinary editing and escaped
keywords, nested/group weights, non-finite rejection, nesting bound, independent batch projections, saliency
softmax selection, upstream inclusive kthvalue threshold, once-only weights, rescale and degenerate tensors.
Extreme nonzero normal scales and tiny nonzero std each reproduced a failure before their numerical repairs.
Max-scaled projection avoids norm underflow/overflow; standardizing before rescaling avoids an overflowing std ratio.

Direct CPU comparison of six expression trees and 20 random prediction sets each (seed 231, C=4/H=W=8) matched the
upstream visitor composition within 2e-5 relative/absolute tolerance (maximum absolute error 7.63e-6), recorded in
`outputs/neutral-reference-parity.log`. Independent review also compared 169 parser cases and 100 random trees;
its only finding was the now-fixed tiny-std clamp. Source blob IDs are in the implementation plan. No exact pixel
equivalence is implied by these finite synthetic checks.

After the final numerical edit, full Python **341 passed**, torch-free **260 passed / 31 skipped**. Scala/runtime
files were unchanged in this foundation wave, so the previous 482 Scala tests and jar remain the last such checks;
they do not validate a Neutral Prompt runtime. Notice collection completed for 112 projects, adding the MIT source
and original licence text while preserving all prior notices. Ledger unchanged at **86 done / 216 todo / 51 na**.

## Neutral Prompt runtime stage 1 (Codex, 2026-10-26; still todo)

`neutral_runtime.py` and Engine leaf encoding/adjustment/install now connect the expression tree to real UNet
predictions through raw `hooks.neutral_prompt`. Each leaf is evaluated on the same input/timestep with its own
text/pooled conditioning and schedule. The negative prompt remains unconditional conditioning. Composition uses
denoised-x0 differences; at rescale=0 it avoids subtracting/reconstructing a large common x0 baseline, which loses
float32 precision near the end of sampling. Ordinary single prompts at rescale=0 bypass the runtime entirely.

Per-stage CPU tests compare native x0 conversion for DDIM, Euler/ancestral/LMS, Heun, DPM2/ancestral, DPM++ SDE,
DPM++ multi/single and UniPC, with epsilon/v/sample predictions where supported. LCM boundary output is checked
separately. Zero-SNR v-prediction and infinite-sigma limits are finite; undefined zero-SNR epsilon fails clearly.
Inverse DDIM uses the input's earlier noise level, reproduced failing against native before its correction.
This does not implement Alternative: its initial zero-noise composition rule remains unresolved and guarded.
Detail Daemon's per-row timestep change was another review finding; it is explicitly guarded pending integration.

```text
python/.venv-smoke/Scripts/python.exe scripts/dev/gpu_check.py python/.venv-smoke/Scripts/python.exe outputs/verify-neutral-runtime-sd15 neutral_sd15
python/.venv-smoke/Scripts/python.exe scripts/dev/gpu_check.py python/.venv-smoke/Scripts/python.exe outputs/verify-neutral-runtime-sdxl neutral_sdxl
```

Each model completed 18 variants / 19 images: baseline/no-op/restored, AND/PERP/SALT/TOPK, nested negative group,
CFG rescale .7, editing, batch 2, CFG1, hires/img2img, Euler, Heun editing, NegPiP and Weight Gradient leaves. All
had no warnings. JSON reports and inspected grids are in the corresponding folders. No-op and restored baseline
both have **zero pixel MAE** on each model. SD1.5/SDXL SALT/TOPK cases visibly added the red scarf; effects/style
depend on model, operator and weight. No upstream pixel parity or reliable semantic isolation is claimed.

Runtime tests cover branch encoding/schedules, ordinary no-op, trailing leaf weights, CFG0/.5/1, XL time IDs/pooled
batch duplication, cleanup after exceptions, logical-stage callbacks, pending-combination rejection and large-shared-
baseline precision. Independent review additionally checked 32 CFG/batch/rescale combinations against the reference
composition and found the two issues described above. All 41 runtime tests pass. Final full Python **382 passed**
(four known torchsde Brownian endpoint-rounding warnings); torch-free **260 passed / 32 skipped**. No Scala source
or jar changed in this stage. Source/dependencies were already attributed; no additional licence refresh was needed.

**Unfinished:** Scala plugin/form/formatter/replay and actual app-service/UI checks; ControlNet/PAG, PhotoMaker,
IP-Adapter/Reference/ControlLLLite, CADS/SAG, CFG Schedule/APG/Mahiro/Latent Modifier/Dynamic Thresholding/Detail Daemon,
and Alternative combinations. Active composed prompts reject these explicitly in `neutral_runtime.validate`
rather than silently using first-leaf residuals/conditioning. Those restrictions are temporary unfinished work,
not a narrower definition of success. Ledger remains **86 done / 216 todo / 51 na**, target `neutral_prompt` todo.

## Neutral Prompt UI and Alternative (Codex, 2026-10-26; remaining combinations still todo)

The registered Scala plugin now exposes enable, CFG rescale, syntax/nesting/cost help and an optional formatter
with all four keywords and signed weights. Saved PNGs contain the effective combined prompt plus `Neutral Prompt`
and `CFG Rescale phi`. Reprocessing saved settings does not append an existing formatter suffix twice, including
when the original main prompt is empty. Missing-class RED preceded the plugin; the empty-prompt replay regression
failed before its fix. The rebuilt expanded form was visually inspected; screenshot process exited 0:
`outputs/verify-neutral-ui/ui-panel.png` and `form-crop.png`.

Actual `RealAlwaysOnCheck ... neutral_prompt` passed **1/1 per SD1.5/SDXL** through plugin -> AppServices -> queue ->
generator -> Python/GPU. Assertions check formatted prompt, hook and both replay keys; saved images inspected
(`outputs/verify-neutral-app-{sd15,sdxl}`). This is app-service verification, not an MCP transport test.

Alternative now encodes each composed decode leaf at the inversion step count, with a separate batch-1 schedule.
Generation retains its own schedule/batch. Wrapper cleanup covers exceptions. Ordinary decode overrides keep the
legacy encoding route when Neutral performs no composition: review found and fixed an accidental embedding-
adjustment change on that no-op path, with a weights-free Revision-like regression.

DDIM inversion can start at exactly zero noise, where direct division by the x0 prediction coefficient is undefined.
The chosen startup rule extends the positive-noise composition: PERP/TOPK preserve their homogeneous noise deltas;
SALT uses centered absolute activations and higher centered moments when the first term ties. Equal multisets or
float64-indistinguishable moments retain original ordering. Rescale's limit projects the delta onto constant and
centered-image components; constant images use noise-space rescale. Numerical-limit tests and the actual inverse
loop with constant analytic predictions cover CFG0/.5/1/2 and phi0/.7. This is an explicit app design for its DDIM
approximation, not a claim of exact upstream img2img Alternative parity. Review's SALT tie counterexample now passes.

```text
python/.venv-smoke/Scripts/python.exe scripts/dev/gpu_check.py python/.venv-smoke/Scripts/python.exe outputs/verify-neutral-alt-sd15 neutral_alt_sd15
python/.venv-smoke/Scripts/python.exe scripts/dev/gpu_check.py python/.venv-smoke/Scripts/python.exe outputs/verify-neutral-alt-sdxl neutral_alt_sdxl
```

Each completed 12 variants without warnings, covering four operators, decode CFG0/.5/1, rescale, editing, NegPiP and
50 decode steps. JSON reports/grids in these folders were inspected. Both models have **zero pixel MAE** for
baseline/no-op, baseline/restored and ordinary Alternative/Neutral-enabled no-op. The startup limit is validated
analytically/native, separately from those particular model configurations. Scarf effects/style vary by operator
and model; these finite checks do not prove reliable semantic isolation.

Final full Scala **483 passed**, Python **396 passed** (four known torchsde endpoint-rounding warnings), torch-free
**260 passed / 32 skipped**. Jar built; runtime module 55 tests. Source/dependency licences were already collected.
Still unfinished: ControlNet/PAG, PhotoMaker, IP-Adapter/Reference/ControlLLLite, CADS/SAG and guidance/timestep
modifiers in `neutral_runtime.validate`. Their guards remain explicit; target stays todo and ledger **86/216/51**.
