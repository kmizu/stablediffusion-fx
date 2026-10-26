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
