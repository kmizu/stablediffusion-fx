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
