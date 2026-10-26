# Latent Upscale implementation plan

**Goal:** Implement extension-index 189 `latent_upscale` as native img2img script with source-resolution VAE encoding, latent interpolation, scheduler selection and fixed denoising-step count.

**Source/spec:** feynlee/latent-upscale at `b9f75f44de15170a38b195692cff28246846da7d`; README, scripts/latent_upscale.py, latent_upscale_overrides/{init,schedulers}.py, LICENSE and recent history inspected 2026-10-26. MIT, Copyright (c) 2023 Ziyue Li. No learned upscaler, additional model weights or new dependency. Source scheduler math already exists in sdfx.sigmas; use that implementation.

**Architecture:** Script plugin adds `latent_upscale` hook with interpolation and schedule values. Engine encodes the original image before target sizing, samples the VAE posterior with seeded per-image generators, interpolates latents, and passes them directly to img2img/inpaint pipelines. Pixel-space overlays/full-resolution crop preparation remain native. All hooks and VAE dtype changes are scoped. Full-step schedule length/strength maps exactly to requested denoising count. Ordinary no-hook generation remains unchanged.

**Original acceptance:** seven meaningful 2-D interpolation variants (bilinear/bicubic, antialiased variants, area, nearest, nearest-exact); six schedules (simple, normal, karras, exponential, polyexponential, automatic); model sampler/strength/batch/prompts; native inpainting/full-res overlay; fixed requested step count. Source exposes linear/trilinear but torch rejects these on 4-D image latents: reject them explicitly with dimensionality explanation, do not silently invent another algorithm. Zero strength is rejected with an actionable message because fixed positive denoising-step count is incompatible with zero noise. Existing built-in sampler schedule fallbacks must be explicit; original exact schedule acceptance uses custom-sigma-capable Euler/DPM++ routes.

**Optional compatibility:** ControlNet/PAG/other adapter pipelines only when their image-latent/masked-image-latent signatures support the same contract; unsupported combinations fail clearly. Not required to claim all unrelated hooks were verified. Upstream removed ddim_uniform in commit 6bcea56; do not expose the dead helper.

**Files:** python/sdfx/latent_upscale.py (validation, full-step mapping, tensor/encoding helpers), engine.py/base.py/hooks.py/dryrun.py (route and feature), LatentUpscalePlugin.scala/StandardPlugins.scala (native form, settings and provenance), PluginsVM/MainWindow replay restoration, focused pytest/munit suites and manual actual-model checker.

- [x] Write failing CPU tests: initial 10 failures because helper module was absent; source latent interpolation (all seven modes), exact requested step counts, config/schedule/unsupported-dimensionality errors, VAE encode-before-resize and dtype restoration. Malformed scheduler list also observed RED before adding its typed validation.
- [x] Implement helpers and img2img/inpaint integration, native script controls and infotext replay.
- [x] Run targeted CPU tests, full torch-free suite; Scala test execution deferred to parent as instructed.
- [ ] Parent: actual SD1.5/SDXL model runs (ordinary and masked/batch), look at comparison grids, app/MCP and native UI validation, then ledger/license notice promotion.

No GPU/sbt execution in this worker. No parent reForge import/launch, no generated docs/notices/ledger/HANDOFF edits.

## Worker evidence and acceptance handoff

- Full torch-free suite, main `.venv-test`, worktree PYTHONPATH, CUDA disabled: **310 passed, 74 skipped**.
- Focused torch/diffusers CPU and nearby extension/img2img/prompt regression suites: **91 passed**, two dependency Swig deprecation warnings (plus its shutdown deprecation). No CUDA device was visible.
- Real torch/diffusers CPU tests use random tiny VAE/UNet components, without tokenizer/model downloads. They run the actual Engine._run pipeline on source 48x32 and target 64x64, batch 2, covering SD1.5 img2img, SD1.5 four-channel inpaint, dedicated nine-channel inpaint, SDXL img2img and SDXL inpaint. Traced VAE input shapes stay source-sized, and decoded output dimensions/batch counts match. These are pipeline-contract tests, not evidence of trained-model visual quality.
- Real scheduler suffix tests cover Euler, Heun, DPM++ 2M and DDIM at .17/.4/1 strengths and seven requested steps. Effective pipeline strength chooses the same integer schedule suffix as the requested strength; it is allowed to move within that discrete interval. Original GenerationParams strength stays unchanged for other hooks.
- Source classification is **reimplemented**, not copied/ported: no upstream source file was copied. Existing sdfx.sigmas already implements the six required schedule families. Upstream MIT notice must be collected by parent using feynlee/latent-upscale, Copyright (c) 2023 Ziyue Li. No optional dependency or weight download is required.

### Parent commands

1. `sbt.bat -batch "plugins/testOnly sdfx.plugins.LatentUpscaleSuite" "ui/testOnly sdfx.ui.LatentUpscaleReplaySuite"` then normal integration tests/build. The native form is the img2img Script selection `latent_upscale`; parent can capture it in the existing real native UI runner.
2. Set PYTHONPATH to `<integrated repo>/python`, clear CUDA_VISIBLE_DEVICES for the authorized GPU process, and run `<.venv-smoke python> scripts/dev/latent_upscale_check.py <checkpoint.safetensors> <existing source PNG> <output-dir> --full` separately for SD1.5 and SDXL. It covers all seven working interpolation choices and all six schedules, plus batch 2, inpainting/full-res, exact seven-vs-two denoising-step callbacks, source-only VAE encoding, and disabled/no-hook restoration. RGB hashes determine restoration; raw PNG hashes are retained separately. Inspect `comparison.png`, `report.json` and individual full-resolution outputs. Existing previous-wave baseline final images are suitable inputs.
3. Actual native script/AppServices and replay: `sbt.bat -batch "mcp/Test/runMain sdfx.mcp.RealLatentUpscaleCheck <python> <checkpoint-dir> <checkpoint-name> <source-image> <output-dir>"` separately per model. Uses isolated AppHome and persistSettings=false; runs real plugin through JobQueue/Generator, persists full GenerationParams, replays it without plugin reprocessing, then replays native infotext plus plugin values and checks exact RGB equality. Writes `native-replay-report.json` and `generation-params.json`.

### Precise limits

Parent acceptance completed: 21 actual cases per SD1.5/SDXL checkpoint passed, including every method/schedule,
source-size VAE traces, exact fixed steps, mask/full-res and disabled/restored RGB equality. Native AppServices
JSON/infotext replay passed both models; public JavaFX Restore/Send-to tests reset target forms before checking
selection/values and mode redirection. Native script card and comparison sheets inspected. Full Scala514 +
assembly, torch Python731/1skip and torch-free312/74skip passed. MIT attribution collected; index189 promoted.
See `docs/VERIFICATION.md` for actual output paths and bounded evidence. The compatibility limits below remain.

Native Inpaint eligibility follows the existing API: `Tab` has only Txt2Img/Img2Img; `Mode.Inpaint.pluginTab` returns `Tab.Img2Img`, and MainWindow builds its scripts through that registry key. Therefore `LatentUpscalePlugin.tabs = Set(Tab.Img2Img)` intentionally includes both img2img and Inpaint, as with SeamlessTilePlugin. The UI regression suite exercises the actual standard registry -> Inpaint form -> replay selection path, including disabled-plugin filtering. Parent native checks verify restoring from txt2img switches to img2img and restoring while Inpaint is active stays there.

Linear/trilinear are broken upstream choices for 4-D image tensors, so the native UI offers seven working modes and explains the two excluded dimensions; raw requests reject them clearly. Zero strength cannot request positive fixed denoising steps and is rejected. Very small strengths that require more than 10,000 scheduler entries are rejected. No learned super-resolution or pixel-sharpening claim is made. Native full-res mask crops resize pixels before VAE encoding, matching the upstream override's crop behavior; whole-image latent upscaling encodes at source resolution. Existing native masked-noise/nothing preprocessing approximations remain as documented in sdfx.inpaint. Dedicated actual inpainting checkpoint and adapter/ControlNet/PAG/offload combinations are not yet real-model verified. Pipelines that cannot accept supplied masked latents reject that optional combination. Upstream old-Karras global override is not recreated as a separate control; current native schedule options remain available. Parent's real SD1.5/SDXL and UI checks are still required before promotion.
