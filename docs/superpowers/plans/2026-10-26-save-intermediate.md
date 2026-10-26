# Save intermediate images

Implement the extension-index target `save_intermediate` from AlUlkesh/sd_save_intermediate_images.
Upstream inspected at `a1d7cecff893280d709767794d5654e8cbc6746f`; README, script and Unlicense are authoritative.
No parent reForge imports. This is a standalone implementation.

## Behaviour and interfaces

- An opt-in generation hook preserves ordinary final images and captures each batch member.
- Denoised uses the scheduler's predicted original sample (or epsilon/v/sample reconstruction), Noisy uses its
  incoming sample. Preview uses the configured preview decoder, falling back to approximate preview.
- Every N logical inference steps, rounded-down start/stop multiples, combined base+hires clock; independent
  iteration clocks. Stop limits captures, never generation. Img2img uses its actual denoising step count.
- Standard and suffix filenames, optional final copies, per-job intermediate directory, PNG metadata and JSON
  manifest. Result `intermediate_artifacts` describes auxiliary files without polluting `images`.
- Optional GIF/WebM/MP4, codec presets, final-image holds, smoothing/duration, base/hires selection and separate
  reproducible Windows ffmpeg recipes. Mixed sizes fit the largest selected frame. ffmpeg executes without a shell,
  is checked before sampling, is cancellable, and temporary video frames are removed in finally.
- Native JavaFX HookPlugin controls, persisted values and infotext; the app's existing preview replaces the web
  viewer and its UI-state persistence replaces the upstream save-defaults button. Debug logs use existing logging.

## Tasks (tests before production)

1. Add pure configuration/cadence/output tests in `python/tests/test_save_intermediate.py`; verify missing-module
   failure, then implement `normalize`, `selected` and `Recorder` in `python/sdfx/save_intermediate.py`.
2. Add CPU scheduler observer tests (DDIM and second-order sigma schedulers), asserting identical scheduler outputs,
   restoration, batch ordering, combined hires clock and cancellation. Implement observer/callback installation in
   `Engine._run`, initialize recorder per request, and finish after ordinary final images.
3. Add `SaveIntermediateSuite.scala` for opt-in controls, hook serialization, registration and infotext. Implement
   `SaveIntermediatePlugin.scala`, register hook in known/real/dry-run features and StandardPlugins.
4. Run targeted CPU pytest with main-repository venv and worktree PYTHONPATH; parent runs Scala tests, SD1.5/SDXL
   baseline equality/captures/videos, native AppServices/MCP and screenshot checks after integration.

## Review focus

Prediction type and scheduler order, hires offsets, all batch seeds, failed saves/video temporary cleanup,
interruption, optional ffmpeg unavailable, no mutation of latent tensors or scheduler return values.

## Acceptance limits

GPU, Scala/native UI and actual ffmpeg encoding are parent-owned verification gates. These must be completed
before ledger promotion. The recorder captures sampling passes only (not inversion or ADetailer repair passes).

Steps use completed logical inference steps (one-based), preserving documented save-every-N semantics while
avoiding the legacy script's zero-based sampler callback quirks. Preview is explicitly the native configured preview
decoder; there is no separate global A1111 live-preview-subject setting. Standard/suffix use the app's atomic naming
instead of A1111 filename-pattern templates. Video export supports the upstream result formats and codec presets;
recipes use relative concat playlists rather than hard-linked duplicate sequences. Smoothing needs at least three
selected frames; shorter sequences retain their selected source frames without motion interpolation. Video duration
includes held frames.

Review timing correction: with smoothing, let N count selected frames plus both endpoint holds. Duration is
`D=max(N, requested seconds)` and source rate is `N/D`, so the default source clock is 1 fps, matching upstream;
the configured output fps must be greater than 1. A requested duration shorter than N seconds is clamped to N.
Without smoothing, D is requested seconds when positive, otherwise `N/fps`. Both encoder commands and generated
recipes share this calculation. Timestamp normalization and explicit source FPS avoid concat-list timebase rounding
and give padding the same source clock as direct encoding. Before interpolation,
clone the final image for `2/source_rate` seconds to satisfy minterpolate lookahead; then trim to D and reset PTS.
Thus the final selected/held image remains available at the endpoint and padding does not lengthen the result.
Encoded duration can still be quantized by the output frame rate/container; real ffmpeg checks remain parent-owned.

Review VAE correction: `_decode` caches the shared VAE's original dtype and restores it in finally, including when
latent conversion or decode raises. CPU injection tests verify both successful and failed force-upcast paths.

## Implementation evidence and parent validation

No dependencies added or installed. ffmpeg is an optional external executable (not distributed by this change).
Upstream source was read, not imported; implementation uses the app's existing Pillow, torch and diffusers.
PNG frames carry `parameters` for native prompt import and `sdfx_intermediate` JSON; manifests retain the complete
generation request, resolved model/vae metadata and output records. On interruption saved frames remain, with an
incomplete manifest; only transient video staging and partial video outputs are removed.
Missing ffmpeg fails before sampling. An unavailable codec or failed video export after sampling emits an explicit
warning and manifest `export_errors`, retaining successful final images and PNGs; cancellation still aborts.

CPU commands (from worktree, with `CUDA_VISIBLE_DEVICES=-1` and worktree `python` in PYTHONPATH):

```powershell
& D:/stable-diffusion-webui-reForge/stablediffusion-fx/python/.venv-smoke/Scripts/python.exe -m pytest python/tests/test_save_intermediate.py python/tests/test_extensions.py python/tests/test_weight_gradient_engine.py python/tests/test_hooks.py python/tests/test_params.py python/tests/test_protocol.py -o addopts='' -q
& D:/stable-diffusion-webui-reForge/stablediffusion-fx/python/.venv-test/Scripts/python.exe -m pytest python/tests/test_save_intermediate.py python/tests/test_extensions.py python/tests/test_hooks.py python/tests/test_protocol.py python/tests/test_params.py python/tests/test_server_dryrun.py -o addopts='' -q
```

Parent must run after integration (normal GPU environment; remove `CUDA_VISIBLE_DEVICES=-1`):

```powershell
& "$env:LOCALAPPDATA/Coursier/data/bin/sbt.bat" -batch 'core/testOnly sdfx.core.ProtocolSuite' 'plugins/testOnly sdfx.plugins.SaveIntermediateSuite' 'engine/testOnly sdfx.engine.GeneratorSuite' ui/assembly
& python/.venv-smoke/Scripts/python.exe scripts/dev/save_intermediate_check.py python/.venv-smoke/Scripts/python.exe D:/stable-diffusion-webui-reForge/models/Stable-diffusion/animekawa_v10.safetensors outputs/verify-intermediate-sd15 --video
& python/.venv-smoke/Scripts/python.exe scripts/dev/save_intermediate_check.py python/.venv-smoke/Scripts/python.exe D:/stable-diffusion-webui-reForge/models/Stable-diffusion/waiIllustriousSDXL_v170.safetensors outputs/verify-intermediate-sdxl --sdxl --video
& 'C:/Program Files/Java/jdk-21/bin/java.exe' -jar target/stablediffusion-fx.jar --dry-run --home outputs/verify-intermediate-ui --screenshot outputs/verify-intermediate-ui.png --exit-after 8
```

Before screenshot, set `expanded.txt2img.plugin.save_intermediate=true` in that home's config/ui-state.json.
Expand Video export for a second screenshot and inspect enabled/disabled controls and overflow. The real-model
script checks enabled/disabled final-pixel equality, all subjects, batch/iteration, suffix, hires/img2img, Heun,
PNG metadata, output auxiliaries, three video formats and cleanup; inspect its contact PNGs and play video outputs.
Also check h265/av1 availability, interpolation/duration/holds and execute generated base/hires recipes using the
local ffmpeg build. Trigger cancellation during sampling/video, verify restoration with a subsequent baseline.

Native AppServices/MCP gate: select `save_intermediate` as AlwaysOn with enabled=true, every_n=2, final_save=true;
generate real SD1.5 and SDXL images. Confirm final gallery count remains batch_size*n_iter, output log names the
persistent manifest, intermediates live under configured outputs/intermediates and outputs/.tmp is cleaned.
Send an explicit `hooks.save_intermediate.output_dir` outside outputs through AppServices; GeneratorSuite asserts
the destination is replaced by the configured root. Direct Python engine clients retain their usual filesystem
output authority. The typed Scala auxiliary result deliberately does not register these artifacts as temp files.

Final agent CPU evidence (2026-10-26): smoke targeted command above **148 passed**, two torchsde floating-endpoint
warnings; no-torch targeted command **118 passed, 24 skipped**. `py_compile` for the recorder/engine/acceptance script
and `git diff --check` passed. Scala/sbt, GPU inference, actual video encoding, native AppServices/MCP and rendered
UI checks were intentionally not run by this worker; parent owns those gates and the extension ledger/notices.

Review-fix verification: new regressions first reproduced same-rate smoothing and failed-decode dtype leakage.
After fixes, refreshed smoke targeted command: **158 passed**, the same two torchsde endpoint warnings. Refreshed
no-torch targeted command: **126 passed, 26 skipped**; a further no-torch recorder-only refresh after explicit
source-FPS normalization: **25 passed, 20 skipped**. `git diff --check` passed. Actual ffmpeg, GPU and sbt/UI remain
parent-owned gates.

Independent CPU real-ffmpeg review (2026-10-26): nine synthetic direct/Windows-recipe cases passed using installed
WinGet ffmpeg/ffprobe. MP4/WebM/GIF smoothing produced 32 frames over 4 seconds; stretched and held MP4 cases
produced 64 frames over 8 seconds. Decoded direct/recipe pixels matched exactly for every case. Interpolated frames
differed from every source frame, endpoints survived, and explicit beginning/end holds were checked. One/two-frame
fallbacks, ordinary timing and requested duration passed. Cancellation after launching the real encoder retained an
incomplete manifest and removed video outputs/staging. GIF initially showed endpoint color errors around 20/255 MAE;
restoring upstream adaptive palette generation/use reduced this below 1/255. A real-ffmpeg color regression failed
before the correction and passed afterward. Recorder suite now **46 passed**, same two torchsde warnings.
Local reproducible runner and JSON evidence are under worktree `outputs/verify-intermediate-ffmpeg/` (ignored artifacts):
`reproduce.py`, `report.json`, retained PNGs, videos and executed recipes. Run that script with worktree `python` in
PYTHONPATH and the main smoke venv. No GPU, model inference or sbt/UI checks were run by this review worker.

## Parent integration accepted

The parent completed 12 actual-generation cases per SD1.5/SDXL checkpoint, with exact final RGB invariance for all
capture subjects and disabled mode. Cadence, iterations/batch, combined hires clock, img2img, Heun, metadata and
three video formats passed; contact sheets were inspected. Native AppServices queue/replay verified persistent
artifacts, four final gallery images, outside hook destination replacement and temp cleanup. Root intermediates
are excluded from gallery scans; unrelated nested names remain supported by regression coverage. Full native
video form inspected. Independent actual HEVC/AV1 uniform/mixed-size tests also passed (four cases). Full Scala491
+ assembly, torch Python707/1skip, no-torch306/60skip passed. Notices include Unlicense and optional external ffmpeg;
index87 promoted. Exact artifacts and bounded checks are recorded in `docs/VERIFICATION.md`.
