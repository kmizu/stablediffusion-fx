# Handoff: bundling the A1111 extension index into stablediffusion-nui

## Active goal reset: ten additional issues then release

On 2026-10-30 the user cleared the previous 20-issue goal and set a new goal to resolve ten issues and release. Treat the new goal as ten further issue resolutions from this reset; #48 and #61 are accepted, making the count 2/10. The earlier ten resolved waves remain project history, not automatic completion of the new objective. Preserve all original scope, real-model and package checks, push and verify before counting, then publish and download/hash-verify the release.

## Current progress: #48 and #61 accepted; #85 and #46 underway

Current #46 work has been integrated in an isolated worktree at `C:/Users/Mizushima/.codex/worktrees/embedding-inspector-integrated/stablediffusion-nui`, with the original worktree retained; it is not yet integrated into main or accepted. #82 was developed separately at `C:/Users/Mizushima/.codex/worktrees/d912/stablediffusion-nui`.

#82 metadata utilities has passed local acceptance and is being integrated: Python 1,535 passed/1 skipped, Scala 1,441 passed/1 skipped, and native/extracted-package SD1.5 and SDXL copy, cancellation and exact generation checks passed. Read-time identity and full-file hash bind a save to the reviewed source. See `metadata-utils-verification.md`, including the retained earlier timeout evidence. Count it only after push and issue closure are verified.

#48 was pushed as commit `b8a623fdad1d78317a10b37149192a59ccff3499`; local and `origin/main` matched and the working tree was clean at verification. [Issue #48](https://github.com/kmizu/stablediffusion-nui/issues/48) closed 2026-10-30 15:17:54 UTC. See `encrypt-image-verification.md` for evidence and limitations, including readable metadata and temporary files, unsupported 16-bit LA, and no actual SVD-model test. #85 is underway in isolated worktree `C:/Users/Mizushima/.codex/worktrees/d919/stablediffusion-nui`, not yet integrated. The release is pending.

#61 img2img Hires Fix was pushed as commit `65b4ddcc11e8c2438c30cdd57b3d7cfc3732e019`; local and remote HEAD matched and the working tree was clean at verification. [Issue #61](https://github.com/kmizu/stablediffusion-nui/issues/61) closed 2026-10-30T16:36:42Z. Acceptance: Python 1,507 passed/1 skipped, Scala 1,437 passed/1 skipped, and native and extracted-package SD1.5/SDXL generation, replay and interruption checks passed with images inspected. See `img2img-hires-fix-verification.md` for source behavior and limits. Accepted count is 2/10; the release remains pending. #46 is underway separately at `C:/Users/Mizushima/.codex/worktrees/embedding-inspector/stablediffusion-nui`; its SDXL pooling and live embedding reload changes remain under validation.

## Historical: Image Encryption (#48), eleventh wave

The following entries record earlier states; the acceptance status above supersedes pending and 0/10 statements.

2026-10-31 ledger update: index 237 is locally `done:core:image-scrambling`; generated totals are 116 done / 186 todo / 51 not applicable. Focused ledger tests passed (`outputs/encrypt-ledger-test.log`). Earlier statements that index 237 is todo are historical. Commit, push, remote verification and issue closure remain pending; count remains 0/10 until closure.

2026-10-31 local acceptance complete: latest full Scala passed 1,427 tests with one skip (`outputs/encrypt-full-scala-latest.log`, exit 0), superseding the pending status in the next entry. Final evidence and package hash are in `encrypt-image-verification.md`. Limits remain: SVD was not tested with an actual SVD model, metadata and temporary files can be readable, and 16-bit LA PNG is unsupported. Commit, push, remote verification and issue closure remain pending; the reset goal count is 0/10 until closure. This is not a release.

2026-10-31 latest compatibility and package evidence: source-derived v1 16-bit grayscale and 8-bit LA v1–v3 cases and same-basename collision fix are included. Python full 1477 passed/1 skipped (`outputs/encrypt-full-python-latest.log`); focused Scala 27 passed (`outputs/encrypt-compat-final.log`). Latest assembly and acceptance ZIP SHA256 `4a911bfecb64ad9bf9c21d1a67b1b6540cb579a774087516bd67001351b03a6b`; package dry-run passed (`outputs/encrypt-package-check-latest.log`), and extracted package real SD1.5/SDXL checks passed with images inspected (`outputs/encrypt-packaged-latest-sd15.log`, `outputs/encrypt-packaged-latest-sdxl.log`; logs identify smoke-venv worker). These supersede earlier package evidence below. Earlier whole Scala rerun was deliberately interrupted after code changed; the latest full run is ongoing (`outputs/encrypt-full-scala-latest.log`) and remains pending. #48 is not complete, current reset count remains 0/10, and this acceptance ZIP is not a release.

2026-10-30 latest acceptance: first whole Scala run failed local-drive/UI checks; focused local-drive 22, UI 25, assembly and mounted controls passed (`outputs/encrypt-failure-recheck.log`); 420/887-pixel screenshots inspected. Whole Scala repeat is running (`outputs/encrypt-full-scala-final.log`) and remains pending. Mounted session real SD1.5/SDXL parity, lock, wrong/correct password, cancellation and recovery passed (`outputs/encrypt-mounted-gpu-sd15.log`, `outputs/encrypt-mounted-gpu-sdxl.log`). Corresponding extracted-package checks passed (`outputs/encrypt-packaged-gpu-sd15.log`, `outputs/encrypt-packaged-gpu-sdxl.log`), using the smoke-venv worker identified in the logs; generated images inspected. Packaged dry-run passed (`outputs/encrypt-package-check.log`); acceptance ZIP SHA256 `9ef8b2cfb7258a582e23e1d5d05dde655c29e2e09720ee5f009d1ed39c679119`. See `encrypt-image-verification.md` for scope and limitations. This is not a release; #48 remains in progress and current reset count is 0/10 pending acceptance, push and verification.

2026-10-30 latest integration: the session and batch controls are mounted in MainWindow. The native dry-run check exercised queued session snapshots, preview refresh, batch conversion, disposal and shutdown at panel widths 420 and 887; screenshots were inspected (`outputs/encrypt-mounted-ui-verified.log`, `outputs/encrypt-mounted-ui-verified/`). Python and JVM exchange 16-bit RGB and RGBA PNGs in both directions with exact samples and Unicode text metadata; the cross-runtime Scala suite passed 11 (`outputs/encrypt-cross-runtime.log`). The full Python smoke-venv suite passed 1473 with one skipped (`outputs/encrypt-full-python-final.log`).

JPEG and WebP conversion may lose metadata, and the batch UI/results warn about it; output is PNG, not the original compressed file. Preserved EXIF may contain embedded thumbnails, which remain readable and can reveal image content. Scrambling is not authenticated encryption, metadata remains readable, and worker temporary plaintext can survive a crash. Scala full tests and assembly are still running; extracted-package acceptance remains pending. Issue #48 is not complete or counted; the reset goal remains 0/10 and no release has been published for it. Earlier notes below describe historical states and are superseded where they say that the session pane is unmounted or Python rejects all 16-bit PNGs.

2026-10-30 metadata and integration: PNG batch conversion now preserves verified iCCP/pHYs, valid EXIF orientation and compatible sRGB/gAMA/cHRM, with raw RGB/RGBA raster samples including 16-bit preserved; malformed or conflicting metadata fails explicitly. Core metadata/legacy/WebP tests passed 23 (`outputs/encrypt-metadata-full.log`). JPEG/WebP metadata can still be lost and now produces an explicit per-result warning and a pre-run panel warning; decoded output is PNG, not original compressed bytes. WebP reader is available in assembled jar (`outputs/encrypt-webp-jar-final.log`), with BSD notice; initial jar lacked runtime dependency, then dependency propagation was fixed. Python codec currently rejects 16-bit rather than losing precision; lossless support is in progress. Upstream v1 grayscale/palette decoding is implemented and tested, with v2/v3 unsupported for those modes. WD locked inputs now fail before interrogation rather than silently processing shuffled pixels (`outputs/encrypt-wd-locked-final.log`); unverified legacy warnings propagate for Temporal, VAE, WD and worker input resolution. Completed Extras/ControlNet worker output temporaries are scrambled with the captured session, though plaintext still exists while worker runs and can remain on crash; actual dry-run native Extras, ControlNet and Temporal integration passed (`outputs/encrypt-agent-final-regression.log`). Independent VAE Tools GPU and native UI acceptance passed SD1.5 and SDXL (`outputs/encrypt-vae-tools-gpu-sd15.log`, `outputs/encrypt-vae-tools-gpu-sdxl.log`, `outputs/encrypt-agents-final.log`, `outputs/encrypt-vae-native-sdxl.log`); readable latent tensors and L/CMYK encoding outputs require explicit per-run consent; reports remain readable for all operations. Further JPEG/WebP metadata work, session mounting and package acceptance remain.

2026-10-30 parallel integration: Temporal Lab now decodes protected input sequences and saves protected frames for DFI, preview, deflicker, fuse and extraction; native session-change preview and real worker check passed after fixing a stale async slot deadlock (`outputs/encrypt-temporal-native-retry.log`). Focused Python temporal/transport tests and Scala temporal tests passed (`outputs/encrypt-temporal-python.log`, `outputs/encrypt-agents-regression-final.log`); legacy unverified passwords are warned and included in UI results. VAE Tools now supports protected RGB/RGBA inputs, RGBA latent PNG and decoded/preview PNG, explicit per-run consent for readable L/CMYK/safetensors latent output, and owner-scoped cancellation. Real SD1.5/SDXL direct Full VAE and native protected UI checks passed (`outputs/encrypt-vae-tools-gpu-sd15.log`, `outputs/encrypt-vae-tools-gpu-sdxl.log`, `outputs/encrypt-agents-final.log`, `outputs/encrypt-vae-native-sdxl.log`); images inspected. MCP deliberately has no desktop password session and rejects marked PNG inputs before re-encoding; VAE service requires an explicit secret argument. Comparison PNG export now captures the session secret; native test covers protected output and stale-session exportTo but not the OS chooser (`outputs/encrypt-agents-final.log`). Custom Code warns it can create arbitrary readable files. Session is cleared on AppServices shutdown without claiming memory erasure. WebP ImageIO 3.12.0 is included with pinned BSD notice and actual alpha fixture test; its runtime dependency had to be added to every project for assembly, after the first assembled-jar test returned null. Rebuilt jar-only Java check passes (`outputs/encrypt-webp-jar-final.log`; prior jshell falsely continued after an exception and is not evidence). PixelScramble now preserves raw RGB/RGBA raster samples; disk 16-bit and ICC/EXIF/DPI are still under investigation. Session pane remains unmounted; issue/release acceptance remains incomplete.

2026-10-30 VAE recovery: automatic VAE-decode recovery now scrambles its RGBA PNG under the captured session password, while preserving the exact safetensors decoder input as readable plaintext; the worker warns explicitly and the standalone session pane discloses this. Protection failures delete the reserved PNG and do not write a tensor (`outputs/encrypt-vae-recovery-failure.log`); 50 tiny-model/codec tests passed (`outputs/encrypt-vae-recovery-regression.log`). Real SD1.5 and SDXL tests injected a decoder failure, checked the PNG marker and decoded format, compared raw tensor bits to captured decoder input, confirmed the warning, and inspected the intentionally Broken preview (`outputs/encrypt-vae-gpu-sd15.log`, `outputs/encrypt-vae-gpu-sdxl.log`). UI session control test passed (`outputs/encrypt-vae-session-ui.log`). This is not full VAE Tools integration: latent L/CMYK and safetensors are data rather than ordinary RGB images, VAE Tools and Temporal Lab still need a coherent read/write and disclosure policy; session pane remains unmounted.

2026-10-30 GIF output policy: Seed Travel, Shift Attention and SVD now require the session plaintext-video choice before making a GIF; the generator also gates img2vid before dispatch. Job-scoped consent is captured with the secret and ordinary sessions remain allowed. Plugin/core/engine focused regressions passed (`outputs/encrypt-gif-regression.log`). Real SD1.5 and SDXL queue runs for Seed Travel and Shift Attention verified early refusal without consent and exact ordinary/protected frame and GIF parity with consent (`outputs/encrypt-gif-gpu-sd15.log`, `outputs/encrypt-gif-gpu-sdxl.log`); first GIF frames were visually inspected and are low quality at 3 steps. SVD has synthetic tests only; no SVD-model acceptance is claimed. Session pane remains unmounted and further side outputs remain open.

2026-10-30 WD actual acceptance groundwork: removed the WD/scramble queue guard. Real WD v3 and SD1.5/SDXL worker checks match captions and decoded output RGB between ordinary and scrambled inputs (`outputs/encrypt-wd-gpu-sd15-retry.log`, `outputs/encrypt-wd-gpu-sdxl.log`). Both actual MainWindow submissions with two protected inputs passed queue, output metadata, decoded image, source-hash and gallery-item checks (`outputs/encrypt-wd-native-sd15.log`, `outputs/encrypt-wd-native-sdxl.log`); decoded images and SD1.5 UI screenshot were visually inspected. The session was enabled programmatically in these tests because its pane remains unmounted. Scala WD/queue/safety 39 and frozen evidence 17 passed (`outputs/encrypt-wd-native-regression.log`); the evidence schema test now explicitly verifies password exclusion. ONNX 1.21.0 and ml_dtypes 0.5.4 were installed in the smoke venv; first real test failed before setup because ml_dtypes was absent, then passed. Other #48 paths, session mounting, package acceptance and real cancellation coverage remain incomplete.

2026-10-30 WD batch groundwork: the worker's owned transaction now decodes marked input snapshots with the job secret and scrambles generated PNGs after metadata insertion but before publication. Scala sends the secret only with the WD execute envelope, outside preparation and params; the JobRunner guard has since been removed after actual-model and native validation, as recorded above. Python full worker routing with the inference provider stubbed verifies decoded input, preserved metadata, protected output and source immutability (`outputs/encrypt-wd-worker-green.log`); wrong passwords, rollback on encoding failure, transport and existing batch regressions passed (`outputs/encrypt-wd-final.log`). Scala transport, queue and diagnostic tests passed 32 (`outputs/encrypt-wd-transport-scala.log`). Transaction staging still contains plaintext while the operation runs and can remain after a crash; this is not secure erasure. This earlier note is superseded by the real-model checks recorded above.

2026-10-30 protected recipes: Save Intermediate can write a password-free JSON recipe plus Windows batch launcher for protected frames. The launcher prompts for the password at execution and warns that the video and temporary frames are readable; staged output is published with an exclusive hard link, preserving an existing or concurrently created target. An actual Windows batch run passed in a Japanese/percent-sign directory (`outputs/encrypt-recipe-windows-final.log`); this exposed an ffmpeg Unicode absolute-path issue, fixed by running with relative paths. Wrong passwords, existing output, concurrent target and temp cleanup tests pass (`outputs/encrypt-recipe-final.log`); focused torch regressions passed (`outputs/encrypt-recipe-regression.log`). Runtime/package paths are recorded in the launcher and can become stale after moving the installation; crash remnants are still possible. The first noninteractive Windows test hung in getpass, then was stopped; redirected input is now read from stdin and the retry passed. Session pane is still unmounted and overall #48 remains incomplete.

2026-10-30 video consent: the standalone session pane now offers an explicit off-by-default plaintext-video switch with temporary-file/crash warning; locking or unlocking resets it and queued snapshots retain their previous value. JSONL carries the permission only with an image password and outside captured generation params. UI control and 27 Scala tests passed (`outputs/encrypt-video-consent-ui-green.log`); Python transport/recorder tests passed (`outputs/encrypt-video-consent-python.log`). Real SD1.5 and SDXL worker Steps Animation with video produced three protected frames and output videos whose decoded RGB matched ordinary videos (`outputs/encrypt-video-gpu-sd15.log`, `outputs/encrypt-video-gpu-sdxl.log`); early 3-step video frames were visually inspected. This does not establish native app/package acceptance; session pane remains unmounted.

Latest: Save Intermediate has groundwork for protected standard PNG frames via an optional in-memory password: the plaintext PNG is serialized in memory and only scrambled bytes are written to its reserved output; failed writes remove the reserved file. Protected Video mode can decode frames for video only when explicitly permitted to export plaintext; protected ffmpeg recipes now prompt for a password at execution and decode frames in a temporary directory. Temporary plaintext video frames are deleted on normal completion or cooperative interruption, but crash remnants remain possible. The secret now travels in the JSONL envelope outside captured generation params and reaches the real worker context; standard Save Intermediate uses it. Steps Animation now saves protected frames when video creation is disabled; protected video requires explicit session-only plaintext-export consent; the standalone session control supplies it but remains unmounted. Native MainWindow dry-run generation also passed after transport changes (`outputs/encrypt-native-generation-transport.log`). Session UI remains unmounted. Its video choice is captured into each queued secret snapshot, sent outside diagnostic generation params, and resets on lock or re-unlock. Protected recipes reference the Python executable and package path of the installation that created them, so moving that installation requires regenerating the recipe. Scala generation/queue/diagnostic tests passed 53 (`outputs/encrypt-transport-scala.log`); Python transport, recording and interruption tests passed (`outputs/encrypt-transport-python-full.log`). Real SD1.5 and SDXL worker Steps Animation frame tests each matched three protected/ordinary frames exactly (`outputs/encrypt-steps-gpu-sd15.log`, `outputs/encrypt-steps-gpu-sdxl.log`); both 3-step examples were visually inspected. Real SD1.5 and SDXL worker standard-frame tests each matched three protected/ordinary frames exactly (`outputs/encrypt-intermediate-gpu-sd15.log`, `outputs/encrypt-intermediate-gpu-sdxl.log`); representative 3-step intermediate images were visually inspected and are unfinished/noisy. Earlier focused tests with torch environment passed (`outputs/encrypt-video-torch.log`); an interruption test observed a plaintext temporary frame before cancellation and confirmed cleanup (`outputs/encrypt-video-cancel.log`); these checks are not full feature or package acceptance. Upstream wrapper forces disk image saves to PNG but NUI video exports use separate encoders; do not claim those are protected.

2026-10-30 continuation: MainWindow real JavaFX Generate plus Python dry-run passes the queued secret snapshot, decoded pixel parity and locked-preview check (`outputs/encrypt-native-generation.log`). Session-change notifications now refresh PreviewPane, GalleryView, ImageDropZone, MaskCanvas, ImageComparisonTab, Inpaint Difference, Extras and ControlNet previews; native UI regressions and mask preservation passed (`outputs/encrypt-auto-views.log`, `outputs/encrypt-difference-protected.log`, `outputs/encrypt-extras-full.log`, `outputs/encrypt-controlnet-green.log`). Session and batch panes remain unmounted. WD batch now supports protected jobs; intermediate PNGs, videos/GIFs, VAE tools and other persistent side outputs still need a coherent full-scope implementation and explicit user-visible limits before enablement. Plaintext worker temporary files and crash remnants remain a limitation. No real-model acceptance for this wave has been performed.

Upstream `viyiviyi/sd-encrypt-image` pinned `3a44d3c264632aa90442bdc50e42af0770289bfd` under `outputs/encrypt-image-source`; AGPL-3.0. It actually permutes pixels deterministically, leaves metadata readable and is not authenticated encryption. Do not market compatibility mode as secure encryption. Source wrapper marks v3 PNG with `Encrypt` and optional `EncryptPwdSha`; old files can lack the verifier, so password correctness is then unknown.

Provisional design `docs/superpowers/specs/2026-10-30-encrypt-image-design.md` and plan `docs/superpowers/plans/2026-10-30-encrypt-image.md`. Independent Python and JVM codecs match 18 pinned-source cases; Python focused21 tests, JVM focused5 passed. `ScrambledPng` implements verified/unknown-password reads and atomic protected PNG writes; in-memory `ImageScrambleSession` snapshots are redacted. `Job.imageSecret` is wired through `Generator.saveAll`/`saveWith` and numbered grids; fake-engine JobQueueSuite18 passed (`outputs/encrypt-queue-green.log`). This is not connected to UI and is not a complete feature. Worker temp images remain plaintext; scripts now use job-scoped image reads for grids and direct inputs, and `Generator.runEngine` decodes init/mask/ControlNet/reference/revision/IP-Adapter/LLLite/PhotoMaker/difference-base inputs to scoped plaintext temps for the worker. Job-scoped caption, restore, novel view, video, image tools and upscale also resolve selected image paths. The difference preview can resolve protected inputs when supplied a secret; an actual Python dry-run worker produced pixel-identical masks (`outputs/encrypt-real-preview.log`). `ScrambleBatch` implements non-overwriting folder PNG/JPEG conversion and explicit unknown-password legacy handling (`outputs/encrypt-batch-green.log`), but is not connected to UI. Standalone JavaFX batch and session control components now pass real control-action checks, but are not mounted or connected to the app. PreviewPane, GalleryView, ImageDropZone and MaskCanvas now read compatible images off the FX thread with revision-guarded session snapshots; native JavaFX tests confirm correct pixels/wrong-password rejection and inpaint-mask retention. Session-change notification now refreshes these controls and comparison; clipboard also resolves images using the session snapshot. The session controls are not yet mounted in MainWindow. Direct Extras upscale/color correction and ControlNet preprocessor preview also use a captured secret; other image views and non-PNG outputs remain unintegrated. Do not enable until those paths and safety limits are addressed. Additional regressions passed: JVM codec5, GeneratorSuite27, JobQueueSuite19, plugin suites49 (`outputs/encrypt-generation-regression.log`, `outputs/encrypt-script-context-final.log`). Prompt Matrix and seamless tile tests verify pixel parity. Worker input tests cover multi-field parity, duplicate input, failures, disabled ControlNet, caption/upscale paths and diagnostics (`outputs/encrypt-tool-final.log`, 54 passed); temp files are plaintext and deletion is not secure erasure. Async input lifecycle and queue/generation regressions passed 58 tests (`outputs/encrypt-preview-regression.log`); The native MainWindow dry-run generation test also passed as noted above. Issue remains OPEN and index237 todo. Count10/20, no release.

## 2026-10-30 update: Inpaint Difference (#67)

Index 243 is the tenth wave toward the twenty-issue release goal. The native Inpaint panel supports original/altered inputs, difference threshold, contours, dilation/erosion and an effective blur/invert preview. It preserves the existing painted mask and queued inputs; preview and input temporary lifetimes are guarded. Pinned MIT source, license and 26 source-derived pixel fixtures are recorded in `inpaint-difference-verification.md`, including EXIF/RGB adaptations and preview limitations.

Final Python regression: 1378 passed, one skipped. Whole Scala: 1367 passed, one skipped before the final invalid-path guard; 41 focused UI tests and assembly passed after it. Both SD1.5 and SDXL passed actual native Generate/Interrupt/recovery and extracted-package RGB comparisons; images/screenshots inspected. Acceptance ZIP SHA256 `d52aa8c566fa93a9f7c9d32241dd9f22c8b180da1f693a4d90d424b6b6b078d4`; it is not a release. Notices cover 156 projects; ledger 115 done, 187 todo, 51 not applicable. See verification document for exact commands, artifacts and limits. Check push and issue closure on GitHub before counting this wave as accepted.

After #67 is pushed and closed, accepted count is 10/20. Select and investigate the next distinct open issue, preserving its full source scope. Do not publish until twenty issues are accepted and the release is built, tested, published and downloaded/hash-verified. Parent reForge remains reference only; never import or launch it.

## 2026-10-30 update: Hires Fix Tweaks

Index297 / #56 is the ninth accepted wave: full native Hires CFG/XYZ, cached1–16 refinements, independent primary/variation/resize seeds, four prompt modes, pre-style network removal, ordered styles, safe template/PNG restoration, output/gallery destination and display preferences. Structured manual LoRAs remain distinct from prompt tags, including Composable provenance. Large seeds remain exact across hook/response/PNG/history; upstream CFG spelling is accepted. AGPL source and licence are pinned. See `hires-fix-tweaks-verification.md` for requirement-by-requirement evidence and explicit visual/adaptation limits.

Final whole Python1340pass1skip; whole Scala1351pass1skip (494s); assembly and final ledger20 tests pass. Actual SD15/SDXL native Generate/Interrupt/Restore and extracted-package manual-LoRA/style/output/cancellation/recovery checks pass; images/screenshots inspected. Real AD four-item detection/inpaint and partial cancellation pass both models; SD15 ControlNet and both-model Hires-only custom LBW exact equivalence pass. Package ZIP SHA256059c394b3dc687b3ad72b66277a4de0a575f6b36aad3df227888f606a4bbadbe; jar SHA256cb206ff0657e279db9a738f7e68717a7356664a13c5a753388ed4b133a597ba5. This is an acceptance package, not a published release. Ledger114done188todo51na. Push/closure receipt is on issue#56; verify HEAD/remote before new work. No test/GPU processes remain after their recorded completion.

Next selected issue is #67 `inpaint_difference` (index243); its current state is at the top of this handoff. Other candidates #48/#85 were only read. Goal remains twenty distinct resolutions then verified release; after #56 closure count9/20.

## 2026-10-30 update: Token Downsampling

Index 295 / #129 is the eighth accepted wave: native two-depth K/V downsampling, requested-step evaluation cutoff, three XYZ axes and PNG Restore. SD1.5/SDXL final GPU tests cover both depths, no-op exactness, img2img/hires and Neutral branches; actual extracted-package MainWindow generation, three interruptions and exact disabled recovery pass both models. Review caught and corrected strength-adjusted cutoff and branch-clock omissions. See token-downsampling-verification.md for hashes/evidence, quality tradeoffs and explicit PAG/HyperTile limits. Whole Python passed; whole Scala 1298 passed/one skip and assembly passed. Notices154; ledger113done189todo51na. No release published. Next: investigate the ninth distinct issue without reducing original scope. Waifu2x #137 commit a44214756f32add81105b356f819ba4665fa6bc0 was verified remote and closed 2026-10-30T03:58:45Z.
## 2026-10-30 update: Waifu2x

Index 321 / #137 is the seventh accepted wave: nine pinned neural upscalers, native Script Anime/Photo/CARN controls and 2x-16x scaling, atomic verified model downloads, exact CPU reference parity, and actual SD1.5/SDXL GPU/hires coverage. Extracted-package native three-family, three-cancellation and neural/diffusion recovery checks passed on both models. Packaged testing exposed a swallowed-interruption race; Generator now synchronously marks the captured queue entry on backend interruption, with a deterministic regression and full engine tests passing. See waifu2x-verification.md for hashes, exact logs and GPU/numerical limits. Final ledger/plugin suites passed 26 tests. Notices: 152 projects; ledger: 112 done, 190 todo, 51 not applicable. The previous VRAM target classification was corrected to core:vram-estimator after final-ledger consistency tests exposed it. No release published for the twenty-issue goal. Next: investigate the eighth distinct issue; preserve complete upstream scope. Publication remains after twenty resolutions.
## 2026-10-30 update: VRAM Estimator

Index 105 / issue #136 has native four-mode CUDA calibration, reserved/allocated peak baselines, four charts, bounded atomic complete/partial profile history and file import/export, strict runtime identity matching and visible-form live estimates. Actual SD1.5/SDXL sweeps and held-out predictions, owned/protected cancellation with unrelated generation preserved, actual bounded CUDA OOM/recovery and assembled-jar native service/view acceptance passed. See vram-estimator-verification.md for exact evidence and limits. Whole Python passed; whole Scala passed 1292 tests with one skip, then the four focused history/session tests and UI compile/assembly passed. MIT notices cover 148 projects. Ledger: 111 done, 191 todo, 51 not applicable. This is the sixth issue of the twenty-issue goal; no release has been published. Next: select and investigate the seventh distinct issue without narrowing its original scope. #121 spans three different style libraries; its complete scope is broader than one prompt selector.

## 2026-10-30 update: StylePile

Index 40 / issue #122 has native Script `stylepile`, full pinned GPL catalog (512 references), searchable lazy previews, weighted style/preset composition, X/Y/A/B/C and noise variables, prompt insertion helpers, custom image/inspiration overlays and interruption-safe sequences. Actual native SD1.5/SDXL configured-style/manual RGB and img2img RGB match exactly; four-image order/seeds and partial interruption/recovery pass. See `stylepile-verification.md`. Whole Scala passed 1289 tests with one skip; assembly and assembled-jar-only native/resource checks passed. All 517 bundled source-file hashes verified in staged Git and jar. Notices cover 147 projects. Current ledger: 110 done, 192 todo, 51 not applicable. This is the fifth issue of the active twenty-issue release goal; release remains unpublished.


## 2026-10-30 update: Steps Animation

Index 57 / issue #120 has native Script `steps_animation`: denoised-step movies, none/blend/MCI interpolation, duration/fps, five codecs, initial-frame skipping, optional export/cleanup, relative output locations, debug logging and bounded interruptible partial export. Actual native Generate/Interrupt/recovery and txt2img/img2img/hires checks passed on SD1.5/SDXL with exact enabled/off final RGB. See `steps-animation-verification.md`. Whole Python passed; whole Scala passed 1269 tests with one skip; assembly passed. Notices cover 146 projects. Current ledger: 109 done, 193 todo, 51 not applicable. This is fourth issue toward the twenty-issue release goal; no release published yet.


## 2026-10-30 update: PNGINFO Injection

Index 336 / issue #98 has native opt-in `pnginfo_injection`, saved Hashes merge, confirmed local resource fingerprints and actual wildcard-source byte provenance. See `pnginfo-injection-verification.md` for real SD1.5/SDXL native PNG, embedding and ADetailer checks, SD1.5 ControlNet check, cache-mutation handling and explicit directory/Civitai limits. Whole Python passed; whole Scala had five Windows drive-probe timeouts and the focused 22-test recheck passed. Assembly and final native checks passed. Current ledger is 108 done, 194 todo, 51 not applicable. This is the third wave of the active twenty-issue goal; release remains unpublished.

## 2026-10-30 update: GPU temperature guard

Index 186 / issue #54 has native AlwaysOn `gpu_temperature_guard`, interruptible step-end hysteresis, bounded NVIDIA/ROCm/WMI temperature queries, physical device/sensor selection and pause/resume/timeout logging. Current ledger: 107 done, 195 todo, 51 not applicable. See `temperature-guard-verification.md` for actual SD1.5/SDXL native Generate/Interrupt/recovery checks, exact guarded/off RGB parity across txt2img/img2img/hires and AMD fixture-only limits. Whole Python regression passed; whole Scala regression passed 1264 tests with one skip. The real Windows bulk scanner takes several minutes; preserve its live process rather than restarting for quiet logs.

The active twenty-issue release progress is tracked in `twenty-issue-release-progress.md`. No release has yet been published for that goal. The separate temporary worktree `sdnui-temperature-native-check` was used for native acceptance while the main checkout's regression ran; its product sources matched the current checkout. Canonical changes and commits remain in this checkout.

## 2026-10-30 update: color correction

Index 261 / issue #15 now has native Extras and script/plugin `color_correction`: A1111, independently implemented AdaIN and Wavelet, optional original-input reference, independent comparison outputs and strength. Current generated ledger: 106 done, 196 todo, 51 not applicable. Read `color-correction-verification.md` for real SD1.5/SDXL and native button checks, broad regression results and numerical/visual limits. This is one issue toward the active twenty-issue release request; no new release has been published.

The GPU/test venvs were recreated under `python/`. GPU runtime uses torch 2.7.1+cu128 and torchvision 0.22.1 with pinned requirements. Run whole Python tests with `CUDA_VISIBLE_DEVICES=-1`; actual model verification runs with GPU visible. License collector accepts only LF/CRLF variants of trusted pinned content, retaining digest checks. Existing test fixtures were updated for wdBatch and current startup/provenance contracts.

Written 2026-10-26 when work moved from Claude Code to Codex. Read this first, then `AGENTS.md`, `README.md`,
`docs/PLUGINS.md` and `docs/EXTENSIONS.md`.

## Current product naming

The product is **Stable Diffusion NUI**, repository `kmizu/stablediffusion-nui` (formerly Stable Diffusion FX / SDFX).
Current builds produce `target/stablediffusion-nui.jar`; the v0.03 Windows package uses
`SDNUI/SDNUI.exe` and `SDNUI-v0.03-windows-x64.zip`. Internal Scala/Python `sdfx` names,
`SDFX_*` environment variables and existing settings stay compatible. Earlier release names,
artifact hashes and verification evidence are historical records and must not be renamed retroactively.
The naming change does not by itself establish that v0.03 acceptance or publication is complete.

## The goal

The owner's request (Japanese, verbatim): 「他のやつでreforceのWeb UI（available）からインストールできるやつは全部コミコミにしたい」
and 「もちろん各プラグインのライセンスで必要なのがあったら置いてな」.

Bundle every extension listed in reForge's *Extensions > Available* tab (the A1111 extension index, 353 entries,
snapshot in `docs/extensions-index.json`) into this app as built-in plugins, and ship the licence notices each one
needs. The owner also asked for **high-quality implementation and UI** (「high qualityな実装とUIを頼むで」): quality wins over
count. Every new plugin is GPU-checked on real models before it is marked done.

The app itself is complete and working: Scala 3 + JavaFX client, Python diffusers engine over stdio JSON Lines, MCP
server, 40 rewrites of the reForge built-ins. Repo: private GitHub `kmizu/stablediffusion-nui`, branch `main`.

## Where things stand

| | Count |
|---|---|
| Index entries done | 105 |
| Index entries todo | 197 |
| Not applicable (web-UI CSS/UX only, accounts, obsolete, external apps, server ops) | 51 |

**Latest verified wave (2026-10-27):** index 122 `core:temporal-lab` provides DFI, deflicker, fuse and frame extraction in a native workspace. The integrated CPU worker, native UI, model-image smoke, Scala and Python checks are recorded in `docs/VERIFICATION.md`, together with source attribution and limitations. The remaining extension issues and final release are separate ongoing work.

**Previous verified wave (2026-10-27):** index 333 `core:nsfw-card-blur` is accepted as native card
previews with shared Show/Blur/Hide. The reduced core gate passed offline, CPU dry-run and actual SD1.5/SDXL
checks; the broad Scala run's two fixture timeouts passed focused serial rechecks. See
[`VERIFICATION.md`](VERIFICATION.md) for precise scope, evidence and limits. For the Windows v0.01 runtime ZIP, release tag and remaining-issue tracker, check the GitHub release and issues.
The requested stopping point is after the download release and issue tracking; this does not complete the
353-entry goal.

**Previous verified wave (2026-10-27):** index 30 `core:config-presets` and index 31 `core:preset-utilities` have
separate native profiles. Config uses mode-specific collections, selected-field overlays and Reapply; Utilities
uses a shared collection, Quick/Detailed save and a choice of current versus component-default apply. The
complete MIT Config starter presets are included, while the unlicensed Utilities source supplies behaviour only.
The corrected implementation passed **964 Scala tests**, **36 native pane checks**, **16 CPU** checks and **19 each**
on real SD1.5 and SDXL models, with trained ControlNet. Python passed 867/1 skip and no-torch 397/82 skips; the
assembled jar passed an isolated Java 21 launch. BuiltinsSuite 20/20 and NativePresetsCoverageSuite 4/4 passed
after the precise two-entry ledger promotion. See [`VERIFICATION.md`](VERIFICATION.md) for the source pins,
artifact hash, parity limits and review history, and
[`native-presets-compatibility.md`](native-presets-compatibility.md) for interchange rules.

`scripts/extensions_matrix.py` is the single ledger. Its `S` dict maps each index entry to `done:<plugin id>`,
`todo:<target>` or `na:<reason>`, and the `IMPLEMENTED` set promotes `todo:<target>` to done. Several entries may
share one target (for example every background remover maps to `remove_background`). To merge a target into another,
edit its `S` entry (done for `pixelization -> pixel_art`, `mask2background -> inpaint_background`).

**Previous verified wave (Codex,2026-10-27):** index253 `core:api-payload` is implemented. Each generation mode has a
collapsed entry to one shared native inspector, with exact submitted request bodies, separate submitted/model
context and optional bounded image diagnostics. Script sub-generations have distinct identities and terminal
states. History, tree traversal, raw previews and diagnostic delivery are bounded; explicit Copy preserves the
whole selected document. Original init-image contents remain omitted even when image diagnostics are enabled.
Final source passed725 Scala tests,867 Python tests/1skip and397 noTorch tests/82skips, assembly, an offline
native pane check and actual CPU18/SD1.5 14/SDXL14 native acceptance checks. Both model runs preserve exact fixed-seed
RGB/body parity. Safe clipboard checks passed and restored the formats captured in these runs. Read the precise
limits, source pins, artifact hash and paths in `docs/VERIFICATION.md`; the353-entry goal remains unfinished.

**Previous verified wave (Codex,2026-10-26):** index320 `core:prevent-interrupt` and index309 `core:hardware-info` are
integrated. The footer protection is process-global/defaultOff and covers actual native/MCP running cancellation
and explicit user restart without blocking shutdown/health recovery or pending-only clearing. Hardware metadata
has native opt-in/remark controls, compact actual-device facts, per-image batch timing, Extras provenance and
bounded informational Restore/Send-to notices with no offline engine startup. Source behavior and original
license limitations are recorded in the owned plans and128-project notices; no unlicensed upstream code was copied.
Final integrated checks:596Scala tests plus assembly,867Python/1skip,397noTorch/82skips, six actual native helpers
(CPU+SD1.5+SDXL for each feature), all successful. Reports/logs and precise limits are in `docs/VERIFICATION.md`.
The completed API payload plan is `docs/superpowers/plans/2026-10-26-core-api-payload.md`.

Done in this session, all GPU-verified unless marked:

- Prompt and sequence scripts (15): dynamic prompts with wildcards, One Button Prompt, seed travel, shift attention,
  LoRA queue and others.
- A1111 prompt editing `[a:b:when]` and alternation `[a|b]`: `python/sdfx/prompt_schedule.py`, with per-step
  embedding swaps in `Engine._schedule_swapper`.
- Sampling hooks (9) in `python/sdfx/sampling_ext.py`: Detail Daemon, Vectorscope CC, Latent Mirroring, Asymmetric
  Tiling, Anti-burn, ReSharpen, Diffusion CG, CFG Schedule, CADS.
- Ultimate SD upscale (`UltimateSdUpscalePlugin.scala`).
- ADetailer (`python/sdfx/adetailer.py`, `Engine._adetailer`, `AdetailerPlugin.scala`), including LAB colour
  matching because baked VAEs lose contrast on the encode/decode round trip.
- LoRA Block Weight (`python/sdfx/lora_blocks.py`, `LoraBlockWeightPlugin.scala`). The `<lora:name:te:unet>` multipliers
  and the `lbw=`, `start=`, `stop=` and `step=` arguments are parsed in `LoraSyntax.parseTag`.
- Checkpoint merger and converter (`python/sdfx/model_tools.py`, `ModelToolPlugins.scala`). Both are blocked over MCP.
- **GPU-verified by Codex on 2026-10-26:** image tools in `python/sdfx/image_tools.py` and `ImageToolPlugins.scala`.
  They cover remove background (rembg or person segmentation), person mask, pixel art, colour enhance, FSR-style CAS
  upscale, depthmap2mask and inpaint background. Real SD1.5 and SDXL images, both masked inpainting paths, and all
  seven plugins through the real Java/MCP app were checked. See `docs/VERIFICATION.md` for scope and evidence.
  Fixed full-range seed overflow and output dimensions in pixel art, an open input file in the engine, and a leaked
  auxiliary mask in the Scala background-inpainting plugin. The 217 todo
  entries remain unfinished. Completing this verification wave does not complete the extension-index goal.
- **Seamless tile inpainting (Codex):** `SeamlessTilePlugin.scala`, id `seamless_tile`. Circularly offsets the input,
  inpaints the central seam in horizontal/vertical/both directions, optionally restores the original arrangement.
  SD1.5 and SDXL checked through the actual app/MCP; repeated tile previews and the form inspected. Neural seam
  repair quality depends on the checkpoint and texture; an ordinary SD1.5 anime checkpoint failed to improve a stone
  texture in some cases. See `docs/VERIFICATION.md`. No new dependency or Python engine hook.
- **Weight Gradient (Codex):** `python/sdfx/weight_gradient.py`, `WeightGradientPlugin.scala`, hook/id `weight_gradient`.
  Full documented syntax with linear/exponential/circular easing, return ramps, step ranges, positive/negative
  prompts, A1111 editing and per-pass clocks. Real SD1.5/SDXL, normal/hires/img2img/Alternative/Heun paths checked;
  constant tags match ordinary weights exactly, and disabling restores the baseline. Actual app-service pipeline
  and UI checked. PhotoMaker combination is explicitly unsupported. No upstream code was copied (no licence).
  See `docs/VERIFICATION.md` and the implementation plan for tested scope and decisions.
- **img2img Alternative quality repair (Codex):** preserves the amplitude recovered by DDIM inversion instead of
  forcing every result to empirical std 1. Noise blending preserves its two endpoints; Randomness=1 matches ordinary
  same-seed generation (matching settings, variation disabled). Decode CFG0/.5 now interpolates correctly, and latent
  inversion explicitly disables pixel-range clipping. Replay infotext now includes original prompts and effective
  decode steps. SD1.5/SDXL real-engine grids, actual app/MCP path and UI were checked. See VERIFICATION.md for limits.
- **NegPiP (Codex):** `python/sdfx/negpip.py`, `NegPipPlugin.scala`, hook/id `negpip`. Separately encodes negative
  terms at absolute weight, preserves attention keys, flips projected values, and masks added padding. Supports
  positive/negative branches, editing, Weight Gradient crossings, batching, PAG, hires, img2img and Alternative.
  SD1.5/SDXL GPU checks include exact no-negative/no-hook restoration; actual app-service path and UI inspected.
  Excessive negative weights can destroy subjects/style. PhotoMaker with negative terms is unsupported. AGPL-3.0
  upstream attribution is collected; no new dependency. See VERIFICATION.md for untested combinations.

UI improvements made this session: `Control.Section` (collapsible groups inside a plugin form), `Plugin.enabledBy`
(greys a control out while its checkbox is off), and the form layout in `W.Form`. Labels longer than 18 characters
go above their control, and the label column stays at least 88 px wide so cards line up.

**Parallel implementation wave (2026-10-26):** `prompt_fusion`, `tokenizer` and `save_intermediate` are integrated.
Prompt Fusion has source grammar/geometry, SDXL pooled conditioning, ordinary-prompt bypass and replay checks.
Tokenizer uses the actual loaded model vocabularies in a native text/ID inspection tab (literal input, not a
simulation of generation conditioning). Intermediate PNG/video artifacts persist separately from final images;
root `outputs/intermediates` is excluded from gallery scans. Both model families, native forms and actual app
queue/replay paths are verified; optional ffmpeg codecs/recipes have independent actual encoding checks. See
the three implementation plans and the final section of `docs/VERIFICATION.md` for exact limits and artifacts.

**Second parallel wave:** `grid_numbers` and `latent_upscale` are accepted after real SD1.5/SDXL, native app/replay,
actual PNG Restore/Send-to, rendered forms and aggregate checks. Grid annotations preserve source images and record
only visible tiles. Latent Upscale encodes before resizing and executes the full requested denoising count.
See the closing VERIFICATION section and their plans.

**Third parallel wave:** `composable_lora` and `core:system-info` are now accepted after source-scope review,
SD1.5/SDXL generation, actual native queue/replay and rendered controls. Composable's source controllers and all
adapter algebra families are implemented; real LoRA/LoCon/LoKr have model evidence, while unavailable trained
LoHa/full/IA3/CP formats have CPU algebra coverage. Keep that distinction. Plots are source-compatible appended
images; trace JSON persists outside `.tmp`. System Info has live diagnostics, native benchmark/history, explicit
metadata selection and preview/submit separation. No cloud submission was sent during acceptance. See both plans
and VERIFICATION for exact scope, paths and limits.

**Fourth native wave:** `core:styles-editor` covers indices163 and248 after SD1.5/SDXL native editing,
data-preservation, category/backup/AES interoperability and exact named-style conditioning checks. The manual
helper's clipboard cleanup failure was fixed before clean final acceptance; it now skips clipboard mutations
when any advertised format cannot be preserved, and always attempts all resource cleanup. See VERIFICATION.

**Native comparison/VAE wave:** `core:image-compare` and `vae_tools` are accepted after independent source/fix
reviews, actual native SD1.5/SDXL image workflows and integrated app/MCP runs. Comparison needs no inference;
it compares existing real-model images without changing them. VAE Tools has Full/Approx NN and pretrained
TAESD/TAESDXL evidence, exact failed decoder-input recovery, quantized image formats and explicit download gates.
Keep codec roundtrip, image reconstruction and approximate previews distinct. See their plans and VERIFICATION.

**Current wave:** the native presets plan for indices 30/31 is complete. Do not restart it as the old shared
`core:presets` target: that name would also mark three unfinished entries done. Index 333 `core:nsfw-card-blur` is implemented and accepted in the native card preview wave.
The 353-entry goal remains active: **104 done, 198 todo, 51 not applicable**, with 138 distinct pending
targets after this promotion. Indices 203 `controlnet_fastload`, 221 `scenario_loader`, 262
`core:state-manager` and 146 `model_presets` remain separate work. See VERIFICATION for core acceptance,
limits and exact evidence. Finish the Windows v0.01 runtime ZIP and release, then document all remaining issues and stop;
do not implement the remaining 198 entries as part of this wave. The independent `core:prompt-tools` audit is under
`outputs/next-core-prompt-tools-audit*`.

For the completed presets wave, the five-entry source audit and manifest remain in main
`outputs/next-core-presets-audit*`, with the full ID map at `outputs/native-config-presets-field-map.md`.
The 132-project notice retains all 130 earlier rows; trained ControlNet assets are local under
`outputs/model-cache/controlnet-acceptance-20261027` and are not distributed. Detailed source reviews, earlier
failed checkpoints, final model reports, Java 21 screenshots and hashes are in [`VERIFICATION.md`](VERIFICATION.md).
The earlier API payload release reports and assembled jar are in `.worktrees/api-payload-integration/outputs`
and `target`.

Neutral Prompt is now marked implemented after an explicit original-scope
acceptance audit; see `docs/superpowers/plans/2026-10-26-neutral-prompt.md` and the closing VERIFICATION section.
All operators, nesting/signed weights, CFG rescale, schedules, native form/formatter/replay and actual AppServices
SD1.5/SDXL generation are verified. Alternative, ControlNet/PAG, IP-Adapter/basic/Plus, Reference, SDXL ControlLLLite,
CADS/CFG Schedule and ordinary PAG batch fixes also have separate tests/matrices. Preserve their invariants.

**Compatibility backlog:** PhotoMaker, SAG, APG/Mahiro/Latent Modifier/Dynamic Thresholding/Detail Daemon with
composed Neutral prompts remain explicitly unsupported in `neutral_runtime.validate`, tests and form help. The
original Runtime task permits clear unsupported combinations; later agent-added notes making every combination
a promotion prerequisite are superseded by the documented acceptance clarification. No original Neutral operator
or model/UI/notice gate was omitted. Keep the seven guards until their own implementations are verified.
The overall 353-entry goal remains active; 198 entries are unfinished. Do not treat this promotion as
completion of that goal or reclassify remaining entries to make it appear complete.

## Remaining targets, most-starred first

sadtalker, facechain, easyphoto, tiled_diffusion, face_swap, segment_anything, animatediff, core:prompt-tools,
deforum, core:tag-autocomplete, mov2mov, core:compile, prompt_library, temporal_kit, train_dreambooth, depth_maps,
regional_prompter, openpose_editor, core:localization, wav2lip, text2video, inpaint_anything, ebsynth_utility,
stable_sr, cn_animation, unprompted, llul, dataset_tag_editor, deoldify, dreamartist, qr_code, infinite_zoom, tipo,
lcm, stylepile, clip_interrogator, prompt_generator, style_selector, aesthetic_gradients, latent_couple,
outpaint_canvas, fabric, manga_maker, model_downloader, multi_subject_render, batch_face_swap, video_loopback,
controlnet_fastload, scenario_loader, core:state-manager, model_organizer, model_keyword, shap_e, prompt_travel, lora_control,
gif2gif, replacer, cd_tuner, riffusion, smart_process, haku_img, infinity_grid,
core:panorama-viewer, loopback_scaler, daam, text2prompt, prompt_translate, prompt_gallery,
refiner, incantations, old_photo_restoration, abysz_lab, aesthetic_score, steps_animation,
nsfw_checker, image_editor, embedding_merge, rich_text, mosaic_outpaint, cross_attention_viz, hypernetwork_training,
inspiration, vram_estimator, sonar, dump_unet, llm_prompt, training_picker, model_presets,
lama_cleaner, kandinsky, hypernetworks, characteristic_guidance, hotshot_xl, live_portrait,
booru_prompt, depth_image_io, creaprompt, embedding_editor, semantic_guidance, custom_diffusion,
encrypt_image, inpaint_difference, depth_library, rpg_diffusion, custom_hires_fix,
img2img_hires_fix, depth_anything, agent_attention, conditioning_highres_fix, hires_fix_tweaks,
panorama_tools, chara_searcher, lua_scripting, deepdanbooru, gpu_temperature_guard, corridor_crawler,
old_sd_firstpasser, temporal, batch_interrogate, token_downsampling, embedding_inspector, demofusion,
prompt_expansion, color_correction, video_extras, clip_io, face_manipulation, ditail,
minecraft_schematic, img2img_workflow, ocr, controlnet_preprocess_extras, triposr, manga_inpainting,
pnginfo_injection, masked_content_extras, metadata_utils, waifu2x

Suggested next steps, by value and fit with what exists:

1. **Native tools**: `core:prompt-tools` and the remaining separate preset-audit targets still need
   implementation and source-scope verification.
2. **Regional prompting**: regional_prompter and latent_couple, via attention-level region masks.
   tiled_diffusion can largely reuse the existing `multidiffusion` hook plus noise inversion and region prompts.
3. **Small, cheap ones**: img2img/custom/conditioning hires fix variants, color_correction,
   waifu2x, deepdanbooru (the WD14 tagger already exists in `extras.caption_image`), batch_interrogate,
   clip_interrogator, aesthetic_score, nsfw_checker.
4. **Big ones**: animatediff, segment_anything, inpaint_anything, deforum, text2video, face_swap (insightface is
   installed in `.venv-smoke`), training (dreambooth, hypernetworks).
5. **`core:*` targets**: these are web-UI features with an equivalent in the JavaFX UI. Tag autocomplete goes in the
   prompt editor. Localization should read A1111 localization JSON.

## How a new plugin is built (follow the existing patterns)

- **Engine side**: add a module under `python/sdfx/`. A sampling hook is wired from `extensions.install()` (UNet
  forward/pre-hooks and `callback_on_step_end` through `_chain_callback`). A non-generation tool is a `BaseEngine`
  method and must be added to `QUEUED_TYPES` in `server.py` (see `image_tool`, `merge_models`). Every new hook name
  goes in three places: `hooks.KNOWN_HOOKS`, `Engine.features()` and `dryrun.features()`.
  `tests/test_extensions.py::test_known_hooks_cover_extensions` checks this.
- **Scala side**: AlwaysOn plugins extend `HookPlugin` (`hookFields`, `infotextKeys`, `xyzAxes`). Scripts extend
  `Plugin` with `run(p, values, ctx)` and Extras extend `ExtrasPlugin.transform`. Register each in
  `StandardPlugins.scala`.
  - **Plugin id rule**: the plugin `id` must equal the matrix target name. `BuiltinsSuite` checks non-core
    `done:<id>` entries in `docs/extensions-status.json` against registered plugins; `core:*` entries are not inference plugins.
  - **Infotext**: use the extension's own keys where it had them, so infotext stays compatible.
- **Writing quality code**: re-implement from documented behaviour. Copy code only from permissively or AGPL/GPL
  licensed sources, and never from repositories without a licence (LoRA Block Weight was done from its README only).
  Write pytest for the math, munit for the Scala side, then GPU-check.
- **Ledger, docs, licences**: add the target to `IMPLEMENTED`, then run `python scripts/extensions_matrix.py` (rewrites
  `docs/EXTENSIONS.md` and `docs/extensions-status.json`). Then run `python scripts/third_party.py` (needs a logged-in
  `gh`; fetches licences into `licenses/` and rewrites `THIRD_PARTY_NOTICES.md`). New optional Python dependencies
  go in the `BASE` list there and as a commented line in `python/requirements.txt`. Do not pin them as hard
  requirements.
- **Commits**: one commit per wave. The message says what was added and how it was verified.

## Environment (Windows 11, RTX 3090, about 9 GB free RAM, so watch memory)

- **sbt**: `%LOCALAPPDATA%\Coursier\data\bin\sbt.bat`. Not on Git Bash's PATH; call it by full path.
  - All tests: `sbt.bat -batch test`.
  - One suite: `sbt.bat -batch "plugins/testOnly sdfx.plugins.AdetailerSuite"`.
  - Fat jar: `sbt.bat -batch ui/assembly` writes `target/stablediffusion-nui.jar`.
- **Java 21**: `C:\Program Files\Java\jdk-21\bin\java.exe`. The `java` on PATH is 17 and is too old for the jar.
- **Python venvs** (in `python/`):
  - `.venv-smoke` is the GPU venv: torch 2.9 cu128, diffusers, ultralytics, mediapipe, insightface, rembg. Use it
    for real-model checks and the full pytest run.
  - `.venv-test` has no torch. It checks that the torch-free parts and the dry-run engine still pass.
- **Checkpoints**: `D:\stable-diffusion-webui-reForge\models\Stable-diffusion`. SD1.5 test model `animekawa_v10`,
  SDXL test model `waiIllustriousSDXL_v170`.
- **LoRAs**: `..\models\Lora`. `20_il-shirobana.safetensors` is SDXL.
- **Shell gotchas**:
  - A token-saver hook blocks long heredocs into `.scala` files, so write them with a file editor.
  - Python text-mode writes on Windows give CRLF. Always write with `newline="\n"`; `.gitattributes` enforces LF.
  - Do not restart a background job the OS killed for low memory unless the owner asks.

## Verifying

| What | Command / file |
|---|---|
| Scala tests | `sbt.bat -batch test` (latest: 725 passed, 2026-10-27) |
| Python tests | `python/.venv-smoke/Scripts/python.exe -m pytest` with `CUDA_VISIBLE_DEVICES=-1` (867 passed, 1 skipped) and `.venv-test` (397 passed, 82 skipped), 2026-10-27 |
| Dry-run smoke | `scripts/smoke.ps1` / `scripts/smoke.sh` |
| Engine on GPU, one case per process | `scripts/dev/gpu_check.py <venv python> <out dir> <case>`; cases `sched_*`, `ext_*`, `ad_*`, `image_*`, `weight_*`, `negpip_*` |
| Script plugins through the real app over MCP | `scripts/dev/write_settings.py`, then `scripts/dev/mcp_scripts.py "C:\Program Files\Java\jdk-21\bin\java.exe" <script ids>` |
| LoRA Block Weight / merger checks | `scripts/dev/lbw_check.py`, `scripts/dev/merge_check.py` |
| UI screenshot | `java -jar target/stablediffusion-nui.jar --dry-run --home <dir> --screenshot out.png --exit-after 8` (also writes `out-panel.png`, the full form). Expand a plugin by putting `{"expanded": {"txt2img.plugin.<id>": true}}` in `<home>/config/ui-state.json` |

The `scripts/dev/` files contain absolute paths of this machine; adjust them if anything moves. `write_settings.py`
creates `config/settings.json` (gitignored); delete it afterwards.

Look at the images, not only the numbers. Vectorscope and Diffusion CG were "working" by every metric but visibly
broken (pseudo-inverse colour directions were too large). The ridge-regularised `rgb_to_latent_dir` fixed them.

## Known issues and decisions

- **IP-Adapter pairing quality**: native basic SDXL IP-Adapter at scale .6 visibly shifts style/contrast on
  waiIllustriousSDXL_v170, even without Neutral Prompt. No-op/restoration remain exact. The basic/Plus grids in
  `outputs/verify-neutral-ip-*` record this limit; compatibility checks do not promise a fixed artistic result.
  Model-offload reload was checked on SD1.5; SDXL reload was checked in GPU mode, not sequential offload.

- **img2img Alternative reconstruction limits**: the strong unintended contrast was traced to empirical noise
  normalization and repaired. Same SDXL DPM++ CFG6 case: near-white 14.19% -> 4.51%, near-black 28.67% -> 14.16%.
  Reconstruction still depends on CFG/count/sampler: matched DDIM/CFG1 was close to source (1.06%/1.71% versus
  1.11%/1.61%), not pixel-identical. It is a DDIM approximation, not the legacy k-diffusion Euler/sigma-adjustment
  implementation. Changing settings can still change contrast/composition. No output recolouring was added.
  Reports: `outputs/verify-alternative-final/alt_quality_{sd15,sdxl}.json`; diagnostics and test plan preserve the
  rejected clipping/CFG hypotheses. Pixel-range clipping is guarded but was already off in this particular model.
- **Weight Gradient clocks**: each sampling pass restarts the ramp and uses its actual logical inference steps
  (second-order callback stages do not advance twice). Existing A1111 prompt-editing hires offsets are unchanged.
  PhotoMaker uses separate text-only encoding; combining it with gradient tags raises an unsupported error.
- **NegPiP strength/compatibility**: SD1.5/SDXL tested; SD2's CLIP route is implemented but not GPU-checked here.
  Strong negative values can remove the subject or break the image. The wrapper preserves native attention
  processors and parameter paths; real ControlNet/IP-Adapter weights together with offload remain unverified.

- **LoRA residue in fp16**: once a LoRA has been loaded into the pipeline, the UNet output differs by about 0.002 in
  fp16 even after the LoRA is disabled or unloaded. Weights, hooks, flags and the memory layout were all verified
  unchanged; the cause is not found. With ancestral samplers this changes a later no-LoRA image with the same seed
  (mean pixel difference about 27). It is not caused by the LoRA Block Weight code. It is worth a look if exact
  reproducibility across a session matters.
- **ADetailer default differs from the extension**: colour correction is on by default, a deliberate quality
  choice. YOLO needs `ultralytics` (optional); `opencv_face` works without it.
- **Model tools and MCP**: the merger and converter write next to model A or the source model and are GUI-only
  (`McpTools.GuiOnlyScripts`).
- **Pixelization**: the A1111 pixelization extension uses a neural model; our `pixel_art` is algorithmic and covers
  both targets.
- **Licences**: the app is AGPL-3.0 because it ports AGPL (A1111, Forge, reForge) and GPL (ComfyUI, Fooocus) code.
  `THIRD_PARTY_NOTICES.md` explains which projects were ported, re-implemented, depended on or used as data.


### Temporal Lab delivery candidate

Temporal Lab implements reference-guided DFI, deflicker, fuse, frame extraction, previews and playback in the native UI. Source algorithms and controls were informed by AbyszOne/Abysz-LAB-Ext at `deff0d4b4a208f862299b14b93e088f613373153`; the pinned GPL-3.0 license is archived under `licenses/AbyszOne__Abysz-LAB-Ext.txt`, and the notice collector records the source. See `docs/VERIFICATION.md` for bounded evidence and limits. Inter Blur is active here, video frame timestamps are computed from frame indices, and complete results use a new directory; full numerical parity with the upstream extension is not claimed.

This isolated issue candidate was assembled from the reviewed Temporal Lab work. Its final integrated checkout and regenerated notices/extension matrix require their own verification before release. Avoid concurrent hostile replacement of path ancestors during processing; video codec allocations can occur before the decoded-frame guard, so the working-memory estimate is not a hard process-memory cap.

License collector first attempt exited1 fetching unrelated pinned AbyszOne/Abysz-LAB-Ext LICENSE; retry session30025 is live. Inspect the retry handle/log before restarting or reporting success.

Current source template format reminder: hr_prompt_mode.get_mode_info uses h=[mode,raw_hr_prompt,optional_remove], n=[mode,raw_hr_negative], a=append_separator, p=prepend_separator, c=marker, quote-swapped JSON. Existing single separator field needs distinct append/prepend values for full source template compatibility. Read source before implementing parser; source also controls save_template and restore_template preferences.
