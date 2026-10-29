# Color Correction Extras: issue #15 investigation

Status: investigated 2026-10-30; not implemented or accepted. This does not promote index 261.

## Authoritative source

- Issue: https://github.com/kmizu/stablediffusion-nui/issues/15
- Upstream: https://github.com/light-and-ray/sd-webui-color-correction-extras
- Examined revision: `0d86f5a55d5648ca693584a1078761cf8ae26395`.
- The README describes bringing A1111 native color correction into Extras, and optional StableSR color correction methods in a Method dropdown.
- GitHub repository metadata reports `license: null`; the root contents have no license file. This is not permission to copy implementation. Use documented behavior and separately licensed algorithm sources; verify model/dependency notices before acceptance.

## Current implementation gap

`ColorEnhancePlugin` stretches chroma, controlled by strength and percentile. It has no reference-image control and is not reference color matching. Do not count it as satisfying this issue.

`python/sdfx/image_tools.py` currently exposes pixel_art, color_enhance, cas_upscale, person_mask, remove_background and depth_mask. No reference matching tool is registered. The native Control API already has ImageFile, so a reference picker can use existing infrastructure.

## Proposed implementation boundary

### Primary-source behavior confirmed

Read-only inspection of upstream `scripts/color_correction_extras.py` at the pinned revision establishes additional requirements omitted from the initial README summary: an absent sample uses the original Extras input captured before processing; optional comparison outputs run every other available method against the same target, rather than chaining methods. Wavelet and AdaIN resize the sample to target size. These are behavior observations, not permission to copy the unlicensed source.

A1111 `modules/processing.py` at `82a973c04367123ae98bd9abdf80d9eda9b910e2` is AGPL-3.0 according to its license endpoint. Its setup converts the reference to OpenCV LAB. Application matches channel histograms in LAB with scikit-image, converts back, then uses a luminosity blend with the original image. Reproduce and verify that blend's argument order and conversion/quantization; a mean/std transfer does not provide parity.

The existing `adetailer.match_colors` is LAB mean/std transfer despite its docstring mentioning A1111 color correction. Reusing it for the new A1111 method would be incorrect. Its existing generation behavior should remain stable; correct the misleading documentation separately when implementing.

The new Extras implementation therefore needs access to the **original pipeline input**, not merely the current transform input, to honor the no-sample fallback after preceding upscalers or restoration. A required sample picker would narrow the source behavior and is not sufficient for acceptance.

Inspection of `ExtrasPlugin.run` shows that existing script plugins process one loaded init image (or generated result) independently, then save outputs. This is not a shared multi-stage Extras chain. Consequently, integrating only a new script would not establish native Extras-tab parity. Inspect and extend the actual `ExtrasTab` execution path as well; do not add a speculative global mutable reference to PluginContext. The inaccurate A1111 comparison in `adetailer.match_colors` has been corrected in its docstring only, preserving numerical behavior.

Provide a separate Extras plugin with id `color_correction`, a reference image picker, method selector and strength. Preserve input dimensions; report the selected method and reference provenance in output metadata. Validate missing or unreadable references before work. Keep reference temporary files alive until the engine finishes and clean them on success and failure.

Before choosing methods, establish A1111's documented color-space/blending behavior and the optional StableSR methods from licensed primary sources. Do not substitute chroma enhancement or label a simple RGB histogram match as A1111 parity. Any unsupported method must be explicit rather than silently falling back.

## Acceptance work still required

### Incremental algorithm evidence (not acceptance)

Real-engine SD1.5 check completed successfully with `scripts/dev/gpu_check.py python/.venv-smoke/Scripts/python.exe outputs/verify-color-sd15 color_correction_sd15`. The engine generated separate warm/cool teapot images using animekawa_v10 and processed all three methods. Source, reference, method outputs, JSON and stderr are retained in that directory. `grid.png` was visually inspected: target composition remains recognizable; A1111 and AdaIN transfer overall color, while Wavelet produces conspicuous local brightness/halo changes when reference geometry differs. Do not describe these methods as equivalent or promise artifact-free transfer. The GPU environment focused suite also passed 27 tests. SDXL and actual native button interaction remain outstanding; Python engine success alone does not close issue #15.

The corresponding `color_correction_sdxl` run also completed successfully with waiIllustriousSDXL_v170 (768x768, 16 steps). `outputs/verify-color-sdxl/grid.png` was visually inspected: all methods retain the source composition and shift toward the blue reference; Wavelet has strong spatial halos and bright smearing on the window and wall. The UI now explains this method difference and RGB output. Native button interaction remains outstanding.

Subsequent native acceptance: `RealColorCorrectionCheck` passed with both SD1.5 and SDXL source images and the real Python engine (not dry-run). The strengthened SDXL run verified each selector method against saved PNG metadata, three distinct output files, and input-reset cleanup. The helper starts the engine explicitly before firing the actual button. Earlier failed attempts were harness startup errors; they do not prove a product failure. Explicit reference interaction and failure recovery still need wider native coverage. Focused GPU-environment tests now pass 30 cases.

Broad Python and Scala regression runs were started afterward. Both have reported failures and are not accepted as passing. Scala reports include local drive probe timeouts in ModelBrowserStateSuite and ModelPreviewCardStateSuite; do not dismiss them without isolated reproduction. The broad Python run's final failure details are pending. Do not promote the ledger or close the issue on the basis of focused tests while this audit is unresolved.

Additional 2026-10-30 checks: native UI compilation and BuiltinsSuite 20/20 passed; ImageToolsSuite 2/2 passed including fallback/comparison cleanup; ui/assembly produced `target/stablediffusion-nui.jar` (sbt-reported hash `b19a455481b607a8ab333c5c9aa3c6cfd8034694`). The current focused Python suite has 16 passing cases. These do not establish rendered UI or real-model acceptance.

Notice regeneration was attempted but failed on the existing pinned Config-Presets license hash (`Zyin055/Config-Presets@e252e60288b1104a45ea12106d92a8d145364fa1`). Diagnose the hash/line-ending mismatch before retrying; do not change the trusted pin merely to make it pass. The collector can write earlier license files before failing. The GPU venv from HANDOFF is absent; checkpoints exist on disk, but runtime/GPU availability must be checked before claiming model verification.

Diagnosis: the pinned API returns 1064 LF bytes, SHA-256 `e5deb76ee659f708965d5d2be291b43efc3ae2a996e116fb629fb8044620915e`; the archived license has 1085 CRLF bytes, SHA-256 `c19b98789c10ffe53568c46f4daae8a49024a02715cf2d4f05de04aa05d3a5f9`. A linewise diff has no content differences. The trusted pin currently describes the archived Windows line endings, so the collector needs explicit byte-preserving/canonical-line-ending handling and regression coverage, rather than bypassing verification. Live `nvidia-smi` confirms RTX 3090 with 24576 MiB; the remaining GPU setup gap is the Python environment, not absence of hardware.

`python/.venv-test` was absent and was recreated. The focused command `$env:PYTHONPATH='python'; python/.venv-test/Scripts/python.exe -m pytest python/tests/test_color_correction.py -q` now passes 19 tests, including all-method engine output persistence and corrupt-reference INVALID_PARAMS handling. All three methods are dispatched through image_tools. Native controls were rendered with a Ready dry-run engine in `outputs/verify-color-extras-ready.png`; that screenshot predates the output selector and does not establish successful processing through button interaction. The current output-selector source compiles, but needs updated visual and interaction acceptance.

The collector now verifies the existing trusted digest against exact bytes or LF/CRLF variants only, with regression coverage rejecting content changes and missing final newlines. Full regeneration completed for 140 projects, including the optional scikit-image and BlendModes notices. This resolves the earlier Config-Presets collection failure without replacing its trusted digest.

StableSR revision `398ee9383777e255540ea027a704c8ce1f32145b`, `scripts/wavelet_color_fix.py`, describes five dilated binomial smoothing levels, replicated borders and low-frequency reference replacement. The implementation here independently expresses that mathematical operation as separable NumPy filters; a direct 2D-kernel test agrees pixel-for-pixel on its fixture. StableSR's root license is S-Lab License 1.0 with a noncommercial restriction, not a permissive license; do not copy or bundle its code. Its script attributes the algorithm helper to pkuliyi2015/sd-webui-stablesr, whose original licensing and notices still need investigation. No upstream numerical parity is claimed: float precision, PIL resize choice and output rounding need explicit comparison. AdaIN now uses normalized sample variance plus epsilon 1e-5 and truncation, with a finite population-variance fallback for a one-pixel input. Wavelet currently rounds output. Both retain exact zero-strength bypass.

- Pin primary algorithm sources and license notices; decide exact supported methods.
- Implement native controls and engine integration, including meaningful boundary and cleanup checks.
- Verify identity/zero-strength behavior, reference distributions, unequal sizes, constant images and invalid parameters.
- Run actual native Extras on real SD1.5 and SDXL generated images and inspect before/reference/after grids.
- Record measured verification and limitations, regenerate notices and ledger only after acceptance, then commit/push the verified wave.

The wider active goal remains twenty resolved issues followed by a verified release. This investigation resolves none of them.
