# img2img Alternative inversion quality repair

> **For agentic workers:** Use superpowers:executing-plans and systematic-debugging for this wave.

**Goal:** Resolve the reproducible SDXL Alternative oversaturation before adding more extension targets.

**Scope:** `python/sdfx/extras.py::invert_latents`, `engine.py::_img2imgalt` blending; analytic tests in
`python/tests/test_img2imgalt.py`; Scala script help/replay fields and BuiltinsSuite regression;
`scripts/dev/gpu_check.py` / `alt_latent_probe.py` diagnostic cases and verification/handoff docs.
No dependency or plugin-id change.

## Evidence and decisions

- Starting commit 473ab0f, clean tree. Previous goal turn was progress (verified implementation and push).
- Real SDXL source: lightness std 24.25, RGB near-white fraction 1.11%, near-black 1.61%.
- Existing Alternative: DPM++ CFG6/decode1 gives 34.91 / 14.19% / 28.67%. Lowering CFG to 1/2, matching decode CFG,
  switching to Euler and DDIM all retain excessive clipping. CFG alone is not the cause.
- Installed DDIMInverseScheduler defaults to `clip_sample=True` when constructed from the DPM++ config. It clips
  predicted clean VAE latents to [-1,1], although valid latent values exceed that range.
- Analytic real-scheduler regression: epsilon=0 and latent=2.5 should return approximately 2.4054 after two steps;
  existing code returns approximately 0.9621. Missing/true clip settings fail; explicit false passes.
- Separate slider-range defect: decode CFG0/.5 is treated as CFG1 because guidance is only evaluated above 1.
  Four analytic failures (SD1.5/SDXL) show unconditional/interpolated predictions are ignored.
- Repair these causes without recolouring outputs, capping the user's CFG, changing sampler defaults, or changing
  the inversion algorithm. Preserve the current DDIM approximation and noise blending/normalization behavior.

## Diagnostic update (do not mistake an analytic bug for the observed root cause)

- Repeating the real grid after the clipping/CFG fixes produced exactly the old images for decode CFG>=1.
  Runtime probe confirmed this loaded model already supplies `clip_sample=False`; the explicit clipping fix covers
  other/missing configurations but is not the source of this particular oversaturation.
- Inverted SDXL noise has std 0.85673. Existing code scales it to std 1.0. Bypassing that empirical normalization
  reduced near-white pixels from 14.19% to 4.51% and near-black from 28.67% to 14.16%, without recolouring pixels.
  Source remains 1.11% / 1.61%, so this is progress in diagnosis, not proof that quality is repaired.
- Matching the inversion endpoint with trailing spacing made little difference; investigate unit conversion and
  reconstruction under matched CFG/count/sampler before changing production normalization.
- Probes are isolated monkeypatches in scripts/dev/alt_latent_probe.py; they do not change persisted engine behavior.
- Further controls: unnormalized inverse state with matched DDIM/CFG1 gives lightness std 24.64, near-white 1.06%,
  near-black 1.71% versus source 24.25/1.11%/1.61%. A 50-step inversion/sampling DPM++ CFG1 is similarly close.
- Ruling: remove empirical global-std normalization from the existing DDIM path. It changes meaningful inverse
  state amplitude and violates the exact Randomness=1 fresh-noise endpoint. Restore the original script's analytic
  blend denominator sqrt((1-r)^2+r^2), preserving both endpoints. Three analytic regressions failed before this
  change and all fourteen inversion/blending tests now pass. This supersedes the initial normalization-preservation
  assumption above; it is an amplitude/endpoint fix, not a pixel recolouring filter.

## Tasks

- [x] Reproduce parameter grid and analytic failures before production changes.
- [x] Disable pixel-range clipping for latent inversion; evaluate CFG whenever scale != 1. Eleven tests pass.
- [x] Repeat parameter grid on real SDXL and SD1.5; inspect images and clipping/colour metrics.
- [x] Verify the real app/MCP script path, default inversion settings and metadata.
- [x] Full Python suites, independent review, updated handoff/verification.

Integration: commit/push this verified repair wave to main under AGENTS.md; Git records the SHA.

## Verification requirements

Reproduce with `gpu_check.py <GPU python> <output> alt_quality_sdxl` / `alt_quality_sd15`.
Compare source and repaired outputs; successful requests alone do not establish visual repair.
Both endpoint/below-one CFG conditioning and SDXL pooled/time-id batch layout are covered analytically.
The upstream reForge script is reference-only; never import or launch the parent application.

## Final evidence

- Source + 12 real comparisons per model succeeded. Randomness=1 pixel MAE=0 on both models with matching settings
  and variation disabled. All comparison sheets inspected; high-CFG reconstruction remains approximate and is
  explicitly documented rather than represented as an exact inverse.
- Same SDXL DPM++ CFG6 near-white fraction 14.19% -> 4.51%, near-black 28.67% -> 14.16%; matched DDIM CFG1 gives
  1.06% / 1.71% versus source 1.11% / 1.61%. No pixel recolouring.
- Fixed missing Decode prompt/negative/effective steps. Scala regression first failed with missing Decode prompt,
  then BuiltinsSuite 18/18 passed; jar rebuilt. The actual MCP script path passed 1/1 for each model and saved all
  replay fields at the default 50 decode steps. Temporary settings restored/removed; isolated UI inspected, exit 0.
- Full Python: GPU venv 314 passed; no-torch venv 247 passed / 17 skipped. Independent math/metadata review found no
  important defect; stale inverse-helper docstring was corrected. Notice/dependency/index sets are unchanged.
- Previous handoff's quality follow-up is resolved to this measured scope; next work is remaining embedding hooks.
  Overall goal remains unfinished: 85 done / 217 todo / 51 na.
