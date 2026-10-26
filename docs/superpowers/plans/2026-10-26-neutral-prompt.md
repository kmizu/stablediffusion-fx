# Neutral Prompt implementation plan

> **For agentic workers:** Use superpowers:executing-plans task by task.

**Goal:** Fully implement extension-index target `neutral_prompt`, preserving AND, AND_PERP, AND_SALT, AND_TOPK,
nested groups, signed branch weights, CFG rescale and scheduled prompt leaves. Keep the target todo until real
SD1.5/SDXL, app/UI and notices gates pass; a parser/math library alone is not an implemented plugin.

**Architecture:** separate a torch-free expression tree/parser and torch-local denoised-image composition from
diffusers integration. Compose denoised x0 deltas relative to an unconditional prediction, not raw token embeddings
or epsilon arrays. Encode each leaf with the existing CLIP/scheduling code. Evaluate each leaf against the same
latent/timestep; preserve the ordinary single-prompt/no-rescale path exactly. Existing JavaFX HookPlugin supplies
enable, CFG rescale and inline syntax help; full source attribution accompanies adapted licensed code.

**Tech stack:** existing Scala 3/JavaFX and Python torch/diffusers; no new dependency.

## Inspected sources (MIT, 2026-10-26)

ljleb/sd-webui-neutral-prompt README and source:
- parser blob `60d997ecfbc2f6ac84a53302f41e9bbef47d0037`
- composition blob `a8605abaca5bc4374d2c4477e9b37e8c65ea49d4`
- prompt transpilation blob `fff7748bb7df68c122b3d39c3f19140d34d17e57`
- UI blob `e3a452b5cd64cf1b2112cf1274a1050cbee59d59`

All four features are required, not one representative keyword. Normal siblings sum weighted deltas. Auxiliary
siblings are resolved against that normal sum: perpendicular projection, per-element saliency winner or top-5%
absolute filtering. Saliency normalizes each vector's full C/H/W absolute values with softmax before comparing.
CFG rescale interpolates both global mean and std toward the unscaled conditional result. Operations are per image,
not across the batch. Upstream combines denoised x0; scheduler conversion needs explicit analytic tests.

## Tasks

- [x] Parser/math foundation RED/GREEN: port MIT parsing/tree semantics, nested groups and ordinary editing;
  tests for all operators, negative/group weights, zero normals, tiny/constant tensors, batch isolation and the
  upstream top-k threshold (95th-percentile kthvalue, includes ties). Reject non-finite weights, bound nesting.
  Files: `python/sdfx/neutral_prompt.py`, `python/tests/test_neutral_prompt.py`; upstream notice and MIT text.
- [ ] Runtime integration: Engine leaf encoding and adjustment, per-step schedules/hires, x0 conversion for
  epsilon/v-prediction and each supported scheduler representation, conditional/unconditional batching, output
  reconstruction, cleanup after errors/cancel. Tests precede hooks; confirm no-negative ordinary no-op exact.
  Inspect all forward-hook interactions; unsupported combinations must be clear rather than silently ignored.
- [ ] Scala plugin, registration, help, replay metadata and supported-features advertisement. Formatter must offer
  all keyword/weight combinations through native controls or equivalent prompt editing. No implicit activation.
- [ ] Real SD1.5/SDXL gates: ordinary baseline/no-op/restored exact, every keyword and nested group, signed weights,
  CFG rescale, batch, schedules, hires/img2img, multiple samplers. Inspect images, app-service path and UI.
- [ ] Independent review, full suites, licences, ledger promotion, handoff/verification, verified commit/push.

## Runtime design questions to resolve with evidence

- Preserve actual scheduler x0 semantics: sigma-scaled Euler/Heun/SDE versus alpha-normalized DPM/DDIM/UniPC.
- A1111 negative prompt remains unconditional conditioning; Neutral Prompt syntax applies to the positive tree.
- PAG, ControlNet residuals and adapters need branch-specific conditioning; do not reuse first-branch residuals
  and call the combination supported. No restriction can be used to declare the overall extension goal complete.
- Existing guidance hooks may alter CFG after composition; establish the ordering with synthetic predictions.

## State

Starting commit `144a149`, clean. Previous goal turn was progress (NegPiP implemented and verified/pushed).
Scope remains 86 done / 216 todo / 51 na. The foundation and runtime are separate reviewable deliverables;
foundation completion must leave `neutral_prompt` todo.

Foundation result: initial missing-module RED, then 18 tests pass. Extreme-scale perpendicular projection and
tiny-variance rescale were separately reproduced failing before repairs. Use max-scaled normals and standardize
before variance multiplication to avoid underflow/overflow. Degenerate zero norms/constant tensors stay finite.
120 direct upstream random-tree composition comparisons passed (max absolute error 7.63e-6); independent reviewer
also compared 169 syntax cases and 100 random trees, finding only the tiny-variance issue (now regression-tested).
Full runtime, x0/scheduler conversion, plugin/form, real-model checks and ledger promotion remain required.
Final foundation suites: 341 Python passed; 260 torch-free passed / 31 skipped. Live notice collection covered
112 projects, adding this MIT source and verbatim licence only. No Scala/runtime code changed in this unit.

## Runtime stage 1 (continuation, not target completion)

- `neutral_runtime.py` Plan/RuntimeHook evaluates CLIP leaves independently, composes x0 deltas, and reconstructs
  a CFG-compatible UNet result. Branch schedules advance by logical steps; every pass gets fresh state. Separate
  contexts/pooled vectors support CFG off, batch >1 and SDXL. Hook/forward cleanup is tested after branch errors.
- Engine encoding/adjustment/install is connected via raw `hooks.neutral_prompt`, with advertised engine/dry-run
  capability. The Scala form is still absent, so this is not an app-complete plugin. No ledger promotion.
- Native scheduler tests compare each stage of Euler/ancestral/LMS, Heun, DPM2/ancestral, DPM++ SDE, DPM++ multi/
  single, UniPC and DDIM, including epsilon/v/sample where native support exists. LCM boundary output and inverse
  DDIM input-noise conversion have separate tests. This proves conversion contracts, not real-model LCM/SD2 quality.
- A low-noise round trip through a large shared x0 baseline loses float32 precision. Runtime with phi=0 composes
  scaled prediction differences directly instead; a large-baseline regression preserves the composed conditional.
- Review exposed missing Detail Daemon handling (per-row timesteps) and zero-SNR v-pred division by zero. Daemon
  is now explicitly guarded pending real branch integration; ZSNR/inf-sigma v coefficients are calculated directly
  and tests fail before the fix/pass after. Undefined zero-SNR epsilon produces a clear unsupported error.
- Inverse DDIM uses the previous input noise level, not the target level; a native comparison reproduced the
  original mismatch then passed after correction. Actual Alternative remains guarded; initial zero-noise inversion
  can have B=0 and needs a considered composition rule before integration.
- SD1.5 and SDXL each passed 18 GPU variants/19 images, no warnings, grids inspected. No-op and disable restoration
  are pixel-identical to baseline. Reports: `outputs/verify-neutral-runtime-{sd15,sdxl}/neutral_{sd15,sdxl}.json`.
- Remaining guards must be resolved with branch-specific evidence rather than called implemented by restriction:
  ControlNet/PAG, PhotoMaker, IP-Adapter, Reference, ControlLLLite, CADS, SAG, CFG Schedule, APG/Mahiro/Latent Modifier,
  Dynamic Thresholding, Detail Daemon and img2img Alternative. No current results prove those combinations.
- Final Python full suite after all edits: 382 passed with four torchsde Brownian endpoint warnings; torch-free
  260 passed / 32 skipped. Runtime test module: 41 passed. No Scala changes, ledger unchanged at 86/216/51.
