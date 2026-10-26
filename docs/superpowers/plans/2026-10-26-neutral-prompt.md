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
- [x] Scala plugin, registration, help, replay metadata and supported-features advertisement. Formatter must offer
  all keyword/weight combinations through native controls or equivalent prompt editing. No implicit activation.
- [x] Real SD1.5/SDXL core gates: ordinary baseline/no-op/restored exact, every keyword and nested group, signed weights,
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

## UI and Alternative stage (continuation, still no target promotion)

- `NeutralPromptPlugin` adds enable/CFG rescale, all-four-operator extra-prompt formatter, signed weights, nesting
  help and replay keys. Actual effective prompt is saved. Replay is idempotent, including an empty main prompt.
  The form shows cost/compatibility and groups additional inputs; UI screenshot inspected after rebuilding.
- `RealAlwaysOnCheck ... neutral_prompt` passed 1/1 each on SD1.5 and SDXL. It asserts effective formatter text,
  hook presence, Neutral Prompt and CFG Rescale phi infotext. Saved PNGs inspected; this is AppServices, not MCP
  transport. `outputs/neutral-app.log` records both; form: `outputs/verify-neutral-ui/form-crop.png`.
- Alternative re-encodes active composed decode prompts at decode step count and batch 1, preserving generation
  branches/batch separately. Override ordinary no-op retains the legacy unadjusted route, including Revision-like
  modifiers (review regression). Each inverse step advances its own branch schedule and restores wrappers in finally.
- At exactly zero inverse noise, composition uses its positive-noise continuation. PERP/TOPK are homogeneous;
  SALT compares centered absolute scores and, on ties, higher centered distribution moments. Equal centered
  multisets retain original order; float64-indistinguishable moments also retain that order. Rescale uses the
  constant/centered-image projection limit, or noise-space rescale when the input image is constant. This is the
  app's explicit DDIM startup design, not an upstream claim of pixel parity. Phi/CFG numerical-limit tests pass.
- Review identified empty-prompt formatter replay duplication, no-op override adjustment drift and SALT first-order
  ties. Each has a regression; all fixed. The scalar DDIM inverse test covers CFG0/.5/1/2 with phi0/.7 from zero noise.
- Actual Alternative: 12 variants per SD1.5/SDXL, including four operators, decode CFG0/.5/1, phi, editing, NegPiP
  and default 50 decode steps. No warnings; ordinary/no-op/restored and Alternative plain/no-op exactly match.
  Reports/grids: `outputs/verify-neutral-alt-{sd15,sdxl}`. These model configs do not prove every zero-noise limit;
  the latter is separately covered by native/analytic tests.
- Final full suites: Scala 483 passed; Python 396 passed (four existing torchsde endpoint warnings), torch-free
  260 passed / 32 skipped. Runtime module now 55 tests. Jar built, UI exited 0. No new source/dependency licence.
- Still finish remaining guards in validate(): ControlNet/PAG, PhotoMaker, IP-Adapter/Reference/ControlLLLite,
  CADS/SAG, CFG Schedule/APG/Mahiro/Latent Modifier/Dynamic Thresholding/Detail Daemon. Keep target todo until
  branch-specific handling/validation is complete. Ledger remains 86 done / 216 todo / 51 na.

## Next integration design: PAG and ControlNet

PAG: resolve ordinary and perturbed positive predictions through the same expression tree independently, both
relative to the ordinary unconditional prediction. Preserve native pipeline PAG/adaptive scale after composition.
Neutral CFG rescale adjusts the CFG result, while PAG retains the difference between unrescaled composed ordinary
and perturbed predictions. With CFG, evaluate [negative, positive, perturbed-positive]. Without pipeline CFG,
PAG's processor divides the batch in two: internally evaluate [negative, positive, perturbed-negative,
perturbed-positive], then return the native two groups. Ignore the perturbed-negative output. Keep processor
objects/cleanup unchanged; test all operators, CFG on/off, rescale, batch, editing and exception restoration.

ControlNet: capture the native call frame for the current step, then recompute branch-specific residuals using
each leaf's current conditioning/pooled vectors. Do not reuse first-leaf residuals. Preserve native scale windows,
guess-mode conditional-only computation/zero unconditional residuals, multiple models and internal CFG expansion.
PAG+ControlNet must either handle both layouts correctly or remain explicitly unfinished until it does.

## PAG / ControlNet integration result

- PAG batching/composition is connected. Ordinary and perturbed leaves resolve separately, preserving native
  adaptive scale. Non-CFG uses four internal groups to retain an ordinary negative prediction while PAG's native
  processor splits two halves. The pipeline still receives its ordinary 2/3 groups. Phi changes CFG but keeps
  the unrescaled composed PAG difference. This ordering is an explicit design choice, covered by reference tests.
- Review found scale=0/no-active-PAG still using an ordinary native batch. Runtime now derives active PAG from
  that actual batch, not merely configuration presence. Failing inactive-layout regression then passes. Actual
  PAG attention processors are CPU-tested; reviewer additionally compared 24 native-processor/pooled/time-ID cases.
- `neutral_controlnet.py` captures the native ControlNet frame and recomputes each leaf's residuals with its text/
  pooled/time-ID conditioning. It preserves native image preprocessing, scale windows and multiple models. Guess
  mode evaluates the positive side and zero-pads negative residuals; PAG groups reuse the corresponding ordinary
  negative/positive residuals (ControlNet itself is not perturbed). Failed replays restore capture state/hooks.
- Frame unit tests prove stale first-leaf residuals are replaced; CPU real small ControlNetModel comparisons with
  nonzero residual layers passed in independent review. Actual Guess-mode behavior is covered by CPU tests, not
  claimed as a separate real-checkpoint GUI mode (the app's existing unit UI has no explicit guess toggle).
- Real SD1.5/SDXL: PAG 12 variants per model; ControlNet 13 variants per model, including CFG1/batch, rescale,
  editing/Heun, multiple models/windows, NegPiP and joint ControlNet+PAG with/without CFG. No warnings; no-op and
  restored baselines are exact on both models. Inspected grids/reports are `outputs/verify-neutral-{pag,cn}-*`.
- SD1.5 checkpoint is A1111 single-file format; the first test wrongly used a diffusers directory and failed to
  load. Test input path corrected to from_single_file, then the full matrix passed. SDXL uses a diffusers directory.
  Original model bytes, source URLs and SHA256s retained in `outputs/controlnet-models/*/source.json`; not bundled.
- Full final suites: Scala 483 passed and jar built; Python 440 passed (four known torchsde endpoint warnings),
  torch-free 260 passed / 33 skipped. Runtime 83 tests + ControlNet 16 tests pass. No new dependency/source licence.
- Remaining guards: PhotoMaker, IP-Adapter/Reference/ControlLLLite, CADS/SAG, CFG Schedule/APG/Mahiro/Latent Modifier/
  Dynamic Thresholding/Detail Daemon. Target still todo; ledger remains 86/216/51.
- Scale-zero live regression completed four cases on each model, including CFG1; no-active-PAG and restored
  composed results match the ordinary composed baseline exactly. Reports: `outputs/verify-neutral-pag-zero-*`.

## Image conditioning design (starting c874033, clean)

IP-Adapter: use native prepare_ip_adapter_image_embeds with CFG at batch 1 to obtain both image branches. Negative
hidden-state features may be encoded from a blank image, not zeros; never fabricate them or duplicate positive
features when pipeline CFG is off. Reuse those prepared features in the native pipeline to avoid a second encoder
pass; separately expand canonical negative/positive features to Runtime's actual CFG/PAG groups. Multiple adapters
and image-token dimensions remain lists/tensors with native shape. Recompute only text conditioning per leaf.

Lifecycle fixes required before unguarding: derived pipelines are constructed before IP load, so synchronize the
base image encoder/feature extractor into all derived pipelines. Rebuild offload placement when components change.
PAG's failure-cleanup processor snapshot must occur after IP load/unload, otherwise cleanup discards loaded IP
processors while IPAdapterState still says they are loaded. Test cached repeated PAG/IP generations and unload.

Then inspect Reference/ControlLLLite root pre/post hooks: Runtime calls saved UNet forward for each leaf, so root
hooks currently execute only around the outer call. Image bank/write passes and per-step activation must run for
the actual leaf conditioning/layout, without recursively re-entering composition. Keep pending guards until their
own tests and real-model checks pass. Overall goal and ledger remain 86/216/51; no target promotion in advance.

## IP-Adapter result (continuation)

- `neutral_ip.py` obtains native negative/positive features once at batch 1 with no gradients, including Plus
  blank-image hidden states; supplies batch-1 features back to the native pipeline and canonical features to
  Runtime's CFG/PAG layout. Negative features are never fabricated or replaced by the positive image.
- Sampling and Alternative share configure_ip_adapter. Inversion now also receives image conditions for plain
  or composed prompts over Decode CFG0/.5/1/2. Ordinary/no-op Alternative outputs match. This fixes an existing
  missing-image-condition route for already-loaded adapters during repeated Alternative runs.
- Derived image encoder/extractor synchronization, offload re-placement and post-IP PAG snapshots are connected.
  Repeated PAG retains IP weights; unloading clears the shared UNet correctly. Missing paths preserve loaded
  adapters; load failures invoke native cleanup, and finally synchronizes components/placement.
- Cache key now includes image_encoder, and Engine._free clears IP state. Both regressions failed before fixes.
  Review then found SD1.5 native unload retaining its feature extractor: new encoder registration now rebuilds
  native-style CLIP crop/resize at encoder.config.image_size when no safety checker is active (the app's mode).
  A real tiny CLIP encoder 16->32 regression failed before fix, then forwards correctly after it.
- Independent review used real UNet/native IP projection/attention and Plus resampler; CFG/batch/PAG/phi reference
  comparisons matched. Its encoder-cache and extractor-size findings are fixed and regression-tested.
- SD1.5/SDXL: 21 cases each, basic/Plus, phi, CFG1/batch, repeated PAG, editing/Heun, hires/img2img, ControlNet,
  Alternative and restore/unload. All no warnings; five baseline/repeat/no-op/restoration comparisons are pixel
  identical on each model. Grids inspected. SDXL basic native IP itself shifts contrast/style on this checkpoint;
  record as pairing quality, not a Neutral Prompt fix or aesthetic guarantee.
- Six focused cases each check same-checkpoint reload while IP is loaded, repeated PAG, NegPiP and unload.
  SD1.5 uses model_offload; SDXL uses GPU. All three comparison pairs match exactly per model. Full reports:
  `outputs/verify-neutral-ip-*`, `outputs/verify-neutral-ip-reload-*`; adapter hashes/URLs in adapter-sources.json.
- Final full suites: Python 467 passed (four known torchsde warnings), torch-free 260 passed / 34 skipped;
  Scala 483 passed and jar built. IP tests 27 passed. No new dependencies/source licences; external model weights
  remain ignored verification assets. Ledger unchanged 86/216/51; finish Reference/LLLite and remaining modifiers.

## Reference / ControlLLLite design (starting ef2d139, clean)

Use explicit branch preparation/cleanup, not recursive calls through the composed UNet. Managed image hooks keep
their attention/projection hooks, while Runtime invokes per-leaf preparation around its saved forward. ControlLLLite
updates its step window first; Reference then writes a fresh bank using that leaf's conditioning and clears it after
the leaf, including errors. Preserve the existing ordinary/no-op route when no composition is active.

Reference fidelity follows explicit positive/negative group layout, including CFG off and PAG's 3/4 internal groups.
At native PAG attention layers, reference conditioning applies only to ordinary groups; the perturbed identity path
stays native. Native PAG processors ignore encoder_hidden_states, so merely passing extra context cannot implement
this. Project current/reference keys/values separately before concatenation: this retains trained spatial length for
ControlLLLite projection hooks. Use native-compatible SDPA normalization/output rules without replacing processor
modules or changing parameter paths. Weight zero must be an exact no-op. Managed reference noising uses scheduler
input noise level (including Heun stages and inverse DDIM), rather than guessing from the target timestep alone.

Required tests: per-leaf bank conditioning and cleanup, LLLite window before bank write, style masks for all layouts,
real PAG attention normal/perturbed behavior, projection input lengths, noise-level contract and no-op/restoration.
Then SD1.5/SDXL Reference matrices and SDXL-only ControlLLLite with real weights, image inspection and full suites.

### Preparatory wave: managed image hooks (not exposed yet)

- Added optional ordered Runtime branch hooks with reverse cleanup, including failed preparation. Reference can
  write using the saved forward and the current leaf conditioning, then clear its bank after that leaf. LLLite can
  update its window without a recursive root call. Managed hooks are deliberately not wired by Engine/extensions
  yet; the Reference/LLLite compatibility guards remain and the extension ledger is unchanged.
- Added native-compatible reference SDPA with separate spatial Q/K/V calls. Combined-context normalization occurs
  before splitting projections (essential for GroupNorm). Explicit positive group masks handle all internal CFG/PAG
  layouts; native PAG identity outputs are preserved. Reference weight zero skips bank writing/injection exactly.
- CPU tests use real diffusers Attention/PAG processors: 37 passed, including native output comparisons with boolean
  and additive masks, LayerNorm/GroupNorm, spatial conditioning, PAG identity preservation, fidelity masks and cleanup.
  Independent read-only review found no material correctness defect within this preparatory scope.
- Full Python: 504 passed / four existing torchsde endpoint warnings. Torch-free: 260 passed / 35 skipped. JUnit
  reports: `outputs/verify-reference-foundation{,-no-torch}.xml`. No Scala/UI changes, dependencies or new licences.
- Not GPU-verified or product-enabled in this wave. Next: factor image-hook installation for forward and inverse
  passes, connect managed hooks through Engine/extras, then run the SD1.5/SDXL matrices before lifting guards.

### Reference / LLLite integration wave (starting 014c5da)

- Connected managed image hooks in Engine's forward runs and Alternative inversion. Helpers register removal
  before installing hooks; inverse setup receives its initialized DDIMInverseScheduler. LLLite activation precedes
  the reference bank write. Each pass installs fresh hooks at its current dimensions. Removed the two guards.
- Retained the ordinary/no-composition root-hook route. Alternative's latent forward pass now preserves its init
  image for LLLite's optional-image fallback; it does not supply that image as a txt2img pipeline argument.
- Independent read-only review found no material wiring issue, including the fallback follow-up. Tests cover
  inverse preparation/cleanup and partial installation failures; the reference-attention file now has 41 tests.
- Real GPU backend matrices: SD1.5 15 cases, SDXL 21 cases; focused pass matrices add 3/5 respectively. All returned
  no result warnings. Nine comparison pairs across the four folders are pixel-identical. Inspected contact sheets
  for all four folders. Reports and external LLLite URL/hash: `outputs/verify-neutral-reference-*`.
- Visual limits are recorded in VERIFICATION.md: existing native SDXL Reference strongly changes saturation/contrast,
  short Alternative cases can be posterized, and the four-step SD1.5 hires example is soft. Finite successful
  execution/equality checks do not establish universal image quality. This wave does not verify GPU cancellation,
  sequential offload or native nine-channel inpainting checkpoints.
- UI compatibility help/cost description and PLUGINS/HANDOFF updated. No new dependencies/source licences;
  the ledger remains 86 done / 216 todo / 51 NA. Next: remaining PhotoMaker, CADS/SAG and guidance/timestep modifiers.
- Final checks: Python 508 passed / four existing torchsde warnings; torch-free 260 passed / 35 skipped; Scala 483
  passed and jar rebuilt. JUnit reports: `outputs/verify-reference-integration{,-no-torch}.xml`.

### CFG Schedule / CADS design (starting fe52290, clean)

Implement `neutral_sampling.Controller` as per-pass conditioning/guidance state. Reuse sampling_ext's documented
CFG curves, scheduler progress, CADS annealing/noise/rescale and normalization. CADS transforms each positive leaf
once per UNet evaluation, before Runtime duplicates CFG/PAG groups or prepares ControlNet/Reference. Optional
negative random draws are shared between leaves of the same feature shape, while retaining each leaf's own payload.
Keep negative signs and padding
metadata intact for NegPiP; do not perturb pooled embeddings (matching the existing CADS contract). Do not mutate
encoded/scheduled tensors. Each pass/inverse gets a seeded generator; disabling/gamma=1 is an exact passthrough.

CFG Schedule changes the scale used by composition and phi, while output reconstruction divides by the native
pipeline's fixed CFG scale. Preserve the raw conditional/PAG difference. Respect the existing CFG-off ignore rule.
Skip the corresponding legacy root hooks only for composition; ordinary/no-op paths must remain exact. Initialize
the same controller against the inverse scheduler for Alternative. Tests precede the Runtime/Engine wiring.

Gate: real native math/layout tests (CFG/PAG/phi, CFG off, zero CFG targets, inverse startup), CADS reproducibility,
shared negatives, metadata preservation, scheduled contexts and no mutation; SD1.5/SDXL matrices and visual QA;
independent review, full suites, docs/compatibility help, verified commit/push. Keep the overall target todo.

### CFG Schedule / CADS implementation and evidence

- Controller connected to forward and Alternative inverse Runtime; native root hooks retained for ordinary/no-op
  paths. Runtime uses effective CFG inside phi/composition and reconstructs the native fixed-scale output, with
  PAG's raw difference preserved. Both guards lifted. Native CFG-off behavior still ignores CFG Schedule.
- Review exposed a shape-only whole-negative cache bug; changed to shared random draws applied to each leaf's
  own payload/metadata. Follow-up RED/GREEN tests also fixed private-sign corruption in the ordinary CADS hook.
  Shared `cads_context` preserves NegPiP signs/zero padding in both paths. Read-only review approved both fixes.
- 42 focused tests pass. Full Python: 550 passed / four existing torchsde warnings. Torch-free: 260 passed / 36
  skipped. Reports: `outputs/verify-neutral-sampling-{tests,no-torch}.xml`.
- SD1.5/SDXL main matrices: 23 cases each. Metadata matrices: 5 each. Total 56, no result warnings; all 20 equality
  pairs are pixel-identical. Contact sheets inspected; style/content/hires limits recorded in VERIFICATION.md.
- A full Scala run exposed a preexisting pending-queue test race: it asserted the last three jobs were pending
  before the worker had dequeued the first. Test-only JobStarted/release latches now make the precondition stable,
  with unconditional release/listener cleanup. No product queue code changed; independent review approved it.
- Compatibility help and handoff updated, no new dependencies/licences. Before promotion, audit the ordinary
  CADS/CFG Schedule root hooks with native PAG layouts; they retain an older half-batch assumption. Remaining
  guarded integrations: PhotoMaker, SAG, APG/Mahiro/Latent Modifier/Dynamic Thresholding/Detail Daemon.
- Next-task CPU probe confirms ordinary three-group PAG errors: CFG Schedule output 11 versus expected 5,
  unexpected CADS negative noise and unequal positive copies (`outputs/neutral-ordinary-pag-probe.json`). Fix these
  before promotion. Final Scala regression: 483 passed and jar rebuilt; targeted queue suite 14 passed.

### Ordinary PAG sampling repair (starting 2f8aced, clean)

Pass a lazy native `do_perturbed_attention_guidance` predicate to CADS/CFG Schedule root hooks. The native
property is authoritative for scale=0 and empty applied layers. CFG Schedule handles [u,c,p] for active PAG and
[u,c] otherwise; shift c and p together so the raw PAG difference survives the effective-CFG rewrite. Constant
CFG is an exact no-op. Preserve non-PAG variable-curve behavior. CADS perturbs canonical n/p once, then copies p
for PAG, keeping default negative conditioning and NegPiP metadata intact. CFG-off PAG has [p,p]. Do not change
the composed controller path or the existing non-PAG random draw layout.

Tests first: batches 1/2, CFG/PAG off/on, gamma1/constant/inactive PAG, signed contexts, tuple/object outputs and
native output formulas. Re-run SD1.5/SDXL ordinary/no-op/composed PAG matrices with restoration and visual QA;
independent review, full suites, docs, verified commit/push. Overall target remains todo.

### Ordinary PAG sampling repair — evidence

- Native activity callbacks wired lazily; CFG handles three groups and preserves the raw PAG difference, CADS
  perturbs canonical contexts before duplication. Constant CFG returns unchanged, including fp16. Original
  non-PAG variable math/draw layout retained; no change to the composed controller path.
- Independent read-only review approved code and installed native pipeline contract. Focused tests: 56 passed;
  full Python 606 passed / four existing torchsde warnings; torch-free 260 passed / 37 skipped. JUnit reports:
  `outputs/verify-ordinary-pag-{tests,no-torch}.xml`. No Scala/UI changes or repeated Scala build this wave.
- SD1.5/SDXL: 24 real GPU cases each, no result warnings. All 16 comparisons are pixel-identical, including
  native/no-op, repeats, constant CFG/zero PAG, restoration and previous-wave non-PAG images. Grids inspected; CADS
  style/off-prompt/short-step hires limits recorded in VERIFICATION.md. Reports: `outputs/verify-sampling-pag-*`.
- The pre-fix CPU counterexample is retained as historical evidence; its contracts now have permanent tests.
  No new dependencies/notices. Remaining guarded integrations unchanged; next implement their branch-aware
  guidance/postprocessing with explicit zero-noise/PAG contracts before promoting the target.
