# Token Downsampling (#129) acceptance record

Eighth accepted wave; remote commit/closure is tracked in twenty-issue-release-progress.md.

## Source, behavior and adaptation

Index 295, feffy380/sd-webui-token-downsampling, MIT pin
f1f1cbae82b75d5658b08abbc3cb8ddc0b876cc5. Its current script supplies two factors
[1,5], disable-after fraction [0,1], three XYZ axes and ToDo infotext keys.
The README max-depth wording describes an older UI and is not the current scope.
Licensed lineage ethansmith2000/comfy-todo pin
ddd75a3e62e0a67f0b9be839a9916b553041494f is also acknowledged. Collector notices
cover 154 projects. No code from unlicensed ImprovedTokenMerge was copied.
No new Python runtime dependency or model download is required.

Native AlwaysOn token_downsampling preserves the two factors, cutoff, three XYZ
axes and source infotext fields, with explicit opt-in. PNG Restore updates actual
form values; an ordinary PNG disables this hook, invalid values are not applied.
Only self-attention K/V inputs are spatially sampled with nearest-exact; full Q and
output shape are preserved. SD1.5 depths 1/2 and SDXL depths 2/4 are independent.
The native adapter captures Transformer2D spatial dimensions so odd/rectangular
sizes do not depend on an ambiguous square-root inference. Tiny outputs clamp to
one pixel. This extends the original helper's behavior at these boundary shapes.

Cutoff counts actual UNet evaluations, including each Neutral Prompt branch. It
uses requested steps rather than the strength-truncated img2img evaluation count.
Each native batch/pass resets the clock. Hires uses its explicit second-pass
steps (or base steps when zero); this is an explicit native adaptation instead of
carrying the exhausted base-pass clock into hires. Separate CFG evaluations and
multistage schedulers can consume the evaluation budget earlier than nominal steps.

PAG and HyperTile have incompatible self-attention layouts and are rejected when
this hook is active. Normal AttnProcessor/SDPA, XFormers and sliced processor input
contracts are preserved; actual GPU acceptance used SDPA. Custom processors and
nonstandard normalization fail explicitly. Mask keys use the same sample indices.
No-op factors/cutoff install no hooks. Original processor identities are retained.
All installed hooks are removed in finally after success, error or interruption.

## Evidence

- 16 focused Python cases pass: normalization, exact resampling, CFG batches,
  odd-size real UNet queries/KV, SD1.5/SDXL depth routing, cutoff/off parity,
  mask indexing, idempotent install/remove, processor error cleanup, requested
  step budget and branch-forward clock. outputs/todo-review-unit.log.
- Final no-torch focused check passed eight applicable cases/eight skips (outputs/todo-final-no-torch.log).
- Pinned MIT pure-function reference comparison: 45 configurations are bit-exact
  (square/rectangular, depths 1/2/4, factors 1/1.5/2/3/5), outputs/todo-reference.json.
  Helper verifies original script SHA256 before compiling just its two pure
  functions, without importing A1111: scripts/dev/check_token_downsampling_reference.py.
- Native plugin tests: opt-in numeric hooks, all three XYZ overrides, metadata
  replay and invalid values. Two cases passed (outputs/todo-replay-compile.log).
- Final whole Python regression exited 0 (outputs/todo-final-python.log).
- Whole Scala passed 1298 tests with one skip; assembly passed
  (outputs/todo-full-scala-build.log). No ignored failures.
- Real RTX3090 SD1.5/SDXL final engine runs: outputs/verify-todo-final-sd15 and
  outputs/verify-todo-final-sdxl. Factor1-only/factor2-only/both, txt2img, img2img,
  hires and two Neutral branches execute. No-op factor and zero-cutoff images
  exactly equal baseline; recovery after active/error cases also exactly equals
  baseline. PAG/HyperTile conflict errors are checked rather than silently ignored.
- Engine logs show actual K/V shapes and cutoff counts. SD1.5 factor 2 halves each
  spatial side, 4096 -> 1024 and 1024 -> 256 tokens. SDXL: 2304 -> 576 and
  576 -> 144. Two-branch 12-step sampling reports 24 evaluations, cutoff .5
  activates exactly six evaluations. Strength .75 with 12 requested steps executes
  nine evaluations and .7 cutoff activates eight, matching the source denominator.
- Source-checkout MainWindow Generate, PNG settings/Restore, three consecutive
  Interrupt operations and exact disabled baseline recovery passed for both
  models (outputs/todo-native-sd15.log and todo-native-sdxl.log).
- Independent review caught the initial strength-adjusted cutoff and branch clock
  omissions. Both were corrected, regression-tested and re-reviewed with no new
  important issue; final GPU checks include both paths. Earlier logs predate fixes.

## Visual/performance limits

SD1.5 shallow and SDXL baseline/both-depth images were inspected. Token reduction
can substantially change composition and detail, especially both depths active
through all steps; it is not a pixel-preserving optimization. Two-step/small hires
checks are functional rather than image-quality claims. Final SDXL single-run
wall times: warmed no-op 2.569s, shallow 2.501s, deep 2.450s, both 2.311s,
recovery 2.577s; branches 4.752s off vs 4.761s on. These include end-to-end overhead
and are not a statistical benchmark or guaranteed speedup. JSON retains all timings.

## Package gate

Current jar SHA256 aea8d471ae0cadee578a6883f1547837413845aff799ee0b72a97b6cf588b0c6.
Windows ZIP SHA256 366df29125bd64f2fec21fe2859a6c12885def420545c1f2f656022b4a3e2772.
Build outputs/todo-package/build-bbda7ffca70b41f39830210d5d5c2472.
Extracted package outputs/todo-extracted/package check ee0aabcaf94b40418556152a796dc8b2/SDNUI.
Bundled-runtime SDNUI.exe dry-run passed; screenshot inspected (outputs/todo-package-launch.log).
The GPU harness uses external JDK21 with the extracted jar/Python because jpackage
omits java.exe; actual executable runtime startup is separately verified. It checks
all tracked Python module hashes against source and retains the unchanged jar hash.
External GPU venv and checkpoint files are test fixtures, not bundled weights.

Both extracted-package native checks exited 0: outputs/todo-packaged-sd15.log and
outputs/todo-packaged-sdxl.log. Active image/settings, PNG Restore, three successive
Interrupt operations and exact disabled baseline recovery all passed. Packaged
SDXL MainWindow screenshot was inspected. No incomplete job output was published.
Ledger after acceptance: 113 done, 189 todo, 51 not applicable. This package is a
local verification artifact; final publication remains after twenty issue resolutions.