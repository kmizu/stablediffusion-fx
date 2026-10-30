# Token Downsampling (#129), eighth wave

## Requirement and source

Resolve index 295 token_downsampling, preserving the actual upstream controls and
XYZ axes, not only the older README's max-depth description. Source MIT wrapper:
feffy380/sd-webui-token-downsampling at f1f1cbae82b75d5658b08abbc3cb8ddc0b876cc5.
Pinned script, README and license are under outputs/token-downsampling-*.
The script exposes factor1/factor2 in [1,5] step .5 (1 disables that depth),
disable_after in [0,1], and all three XYZ axes/infotext keys. It reduces only
self-attention K/V via nearest-exact spatial resampling; full Q/output length stays.
SD1.5 depths are 1/2; SDXL depths 2/4. Its clock counts UNet forward evaluations
and enables the first floor(disable_after * steps) calls, reset per batch.

Reference lineage: ethansmith2000/comfy-todo at
ddd75a3e62e0a67f0b9be839a9916b553041494f (MIT; verify pinned license).
ImprovedTokenMerge at 81928e4fe47b2be47a983dcfdc0586c3cb138e61 has no visible
license in its root listing: do not copy its code. Document behavior from licensed
wrapper/reference instead. Re-implement for diffusers, never import parent reForge.

## Native design

Use one opt-in AlwaysOn HookPlugin id/hook token_downsampling, two factor sliders
and disable-after slider; defaults factor1=1, factor2=1, cutoff=1. Preserve source
ToDo factor / ToDo factor2 / ToDo disable after infotext and three numeric XYZ axes.
Do not invent a max-depth knob that the current upstream no longer exposes.

A dedicated Python hook reads current UNet spatial dimensions each evaluation,
identifies self-attention depths, resamples only K/V input using nearest-exact,
and keeps queries intact. Preserve processor identity and remove every installed
hook in finally on success/error/cancel. Validate finite ranges before inference.
Handle odd/non-square latent shapes from actual depth geometry; reject ambiguous
layouts rather than silently reshape. Bound reduced dimensions to at least one.
Factor1=factor2=1 and cutoff=0 must produce baseline-identical output.

Reset evaluation clock for each native generation pass/batch, explicitly document
this hires adaptation; it avoids carrying an exhausted base-pass clock into hires.
Evaluate processors/masks and interactions (especially HyperTile, PAG and branch
sampling) against actual diffusers contracts. Support valid combinations and report
proven incompatibilities before mutation; never silently claim a skipped hook active.

## Implementation and verification

1. Red tests: exact nearest-exact K/V/reference behavior, full query length, both
   model depth sets, cutoff boundaries, factors1/2, odd/non-square shapes, CFG batch,
   no-op exactness, no mutation leaks on failure, repeated install/remove and
   supported attention processor invocation. Test invalid/NaN settings early.
2. Implement Python hook, engine pass integration and cleanup; register hook in
   KNOWN_HOOKS, Engine.features and dryrun.features. Add native plugin registration,
   controls, infotext replay/XYZ integration tests. No new model/download dependency.
3. Actual SD1.5/SDXL native Generate and engine txt2img/img2img/hires: record output
   images, disabled/off parity, active K/V token counts and cutoff evidence. Measure
   timing without promising a speedup, inspect images for quality tradeoffs. Test
   interrupt/error cleanup and following baseline recovery. Exercise both factors.
4. Verify source/reference licenses, add collector notices; run appropriate full
   Python/Scala regressions, assembly and packaged native checks. Independent review.
5. Only after acceptance promote index 295, regenerate docs, run final ledger tests,
   commit one wave, push main and verify remote equality/clean tree; close #129.
   Goal becomes 8/20. Final release is still after twenty distinct resolutions.