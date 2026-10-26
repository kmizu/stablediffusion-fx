# Weight Gradient implementation plan

> **For agentic workers:** Use superpowers:executing-plans for this wave.

**Goal:** Implement extension-index target `weight_gradient` from DingoBite/weight_gradient's README.

**Architecture:** A pure Python parser renders gradient tags as ordinary weighted prompts per step, then composes
them with the existing A1111 editing/alternation scheduler. Existing embedding swaps handle CFG/PAG/SDXL pooled
conditioning. An opt-in Scala AlwaysOn plugin emits `hooks.weight_gradient`; disabled syntax stays literal.

**Spec:** Upstream README retrieved 2026-10-26. No published licence found; do not read/copy upstream implementation.
Canonical file: `https://github.com/DingoBite/weight_gradient/blob/master/readme.md`, Git blob
`addaac1a522d33fcf896b29c04e84fe129d68cef` (retrieved with GitHub's `/readme` API, not guessed filename casing).
Syntax `{tokens : start-end : startWeight-endWeight[-returnWeight] : mode}`, empty interval = whole pass;
linear `a`, exponential `e/ei/eo`, circular `c/ci/co`; four-decimal weights; fractional step indices floor.
Exponential/circular ease-in-out curves are identified by reproducing README numeric examples, not source code.

**Decisions:** Gradient clocks restart in each sampling pass, including hires; ordinary A1111 editing keeps existing
hires offsets. Signed weights are supported. Escaped/malformed tags remain literal. Balanced emphasis/schedules
inside token text must not be split at their colons. No new dependency. Existing architecture and desktop UI retained.

## Verification and tasks

- [x] Pure parser tests RED/GREEN: README examples, all seven curves, fractional/percentage/absolute intervals,
  three-point return, boundary/one-step behavior, multiple/escaped/malformed tags and nested emphasis/editing.
  Create `python/sdfx/weight_gradient.py`, `python/tests/test_weight_gradient.py`.
- [x] Engine tests RED/GREEN: opt-in positive/negative schedules, disabled behavior, batch conditioning, CFG/PAG
  swap callback and hires step counts. Wire `_encode_scheduled` and ensure hires re-encoding for gradient syntax.
  Register feature in hooks, Engine.features, dryrun.features.
- [x] Scala plugin and tests RED/GREEN: default disabled, syntax help, hook registration/infotext.
- [x] GPU SD1.5/SDXL checks compare a constant tag to equivalent weighted prompt, nonconstant/negative/hires cases,
  and restored baseline after disabling. Inspect images and real app/UI before promoting ledger.
- [x] Refresh notices, run full relevant suites, independent review and update handoff.

Integration: commit/push this verified wave to main under AGENTS.md; Git records the resulting SHA.

## Review focus

- Avoid treating the always-empty hook configuration `{}` as disabled.
- Preserve normal prompt-schedule grammar and hires offsets.
- Recompute hires gradient schedules when second-pass step count differs even with inherited prompt text.
- Long prompts may require schedule-wide conditioning padding; preserve the existing behavior.
- Never claim deterministic effects or feature completion from a single visually broken output.

## Ledger

- Start: 84dee44, clean worktree; previous goal turn was progress (verified changes and pushes).
- Remaining scope unchanged: 218 entries / 153 targets before this wave; no overall completion claim.
- Parser examples and seven curves passed after the missing-module RED. Engine opt-in, batching, pooled CFG/PAG
  updates and inherited hires schedules passed after six failing tests. Img2img actual-step clock was separately
  reproduced failing and fixed. PhotoMaker's text-only path explicitly rejects gradient syntax instead of silently
  ignoring the feature. Scala plugin missing-class RED -> plugin + builtins 19 green.
- First GPU checks: ten images each on SD1.5 and SDXL. Constant-tag and after-disable output exactly match baseline;
  changing gradient differs from both fixed endpoints. Negative/return/hires/img2img outputs inspected. Java
  AppServices pipeline also passed with the plugin enabled on both models; UI form inspected in an isolated home.
- Independent review found two P2 timing issues: img2img Alternative runs full steps despite denoising strength;
  second-order schedulers invoke callbacks twice per inference step. Added regressions and observed five failures,
  then fixed full-step Alternative clock and stage-aware callback timing. Tested actual diffusers Euler, Heun,
  KDPM2, KDPM2 ancestral and DPM++ SDE scheduler timestep arrays. Final GPU checks include Heun and Alternative.
- Ruling: gradients count logical inference steps and restart at each pass, while A1111 editing keeps prior
  offset semantics. This is a documented implementation choice; not a claim that every upstream hires edge case
  has been compared. Source README does not define PhotoMaker or hires interaction.
- Final model checks passed fifteen images per model, including Heun constant-control and Alternative timing.
  Full suites: Scala 481, GPU-venv Python 300, no-torch Python 247 passed/16 skipped. Final UI screenshot/exit 0.
- SDXL Alternative was visibly oversaturated: focused default-inversion comparison also reproduced it with
  ordinary fixed weights and no gradient hook. Record it as an existing quality follow-up, not a successful
  aesthetic result or a Weight Gradient regression. It is the next handoff action before adding more targets.
- Final notices refresh: 110 projects, only the new reimplemented/unlicensed source row added; no existing licence
  notices lost. Ledger is 85 done / 217 todo / 51 na. Post-promotion BuiltinsSuite passed all 18 tests. Full extension
  goal remains active and unfinished; the new feature's verified scope is described in VERIFICATION.md.
