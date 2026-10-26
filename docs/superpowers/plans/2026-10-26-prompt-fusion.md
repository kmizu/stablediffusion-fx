# Prompt Fusion standalone port

## Goal and authoritative source

Implement extension-index entry 72, target `prompt_fusion`, from
https://github.com/ljleb/prompt-fusion-extension, commit
`1d86996ac4c3d32efaacb8a115600aa6fde6dab6` (inspected 2026-10-26).
The upstream `readme.md`, `scripts/promptlang.py`, parser, AST, interpolation math,
geometry and parser corpus were inspected before implementation. Upstream is MIT
(copyright line is literally `Copyright (c) 2003`). Preserve that licence in
`python/sdfx/fusion/LICENSE` and collect the normal third-party notice before promotion.

Upstream adds interpolation inside one image's sampling, rather than producing a
sequence of images. Required source behaviors are linear, Bezier and Catmull-Rom
embedding curves, contextual attention interpolation, alternation interpolation,
weighted means, variables and functions, nesting and native prompt syntax. Upstream
settings are enable, slerp scale, negative-prompt origin and slerp epsilon.

## Architecture and implementation sequence

1. Add CPU tests first; verify failures for the missing module and missing native routing.
2. Port the MIT parser/AST/curve implementation under `sdfx.fusion`; remove all A1111
   imports, global settings and executable visualization code. Retain upstream source
   provenance and licence. No reForge import, launch, monkeypatch or installation.
3. Implement `sdfx.prompt_fusion`: build leaf prompt database, encode through the native
   engine, interpolate each logical step, and return `ScheduledEmbeds`. SDXL interpolates
   both cross-attention and pooled vectors. Encode batch 1, then repeat the final
   conditioning so spherical geometry never couples unrelated batch elements.
4. Register hook/features in all three required registries. Reuse existing CFG/PAG
   schedule swapping and actual-pass clocks, including independent hires conditioning.
5. Add native AlwaysOn controls with examples, geometry controls, costs, compatibility
   explanations, validation and saved infotext; add munit coverage and registration.
6. Parent integration: Scala tests/build, real SD1.5 + SDXL generation, actual app/MCP
   replay and visual native-form inspection, notices and ledger only after those gates.

## Preserved semantics and deliberate repairs

- The parser/AST/curve functions are ported rather than replaced with a subset grammar.
  The entire upstream parser corpus is retained as a CPU regression test.
- Ordinary prompts bypass Fusion parsing, preserving exact native whitespace and
  tokenization. Enabling the plugin alone has no effect on ordinary prompt encoding.
- Fusion interpolation clocks start at zero, as in upstream `_sample_tensor_schedules`.
  Explicit integer curve position `3` becomes interpolation step 4 in the zero-based
  clock; `[a:b: , 3]` at five steps yields coefficients `0, .25, .5, .75, 1`.
  Native callbacks remain one-based logical-step schedules.
- Native A1111 editing omits step-zero boundaries. Fusion-generated contextual attention
  windows require those boundaries, so `leaf_texts` shifts editing endpoints by one
  locally. Ordinary scheduling is unchanged. The contextual ramp regression caught this
  discrepancy before integration.
- Each hires/img2img pass restarts the Fusion clock using its actual logical sample count.
  Existing native scheduling runs for the remaining syntax and Weight Gradient.
- Upstream weighted means used in-place multiplication/addition and therefore corrupted
  cached control points on repeated evaluation. The port uses out-of-place arithmetic.
- Slerp falls back to linear interpolation when either vector norm is zero, preventing
  division-by-zero NaNs. Finite scale/epsilon and finite results are validated.
- Empty conditioning pads short token sequences by whole CLIP chunks. SDXL's native
  zero-negative behavior is retained. Positive slerp can use the time-varying interpolated
  negative prompt as origin; negative slerp uses the empty prompt.
- Expansion over 512 leaf conditions reports an error. Recursive/invalid variables and
  malformed numeric evaluation become explicit engine parameter errors.

## Compatibility boundaries

PhotoMaker's text-only encoder is unsupported with active Fusion syntax. NegPiP signed
attention channels cannot be interpolated as ordinary embeddings: actual negative terms
report an unsupported error; enabling NegPiP with no such terms preserves its normal path.
Neutral operators should surround Fusion expressions, so Neutral's existing per-branch
encoding invokes Fusion. Operators introduced inside a Fusion variable report an error.
Existing Neutral guards remain intact. No new promise is made about those guarded hooks.

Img2img Alternative's inverse pass has no scheduled-conditioning callbacks. All effective
inverse prompt routes (inherited main prompts and either override field) explicitly reject
Fusion syntax rather than freezing its first context. Ordinary inverse prompts remain
unchanged, including when the plugin is enabled without active Fusion syntax. Its normal
forward sampler already accepts scheduled conditioning, but that does not establish inverse
sampling support. Other
attention adapters continue through native `ScheduledEmbeds` and `_adjust_embeds`;
real combined model weights require independent GPU verification.

## Validation performed here

CPU only, `CUDA_VISIBLE_DEVICES=-1`, worktree `PYTHONPATH=python`, shared `.venv-smoke`.
Initial four tests failed because `sdfx.prompt_fusion` did not exist; the native engine
routing test then failed because the engine returned an unscheduled dictionary.
Eighteen Fusion tests pass, including the full upstream parser corpus (many cases),
native routing and batching, all curves, weighted means, negative/pooled tensor paths,
alternation speed, contextual attention, long sequence padding, zero-norm spherical
geometry, ordinary-prompt bypass, inverse-route guards and torch-free feature advertisement.
The targeted CPU regression set passed 117 tests in the torch venv and 82 tests with
21 skipped in the no-torch venv; these are focused sets, not claims of a full-suite run.

Run targeted regressions:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONPATH = "$PWD\python"
& D:\stable-diffusion-webui-reForge\stablediffusion-fx\python\.venv-smoke\Scripts\python.exe -m pytest python/tests/test_prompt_fusion.py python/tests/test_prompt_fusion_features.py python/tests/test_prompt_fusion_upstream.py python/tests/test_extensions.py python/tests/test_weight_gradient.py python/tests/test_prompt_schedule.py python/tests/test_negpip.py python/tests/test_neutral_prompt.py -q
```

## Parent verification required before promotion

- `sbt.bat -batch "plugins/testOnly sdfx.plugins.PromptFusionSuite"`, full tests and UI assembly.
- SD1.5 and SDXL baseline with hook disabled/enabled but ordinary text: exact same output.
- Active `[cat:dog: , 10]`, Bezier and Catmull three-point curves, `[red:blue::mean]`,
  `[red|blue:0.5]`, `(red scarf:1,2)`, variable/function syntax on both prompt branches.
- Same-seed constant endpoint/mean identity; explicit nonzero slerp and negative-origin
  toggling; long endpoint versus short negative text; batch 2; Euler/Heun/PAG.
- Hires with distinct second-pass steps, img2img actual-step clocks, Neutral outer
  composition and Weight Gradient. Confirm unsupported PhotoMaker/NegPiP paths fail clearly.
- Actual AppServices/MCP request includes hook and saved infotext; replay preserves exact
  prompt text and all three geometry fields. Inspect real images and complete native form.
- UI dry-run screenshot with `txt2img.plugin.prompt_fusion` expanded. Enable/disable
  geometry controls, read all examples and check narrow/window scrolling.
- Collect MIT notice using the authoritative upstream URL; add `prompt_fusion` to
  `IMPLEMENTED` and regenerate matrices only after these checks succeed.

No GPU, sbt, app launch, ledger or main verification-document changes were performed in
this isolated implementation task. No additional dependency is required.

## Independent-review numeric validation follow-up

The reviewer reproduced a raw `OverflowError` from a curve-position variable containing
`1e309`. The same nonfinite variable in an attention ramp could render an invalid weight
that ordinary attention parsing silently ignored. Ten Engine-route RED tests reproduced
both paths with positive/negative overflowing exponent notation, NaN, infinity and a
400-digit integer. `_eval_int_or_float` now rejects values outside finite float range
before curve arithmetic or attention rendering, while preserving valid integer types and
their step semantics. The adapter also maps unexpected arithmetic `OverflowError` to
`INVALID_PARAMS`. Ordinary no-op routing and the source grammar remain unchanged.
After the fix, the targeted torch CPU set passes 127 tests; the no-torch set passes
82 tests with 21 skipped. No GPU, Scala or app commands were run for this follow-up.

## Parent-owned GPU acceptance harness

`scripts/dev/prompt_fusion_check.py` runs one local checkpoint in a standalone engine
subprocess. Invoke it once per model family with a fresh output directory:

```powershell
& python/.venv-smoke/Scripts/python.exe scripts/dev/prompt_fusion_check.py python/.venv-smoke/Scripts/python.exe D:/stable-diffusion-webui-reForge/models/Stable-diffusion/animekawa_v10.safetensors outputs/verify-prompt-fusion-sd15
& python/.venv-smoke/Scripts/python.exe scripts/dev/prompt_fusion_check.py python/.venv-smoke/Scripts/python.exe D:/stable-diffusion-webui-reForge/models/Stable-diffusion/waiIllustriousSDXL_v170.safetensors outputs/verify-prompt-fusion-sdxl --sdxl
```

The modest default is twenty generation cases at eight base steps (512 SD1.5 / 768
SDXL), plus six error cases that must stop before sampling. The source grammar cases
exercise functions/variables, linear/Bezier/Catmull curves, weighted mean, alternation,
contextual attention, spherical geometry with both origins, negative-branch curves and
long conditioning. Native paths include batch two, Heun, distinct hires prompts and
pass length, and img2img. SDXL generation exercises cross-attention and pooled vectors
through the native pipeline; the script does not expose embedding tensors over protocol.

Exact RGB identity is required for ordinary enabled/disabled, function expansion,
constant curves and restoration after all six error cases. Active linear interpolation
must have a pixel effect. Every output must have the correct dimensions/count and
non-flat pixels. Differences for other active variants are recorded without pretending
that a numeric threshold establishes artistic quality. Guards check both code and
message for PhotoMaker, signed NegPiP, inherited/override Alternative inversion and
nonfinite position/attention variables.

Artifacts are incremental `results.json`, complete request/event/response
`protocol.jsonl`, `stderr.log`, the original result PNGs and labeled `contact.png`.
Failures preserve partial artifacts and exit nonzero; process cleanup runs even if
artifact rendering fails. Human visual review of the sheet and original PNGs is still
required. Native UI/MCP/replay remains a separate parent gate.

Only CPU harness tests and `--help` were run while preparing this script. Thirty-one
focused Fusion/harness tests passed in the torch CPU venv; five harness/feature tests
passed in the no-torch venv. The GPU acceptance commands above have **not** been run
by the implementing agent.

## Parent integration accepted

The parent ran the harness successfully on both real checkpoints: 20 generation cases and six structured errors
per model. Baseline/ordinary/constant/variable-function/restored RGB comparisons were exact. Contact sheets were
visually inspected. Native AppServices queue, saved parameters and replay passed on both model families; expanded
native form screenshots were inspected. Full Scala491 + assembly, torch Python707/1skip, no-torch306/60skip passed.
MIT source notices were collected and index72 promoted. Evidence and unsupported-combination boundaries are in
the parallel-wave section of `docs/VERIFICATION.md`; no claim of universal extension compatibility is made.
