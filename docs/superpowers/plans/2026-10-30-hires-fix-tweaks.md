# Hires Fix Tweaks (#56), ninth wave

## Source and scope

Index297 is w-e-w/sd-webui-hires-fix-tweaks, AGPL3, pinned
28902f742a795afc31d86641ad85466afdd280e6. Complete Python sources, README and
license are retained under outputs/hires-fix-tweaks-source. #81 and #94 were
examined as candidates only; do not count or close them. Goal currently8/20.

Current native Hires already supports separate CFG/sampler/schedule/prompts, but
not the complete upstream feature set. Do not accept #56 just for exposing CFG.
Required behavior:
- Hires CFG 0=same, including its named XYZ axis.
- Generate1..16 separate hires batches from each SAME base latent/image, preserving
  order (base batch -> hires repeat -> item). Never rerun base for each repeat.
- Optional hires seed:0 inherits base, -1 chooses random, positive fixed; dedicated
  variation seed/strength/resize dimensions, random/reuse buttons, per-result
  base and hires seed metadata. Preserve independently resolved random seeds.
- Positive and negative prompt modes Default/Append/Prepend/Prompt S/R. Configurable
  separators ({newline}) and marker, doubled-marker escaping, multiline search and
  replacement, ordered substitutions, base insertion-marker removal. Preserve
  source whitespace behavior via licensed reference comparisons.
- Remove first-pass extra-network tags only before constructing hires positive
  prompt; do not erase embedding words, manual LoRA settings or style-added tags.
  Actual pass LoRA activation must match prompts, not merely remove visible text.
- Independent hires style selection: enable+empty means remove inherited styles;
  multiple styles and Apply-to-prompt action; work before/after styles in exactly
  the source order. Prompt-mode template save/restore settings and source metadata
  compatibility, with safe validation and actual native form restoration.
- Separate hires output directory with explicit existing global-output precedence,
  collision-safe final naming and accurate gallery/payload paths; never redirect
  temporary files or change shared global settings during generation.
- Native equivalent presentation preferences for visibility of CFG/batch-seed/
  prompt modes/remove networks/styles, radio vs combo prompt modes, and checkpoint
  display path mode (default/hires-only/global). Display labels must retain stable
  actual checkpoint identities and duplicate-name disambiguation. Do not create
  unrelated checkpoint switching behavior merely for a label preference.

## Architecture and invariants

Preserve existing sdfx/schema/default settings compatibility. One registered target
hires_fix_tweaks, native controls integrated with the Hires area rather than an
unrelated script that reruns base generation. Existing Hires CFG remains one value.
Use immutable per-job configuration with a hook/object boundary and explicit
engine feature registration where new hook names are used. Host prepare performs
prompt-mode transformation BEFORE style expansion/LoRA extraction, with raw and
resolved template/provenance distinguished. Base and hires LoRA lists must be
separate for active tweaks; ordinary disabled behavior remains unchanged.

Python preserves one base result, caches its upscaled source, and iterates hires
seeds without modifying shared pipeline/global state. Existing latent path must
stay latent; do not replace it with decoded img2img. Variation/resize noise must
be applied to the actual hires noising operation, not just metadata. Preserve
batch-size, ControlNet, prompt scheduling, ADetailer/postprocessing and interruption
semantics; partial completed images survive, unfinished outputs do not. Accurate
progress and output counts include all repeats. Validate total work/output limits
before expensive execution and keep model/LoRA/hook cleanup in finally paths.

## Execution and acceptance

1. Source behavior matrix and pure helpers with RED tests: prompt modes/escaping,
   separators, metadata templates, seed/repeat order and variation inheritance.
2. Native controls/preferences and form/persistence/replay/XYZ tests, including
   checkpoint label identity and style Apply behavior. Extend schema only where
   needed with backward compatible defaults and corresponding preset contracts.
3. Engine repeat/noise/pass-LoRA integration and output routing. Tests exercise
   same-source reuse, pass-specific settings, disabled exactness, cancellation,
   output cardinality/metadata and following generation recovery.
4. Actual SD1.5 and SDXL native Generate/Interrupt, base-once/multiple-hires,
   prompt/style/LoRA cases, seed/variation changes, latent and pixel upscalers,
   explicit destination, metadata Restore, and image inspection. Record adaptations
   or upstream bugs (such as upstream subseed randomness checking hr_seed instead
   of hr_subseed) explicitly; do not silently narrow scope to convenient cases.
5. Add AGPL notice, independent review, complete Python/Scala regressions and
   extracted-package acceptance. Promote ledger only after every scope item has
   evidence, run final ledger gates, one wave commit/push and verified #56 closure.
   Count9/20 only then; publish only after20 distinct accepted resolutions.