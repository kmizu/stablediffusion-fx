# Composable LoRA / LyCORIS with steps

## Parent acceptance completed

Actual standalone GPU matrix passed31SD1.5/33SDXL cases plus four compatibility guards each, including trained
LoCon/LoKr. Other source adapter families retain explicit CPU-algebra coverage rather than claimed trained GPU
evidence. Actual native queue/JSON/PNG replay passed both models, baseline RGB restoration was exact, plots and
persistent traces survived correct cleanup, and false positive-AND warnings were repaired. Public native Restore/
Send-to replaced all six checkbox settings; the expanded form and both model contact sheets were inspected.
Full Scala522 + assembly, Python795/1skip and torch-free353/78skip passed. AGPL notice retained; index78 promoted.
Exact source scope, paths, known high-guidance artifacts and compatibility limits are in `docs/VERIFICATION.md`.
The overall extension-index goal remains active; the worker evidence below is historical to its own phase.

## Source and binding scope

Extension-index entry 78, target `composable_lora`, authoritative source
https://github.com/a2569875/stable-diffusion-webui-composable-lora, pinned
`a03d40eb3a3bc63728f46f1c00229310734c8fef`, AGPL-3.0.
Inspected README (including all controller examples), the seven UI controls, runtime,
step parser/evaluator, LyCORIS converter and recent history. The July 2023 `fix weights`
change explicitly restores base weights before applying patches; this port must never
mutate/fuse model parameters in the first place. Preserve notices through normal parent
collection, not generated-document edits in this implementation branch.

Required behavior:

- Scope LoRA/LyCORIS to its `AND` subprompt, including repeated adapters in several blocks.
- Independently include/exclude LoRA in negative text encoding and negative denoising.
  Preserve upstream's single-block exception unless "Don't use LoRA in uc if there're no
  subprompts" is selected.
- Optional step syntax `[A:B:N]`, add/remove forms, nested start/end, fractional positions
  and alternation; inclusive source endpoints and strongest-absolute-weight duplicate rule.
- `#increment`, `#decrease`, `#cmd(...)`, including literal `\u0023`, escaped code,
  contextual lifecycle, all documented math helpers, command context fields, function
  definitions and `enable_prepare_step` text-encoder behavior.
- All seven source controls, including weight plots and local/global hires clocks.
- Native LoRA, convolutional LoCon (including CP middle kernels), LoHa, LoKr, full-rank
  deltas and IA3 are core source formats, not optional compatibility work.

## Architecture

1. Port the AGPL controller grammar/math into a torch-free standalone module, replacing
   A1111 extra-network parsing with native data objects. Tests first for source examples,
   boundaries, duplicate weights, math commands, command failures and numeric finiteness.
2. Add standalone adapter tensor loading/mapping and module-local output patches. Standard
   low-rank adapters use efficient sequential factors; nonstandard families use their
   actual algebra. Test tiny Linear/Conv2d modules, every family, separate CFG rows,
   exception cleanup, state isolation and exact unchanged base parameters.
3. Encode each branch with its own text routing; reuse native prompt schedules, Fusion,
   Weight Gradient and NegPiP. Reuse Neutral's branch prediction composition so `AND`
   combines image predictions rather than concatenating unrelated prompts. Carry routing
   metadata per branch and prepare/remove patches in the existing branch-hook lifecycle.
4. Preserve prompt tags through Scala Generator.prepare only when this hook is enabled;
   resolve unique model paths without adding duplicate tag weights together. Ordinary
   LoRA extraction and existing LoRA Block Weight behavior remain unchanged when disabled.
5. Native controls/help/replay plus all hook/feature registrations; weight plot artifacts
   are diagnostic images distinct from generated final images. Reversible hooks are
   installed only for one job and removed even on failures/cancellation.
6. Prepare protocol GPU acceptance harness using local models; parent exclusively runs
   real-model, Scala, native UI/replay and notice/ledger acceptance gates.

## Command execution

Pure math expressions use a restricted AST path and require no setting. Advanced Python
statements, functions and native object inspection use the project's existing explicit
Settings > Allow custom code / `SDFX_ALLOW_CODE=1` boundary. Reject disabled advanced code
clearly instead of silently falling back to the original weight. Native command objects
replace A1111-specific objects; parameter names and their meaning are preserved.

## Verification targets

CPU tests use only the main checkout's absolute venv executables, worktree PYTHONPATH and
`CUDA_VISIBLE_DEVICES=-1`. No GPU, sbt, installation or global configuration changes here.
The parent validates SD1.5 `20_yui.safetensors` / `slimegirl11.safetensors` and SDXL
`alraune_v1_illustrious_b-000007.safetensors` / `sdxl-vbs-SeifukuKh03.safetensors`, plus
actual LoCon `sdxl-vbs-BNSeifukuKh03.safetensors` and LoKr
`MonsterGirlXL_ANI31_lokr_V43P1NF.safetensors`. No local LoHa checkpoint is available;
its actual tensor algebra receives meaningful CPU tests.

Real acceptance must cover exact no-hook/no-adapter restoration, branch isolation,
shared adapters with distinct weights, both negative settings, all scheduling/controller
forms, plots, separate TE/UNet multipliers, batching, SDXL pooled conditioning, hires
clocks, img2img, ordinary Fusion/Neutral invariants and advanced Python opt-in. Inspect
actual images and native form at normal/minimum windows. Explicit unsupported integration
guards are allowed for third-party or inverse routes that have not been wired; they must
not replace any source operator, controller or core adapter family.

## Progress and evidence

The source-scoped CPU implementation is prepared for parent integration and acceptance;
this is not an extension-ledger promotion or a claim of real-image/native-UI acceptance.

Implemented paths:

- `sdfx/clora/controllers.py`: attributed AGPL parser/controller port, including all
  documented math helpers and step/prepare semantics. Pure math commands need no opt-in;
  statements, functions, attribute/native-object access use `SDFX_ALLOW_CODE=1`.
  Command failures and nonfinite results produce explicit parameter errors rather than
  silently restoring the original weight. Equal start/end windows avoid division by zero.
  The documented real cube-root helper also accepts negative inputs; upstream's fractional
  Python power incorrectly produced a complex number there (covered by a RED/GREEN test).
- `patches.py`: efficient sequential factors for ordinary LoRA/LoCon, CP middle kernels,
  actual LoHa/Hadamard and LoKr/Kronecker reconstruction, full deltas, IA3 and sparse
  weight residuals. Full-factor LoKr ignores alpha exactly as upstream does; decomposed
  factors use alpha/rank. No base parameter is fused or rewritten.
- Mapping accepts native/Kohya names, both CLIP encoder prefixes and legacy SGM block
  names. Root conv/time/additional-embedding layers have explicit native mappings;
  generic SGM conversion uses the installed diffusers helper. Unmatched layers/tensor
  families are errors, not ignored adapter fragments.
- `runtime.py`: per-job module hooks, per-branch text and denoiser routing, CFG/PAG batch
  layout, current-pass negative scope and current-pass single-block exception. Distinct
  hires prompts carry their own complete positive/negative source metadata; future hires
  weights cannot leak into the base negative path. Explicit unplaced `LoraRef`s are
  distinguished from prompt-placed adapters. Global hires totals also reach prepare-step
  commands in both passes; inactive hires settings do not extend img2img clocks. Every
  scope and hook is removed on error.
- `Engine`: source-tag-preserving encoding, native scheduled embeddings, per-pass branch
  hook clocks and a final cleanup boundary. Ordinary paths remain unchanged when absent
  or disabled. The existing Neutral composition math and its compatibility guards are
  reused, with a small optional `set_branch` lifecycle callback.
- Native `ComposableLoraPlugin`: all seven source controls, examples and execution help.
  Generator resolves repeated adapter files once while retaining branch weights/tags.
  PNG infotext restores all six option fields into both requests and the native form.
  The parent must merge this focused MainWindow restoration with its newer
  `restorePlugins` helper rather than replacing other tools' restoration paths.
- Weight plots are additional diagnostic gallery images (as upstream appends a plot),
  with adjacent JSON recording actual per-pass/branch/adapter weight means and layer
  ranges. Ordinary final images are untouched; enabling the plot adds one result image.

## Review-driven repairs and CPU evidence

Initial controller/patch/runtime tests failed for missing implementations; the Engine
routing regression failed for missing composition metadata. Independent read-only review
then reproduced three defects with RED tests: legacy SGM input-conv mapping, omitted
sparse LoRA/LoCon residuals, and incorrect full-factor LoKr alpha. All were repaired and
their regressions pass. A fourth RED/GREEN regression fixed native distinct-hires negative
weight/single-block scope leakage. No unsupported core format was removed to fix a test.

CPU tests cover source step/ramp/switch/alternation rules, repeated weights, custom-code
gating, function commands, every documented native command context field, prepare-step
opt-in, all implemented tensor families, real module CFG row isolation, exact parameter
restoration and exception cleanup. Two tests instantiate actual tiny diffusers UNets on
CPU: ordinary conditioning and SDXL `text_time`/pooled conditioning, batch two, comparing
composition against independently scoped predictions.

Focused regression runs include the existing Neutral runtime, Fusion source corpus and
engine paths, LoRA Block Weight and extension hook tests. They pass with only the four
known torchsde endpoint warnings: **194 passed** in the torch CPU venv. The selected
no-torch set passed **50 tests with 8 skipped**. Final commit is reported to the parent;
no full-suite, Scala, GPU generation or rendered app claim is made by this agent.

Header-only checks against canonical SDXL models on torch's `meta` device found zero
unmatched groups: alraune 986; Seifuku 788; BN LoCon 788; MonsterGirl LoKr 788. No model
or adapter tensor data was loaded for that check. Separately, the parent ran actual
GPU-loaded-pipe mapping preflight for all six local files. Its report is
`outputs/verify-clora-mapping/report.json` in the main checkout, recorded against
`patches.py` SHA256 `aa86876dff68508519f40d23b068dbc425e30b7c15bfbf88b0408c301a3f885f`:
SD1.5 files each map 264 modules; SDXL alraune maps 986; the other three map 788 each.
That report explicitly covers loading/key mapping/Patch construction, not adapter
forward, output shapes, matrix multiplication or image generation.

## Explicit compatibility limits

These are integration boundaries, not omissions from the source core:

- Native LoRA Block Weight/start/stop fields with Composable LoRA are rejected. Existing
  behavior when disabled is unchanged; source controllers and `cmd layer_name` implement
  this feature's own time/layer control. Separate nonzero TE/UNet multipliers work; the
  zero-TE/nonzero-UNet ratio case is explicitly unsupported.
- PhotoMaker's separate encoder and img2img Alternative's separate inversion route are
  rejected when actual adapters are present. NegPiP signed marker channels are rejected
  when signed terms occur. The existing seven composed-Neutral guards remain intact.
- No general compatibility promise is made for every external adapter, offload or
  attention processor. Ordinary Fusion, extended Neutral, Heun, batch, hires and img2img
  have harness cases; source functionality does not depend on accepting unrelated hooks.
- Core LoHa, IA3, full-rank and CP math has meaningful CPU evidence. There is no local
  trained LoHa/IA3/full-rank/CP adapter checkpoint, so real-image validation of those
  particular formats remains unmeasured rather than relabeled optional.

## Parent GPU and native acceptance commands

From the integrated checkout, use fresh output directories and run sequentially:

```powershell
& python/.venv-smoke/Scripts/python.exe scripts/dev/composable_lora_check.py python/.venv-smoke/Scripts/python.exe D:/stable-diffusion-webui-reForge/models/Stable-diffusion/animekawa_v10.safetensors outputs/verify-composable-sd15 --allow-code
& python/.venv-smoke/Scripts/python.exe scripts/dev/composable_lora_check.py python/.venv-smoke/Scripts/python.exe D:/stable-diffusion-webui-reForge/models/Stable-diffusion/waiIllustriousSDXL_v170.safetensors outputs/verify-composable-sdxl --sdxl --allow-code
```

The harness uses the local adapter paths listed above by default, with explicit override
flags. It runs 30 SD1.5 / 32 SDXL short generation cases, plus the source Python-function
case when `--allow-code` is selected. Four guarded-error cases run before the final exact
restoration check; without code opt-in, a fifth verifies the existing execution gate.
Default sampling is eight steps at 512/768 square, with small hires/img2img passes.

Assertions include exact ordinary no-adapter/disabled/restored pixels, exact zero-adapter
AND composition, actual adapter output effect, image counts/dimensions, non-flat outputs,
all adapter hashes, real per-branch plot weights and continued global hires steps.
Artifacts include complete `protocol.jsonl`, incremental `results.json`, stderr, original
PNGs, labeled contact sheet, plots and trace JSON. Review the actual images; numeric
checks do not establish identity fidelity or spatial separation.

Parent-only remaining gates: munit suites (ComposableLoraSuite and GeneratorSuite), full
required regression sets, UI assembly, both real-model harness runs, actual AppServices
queue/replay, PNG-to-native-form restoration, rendered controls at normal/minimum window
sizes, source notice collection and ledger promotion. The advanced command setting is
local authorization and is never enabled by loading an image or its saved metadata.


### Native trace retention follow-up

The source-compatible appended plot remains an ordinary image; effective-weight JSON is a persistent auxiliary output under `outputs/intermediates/composable-lora/<uuid>.json`, independent of request `.tmp` image paths. Native prepare injects the persistent output root only into sent hook configuration, so recorded JSON/PNG parameters do not acquire machine paths. `GenerationResult.composableLoraTrace` types the retained manifest path, and Generator logs `Composable LoRA trace: <path>` without registering it as a temporary image or deleting it with images. The normal gallery already excludes the root `intermediates` directory.

RED: new Python retention test showed the old JSON inside `.tmp`; GREEN: runtime suite 13 passed, all Composable CPU suites 37 passed; targeted Composable + existing Neutral/Fusion/LBW regression 158 passed with four known torchsde endpoint warnings. Scala ProtocolSuite and GeneratorSuite add backward decode/roundtrip and real fake-engine filesystem retention/logging/image cleanup assertions; root must execute them (child does not run sbt).

Manual native AppServices helper (root GPU only), once for each family:

```
ui/Test/runMain sdfx.ui.RealComposableLoraCheck <python> <checkpoint-directory> <checkpoint-name> <lora-directory> <lora-name> <output-directory>
```

Use SD15 `animekawa_v10` with `20_yui`, SDXL `waiIllustriousSDXL_v170` with `alraune_v1_illustrious_b-000007`, and the main absolute smoke Python. It verifies actual engine feature/loading, source placement through native prepare and unique adapter resolution, parameter JSON and PNG infotext replay, exact no-adapter/restored ordinary pixels, actual adapter effect, source additional plot count, nonempty persistent trace records/logs, and no request-temp orphans. It writes `native-composable-report.json`; it never persists settings. This helper uses AppServices directly and does not claim visual UI screenshot coverage.
