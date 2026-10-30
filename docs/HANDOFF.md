# Handoff: bundling the A1111 extension index into stablediffusion-nui

## 2026-10-30 update: VRAM Estimator

Index 105 / issue #136 has native four-mode CUDA calibration, reserved/allocated peak baselines, four charts, bounded atomic complete/partial profile history and file import/export, strict runtime identity matching and visible-form live estimates. Actual SD1.5/SDXL sweeps and held-out predictions, owned/protected cancellation with unrelated generation preserved, actual bounded CUDA OOM/recovery and assembled-jar native service/view acceptance passed. See vram-estimator-verification.md for exact evidence and limits. Whole Python passed; whole Scala passed 1292 tests with one skip, then the four focused history/session tests and UI compile/assembly passed. MIT notices cover 148 projects. Ledger: 111 done, 191 todo, 51 not applicable. This is the sixth issue of the twenty-issue goal; no release has been published. Next: select and investigate the seventh distinct issue without narrowing its original scope. #121 spans three different style libraries; its complete scope is broader than one prompt selector.

## 2026-10-30 update: StylePile

Index 40 / issue #122 has native Script `stylepile`, full pinned GPL catalog (512 references), searchable lazy previews, weighted style/preset composition, X/Y/A/B/C and noise variables, prompt insertion helpers, custom image/inspiration overlays and interruption-safe sequences. Actual native SD1.5/SDXL configured-style/manual RGB and img2img RGB match exactly; four-image order/seeds and partial interruption/recovery pass. See `stylepile-verification.md`. Whole Scala passed 1289 tests with one skip; assembly and assembled-jar-only native/resource checks passed. All 517 bundled source-file hashes verified in staged Git and jar. Notices cover 147 projects. Current ledger: 110 done, 192 todo, 51 not applicable. This is the fifth issue of the active twenty-issue release goal; release remains unpublished.


## 2026-10-30 update: Steps Animation

Index 57 / issue #120 has native Script `steps_animation`: denoised-step movies, none/blend/MCI interpolation, duration/fps, five codecs, initial-frame skipping, optional export/cleanup, relative output locations, debug logging and bounded interruptible partial export. Actual native Generate/Interrupt/recovery and txt2img/img2img/hires checks passed on SD1.5/SDXL with exact enabled/off final RGB. See `steps-animation-verification.md`. Whole Python passed; whole Scala passed 1269 tests with one skip; assembly passed. Notices cover 146 projects. Current ledger: 109 done, 193 todo, 51 not applicable. This is fourth issue toward the twenty-issue release goal; no release published yet.


## 2026-10-30 update: PNGINFO Injection

Index 336 / issue #98 has native opt-in `pnginfo_injection`, saved Hashes merge, confirmed local resource fingerprints and actual wildcard-source byte provenance. See `pnginfo-injection-verification.md` for real SD1.5/SDXL native PNG, embedding and ADetailer checks, SD1.5 ControlNet check, cache-mutation handling and explicit directory/Civitai limits. Whole Python passed; whole Scala had five Windows drive-probe timeouts and the focused 22-test recheck passed. Assembly and final native checks passed. Current ledger is 108 done, 194 todo, 51 not applicable. This is the third wave of the active twenty-issue goal; release remains unpublished.

## 2026-10-30 update: GPU temperature guard

Index 186 / issue #54 has native AlwaysOn `gpu_temperature_guard`, interruptible step-end hysteresis, bounded NVIDIA/ROCm/WMI temperature queries, physical device/sensor selection and pause/resume/timeout logging. Current ledger: 107 done, 195 todo, 51 not applicable. See `temperature-guard-verification.md` for actual SD1.5/SDXL native Generate/Interrupt/recovery checks, exact guarded/off RGB parity across txt2img/img2img/hires and AMD fixture-only limits. Whole Python regression passed; whole Scala regression passed 1264 tests with one skip. The real Windows bulk scanner takes several minutes; preserve its live process rather than restarting for quiet logs.

The active twenty-issue release progress is tracked in `twenty-issue-release-progress.md`. No release has yet been published for that goal. The separate temporary worktree `sdnui-temperature-native-check` was used for native acceptance while the main checkout's regression ran; its product sources matched the current checkout. Canonical changes and commits remain in this checkout.

## 2026-10-30 update: color correction

Index 261 / issue #15 now has native Extras and script/plugin `color_correction`: A1111, independently implemented AdaIN and Wavelet, optional original-input reference, independent comparison outputs and strength. Current generated ledger: 106 done, 196 todo, 51 not applicable. Read `color-correction-verification.md` for real SD1.5/SDXL and native button checks, broad regression results and numerical/visual limits. This is one issue toward the active twenty-issue release request; no new release has been published.

The GPU/test venvs were recreated under `python/`. GPU runtime uses torch 2.7.1+cu128 and torchvision 0.22.1 with pinned requirements. Run whole Python tests with `CUDA_VISIBLE_DEVICES=-1`; actual model verification runs with GPU visible. License collector accepts only LF/CRLF variants of trusted pinned content, retaining digest checks. Existing test fixtures were updated for wdBatch and current startup/provenance contracts.

Written 2026-10-26 when work moved from Claude Code to Codex. Read this first, then `AGENTS.md`, `README.md`,
`docs/PLUGINS.md` and `docs/EXTENSIONS.md`.

## Current product naming

The product is **Stable Diffusion NUI**, repository `kmizu/stablediffusion-nui` (formerly Stable Diffusion FX / SDFX).
Current builds produce `target/stablediffusion-nui.jar`; the v0.03 Windows package uses
`SDNUI/SDNUI.exe` and `SDNUI-v0.03-windows-x64.zip`. Internal Scala/Python `sdfx` names,
`SDFX_*` environment variables and existing settings stay compatible. Earlier release names,
artifact hashes and verification evidence are historical records and must not be renamed retroactively.
The naming change does not by itself establish that v0.03 acceptance or publication is complete.

## The goal

The owner's request (Japanese, verbatim): 「他のやつでreforceのWeb UI（available）からインストールできるやつは全部コミコミにしたい」
and 「もちろん各プラグインのライセンスで必要なのがあったら置いてな」.

Bundle every extension listed in reForge's *Extensions > Available* tab (the A1111 extension index, 353 entries,
snapshot in `docs/extensions-index.json`) into this app as built-in plugins, and ship the licence notices each one
needs. The owner also asked for **high-quality implementation and UI** (「high qualityな実装とUIを頼むで」): quality wins over
count. Every new plugin is GPU-checked on real models before it is marked done.

The app itself is complete and working: Scala 3 + JavaFX client, Python diffusers engine over stdio JSON Lines, MCP
server, 40 rewrites of the reForge built-ins. Repo: private GitHub `kmizu/stablediffusion-nui`, branch `main`.

## Where things stand

| | Count |
|---|---|
| Index entries done | 105 |
| Index entries todo | 197 |
| Not applicable (web-UI CSS/UX only, accounts, obsolete, external apps, server ops) | 51 |

**Latest verified wave (2026-10-27):** index 122 `core:temporal-lab` provides DFI, deflicker, fuse and frame extraction in a native workspace. The integrated CPU worker, native UI, model-image smoke, Scala and Python checks are recorded in `docs/VERIFICATION.md`, together with source attribution and limitations. The remaining extension issues and final release are separate ongoing work.

**Previous verified wave (2026-10-27):** index 333 `core:nsfw-card-blur` is accepted as native card
previews with shared Show/Blur/Hide. The reduced core gate passed offline, CPU dry-run and actual SD1.5/SDXL
checks; the broad Scala run's two fixture timeouts passed focused serial rechecks. See
[`VERIFICATION.md`](VERIFICATION.md) for precise scope, evidence and limits. For the Windows v0.01 runtime ZIP, release tag and remaining-issue tracker, check the GitHub release and issues.
The requested stopping point is after the download release and issue tracking; this does not complete the
353-entry goal.

**Previous verified wave (2026-10-27):** index 30 `core:config-presets` and index 31 `core:preset-utilities` have
separate native profiles. Config uses mode-specific collections, selected-field overlays and Reapply; Utilities
uses a shared collection, Quick/Detailed save and a choice of current versus component-default apply. The
complete MIT Config starter presets are included, while the unlicensed Utilities source supplies behaviour only.
The corrected implementation passed **964 Scala tests**, **36 native pane checks**, **16 CPU** checks and **19 each**
on real SD1.5 and SDXL models, with trained ControlNet. Python passed 867/1 skip and no-torch 397/82 skips; the
assembled jar passed an isolated Java 21 launch. BuiltinsSuite 20/20 and NativePresetsCoverageSuite 4/4 passed
after the precise two-entry ledger promotion. See [`VERIFICATION.md`](VERIFICATION.md) for the source pins,
artifact hash, parity limits and review history, and
[`native-presets-compatibility.md`](native-presets-compatibility.md) for interchange rules.

`scripts/extensions_matrix.py` is the single ledger. Its `S` dict maps each index entry to `done:<plugin id>`,
`todo:<target>` or `na:<reason>`, and the `IMPLEMENTED` set promotes `todo:<target>` to done. Several entries may
share one target (for example every background remover maps to `remove_background`). To merge a target into another,
edit its `S` entry (done for `pixelization -> pixel_art`, `mask2background -> inpaint_background`).

**Previous verified wave (Codex,2026-10-27):** index253 `core:api-payload` is implemented. Each generation mode has a
collapsed entry to one shared native inspector, with exact submitted request bodies, separate submitted/model
context and optional bounded image diagnostics. Script sub-generations have distinct identities and terminal
states. History, tree traversal, raw previews and diagnostic delivery are bounded; explicit Copy preserves the
whole selected document. Original init-image contents remain omitted even when image diagnostics are enabled.
Final source passed725 Scala tests,867 Python tests/1skip and397 noTorch tests/82skips, assembly, an offline
native pane check and actual CPU18/SD1.5 14/SDXL14 native acceptance checks. Both model runs preserve exact fixed-seed
RGB/body parity. Safe clipboard checks passed and restored the formats captured in these runs. Read the precise
limits, source pins, artifact hash and paths in `docs/VERIFICATION.md`; the353-entry goal remains unfinished.

**Previous verified wave (Codex,2026-10-26):** index320 `core:prevent-interrupt` and index309 `core:hardware-info` are
integrated. The footer protection is process-global/defaultOff and covers actual native/MCP running cancellation
and explicit user restart without blocking shutdown/health recovery or pending-only clearing. Hardware metadata
has native opt-in/remark controls, compact actual-device facts, per-image batch timing, Extras provenance and
bounded informational Restore/Send-to notices with no offline engine startup. Source behavior and original
license limitations are recorded in the owned plans and128-project notices; no unlicensed upstream code was copied.
Final integrated checks:596Scala tests plus assembly,867Python/1skip,397noTorch/82skips, six actual native helpers
(CPU+SD1.5+SDXL for each feature), all successful. Reports/logs and precise limits are in `docs/VERIFICATION.md`.
The completed API payload plan is `docs/superpowers/plans/2026-10-26-core-api-payload.md`.

Done in this session, all GPU-verified unless marked:

- Prompt and sequence scripts (15): dynamic prompts with wildcards, One Button Prompt, seed travel, shift attention,
  LoRA queue and others.
- A1111 prompt editing `[a:b:when]` and alternation `[a|b]`: `python/sdfx/prompt_schedule.py`, with per-step
  embedding swaps in `Engine._schedule_swapper`.
- Sampling hooks (9) in `python/sdfx/sampling_ext.py`: Detail Daemon, Vectorscope CC, Latent Mirroring, Asymmetric
  Tiling, Anti-burn, ReSharpen, Diffusion CG, CFG Schedule, CADS.
- Ultimate SD upscale (`UltimateSdUpscalePlugin.scala`).
- ADetailer (`python/sdfx/adetailer.py`, `Engine._adetailer`, `AdetailerPlugin.scala`), including LAB colour
  matching because baked VAEs lose contrast on the encode/decode round trip.
- LoRA Block Weight (`python/sdfx/lora_blocks.py`, `LoraBlockWeightPlugin.scala`). The `<lora:name:te:unet>` multipliers
  and the `lbw=`, `start=`, `stop=` and `step=` arguments are parsed in `LoraSyntax.parseTag`.
- Checkpoint merger and converter (`python/sdfx/model_tools.py`, `ModelToolPlugins.scala`). Both are blocked over MCP.
- **GPU-verified by Codex on 2026-10-26:** image tools in `python/sdfx/image_tools.py` and `ImageToolPlugins.scala`.
  They cover remove background (rembg or person segmentation), person mask, pixel art, colour enhance, FSR-style CAS
  upscale, depthmap2mask and inpaint background. Real SD1.5 and SDXL images, both masked inpainting paths, and all
  seven plugins through the real Java/MCP app were checked. See `docs/VERIFICATION.md` for scope and evidence.
  Fixed full-range seed overflow and output dimensions in pixel art, an open input file in the engine, and a leaked
  auxiliary mask in the Scala background-inpainting plugin. The 217 todo
  entries remain unfinished. Completing this verification wave does not complete the extension-index goal.
- **Seamless tile inpainting (Codex):** `SeamlessTilePlugin.scala`, id `seamless_tile`. Circularly offsets the input,
  inpaints the central seam in horizontal/vertical/both directions, optionally restores the original arrangement.
  SD1.5 and SDXL checked through the actual app/MCP; repeated tile previews and the form inspected. Neural seam
  repair quality depends on the checkpoint and texture; an ordinary SD1.5 anime checkpoint failed to improve a stone
  texture in some cases. See `docs/VERIFICATION.md`. No new dependency or Python engine hook.
- **Weight Gradient (Codex):** `python/sdfx/weight_gradient.py`, `WeightGradientPlugin.scala`, hook/id `weight_gradient`.
  Full documented syntax with linear/exponential/circular easing, return ramps, step ranges, positive/negative
  prompts, A1111 editing and per-pass clocks. Real SD1.5/SDXL, normal/hires/img2img/Alternative/Heun paths checked;
  constant tags match ordinary weights exactly, and disabling restores the baseline. Actual app-service pipeline
  and UI checked. PhotoMaker combination is explicitly unsupported. No upstream code was copied (no licence).
  See `docs/VERIFICATION.md` and the implementation plan for tested scope and decisions.
- **img2img Alternative quality repair (Codex):** preserves the amplitude recovered by DDIM inversion instead of
  forcing every result to empirical std 1. Noise blending preserves its two endpoints; Randomness=1 matches ordinary
  same-seed generation (matching settings, variation disabled). Decode CFG0/.5 now interpolates correctly, and latent
  inversion explicitly disables pixel-range clipping. Replay infotext now includes original prompts and effective
  decode steps. SD1.5/SDXL real-engine grids, actual app/MCP path and UI were checked. See VERIFICATION.md for limits.
- **NegPiP (Codex):** `python/sdfx/negpip.py`, `NegPipPlugin.scala`, hook/id `negpip`. Separately encodes negative
  terms at absolute weight, preserves attention keys, flips projected values, and masks added padding. Supports
  positive/negative branches, editing, Weight Gradient crossings, batching, PAG, hires, img2img and Alternative.
  SD1.5/SDXL GPU checks include exact no-negative/no-hook restoration; actual app-service path and UI inspected.
  Excessive negative weights can destroy subjects/style. PhotoMaker with negative terms is unsupported. AGPL-3.0
  upstream attribution is collected; no new dependency. See VERIFICATION.md for untested combinations.

UI improvements made this session: `Control.Section` (collapsible groups inside a plugin form), `Plugin.enabledBy`
(greys a control out while its checkbox is off), and the form layout in `W.Form`. Labels longer than 18 characters
go above their control, and the label column stays at least 88 px wide so cards line up.

**Parallel implementation wave (2026-10-26):** `prompt_fusion`, `tokenizer` and `save_intermediate` are integrated.
Prompt Fusion has source grammar/geometry, SDXL pooled conditioning, ordinary-prompt bypass and replay checks.
Tokenizer uses the actual loaded model vocabularies in a native text/ID inspection tab (literal input, not a
simulation of generation conditioning). Intermediate PNG/video artifacts persist separately from final images;
root `outputs/intermediates` is excluded from gallery scans. Both model families, native forms and actual app
queue/replay paths are verified; optional ffmpeg codecs/recipes have independent actual encoding checks. See
the three implementation plans and the final section of `docs/VERIFICATION.md` for exact limits and artifacts.

**Second parallel wave:** `grid_numbers` and `latent_upscale` are accepted after real SD1.5/SDXL, native app/replay,
actual PNG Restore/Send-to, rendered forms and aggregate checks. Grid annotations preserve source images and record
only visible tiles. Latent Upscale encodes before resizing and executes the full requested denoising count.
See the closing VERIFICATION section and their plans.

**Third parallel wave:** `composable_lora` and `core:system-info` are now accepted after source-scope review,
SD1.5/SDXL generation, actual native queue/replay and rendered controls. Composable's source controllers and all
adapter algebra families are implemented; real LoRA/LoCon/LoKr have model evidence, while unavailable trained
LoHa/full/IA3/CP formats have CPU algebra coverage. Keep that distinction. Plots are source-compatible appended
images; trace JSON persists outside `.tmp`. System Info has live diagnostics, native benchmark/history, explicit
metadata selection and preview/submit separation. No cloud submission was sent during acceptance. See both plans
and VERIFICATION for exact scope, paths and limits.

**Fourth native wave:** `core:styles-editor` covers indices163 and248 after SD1.5/SDXL native editing,
data-preservation, category/backup/AES interoperability and exact named-style conditioning checks. The manual
helper's clipboard cleanup failure was fixed before clean final acceptance; it now skips clipboard mutations
when any advertised format cannot be preserved, and always attempts all resource cleanup. See VERIFICATION.

**Native comparison/VAE wave:** `core:image-compare` and `vae_tools` are accepted after independent source/fix
reviews, actual native SD1.5/SDXL image workflows and integrated app/MCP runs. Comparison needs no inference;
it compares existing real-model images without changing them. VAE Tools has Full/Approx NN and pretrained
TAESD/TAESDXL evidence, exact failed decoder-input recovery, quantized image formats and explicit download gates.
Keep codec roundtrip, image reconstruction and approximate previews distinct. See their plans and VERIFICATION.

**Current wave:** the native presets plan for indices 30/31 is complete. Do not restart it as the old shared
`core:presets` target: that name would also mark three unfinished entries done. Index 333 `core:nsfw-card-blur` is implemented and accepted in the native card preview wave.
The 353-entry goal remains active: **104 done, 198 todo, 51 not applicable**, with 138 distinct pending
targets after this promotion. Indices 203 `controlnet_fastload`, 221 `scenario_loader`, 262
`core:state-manager` and 146 `model_presets` remain separate work. See VERIFICATION for core acceptance,
limits and exact evidence. Finish the Windows v0.01 runtime ZIP and release, then document all remaining issues and stop;
do not implement the remaining 198 entries as part of this wave. The independent `core:prompt-tools` audit is under
`outputs/next-core-prompt-tools-audit*`.

For the completed presets wave, the five-entry source audit and manifest remain in main
`outputs/next-core-presets-audit*`, with the full ID map at `outputs/native-config-presets-field-map.md`.
The 132-project notice retains all 130 earlier rows; trained ControlNet assets are local under
`outputs/model-cache/controlnet-acceptance-20261027` and are not distributed. Detailed source reviews, earlier
failed checkpoints, final model reports, Java 21 screenshots and hashes are in [`VERIFICATION.md`](VERIFICATION.md).
The earlier API payload release reports and assembled jar are in `.worktrees/api-payload-integration/outputs`
and `target`.

Neutral Prompt is now marked implemented after an explicit original-scope
acceptance audit; see `docs/superpowers/plans/2026-10-26-neutral-prompt.md` and the closing VERIFICATION section.
All operators, nesting/signed weights, CFG rescale, schedules, native form/formatter/replay and actual AppServices
SD1.5/SDXL generation are verified. Alternative, ControlNet/PAG, IP-Adapter/basic/Plus, Reference, SDXL ControlLLLite,
CADS/CFG Schedule and ordinary PAG batch fixes also have separate tests/matrices. Preserve their invariants.

**Compatibility backlog:** PhotoMaker, SAG, APG/Mahiro/Latent Modifier/Dynamic Thresholding/Detail Daemon with
composed Neutral prompts remain explicitly unsupported in `neutral_runtime.validate`, tests and form help. The
original Runtime task permits clear unsupported combinations; later agent-added notes making every combination
a promotion prerequisite are superseded by the documented acceptance clarification. No original Neutral operator
or model/UI/notice gate was omitted. Keep the seven guards until their own implementations are verified.
The overall 353-entry goal remains active; 198 entries are unfinished. Do not treat this promotion as
completion of that goal or reclassify remaining entries to make it appear complete.

## Remaining targets, most-starred first

sadtalker, facechain, easyphoto, tiled_diffusion, face_swap, segment_anything, animatediff, core:prompt-tools,
deforum, core:tag-autocomplete, mov2mov, core:compile, prompt_library, temporal_kit, train_dreambooth, depth_maps,
regional_prompter, openpose_editor, core:localization, wav2lip, text2video, inpaint_anything, ebsynth_utility,
stable_sr, cn_animation, unprompted, llul, dataset_tag_editor, deoldify, dreamartist, qr_code, infinite_zoom, tipo,
lcm, stylepile, clip_interrogator, prompt_generator, style_selector, aesthetic_gradients, latent_couple,
outpaint_canvas, fabric, manga_maker, model_downloader, multi_subject_render, batch_face_swap, video_loopback,
controlnet_fastload, scenario_loader, core:state-manager, model_organizer, model_keyword, shap_e, prompt_travel, lora_control,
gif2gif, replacer, cd_tuner, riffusion, smart_process, haku_img, infinity_grid,
core:panorama-viewer, loopback_scaler, daam, text2prompt, prompt_translate, prompt_gallery,
refiner, incantations, old_photo_restoration, abysz_lab, aesthetic_score, steps_animation,
nsfw_checker, image_editor, embedding_merge, rich_text, mosaic_outpaint, cross_attention_viz, hypernetwork_training,
inspiration, vram_estimator, sonar, dump_unet, llm_prompt, training_picker, model_presets,
lama_cleaner, kandinsky, hypernetworks, characteristic_guidance, hotshot_xl, live_portrait,
booru_prompt, depth_image_io, creaprompt, embedding_editor, semantic_guidance, custom_diffusion,
encrypt_image, inpaint_difference, depth_library, rpg_diffusion, custom_hires_fix,
img2img_hires_fix, depth_anything, agent_attention, conditioning_highres_fix, hires_fix_tweaks,
panorama_tools, chara_searcher, lua_scripting, deepdanbooru, gpu_temperature_guard, corridor_crawler,
old_sd_firstpasser, temporal, batch_interrogate, token_downsampling, embedding_inspector, demofusion,
prompt_expansion, color_correction, video_extras, clip_io, face_manipulation, ditail,
minecraft_schematic, img2img_workflow, ocr, controlnet_preprocess_extras, triposr, manga_inpainting,
pnginfo_injection, masked_content_extras, metadata_utils, waifu2x

Suggested next steps, by value and fit with what exists:

1. **Native tools**: `core:prompt-tools` and the remaining separate preset-audit targets still need
   implementation and source-scope verification.
2. **Regional prompting**: regional_prompter and latent_couple, via attention-level region masks.
   tiled_diffusion can largely reuse the existing `multidiffusion` hook plus noise inversion and region prompts.
3. **Small, cheap ones**: img2img/custom/conditioning hires fix variants, color_correction,
   waifu2x, deepdanbooru (the WD14 tagger already exists in `extras.caption_image`), batch_interrogate,
   clip_interrogator, aesthetic_score, nsfw_checker.
4. **Big ones**: animatediff, segment_anything, inpaint_anything, deforum, text2video, face_swap (insightface is
   installed in `.venv-smoke`), training (dreambooth, hypernetworks).
5. **`core:*` targets**: these are web-UI features with an equivalent in the JavaFX UI. Tag autocomplete goes in the
   prompt editor. Localization should read A1111 localization JSON.

## How a new plugin is built (follow the existing patterns)

- **Engine side**: add a module under `python/sdfx/`. A sampling hook is wired from `extensions.install()` (UNet
  forward/pre-hooks and `callback_on_step_end` through `_chain_callback`). A non-generation tool is a `BaseEngine`
  method and must be added to `QUEUED_TYPES` in `server.py` (see `image_tool`, `merge_models`). Every new hook name
  goes in three places: `hooks.KNOWN_HOOKS`, `Engine.features()` and `dryrun.features()`.
  `tests/test_extensions.py::test_known_hooks_cover_extensions` checks this.
- **Scala side**: AlwaysOn plugins extend `HookPlugin` (`hookFields`, `infotextKeys`, `xyzAxes`). Scripts extend
  `Plugin` with `run(p, values, ctx)` and Extras extend `ExtrasPlugin.transform`. Register each in
  `StandardPlugins.scala`.
  - **Plugin id rule**: the plugin `id` must equal the matrix target name. `BuiltinsSuite` checks non-core
    `done:<id>` entries in `docs/extensions-status.json` against registered plugins; `core:*` entries are not inference plugins.
  - **Infotext**: use the extension's own keys where it had them, so infotext stays compatible.
- **Writing quality code**: re-implement from documented behaviour. Copy code only from permissively or AGPL/GPL
  licensed sources, and never from repositories without a licence (LoRA Block Weight was done from its README only).
  Write pytest for the math, munit for the Scala side, then GPU-check.
- **Ledger, docs, licences**: add the target to `IMPLEMENTED`, then run `python scripts/extensions_matrix.py` (rewrites
  `docs/EXTENSIONS.md` and `docs/extensions-status.json`). Then run `python scripts/third_party.py` (needs a logged-in
  `gh`; fetches licences into `licenses/` and rewrites `THIRD_PARTY_NOTICES.md`). New optional Python dependencies
  go in the `BASE` list there and as a commented line in `python/requirements.txt`. Do not pin them as hard
  requirements.
- **Commits**: one commit per wave. The message says what was added and how it was verified.

## Environment (Windows 11, RTX 3090, about 9 GB free RAM, so watch memory)

- **sbt**: `%LOCALAPPDATA%\Coursier\data\bin\sbt.bat`. Not on Git Bash's PATH; call it by full path.
  - All tests: `sbt.bat -batch test`.
  - One suite: `sbt.bat -batch "plugins/testOnly sdfx.plugins.AdetailerSuite"`.
  - Fat jar: `sbt.bat -batch ui/assembly` writes `target/stablediffusion-nui.jar`.
- **Java 21**: `C:\Program Files\Java\jdk-21\bin\java.exe`. The `java` on PATH is 17 and is too old for the jar.
- **Python venvs** (in `python/`):
  - `.venv-smoke` is the GPU venv: torch 2.9 cu128, diffusers, ultralytics, mediapipe, insightface, rembg. Use it
    for real-model checks and the full pytest run.
  - `.venv-test` has no torch. It checks that the torch-free parts and the dry-run engine still pass.
- **Checkpoints**: `D:\stable-diffusion-webui-reForge\models\Stable-diffusion`. SD1.5 test model `animekawa_v10`,
  SDXL test model `waiIllustriousSDXL_v170`.
- **LoRAs**: `..\models\Lora`. `20_il-shirobana.safetensors` is SDXL.
- **Shell gotchas**:
  - A token-saver hook blocks long heredocs into `.scala` files, so write them with a file editor.
  - Python text-mode writes on Windows give CRLF. Always write with `newline="\n"`; `.gitattributes` enforces LF.
  - Do not restart a background job the OS killed for low memory unless the owner asks.

## Verifying

| What | Command / file |
|---|---|
| Scala tests | `sbt.bat -batch test` (latest: 725 passed, 2026-10-27) |
| Python tests | `python/.venv-smoke/Scripts/python.exe -m pytest` with `CUDA_VISIBLE_DEVICES=-1` (867 passed, 1 skipped) and `.venv-test` (397 passed, 82 skipped), 2026-10-27 |
| Dry-run smoke | `scripts/smoke.ps1` / `scripts/smoke.sh` |
| Engine on GPU, one case per process | `scripts/dev/gpu_check.py <venv python> <out dir> <case>`; cases `sched_*`, `ext_*`, `ad_*`, `image_*`, `weight_*`, `negpip_*` |
| Script plugins through the real app over MCP | `scripts/dev/write_settings.py`, then `scripts/dev/mcp_scripts.py "C:\Program Files\Java\jdk-21\bin\java.exe" <script ids>` |
| LoRA Block Weight / merger checks | `scripts/dev/lbw_check.py`, `scripts/dev/merge_check.py` |
| UI screenshot | `java -jar target/stablediffusion-nui.jar --dry-run --home <dir> --screenshot out.png --exit-after 8` (also writes `out-panel.png`, the full form). Expand a plugin by putting `{"expanded": {"txt2img.plugin.<id>": true}}` in `<home>/config/ui-state.json` |

The `scripts/dev/` files contain absolute paths of this machine; adjust them if anything moves. `write_settings.py`
creates `config/settings.json` (gitignored); delete it afterwards.

Look at the images, not only the numbers. Vectorscope and Diffusion CG were "working" by every metric but visibly
broken (pseudo-inverse colour directions were too large). The ridge-regularised `rgb_to_latent_dir` fixed them.

## Known issues and decisions

- **IP-Adapter pairing quality**: native basic SDXL IP-Adapter at scale .6 visibly shifts style/contrast on
  waiIllustriousSDXL_v170, even without Neutral Prompt. No-op/restoration remain exact. The basic/Plus grids in
  `outputs/verify-neutral-ip-*` record this limit; compatibility checks do not promise a fixed artistic result.
  Model-offload reload was checked on SD1.5; SDXL reload was checked in GPU mode, not sequential offload.

- **img2img Alternative reconstruction limits**: the strong unintended contrast was traced to empirical noise
  normalization and repaired. Same SDXL DPM++ CFG6 case: near-white 14.19% -> 4.51%, near-black 28.67% -> 14.16%.
  Reconstruction still depends on CFG/count/sampler: matched DDIM/CFG1 was close to source (1.06%/1.71% versus
  1.11%/1.61%), not pixel-identical. It is a DDIM approximation, not the legacy k-diffusion Euler/sigma-adjustment
  implementation. Changing settings can still change contrast/composition. No output recolouring was added.
  Reports: `outputs/verify-alternative-final/alt_quality_{sd15,sdxl}.json`; diagnostics and test plan preserve the
  rejected clipping/CFG hypotheses. Pixel-range clipping is guarded but was already off in this particular model.
- **Weight Gradient clocks**: each sampling pass restarts the ramp and uses its actual logical inference steps
  (second-order callback stages do not advance twice). Existing A1111 prompt-editing hires offsets are unchanged.
  PhotoMaker uses separate text-only encoding; combining it with gradient tags raises an unsupported error.
- **NegPiP strength/compatibility**: SD1.5/SDXL tested; SD2's CLIP route is implemented but not GPU-checked here.
  Strong negative values can remove the subject or break the image. The wrapper preserves native attention
  processors and parameter paths; real ControlNet/IP-Adapter weights together with offload remain unverified.

- **LoRA residue in fp16**: once a LoRA has been loaded into the pipeline, the UNet output differs by about 0.002 in
  fp16 even after the LoRA is disabled or unloaded. Weights, hooks, flags and the memory layout were all verified
  unchanged; the cause is not found. With ancestral samplers this changes a later no-LoRA image with the same seed
  (mean pixel difference about 27). It is not caused by the LoRA Block Weight code. It is worth a look if exact
  reproducibility across a session matters.
- **ADetailer default differs from the extension**: colour correction is on by default, a deliberate quality
  choice. YOLO needs `ultralytics` (optional); `opencv_face` works without it.
- **Model tools and MCP**: the merger and converter write next to model A or the source model and are GUI-only
  (`McpTools.GuiOnlyScripts`).
- **Pixelization**: the A1111 pixelization extension uses a neural model; our `pixel_art` is algorithmic and covers
  both targets.
- **Licences**: the app is AGPL-3.0 because it ports AGPL (A1111, Forge, reForge) and GPL (ComfyUI, Fooocus) code.
  `THIRD_PARTY_NOTICES.md` explains which projects were ported, re-implemented, depended on or used as data.


### Temporal Lab delivery candidate

Temporal Lab implements reference-guided DFI, deflicker, fuse, frame extraction, previews and playback in the native UI. Source algorithms and controls were informed by AbyszOne/Abysz-LAB-Ext at `deff0d4b4a208f862299b14b93e088f613373153`; the pinned GPL-3.0 license is archived under `licenses/AbyszOne__Abysz-LAB-Ext.txt`, and the notice collector records the source. See `docs/VERIFICATION.md` for bounded evidence and limits. Inter Blur is active here, video frame timestamps are computed from frame indices, and complete results use a new directory; full numerical parity with the upstream extension is not claimed.

This isolated issue candidate was assembled from the reviewed Temporal Lab work. Its final integrated checkout and regenerated notices/extension matrix require their own verification before release. Avoid concurrent hostile replacement of path ancestors during processing; video codec allocations can occur before the decoded-frame guard, so the working-memory estimate is not a hard process-memory cap.
