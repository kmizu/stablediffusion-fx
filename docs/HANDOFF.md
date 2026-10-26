# Handoff: bundling the A1111 extension index into stablediffusion-fx

Written 2026-10-26 when work moved from Claude Code to Codex. Read this first, then `AGENTS.md`, `README.md`,
`docs/PLUGINS.md` and `docs/EXTENSIONS.md`.

## The goal

The owner's request (Japanese, verbatim): 「他のやつでreforceのWeb UI（available）からインストールできるやつは全部コミコミにしたい」
and 「もちろん各プラグインのライセンスで必要なのがあったら置いてな」.

Bundle every extension listed in reForge's *Extensions > Available* tab (the A1111 extension index, 353 entries,
snapshot in `docs/extensions-index.json`) into this app as built-in plugins, and ship the licence notices each one
needs. The owner also asked for **high-quality implementation and UI** (「high qualityな実装とUIを頼むで」): quality wins over
count. Every new plugin is GPU-checked on real models before it is marked done.

The app itself is complete and working: Scala 3 + JavaFX client, Python diffusers engine over stdio JSON Lines, MCP
server, 40 rewrites of the reForge built-ins. Repo: private GitHub `kmizu/stablediffusion-fx`, branch `main`.

## Where things stand

| | Count |
|---|---|
| Index entries done | 84 |
| Index entries todo | 218 (153 distinct targets) |
| Not applicable (web-UI CSS/UX only, accounts, obsolete, external apps, server ops) | 51 |

`scripts/extensions_matrix.py` is the single ledger. Its `S` dict maps each index entry to `done:<plugin id>`,
`todo:<target>` or `na:<reason>`, and the `IMPLEMENTED` set promotes `todo:<target>` to done. Several entries may
share one target (for example every background remover maps to `remove_background`). To merge a target into another,
edit its `S` entry (done for `pixelization -> pixel_art`, `mask2background -> inpaint_background`).

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
  auxiliary mask in the Scala background-inpainting plugin. **Next task: prompt-embedding hooks** below; the 218 todo
  entries remain unfinished. Completing this verification wave does not complete the extension-index goal.
- **Seamless tile inpainting (Codex):** `SeamlessTilePlugin.scala`, id `seamless_tile`. Circularly offsets the input,
  inpaints the central seam in horizontal/vertical/both directions, optionally restores the original arrangement.
  SD1.5 and SDXL checked through the actual app/MCP; repeated tile previews and the form inspected. Neural seam
  repair quality depends on the checkpoint and texture; an ordinary SD1.5 anime checkpoint failed to improve a stone
  texture in some cases. See `docs/VERIFICATION.md`. No new dependency or Python engine hook.

UI improvements made this session: `Control.Section` (collapsible groups inside a plugin form), `Plugin.enabledBy`
(greys a control out while its checkbox is off), and the form layout in `W.Form`. Labels longer than 18 characters
go above their control, and the label column stays at least 88 px wide so cards line up.

## Remaining targets, most-starred first

sadtalker, facechain, easyphoto, tiled_diffusion, face_swap, segment_anything, animatediff, core:prompt-tools,
deforum, core:tag-autocomplete, mov2mov, core:compile, prompt_library, temporal_kit, train_dreambooth, depth_maps,
regional_prompter, openpose_editor, core:localization, wav2lip, text2video, inpaint_anything, ebsynth_utility,
stable_sr, cn_animation, unprompted, llul, dataset_tag_editor, deoldify, dreamartist, qr_code, infinite_zoom, tipo,
lcm, stylepile, clip_interrogator, prompt_generator, style_selector, aesthetic_gradients, latent_couple,
outpaint_canvas, fabric, manga_maker, model_downloader, multi_subject_render, batch_face_swap, video_loopback,
core:presets, model_organizer, core:system-info, model_keyword, shap_e, prompt_travel, prompt_fusion, lora_control,
negpip, gif2gif, replacer, cd_tuner, riffusion, core:api-payload, smart_process, haku_img, infinity_grid,
neutral_prompt, core:panorama-viewer, loopback_scaler, daam, tokenizer, text2prompt, prompt_translate, prompt_gallery,
composable_lora, refiner, incantations, old_photo_restoration, abysz_lab, aesthetic_score, steps_animation,
nsfw_checker, image_editor, embedding_merge, rich_text, mosaic_outpaint, cross_attention_viz, hypernetwork_training,
inspiration, vram_estimator, sonar, save_intermediate, dump_unet, llm_prompt, training_picker, model_presets,
lama_cleaner, kandinsky, hypernetworks, core:styles-editor, characteristic_guidance, hotshot_xl, live_portrait,
booru_prompt, latent_upscale, depth_image_io, creaprompt, embedding_editor, semantic_guidance, custom_diffusion,
core:image-compare, encrypt_image, inpaint_difference, depth_library, rpg_diffusion, custom_hires_fix,
img2img_hires_fix, depth_anything, agent_attention, conditioning_highres_fix, hires_fix_tweaks,
panorama_tools, chara_searcher, lua_scripting, deepdanbooru, gpu_temperature_guard, corridor_crawler,
old_sd_firstpasser, temporal, grid_numbers, batch_interrogate, token_downsampling, embedding_inspector, demofusion,
prompt_expansion, weight_gradient, color_correction, video_extras, clip_io, face_manipulation, ditail,
minecraft_schematic, img2img_workflow, ocr, controlnet_preprocess_extras, triposr, manga_inpainting,
pnginfo_injection, masked_content_extras, metadata_utils, hardware_info, waifu2x, core:nsfw-card-blur, vae_tools,
core:prevent-interrupt

Suggested next steps, by value and fit with what exists:

1. **Prompt-embedding hooks**: negpip, prompt_fusion, weight_gradient, neutral_prompt and composable_lora. They build
   on `prompt_schedule.py` and `_schedule_swapper`.
2. **Regional prompting**: regional_prompter and latent_couple, via attention-level region masks.
   tiled_diffusion can largely reuse the existing `multidiffusion` hook plus noise inversion and region prompts.
3. **Small, cheap ones**: tokenizer (engine `tokenize` request), latent_upscale,
   img2img/custom/conditioning hires fix variants, grid_numbers, save_intermediate, color_correction, vae_tools,
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
  - **Plugin id rule**: the plugin `id` must equal the matrix target name. `BuiltinsSuite` checks every
    `done:<id>` in `docs/extensions-status.json` against the registered plugins.
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
  - Fat jar: `sbt.bat -batch ui/assembly` writes `target/stablediffusion-fx.jar`.
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
| Scala tests | `sbt.bat -batch test` (6 modules, about 470 tests, all green at handoff) |
| Python tests | `python/.venv-smoke/Scripts/python.exe -m pytest` (254 green) and `.venv-test` (221 green, 15 skipped) |
| Dry-run smoke | `scripts/smoke.ps1` / `scripts/smoke.sh` |
| Engine on GPU, one case per process | `scripts/dev/gpu_check.py <venv python> <out dir> <case>`; cases `sched_*`, `ext_*`, `ad_*` |
| Script plugins through the real app over MCP | `scripts/dev/write_settings.py`, then `scripts/dev/mcp_scripts.py "C:\Program Files\Java\jdk-21\bin\java.exe" <script ids>` |
| LoRA Block Weight / merger checks | `scripts/dev/lbw_check.py`, `scripts/dev/merge_check.py` |
| UI screenshot | `java -jar target/stablediffusion-fx.jar --dry-run --home <dir> --screenshot out.png --exit-after 8` (also writes `out-panel.png`, the full form). Expand a plugin by putting `{"expanded": {"txt2img.plugin.<id>": true}}` in `<home>/config/ui-state.json` |

The `scripts/dev/` files contain absolute paths of this machine; adjust them if anything moves. `write_settings.py`
creates `config/settings.json` (gitignored); delete it afterwards.

Look at the images, not only the numbers. Vectorscope and Diffusion CG were "working" by every metric but visibly
broken (pseudo-inverse colour directions were too large). The ridge-regularised `rgb_to_latent_dir` fixed them.

## Known issues and decisions

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
