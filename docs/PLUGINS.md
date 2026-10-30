# Plugin coverage vs. reForge built-ins

Every built-in script (`scripts/`) and built-in extension (`extensions-builtin/`) of the parent
reForge repository, and where its rewrite lives in stablediffusion-fx.

Kinds: **Script** = selected in the Script dropdown, runs the whole job. **AlwaysOn** = accordion that
tweaks every generation (sends `params.hooks` to the engine). **Extras** = post-processing of an image.
**Core** = built into the app rather than a plugin.

## Scripts (`scripts/`)

| reForge | stablediffusion-fx | Kind | Engine side |
|---|---|---|---|
| xyz_grid.py | `xyz_grid` X/Y/Z plot | Script | - |
| prompt_matrix.py | `prompt_matrix` | Script | - |
| prompts_from_file.py | `prompts_from_file` | Script | - |
| loopback.py | `loopback` | Script | - |
| sd_upscale.py | `sd_upscale` | Script | `upscale` |
| poor_mans_outpainting.py | `outpaint` | Script | - |
| outpainting_mk_2.py | `outpaint_mk2` | Script | - (noise-matched fill in Scala) |
| img2imgalt.py | `img2img_alt` | Script | hook `img2imgalt` (noise inversion) |
| custom_code.py | `custom_code` | Script | `run_code` request, disabled unless `allow_custom_code` |
| processing_autosized_crop.py | `autosized_crop` | Extras | - |
| postprocessing_upscale.py | `upscale_extras` | Extras | `upscale` |
| postprocessing_split_oversized.py | `split_oversized` | Extras | - |
| postprocessing_create_flipped_copies.py | `flipped_copies` | Extras | - |
| postprocessing_focal_crop.py | `focal_crop` | Extras | - (edge/entropy + optional face boxes) |
| postprocessing_gfpgan.py | `face_restore` (AlwaysOn) / `face_restore_extras` (GFPGAN) | AlwaysOn + Extras | `restore_faces` |
| postprocessing_codeformer.py | `face_restore` (AlwaysOn) / `face_restore_extras` (CodeFormer) | AlwaysOn + Extras | `restore_faces` |
| postprocessing_caption.py | `caption` | Extras | `caption` (BLIP) |

## Built-in extensions (`extensions-builtin/`)

| reForge | stablediffusion-fx | Kind | Engine side |
|---|---|---|---|
| Lora | Core: `<lora:name:w>` syntax + LoRA picker | Core | `load_lora_weights` |
| LDSR | `LDSR` upscaler | Core (upscaler) | diffusers `LDMSuperResolutionPipeline` |
| ScuNET, SwinIR | any SCUNet/SwinIR file in `models/upscalers` | Core (upscaler) | spandrel |
| canvas-zoom-and-pan | Core UI: zoom/pan preview + mask canvas | Core | - |
| extra-options-section | Core UI: quick settings in the top bar | Core | - |
| mobile | n/a (desktop app) | - | - |
| prompt-bracket-checker | Core UI: bracket warning under prompts | Core | - |
| sd_forge_controlnet | Core: ControlNet units | Core | ControlNet pipelines |
| sd_forge_controlnet_example | n/a (example code for extension authors) — see README plugin guide | - | - |
| forge_preprocessor_inpaint | preprocessor `inpaint` | ControlNet preprocessor | `controlnet.py` |
| forge_preprocessor_recolor | preprocessors `recolor_luminance`, `recolor_intensity` | ControlNet preprocessor | `controlnet.py` |
| forge_preprocessor_tile | preprocessors `tile_resample`, `tile_colorfix`, `blur_gaussian` | ControlNet preprocessor | `controlnet.py` |
| forge_preprocessor_marigold | preprocessors `depth_marigold`, `normal_marigold` | ControlNet preprocessor | diffusers Marigold |
| forge_preprocessor_normalbae | preprocessor `normal_bae` | ControlNet preprocessor | controlnet_aux (optional) |
| forge_legacy_preprocessors | canny, depth (DPT), openpose, lineart, softedge, mlsd, ... | ControlNet preprocessor | cv2 / transformers / controlnet_aux |
| forge_preprocessor_reference | `reference` (reference-only attention) | AlwaysOn | hook `reference` |
| forge_preprocessor_revision | `revision` (SDXL clip-vision pooled conditioning) | AlwaysOn | hook `revision` |
| sd_forge_ipadapter | `ip_adapter` | AlwaysOn | `load_ip_adapter` |
| sd_forge_controlllite | `controllllite` | AlwaysOn | hook `controllllite` |
| sd_forge_photomaker | `photomaker` | AlwaysOn | hook `photomaker` (needs `photomaker` package) |
| sd_forge_fooocus_inpaint | `fooocus_inpaint` | AlwaysOn | hook `fooocus_inpaint` |
| sd_forge_freeu | `freeu` | AlwaysOn | `enable_freeu` |
| sd_forge_dynamic_thresholding | `dynamic_thresholding` | AlwaysOn | UNet output hook |
| sd_forge_hypertile | `hypertile` | AlwaysOn | attention tiling hook |
| sd_forge_sag | `sag` | AlwaysOn | hook `sag` |
| sd_forge_kohya_hrfix | `kohya_hrfix` (Deep Shrink) | AlwaysOn | hook `kohya_hrfix` |
| sd_forge_latent_modifier | `latent_modifier` | AlwaysOn | hook `latent_modifier` |
| sd_forge_multidiffusion | `multidiffusion` (tiled diffusion) | AlwaysOn | hook `multidiffusion` |
| sd_forge_neveroom | `neveroom` | AlwaysOn | hook `neveroom` |
| sd_forge_svd | `svd` (img2vid) | Script (img2img) | `img2vid` request |
| sd_forge_z123 | `zero123` (novel view) | Script (img2img) | `novel_view` request |
| soft-inpainting | `soft_inpainting` | AlwaysOn | hook `soft_inpainting` |
| sd_webui_random_resolutions | `random_resolutions` | AlwaysOn | - |
| reForge-APGIsYourCFG | `apg` | AlwaysOn | guidance hook `apg` |
| reForge-RescaleCFG | `rescale_cfg` | AlwaysOn | `guidance_rescale` |
| reForge-advanced_model_sampling(+_backported) | `model_sampling` | AlwaysOn | hook `model_sampling` |
| mahiro_reforge | `mahiro` | AlwaysOn | guidance hook `mahiro` |
| (PAG, forge2) | `pag` | AlwaysOn | diffusers PAG |

## Verification status (RTX 3090, 2026-10-25)

Extension-index addition (2026-10-26): `seamless_tile` — img2img Script, **Seamless tile inpainting**. Moves opposite
edges to the centre, inpaints a band around their join, optionally shifts back. Choose Horizontal, Vertical or Both;
mask width is the total band across the joined edges. Best with texture prompts and an inpainting checkpoint.
Both directions can retain small seams. Implementation/real-model evidence is in [VERIFICATION.md](VERIFICATION.md).

`weight_gradient` — AlwaysOn **Weight Gradient**. Enable it, then use `{words::0-1:e}` in either prompt to change
their weight over a sampling pass. Optional step range and return weight, seven easing curves, A1111 prompt editing,
hires and img2img are supported. The form includes syntax help. PhotoMaker cannot be combined with these tags.

`negpip` — AlwaysOn **NegPiP**. Negative weights such as `(black hair:-1)` suppress a concept in the positive
prompt and boost it in the negative prompt. Keys stay positive while value projections reverse their contribution.
Supports prompt editing, Weight Gradient, hires and img2img; negative weights with PhotoMaker are unsupported.
Effects depend on the checkpoint, and excessive weights can damage the image. SD1.5/SDXL were GPU-checked.

`neutral_prompt` — AlwaysOn **Neutral Prompt** (implemented, with the compatibility limits below). Compose
positive prompts with AND, AND_PERP, AND_SALT or AND_TOPK, signed `:weight` suffixes and nested `[groups]`.
The form offers CFG rescale and an optional extra-prompt formatter; saved images contain the effective combined
prompt. Hires, img2img/Alternative, editing, Weight Gradient and NegPiP are checked on SD1.5/SDXL. ControlNet and
PAG, including their joint use, are checked with per-branch conditioning. IP-Adapter (basic/Plus) also supplies
native positive/negative image features to every branch, including CFG off/PAG and Alternative. Reference and
SDXL ControlLLLite now prepare their image conditioning per branch, including Alternative inversion and PAG.
Reference adds a reference-image model evaluation for each leaf. PhotoMaker, SAG, APG/Mahiro, Latent Modifier,
Dynamic Thresholding and Detail Daemon currently report an explicit unsupported-combination
error while their branch-specific integration is unfinished.
CFG Schedule now applies its current scale before Neutral rescale/output reconstruction; CADS perturbs each leaf's
text once, before CFG/PAG copies and image-condition preparation, with seeded per-pass state. Both also work during
Alternative inversion. Native CFG-off behavior ignores CFG Schedule. NegPiP signs and masked padding are retained.
Ordinary CADS/CFG Schedule also handle native PAG groups: the positive noisy context is shared, and scheduled CFG
preserves PAG's conditional/perturbed difference. Zero-scale PAG follows the ordinary non-PAG path.

Every plugin has unit tests (Scala `plugins`, `engine`, `mcp`, `ui` suites; Python `tests/`) and runs end to end in
dry-run mode. Real-model GPU runs:

| Verified on a real model | Plugins |
|---|---|
| SD1.5 and SDXL | apg, mahiro, rescale_cfg, model_sampling, kohya_hrfix, multidiffusion (both methods), latent_modifier, neveroom, reference, img2img_alt, soft_inpainting, freeu, pag, dynamic_thresholding, hypertile |
| SD1.5 only (by design) | sag (needs DDIM / UniPC / DPM++ samplers; other samplers fall back with a warning) |
| SDXL only (by design) | revision |
| Model-backed, verified (outputs checked visually) | face_restore / face_restore_extras (GFPGAN v1.4, CodeFormer), caption (BLIP, WD14 tagger), LDSR upscaler, depth / normal Marigold, recolor / tile / blur preprocessors, ip_adapter (SD1.5 ip-adapter_sd15), controllllite (SDXL canny anime), fooocus_inpaint (inpaint_v26 patch + head), photomaker (v1), svd (stable-video-diffusion-img2vid fp16, 14-frame GIF), zero123 (Zero123-XL diffusers, threestudio layout), custom_code |

Notes from the real-model runs:
- PhotoMaker: needs `pip install --no-deps "git+https://github.com/TencentARC/PhotoMaker.git"`; only v1 adapters (v2 needs
  InsightFace embeddings). The prompt must contain the trigger word after the class word, e.g. `a woman img`.
- Zero123 expects one object on a plain white background; azimuth 0 reproduces the input view.
- SVD / Zero123 folders downloaded with only `*.fp16.*` weights are detected and loaded with `variant="fp16"`.

Per-plugin evidence of the final real-model runs (which run, duration, result): [VERIFICATION.md](VERIFICATION.md).

### Prompt Fusion, Tokenizer and intermediate capture

- Enable **Prompt Fusion** to use embedding curves (`[red:blue: , ]`, `:bezier`, `:catmull`), means,
  contextual attention ramps and `$name` variables/functions in either prompt. Slerp controls its geometry;
  each actual sampling pass has its own clock. Ordinary prompts bypass Fusion. Active Fusion syntax combined
  with PhotoMaker, signed NegPiP terms or Alternative inversion returns an explicit unsupported error.
- The **Tokenizer** tab inspects actual loaded CLIP vocabularies, with both streams for SDXL. Enter literal text
  or comma-separated IDs; colored Unicode groups, full IDs and a table can be copied and scrolled. No automatic
  BOS/EOS is added. Attention/scheduling/LoRA/textual-inversion syntax is inspected literally, without expansion.
  This tool describes vocabulary tokens rather than predicting generation chunks.
- **Save intermediate images** records denoised estimates, incoming noisy samples or the native preview at the
  chosen cadence. Hires shares the base step clock; each iteration restarts it. Frames, optional final copies,
  manifests and videos persist in `outputs/intermediates`, separate from the final gallery. Inversion and
  ADetailer repair passes are not captured. Video mode needs a separately installed `ffmpeg` on PATH; it supports
  GIF, WebM and MP4 (H264/HEVC/AV1 subject to the executable's encoders), optional interpolation and executable
  Windows recipes. Encoding failures retain PNGs and final images with an explicit export error in the manifest.
  Interpolation uses at least one second per selected source frame (including endpoint holds).

- **Grid image numbers** annotates grid tiles with the numeric prefix of their actual saved filenames. Optional
  X/Y/Z information uses the effective axis labels/values; text fits or is omitted on tiny cells. Source images
  remain unchanged. Enabled ordinary batches also save a grid; script grids keep their existing layouts.
  Both options restore from PNG parameters, including explicit false values.
- **Latent Upscale** is an img2img/Inpaint script. It VAE-encodes before target resizing, interpolates latents using
  seven 2-D methods, and performs the full requested denoising-step count. Choose one of six noise schedules;
  Automatic retains the main schedule. Full-resolution masking retains native crop/overlay preparation. It needs
  no learned upscaler or extra download. Strength must be positive and large enough for the 10,000-entry schedule
  bound. Upstream linear/trilinear are invalid for 4-D image latents and are rejected explicitly. Unsupported sampler
  schedules report the native fallback. PNG Restore and Send to img2img/Inpaint restore the actual script controls.

- **Composable LoRA / LyCORIS** keeps each adapter with its own `AND` branch and supports per-step switching,
  ramps and `#cmd` controllers. Negative text-encoder/denoiser switches retain the source single-prompt exception;
  enable its exclusion option to apply those switches to one prompt too. Math expressions work normally; Python
  statements/functions use the existing Allow custom code setting. It supports LoRA, LoCon/CP, LoHa, LoKr,
  full-rank and IA3 algebra. Plots append a diagnostic gallery image; trace JSON persists in
  `outputs/intermediates/composable-lora` and its path appears in the log. Branches combine predictions rather than
  defining spatial regions; strong multi-branch guidance can oversaturate. PhotoMaker, Alternative inversion,
  signed NegPiP and native LoRA Block Weight/start/stop fields have explicit compatibility errors.
- **System Info** shows live runtime/memory and model/network inventory, plus fixed-scene benchmarks, local
  history and JSON export. Polling pauses while hidden. Generation metadata is off until fields are selected.
  Quick/turbo is a short workload; larger presets stop on errors and can be interrupted. Optional legacy submission
  requires a separate payload preview and Submit action; no data is sent by refresh, benchmark or export. UDP
  transmission cannot confirm server receipt. The optional preinstalled driver utility is never installed/bundled.
- **Styles** edits prompt styles and categories with autosave, notes, search/regex, literal replacement and
  reorder. Unchanged CSV records and foreign columns retain their bytes; conflicting external changes require
  Reload. Category edits update generation styles immediately. Changed styles have ten-minute backups (latest24),
  manual backup/import/export and optional AES-Crypt v2 backup encryption. Active CSV files remain plaintext.
  Extended presets separately save positive/negative prompts, size, checkpoint and VAE with UUID identities;
  Apply loads model then VAE and leaves unrelated generation settings intact. Passwords are never saved.
- **Comparison** shows two images with a directional wipe, B opacity, shared zoom/pan, swap and reset. Send from
  img2img/Inpaint/Extras or compare the current and previous txt2img jobs. Optional Ctrl+Alt mouse sends use the
  pixels actually displayed. Metadata differences are token/field based. Export writes a new PNG and refuses
  source aliases; loading errors retain the displayed pair. Formats follow the installed JVM image readers.
- **VAE Tools** encodes/decodes original files through the selected checkpoint VAE or optional TAESD. L PNG is an
  exact tensor container; RGBA/CMYK are quantized and JPEG adds loss. Full/TAESD encode; Approx NN and cheap decode
  are optional previews. Downloads require explicit enablement. Generation VAE failures retain exact decoder-input
  safetensors and a quantized recovery PNG, and clearly mark the approximate image Broken. Recovery artifacts stay
  separate from final gallery scans. Tiling and precision changes restore their previous state after each request.
- **Prevent interruption** is a runtime-only footer switch, shared by GUI and MCP clients of the same engine.
  When On, active Interrupt/Esc, running-job cancellation and explicit user restart are rejected with a visible
  reason; existing work continues. Turn it Off to interrupt normally. Pending-only queue clearing, shutdown and
  automatic health recovery keep their normal behavior. A new engine starts Off; another frontend's change
  appears within five seconds. Admission is checked by the engine even before the displayed switch refreshes.
- **Hardware image metadata** in System Info optionally adds compact hardware facts, per-image time and a literal
  remark to generated PNGs and Extras output. It starts Off and is independent of the selectable System Info JSON
  fields. Generation timing averages each batch and resets between iterations; Extras measures its whole operation.
  Queue wait, model preloading and final file publication are excluded. Restore/Send to can show up to two
  informational device-difference notices per window; they never change the local device and work offline without
  starting an engine. Unavailable hardware facts are recorded as unknown rather than invented.
- **API payload** opens a shared read-only inspector from the collapsed section in each generation mode. It
  retains the exact native request body plus separate submitted/model context and image diagnostics, with distinct
  job/sub-generation history. Tree actions include recursive expansion, marks and JSON paths; raw preview is capped
  at64KiB, while explicit Copy uses the complete selected retained document. The shared image-content preference
  affects future submissions only; init-image contents are always omitted. Limits produce visible diagnostics or
  continuation, and captured model facts are labelled snapshots rather than proof of execution identity. Opening
  the inspector never starts the engine. History is memory-only; this is not an HTTP request or replay editor.

## PNG resource hashes

AlwaysOn `pnginfo_injection` is opt-in. It merges confirmed loaded model, explicit VAE, LoRA, referenced textual inversion, single-file ControlNet and ADetailer fingerprints into saved PNG `Hashes`. Dynamic Prompts also records positive/negative wildcard source files as `wildcards_file:<name>`. These entries do not assert Civitai bundle identity. Missing, ambiguous or changed cached resources are warned and omitted. See [verification](pnginfo-injection-verification.md).

## Steps Animation

Script `steps_animation` captures denoised steps during one txt2img or img2img generation and optionally exports a movie with external ffmpeg. Choose duration, frame rate, none/blend/MCI interpolation and codec. Batch size/count must be one; Save intermediate images must be disabled. Frames are retained unless deletion is requested and export succeeds. Optional partial export preserves the job's cancelled status; interrupt again to stop encoding. Paths are relative to the generation output folder. See [verification](steps-animation-verification.md) for codec, model and interrupt coverage.
