# Native configuration preset compatibility

This document describes the source interchange contract. The native library is a separate
`sdfx-native-presets` v1 file. Source import requires the explicit format
`config-presets` or `preset-utilities`; Config also requires an explicit txt2img
or img2img mode. A filename never chooses either format.

Source pins: Config-Presets index 30, MIT, commit
`e252e60288b1104a45ea12106d92a8d145364fa1`, especially
`scripts/config_presets.py:592-980`; Preset Utilities index 31, archived with no
tracked licence, commit `e301734567f37ac53c6225aec3a5c9858b357e7e`,
especially `scripts/zpreset_utils.py:108-208,378-435,510-702`. The audit,
manifest and complete read-only source trees live in the main repository's
`outputs/next-core-presets-audit*`; the corrected ID inventory is
`outputs/native-config-presets-field-map.md`. No index-31 preset data, code or
other asset is packaged here.

## Field meaning

Config has independent `config-txt2img` and `config-img2img` collections.
Inpaint reads the img2img source collection, but its own destination form is
separate. Utilities has one shared collection. A saved field, a tracked field,
a source alias, and a field available in the destination mode are distinct.
Unavailable source entries remain in source provenance and produce diagnostics.
An explicit `SourceBinding` can connect a source key to an already registered
native field. Labels and partial ID prefixes do not authorize a plugin mapping.

The pure metadata accessors `resolveSourceId`, `sourceIdsForField`, and
`exportTrackedIds` use the same verified routes and binding conflict checks as
record import/export. Resolution retains canonical keys for known source IDs
even when the current form has no such field; the caller compares those keys
with its schema before offering apply. Reverse metadata lists verified explicit
aliases first, then known source IDs. The tracked-ID exporter requires every
tracked key to be representable, reports omissions, and emits the standard
Tiling ID only for its complete five-field group. An explicitly selected
supported-only export omits unsupported fields; byte rendering and file limits
are checked separately by `PresetSourceFiles.writeTrackedIds`.
`sourceFieldIds(record)` reads the saved source object's decoded field names in
their original order through the strict raw-JSON reader. It returns an empty
list for a native record without source provenance and rejects invalid or
oversized saved source objects. It never converts or rewrites source values.

| Source 30 IDs (prefix `txt2img_` or `img2img_` unless shown) | Source 31 labels | Native key / condition |
| --- | --- | --- |
| `sampling`, `scheduler`, `steps` | Sampling method, Sampling steps/Steps | `generation.sampler`, `scheduler`, `steps`; sampler names are not split by guessed suffix |
| `width`, `height`, `batch_count`, `batch_size`, `cfg_scale` | Width, Height, Batch count, Batch size, CFG Scale | Same named generation fields in camel case; declared destination ranges apply |
| `restore_faces` | Restore faces | `plugin.face_restore.enabled`; restoration also needs a selected model |
| `enable_hr`, `hr-checkbox`, `show_second_pass` | Hires. fix, Hires fix, Highres. fix | `generation.hiresEnabled`, Txt2Img only |
| `hr_scale`, `hr_upscaler`, `sampling_alt`, `hires_steps`, `steps_alt` | Upscale by, Upscaler, Hires. steps/Hires steps | `generation.hiresScale`, `hiresUpscaler`, `hiresSteps`; `sampling_alt` means upscaler in the pinned source |
| `denoising_strength` | Denoising strength | Logical `generation.denoisingStrength`: hires denoise in Txt2Img, ordinary denoise in Img2Img/Inpaint |
| `prompt`, `neg_prompt`, `seed`, `subseed`, `subseed_strength` | Prompt, Negative prompt, Seed, Variation seed, Variation strength | Corresponding generation fields; seeds use exact signed Long conversion |
| `subseed_show` | Extra | `generation.variationEnabled`, a processing gate; source default false and legacy native initial value true |
| `hr_resize_x/y`, `hr_sampler`, bare `hires_prompt`, `hires_neg_prompt` | Resize width/height to, Hires sampling method | Txt2Img hires fields; “Use same sampler” maps to native empty string |
| `resize_mode` | Resize mode | `generation.resizeMode`; pinned 0/1/2 table, adapted to the supplied field kind rather than current choice order |
| `mask_blur`, `inpainting_fill`, `mask_mode`, `inpaint_full_res`, `inpaint_full_res_padding` | Mask blur, Masked content, Mask mode, Inpaint area, Only masked padding, pixels | Inpaint fields; fixed source indices/labels are converted to declared native Integer/Flag/Choice types |
| `script_list`, verified XYZ axis keys | Script and verified XYZ labels | `script.selected`, `plugin.xyz_grid.*`; only known script titles and exact axes |
| ControlNet-0/1/2 enable, preprocessor, model, weight, start/end guidance IDs | Explicit user bindings | `controlnet.0/1/2.*`; slot identity and disabled drafts survive |
| `tiling` | Tiling | Only when asymmetric tiling's five controls exist: true sets enabled/X/Y/start=0/stop=-1; false changes enabled only |

Config's default tracked set includes the available standard generation controls
from its fixed tracked list. Prompt and Seed are examples in its commented
custom-ID templates and enter tracking only when selected. Utilities' tracked
set includes available standard labels, including prompt, seed and Extra.
Utilities' independently authored `Reset common controls` contains only
standard keys present in all three mode schemas with identical declared
defaults. Denoising strength is omitted because its mode defaults differ.

The full Config first-use dataset has 14 txt2img and four img2img entries in
source order, including None, both Flux entries, SDXL with Refiner and the 4K
entry. It retains unsupported raw fields such as distilled CFG and refiner
enable. The 4K entry's width 768, height 432 and upscale factor 5 are retained
as representable values in the core codec. The native hires slider's bound is
owned by the separate UI field-catalog work; acceptance must verify that it
admits 5. The factor is never silently clamped.

## Unavailable or ambiguous fields

Refiner enable/switch, Flux distilled CFG, source seed-resize dimensions,
quicksettings checkpoint/VAE/clip skip, SD.Next presentation toggles, source
styles multiselect, img2img Scale and batch directories, mask transparency,
image CFG, source widget JSON objects, and unavailable ControlNet controls
have no automatic native mutation. Neither profile loads a checkpoint or VAE.
Old Utilities Firstpass width/height do not become Upscaler/Upscale by merely
because its source renamed the labels: numeric dimensions have different
meaning. Unknown script titles and source option indices without a fixed table
also stay unavailable.

The 189 txt2img and 232 img2img IDs in Config's generated commented
templates are inactive examples. ADetailer, Tiled Diffusion/VAE, region tools,
StableSR, Soft Inpainting, Loopback, SD upscale and Ultimate SD Upscale IDs
must remain unresolved until an exact field-level binding is configured and
validated against an actual native control. The source's full synonym pairs
are declared in `PresetCompatibility.configSynonyms`, including
the ControlNet spelling correction, ADetailer masked-area/padding changes,
Tiled VAE renames and enable-checkbox changes. An exact declared synonym can
resolve through a verified explicit endpoint binding; unbound pairs stay
opaque. This does not turn a renamed third-party control into active support.
The historical Utilities additional-components file contains a Z-label
mapping to Y's ID; ambiguous labels and wrong IDs are reported rather than
silently choosing the first control.

## Import, export and limits

Source JSON is a dictionary of exact preset names to sparse field objects.
The strict native reader rejects duplicate decoded keys, invalid UTF-8,
unpaired surrogates and excess nesting/nodes before mapping. Known scalar
values become typed atoms; unknown, malformed-for-field and unavailable
source entries retain their raw JSON. Import leaves `dirty` empty. Source
export reuses every unedited raw key/value token, including known aliases and
large numeric lexemes. Editing a known field marks its canonical key dirty;
export then uses a verified built-in or explicit reverse binding. Full export
refuses a native-only field without a source key. The separate supported-fields
export requires an explicit omission review in the UI. Native v1 export is the
lossless authoritative format.

Reverse conversion follows the selected source route. A custom declarative
binding uses plain typed scalar semantics; standard script titles,
same-sampler and fixed option labels are transformed only for their verified
source keys. Config's bare `resize_mode` remains bare after edits. Utilities
choice fields export labels such as “Resize and fill”, “Inpaint not masked” and
“Only masked”; Config's fixed indexed fields use the pinned index table.
Config XYZ export is limited to the six verified axis type/value IDs;
Utilities' supported XYZ labels reverse to their exact labels. A reverse
binding is checked against the same conflict rules as import.

`Tiling` is represented as one compound source value. A true value needs
enabled, X, Y, start=0 and stop=-1 in the native record. False changes only
enabled on import, preserving disabled drafts. Full source export refuses a
partial Tiling=true configuration. Disabled draft constituents require their
own verified source bindings for full export; otherwise it refuses that export.
A separately chosen supported-fields export may omit fields only with a
visible omission review. The complete native library stores every
constituent independently.

The exact declared Config synonym graph is used only when a verified source
endpoint is bound. An old Tiled VAE or ADetailer spelling can resolve through
its explicitly bound counterpart; an unbound pair remains opaque. Conflicting
values from both spellings suppress that native field and produce a diagnostic.

Per file: 4,194,304 UTF-8 bytes; JSON nesting 32; lexical nodes including
keys 100,000; collections 3; total records 1,000; fields per record 2,048;
registered fields per mode 4,096; key/ID 512 UTF-8 bytes; scalar text
262,144 UTF-8 bytes. Custom mapping/text files are capped at 262,144 bytes
and 2,048 noncomment bindings. Names are case-sensitive, at most 256 code
points, and cannot be blank or contain controls. Limit failures reject the
whole operation. Preview is limited to 4,096 characters and diagnostics to
200 visible rows with an exact total; exact export retains data within file
limits.

The native action “Reload preset tools” reloads files, bindings and editor
metadata. It does not restart the engine, cancel generation or hot-reload
external plugin jars. This differs from the archived Utilities' browser/UI
restart path. Source import/export is an explicit file operation, separate
from the revision-aware native library.

## Native acceptance (2026-10-27)

The reviewed product-source checkpoint is `.worktrees/core-config-presets` HEAD
`d6248d4c71f5b00e1e6398871d5470ca22034715`. The earlier native pane 32-check run
(`.worktrees/config-presets-pane/outputs/verify-native-presets-pane-review-round1-third/`)
confirmed that the actual 4K widget accepts hires scale 5, saves/reloads it with
768x432, and keeps the collection and destination distinct. No 4K inference was
attempted. The final CPU 16-check and SD1.5/SDXL 19-check reports are in the
canonical worktree's `outputs/verify-native-presets-{cpu,sd15,sdxl}-final/`.
They verify actual native Save → Reload → Apply with fixed-seed submitted-body
and decoded-RGB parity in the supported modes. Inpaint parity normalizes only
the distinct temporary mask path after the admitted path, PNG hash and pixels
have been captured; its raw submitted bodies are retained separately.

The pane run also checked the complete first-use 14+4 collection, partial
source import/opaque retention, verified Additional label binding, raw native
export after long Prompt/alias values, 210 paged omission entries, nine paged
apply warnings, and the four independent tracked/checked/saved/available flags.
Extra presents source default false, native initial true and its current value
as separate facts. In Txt2Img, an Extra-disabled `.4` variation draft submits
effective zero and is restored after restart; positive strength changes actual
pixels. Img2Img/Inpaint submit their strength and retain drafts, but their
current engine reports the image effect unsupported. Three disabled ControlNet
slot drafts survive save/restart without activating a unit; separately, real
trained `sd15-canny` and `sdxl-canny` units preserve active model/image/weight
and queued state through preset operations. Local model assets are pinned by the
main `outputs/model-cache/controlnet-acceptance-20261027` manifest and are not
distributed with the app.

The source-30 MIT notice covers the complete 18-record starter data and is
archived as `licenses/Zyin055__Config-Presets.txt` (1,085 bytes; SHA-256
`c19b98789c10ffe53568c46f4daae8a49024a02715cf2d4f05de04aa05d3a5f9`).
Source 31 remains an independently implemented behaviour/schema reference:
no code, prose, assets or default JSON from its unlicensed tracked tree is
bundled. The product added no Python dependency or engine body-shape change.
The regression and image-inspection scope is recorded in [`VERIFICATION.md`](VERIFICATION.md). Compatibility classification
in the 421-ID inventory below does not itself make an inactive source example
an active native field.

The whole-wave checkpoint review (`.worktrees/core-config-presets/outputs/presets-wave-integration-review.md`)
found two Important gaps in this *source interchange* contract at `7b340bc`:
opaque/default Refiner and Flux keys are retained but omitted from preset
Select/Reapply diagnostics (I1), and valid fixed resize/masked-content source
choices can import as Text despite actual native Integer-backed Choice fields
(I2). The native-capture/model parity above does not cover those conversions.
The companion review at
`.worktrees/core-config-presets/outputs/presets-wave-integration-rereview-round1.md`
closes both findings and **approves the source-first checkpoint**. I1 adds one
generic `opaque-unapplied` warning per retained key on Select/Reapply, including
after native Reload, while retaining raw data and typed apply atomicity. Focused
pure 19/19 and controller 24/24 tests passed. I2 converts through the pinned
semantic table to the destination's declared Integer or Text Choice atom and
requires membership in its declared options; Compatibility 47/47 and actual
NativePresetFields schema 3/3 tests passed. A later combined native Pane run
passed 36/36 actual workflows with Refiner/Flux/unknown-key warnings, the last
of 210 paged opaque diagnostics, and Config/Utilities numeric and label source
choices applied to real Img2Img/Inpaint VMs. See
`.worktrees/config-presets-pane/outputs/presets-wave-fixes-native.log` and
`outputs/verify-native-presets-wave-fixes/native-presets-pane-report.json` in
that worktree. Existing SD1.5/SDXL native-capture/model results remain valid
for their unchanged scope; those runs did not test source enum import. Corrected
canonical source `d6248d4` passed 964 full Scala tests and assembled a jar with
SHA-256 `0520F97B4C0EF429C67993C2D3F0089AB291E55A9E60F99398CF4B4AC45AE33C`.
The corrected packaged jar passed an isolated Java 21.0.3 dry-run and root
inspected its screenshot. The ledger promotes only indices 30/31 to
`core:config-presets` and `core:preset-utilities`: 103 done, 199 todo, 51 not
applicable, 139 distinct pending targets, all 353 entries retained. Sources
203/221/262 and index 146 remain todo. Post-promotion BuiltinsSuite 20/20 and
NativePresetsCoverageSuite 4/4 passed. See `outputs/presets-source-promotion.json`
and `outputs/presets-postpromotion-checks.log` in the canonical worktree.
The remaining 199 todo entries require their own implementation and verification;
the completed indices 30/31 preset plan is not their implementation.

## Config commented template ID inventory

All 421 IDs below are **inactive comments in the pinned MIT source**. “Conditional adapter” means import can recognize the ID only after a user explicitly tracks/imports it and the exact destination field exists; Tiling also requires its complete five-control native group. “Unavailable” means retained raw with a diagnostic. “Explicit binding” means the source name alone proves no native control identity; a user must supply a verified binding to an existing registered field. These labels describe compatibility classification, not active extension support.

| Mode | Source ID | Classification |
| --- | --- | --- |
| txt2img | `txt2img_prompt` | Conditional adapter |
| txt2img | `txt2img_neg_prompt` | Conditional adapter |
| txt2img | `txt2img_styles` | Unavailable: no equivalent control |
| txt2img | `txt2img_seed` | Conditional adapter |
| txt2img | `txt2img_subseed_show` | Conditional adapter |
| txt2img | `txt2img_subseed` | Conditional adapter |
| txt2img | `txt2img_subseed_strength` | Conditional adapter |
| txt2img | `txt2img_seed_resize_from_w` | Unavailable: no equivalent control |
| txt2img | `txt2img_seed_resize_from_h` | Unavailable: no equivalent control |
| txt2img | `txt2img_tiling` | Conditional adapter |
| txt2img | `txt2img_hr_resize_x` | Conditional adapter |
| txt2img | `txt2img_hr_resize_y` | Conditional adapter |
| txt2img | `hr_sampler` | Conditional adapter |
| txt2img | `hires_prompt` | Conditional adapter |
| txt2img | `hires_neg_prompt` | Conditional adapter |
| txt2img | `script_list` | Conditional adapter |
| txt2img | `script_txt2img_xyz_plot_x_type` | Conditional adapter |
| txt2img | `script_txt2img_xyz_plot_y_type` | Conditional adapter |
| txt2img | `script_txt2img_xyz_plot_z_type` | Conditional adapter |
| txt2img | `script_txt2img_xyz_plot_x_values` | Conditional adapter |
| txt2img | `script_txt2img_xyz_plot_y_values` | Conditional adapter |
| txt2img | `script_txt2img_xyz_plot_z_values` | Conditional adapter |
| txt2img | `cd_txt2img_divisions` | Explicit binding |
| txt2img | `cd_txt2img_positions` | Explicit binding |
| txt2img | `cd_txt2img_weights` | Explicit binding |
| txt2img | `cd_txt2img_end_at_this_step` | Explicit binding |
| txt2img | `txt2img_controlnet_ControlNet-0_controlnet_control_step_slider` | Unavailable: no equivalent ControlNet property |
| txt2img | `txt2img_controlnet_ControlNet-1_controlnet_control_step_slider` | Unavailable: no equivalent ControlNet property |
| txt2img | `txt2img_controlnet_ControlNet-2_controlnet_control_step_slider` | Unavailable: no equivalent ControlNet property |
| txt2img | `txt2img_controlnet_ControlNet-0_controlnet_enable_checkbox` | Conditional adapter |
| txt2img | `txt2img_controlnet_ControlNet-0_controlnet_low_vram_checkbox` | Unavailable: no equivalent ControlNet property |
| txt2img | `txt2img_controlnet_ControlNet-0_controlnet_pixel_perfect_checkbox` | Unavailable: no equivalent ControlNet property |
| txt2img | `txt2img_controlnet_ControlNet-0_controlnet_preprocessor_preview_checkbox` | Unavailable: no equivalent ControlNet property |
| txt2img | `txt2img_controlnet_ControlNet-0_controlnet_type_filter_radio` | Unavailable: no equivalent ControlNet property |
| txt2img | `txt2img_controlnet_ControlNet-0_controlnet_preprocessor_dropdown` | Conditional adapter |
| txt2img | `txt2img_controlnet_ControlNet-0_controlnet_model_dropdown` | Conditional adapter |
| txt2img | `txt2img_controlnet_ControlNet-0_controlnet_control_weight_slider` | Conditional adapter |
| txt2img | `txt2img_controlnet_ControlNet-0_controlnet_start_control_step_slider` | Conditional adapter |
| txt2img | `txt2img_controlnet_ControlNet-0_controlnet_ending_control_step_slider` | Conditional adapter |
| txt2img | `txt2img_controlnet_ControlNet-0_controlnet_control_mode_radio` | Unavailable: no equivalent ControlNet property |
| txt2img | `txt2img_controlnet_ControlNet-0_controlnet_resize_mode_radio` | Unavailable: no equivalent ControlNet property |
| txt2img | `txt2img_controlnet_ControlNet-0_controlnet_automatically_send_generated_images_checkbox` | Unavailable: no equivalent ControlNet property |
| txt2img | `txt2img_controlnet_ControlNet-1_controlnet_enable_checkbox` | Conditional adapter |
| txt2img | `txt2img_controlnet_ControlNet-1_controlnet_low_vram_checkbox` | Unavailable: no equivalent ControlNet property |
| txt2img | `txt2img_controlnet_ControlNet-1_controlnet_pixel_perfect_checkbox` | Unavailable: no equivalent ControlNet property |
| txt2img | `txt2img_controlnet_ControlNet-1_controlnet_preprocessor_preview_checkbox` | Unavailable: no equivalent ControlNet property |
| txt2img | `txt2img_controlnet_ControlNet-1_controlnet_type_filter_radio` | Unavailable: no equivalent ControlNet property |
| txt2img | `txt2img_controlnet_ControlNet-1_controlnet_preprocessor_dropdown` | Conditional adapter |
| txt2img | `txt2img_controlnet_ControlNet-1_controlnet_model_dropdown` | Conditional adapter |
| txt2img | `txt2img_controlnet_ControlNet-1_controlnet_control_weight_slider` | Conditional adapter |
| txt2img | `txt2img_controlnet_ControlNet-1_controlnet_start_control_step_slider` | Conditional adapter |
| txt2img | `txt2img_controlnet_ControlNet-1_controlnet_ending_control_step_slider` | Conditional adapter |
| txt2img | `txt2img_controlnet_ControlNet-1_controlnet_control_mode_radio` | Unavailable: no equivalent ControlNet property |
| txt2img | `txt2img_controlnet_ControlNet-1_controlnet_resize_mode_radio` | Unavailable: no equivalent ControlNet property |
| txt2img | `txt2img_controlnet_ControlNet-1_controlnet_automatically_send_generated_images_checkbox` | Unavailable: no equivalent ControlNet property |
| txt2img | `txt2img_controlnet_ControlNet-2_controlnet_enable_checkbox` | Conditional adapter |
| txt2img | `txt2img_controlnet_ControlNet-2_controlnet_low_vram_checkbox` | Unavailable: no equivalent ControlNet property |
| txt2img | `txt2img_controlnet_ControlNet-2_controlnet_pixel_perfect_checkbox` | Unavailable: no equivalent ControlNet property |
| txt2img | `txt2img_controlnet_ControlNet-2_controlnet_preprocessor_preview_checkbox` | Unavailable: no equivalent ControlNet property |
| txt2img | `txt2img_controlnet_ControlNet-2_controlnet_type_filter_radio` | Unavailable: no equivalent ControlNet property |
| txt2img | `txt2img_controlnet_ControlNet-2_controlnet_preprocessor_dropdown` | Conditional adapter |
| txt2img | `txt2img_controlnet_ControlNet-2_controlnet_model_dropdown` | Conditional adapter |
| txt2img | `txt2img_controlnet_ControlNet-2_controlnet_control_weight_slider` | Conditional adapter |
| txt2img | `txt2img_controlnet_ControlNet-2_controlnet_start_control_step_slider` | Conditional adapter |
| txt2img | `txt2img_controlnet_ControlNet-2_controlnet_ending_control_step_slider` | Conditional adapter |
| txt2img | `txt2img_controlnet_ControlNet-2_controlnet_control_mode_radio` | Unavailable: no equivalent ControlNet property |
| txt2img | `txt2img_controlnet_ControlNet-2_controlnet_resize_mode_radio` | Unavailable: no equivalent ControlNet property |
| txt2img | `txt2img_controlnet_ControlNet-2_controlnet_automatically_send_generated_images_checkbox` | Unavailable: no equivalent ControlNet property |
| txt2img | `MD-t2i-enabled-checkbox` | Explicit binding |
| txt2img | `MD-t2i-overwrite-image-size` | Explicit binding |
| txt2img | `MD-overwrite-width-t2i` | Explicit binding |
| txt2img | `MD-overwrite-height-t2i` | Explicit binding |
| txt2img | `MD-t2i-method` | Explicit binding |
| txt2img | `MD-t2i-control-tensor-cpu` | Explicit binding |
| txt2img | `MD-t2i-latent-tile-width` | Explicit binding |
| txt2img | `MD-t2i-latent-tile-height` | Explicit binding |
| txt2img | `MD-t2i-latent-tile-overlap` | Explicit binding |
| txt2img | `MD-t2i-latent-tile-batch-size` | Explicit binding |
| txt2img | `MD-t2i-enable-bbox-control` | Explicit binding |
| txt2img | `MD-t2i-draw-background` | Explicit binding |
| txt2img | `MD-t2i-cfg-name` | Explicit binding |
| txt2img | `MD-bbox-t2i-0-enable` | Explicit binding |
| txt2img | `MD-t2i-0-blend-mode` | Explicit binding |
| txt2img | `MD-t2i-0-feather` | Explicit binding |
| txt2img | `MD-t2i-0-x` | Explicit binding |
| txt2img | `MD-t2i-0-y` | Explicit binding |
| txt2img | `MD-t2i-0-w` | Explicit binding |
| txt2img | `MD-t2i-0-h` | Explicit binding |
| txt2img | `MD-t2i-0-prompt` | Explicit binding |
| txt2img | `MD-t2i-0-neg-prompt` | Explicit binding |
| txt2img | `MD-t2i-0-seed` | Explicit binding |
| txt2img | `MD-bbox-t2i-1-enable` | Explicit binding |
| txt2img | `MD-t2i-1-blend-mode` | Explicit binding |
| txt2img | `MD-t2i-1-feather` | Explicit binding |
| txt2img | `MD-t2i-1-x` | Explicit binding |
| txt2img | `MD-t2i-1-y` | Explicit binding |
| txt2img | `MD-t2i-1-w` | Explicit binding |
| txt2img | `MD-t2i-1-h` | Explicit binding |
| txt2img | `MD-t2i-1-prompt` | Explicit binding |
| txt2img | `MD-t2i-1-neg-prompt` | Explicit binding |
| txt2img | `MD-t2i-1-seed` | Explicit binding |
| txt2img | `MD-bbox-t2i-2-enable` | Explicit binding |
| txt2img | `MD-t2i-2-blend-mode` | Explicit binding |
| txt2img | `MD-t2i-2-feather` | Explicit binding |
| txt2img | `MD-t2i-2-x` | Explicit binding |
| txt2img | `MD-t2i-2-y` | Explicit binding |
| txt2img | `MD-t2i-2-w` | Explicit binding |
| txt2img | `MD-t2i-2-h` | Explicit binding |
| txt2img | `MD-t2i-2-prompt` | Explicit binding |
| txt2img | `MD-t2i-2-neg-prompt` | Explicit binding |
| txt2img | `MD-t2i-2-seed` | Explicit binding |
| txt2img | `MDV-t2i-enabled-checkbox` | Explicit binding |
| txt2img | `MD-t2i-vae2gpu` | Explicit binding |
| txt2img | `MD-t2i-enc-size` | Explicit binding |
| txt2img | `MD-t2i-dec-size` | Explicit binding |
| txt2img | `MD-t2i-fastenc` | Explicit binding |
| txt2img | `MD-t2i-fastenc-colorfix` | Explicit binding |
| txt2img | `MD-t2i-fastdec` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_main_accordion-checkbox` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_model` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_prompt` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_negative_prompt` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_confidence` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_mask_min_ratio` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_mask_max_ratio` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_x_offset` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_y_offset` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_dilate_erode` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_mask_merge_invert` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_mask_blur` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_denoising_strength` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_inpaint_only_masked` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_inpaint_only_masked_padding` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_use_inpaint_width_height` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_inpaint_width` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_inpaint_height` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_use_steps` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_steps` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_use_cfg_scale` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_cfg_scale` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_use_checkpoint` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_use_vae` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_use_sampler` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_sampler` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_scheduler` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_use_noise_multiplier` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_noise_multiplier` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_use_clip_skip` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_clip_skip` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_restore_face` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_controlnet_model` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_controlnet_weight` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_controlnet_guidance_start` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_controlnet_guidance_end` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_model_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_prompt_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_negative_prompt_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_confidence_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_mask_min_ratio_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_mask_max_ratio_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_x_offset_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_y_offset_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_dilate_erode_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_mask_merge_invert_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_mask_blur_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_denoising_strength_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_inpaint_only_masked_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_inpaint_only_masked_padding_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_use_inpaint_width_height_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_inpaint_width_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_inpaint_height_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_use_steps_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_steps_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_use_cfg_scale_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_cfg_scale_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_use_checkpoint_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_use_vae_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_use_sampler_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_sampler_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_scheduler_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_use_noise_multiplier_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_noise_multiplier_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_use_clip_skip_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_clip_skip_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_restore_face_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_controlnet_model_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_controlnet_weight_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_controlnet_guidance_start_2nd` | Explicit binding |
| txt2img | `script_txt2img_adetailer_ad_controlnet_guidance_end_2nd` | Explicit binding |
| img2img | `img2img_prompt` | Conditional adapter |
| img2img | `img2img_neg_prompt` | Conditional adapter |
| img2img | `img2img_mask_mode` | Conditional adapter |
| img2img | `img2img_mask_blur` | Conditional adapter |
| img2img | `img2img_mask_alpha` | Unavailable: no equivalent control |
| img2img | `img2img_inpainting_fill` | Conditional adapter |
| img2img | `img2img_inpaint_full_res` | Conditional adapter |
| img2img | `img2img_inpaint_full_res_padding` | Conditional adapter |
| img2img | `resize_mode` | Conditional adapter |
| img2img | `img2img_scale` | Unavailable: no equivalent control |
| img2img | `img2img_seed` | Conditional adapter |
| img2img | `img2img_subseed_show` | Conditional adapter |
| img2img | `img2img_subseed` | Conditional adapter |
| img2img | `img2img_subseed_strength` | Conditional adapter |
| img2img | `img2img_seed_resize_from_w` | Unavailable: no equivalent control |
| img2img | `img2img_seed_resize_from_h` | Unavailable: no equivalent control |
| img2img | `img2img_tiling` | Conditional adapter |
| img2img | `img2img_batch_input_dir` | Unavailable: no equivalent control |
| img2img | `img2img_batch_output_dir` | Unavailable: no equivalent control |
| img2img | `img2img_batch_inpaint_mask_dir` | Unavailable: no equivalent control |
| img2img | `soft_inpainting_enabled-checkbox` | Explicit binding |
| img2img | `mask_blend_power` | Explicit binding |
| img2img | `mask_blend_scale` | Explicit binding |
| img2img | `inpaint_detail_preservation` | Explicit binding |
| img2img | `composite_mask_influence` | Explicit binding |
| img2img | `composite_difference_threshold` | Explicit binding |
| img2img | `composite_difference_contrast` | Explicit binding |
| img2img | `script_list` | Conditional adapter |
| img2img | `script_img2img_xyz_plot_x_type` | Conditional adapter |
| img2img | `script_img2img_xyz_plot_y_type` | Conditional adapter |
| img2img | `script_img2img_xyz_plot_z_type` | Conditional adapter |
| img2img | `script_img2img_xyz_plot_x_values` | Conditional adapter |
| img2img | `script_img2img_xyz_plot_y_values` | Conditional adapter |
| img2img | `script_img2img_xyz_plot_z_values` | Conditional adapter |
| img2img | `script_loopback_loops` | Explicit binding |
| img2img | `script_loopback_final_denoising_strength` | Explicit binding |
| img2img | `script_sd_upscale_overlap` | Explicit binding |
| img2img | `script_sd_upscale_scale_factor` | Explicit binding |
| img2img | `script_sd_upscale_upscaler_index` | Explicit binding |
| img2img | `cd_img2img_divisions` | Explicit binding |
| img2img | `cd_img2img_positions` | Explicit binding |
| img2img | `cd_img2img_weights` | Explicit binding |
| img2img | `cd_img2img_end_at_this_step` | Explicit binding |
| img2img | `img2img_controlnet_ControlNet-0_controlnet_control_step_slider` | Unavailable: no equivalent ControlNet property |
| img2img | `img2img_controlnet_ControlNet-1_controlnet_control_step_slider` | Unavailable: no equivalent ControlNet property |
| img2img | `img2img_controlnet_ControlNet-2_controlnet_control_step_slider` | Unavailable: no equivalent ControlNet property |
| img2img | `img2img_controlnet_ControlNet-0_controlnet_enable_checkbox` | Conditional adapter |
| img2img | `img2img_controlnet_ControlNet-0_controlnet_low_vram_checkbox` | Unavailable: no equivalent ControlNet property |
| img2img | `img2img_controlnet_ControlNet-0_controlnet_pixel_perfect_checkbox` | Unavailable: no equivalent ControlNet property |
| img2img | `img2img_controlnet_ControlNet-0_controlnet_preprocessor_preview_checkbox` | Unavailable: no equivalent ControlNet property |
| img2img | `img2img_controlnet_ControlNet-0_controlnet_type_filter_radio` | Unavailable: no equivalent ControlNet property |
| img2img | `img2img_controlnet_ControlNet-0_controlnet_preprocessor_dropdown` | Conditional adapter |
| img2img | `img2img_controlnet_ControlNet-0_controlnet_model_dropdown` | Conditional adapter |
| img2img | `img2img_controlnet_ControlNet-0_controlnet_control_weight_slider` | Conditional adapter |
| img2img | `img2img_controlnet_ControlNet-0_controlnet_start_control_step_slider` | Conditional adapter |
| img2img | `img2img_controlnet_ControlNet-0_controlnet_ending_control_step_slider` | Conditional adapter |
| img2img | `img2img_controlnet_ControlNet-0_controlnet_control_mode_radio` | Unavailable: no equivalent ControlNet property |
| img2img | `img2img_controlnet_ControlNet-0_controlnet_resize_mode_radio` | Unavailable: no equivalent ControlNet property |
| img2img | `img2img_controlnet_ControlNet-0_controlnet_automatically_send_generated_images_checkbox` | Unavailable: no equivalent ControlNet property |
| img2img | `img2img_controlnet_ControlNet-1_controlnet_enable_checkbox` | Conditional adapter |
| img2img | `img2img_controlnet_ControlNet-1_controlnet_low_vram_checkbox` | Unavailable: no equivalent ControlNet property |
| img2img | `img2img_controlnet_ControlNet-1_controlnet_pixel_perfect_checkbox` | Unavailable: no equivalent ControlNet property |
| img2img | `img2img_controlnet_ControlNet-1_controlnet_preprocessor_preview_checkbox` | Unavailable: no equivalent ControlNet property |
| img2img | `img2img_controlnet_ControlNet-1_controlnet_type_filter_radio` | Unavailable: no equivalent ControlNet property |
| img2img | `img2img_controlnet_ControlNet-1_controlnet_preprocessor_dropdown` | Conditional adapter |
| img2img | `img2img_controlnet_ControlNet-1_controlnet_model_dropdown` | Conditional adapter |
| img2img | `img2img_controlnet_ControlNet-1_controlnet_control_weight_slider` | Conditional adapter |
| img2img | `img2img_controlnet_ControlNet-1_controlnet_start_control_step_slider` | Conditional adapter |
| img2img | `img2img_controlnet_ControlNet-1_controlnet_ending_control_step_slider` | Conditional adapter |
| img2img | `img2img_controlnet_ControlNet-1_controlnet_control_mode_radio` | Unavailable: no equivalent ControlNet property |
| img2img | `img2img_controlnet_ControlNet-1_controlnet_resize_mode_radio` | Unavailable: no equivalent ControlNet property |
| img2img | `img2img_controlnet_ControlNet-1_controlnet_automatically_send_generated_images_checkbox` | Unavailable: no equivalent ControlNet property |
| img2img | `img2img_controlnet_ControlNet-2_controlnet_enable_checkbox` | Conditional adapter |
| img2img | `img2img_controlnet_ControlNet-2_controlnet_low_vram_checkbox` | Unavailable: no equivalent ControlNet property |
| img2img | `img2img_controlnet_ControlNet-2_controlnet_pixel_perfect_checkbox` | Unavailable: no equivalent ControlNet property |
| img2img | `img2img_controlnet_ControlNet-2_controlnet_preprocessor_preview_checkbox` | Unavailable: no equivalent ControlNet property |
| img2img | `img2img_controlnet_ControlNet-2_controlnet_type_filter_radio` | Unavailable: no equivalent ControlNet property |
| img2img | `img2img_controlnet_ControlNet-2_controlnet_preprocessor_dropdown` | Conditional adapter |
| img2img | `img2img_controlnet_ControlNet-2_controlnet_model_dropdown` | Conditional adapter |
| img2img | `img2img_controlnet_ControlNet-2_controlnet_control_weight_slider` | Conditional adapter |
| img2img | `img2img_controlnet_ControlNet-2_controlnet_start_control_step_slider` | Conditional adapter |
| img2img | `img2img_controlnet_ControlNet-2_controlnet_ending_control_step_slider` | Conditional adapter |
| img2img | `img2img_controlnet_ControlNet-2_controlnet_control_mode_radio` | Unavailable: no equivalent ControlNet property |
| img2img | `img2img_controlnet_ControlNet-2_controlnet_resize_mode_radio` | Unavailable: no equivalent ControlNet property |
| img2img | `img2img_controlnet_ControlNet-2_controlnet_automatically_send_generated_images_checkbox` | Unavailable: no equivalent ControlNet property |
| img2img | `MD-i2i-enabled-checkbox` | Explicit binding |
| img2img | `MD-i2i-keep-input-size` | Explicit binding |
| img2img | `MD-i2i-method` | Explicit binding |
| img2img | `MD-i2i-control-tensor-cpu` | Explicit binding |
| img2img | `MD-i2i-latent-tile-width` | Explicit binding |
| img2img | `MD-i2i-latent-tile-height` | Explicit binding |
| img2img | `MD-i2i-latent-tile-overlap` | Explicit binding |
| img2img | `MD-i2i-latent-tile-batch-size` | Explicit binding |
| img2img | `MD-i2i-upscaler-index` | Explicit binding |
| img2img | `MD-i2i-upscaler-factor` | Explicit binding |
| img2img | `MD-i2i-noise-inverse` | Explicit binding |
| img2img | `MD-i2i-noise-inverse-steps` | Explicit binding |
| img2img | `MD-i2i-noise-inverse-retouch` | Explicit binding |
| img2img | `MD-i2i-noise-inverse-renoise-strength` | Explicit binding |
| img2img | `MD-i2i-noise-inverse-renoise-kernel` | Explicit binding |
| img2img | `MD-i2i-enable-bbox-control` | Explicit binding |
| img2img | `MD-i2i-draw-background` | Explicit binding |
| img2img | `MD-i2i-cfg-name` | Explicit binding |
| img2img | `MD-bbox-i2i-0-enable` | Explicit binding |
| img2img | `MD-i2i-0-blend-mode` | Explicit binding |
| img2img | `MD-i2i-0-feather` | Explicit binding |
| img2img | `MD-i2i-0-x` | Explicit binding |
| img2img | `MD-i2i-0-y` | Explicit binding |
| img2img | `MD-i2i-0-w` | Explicit binding |
| img2img | `MD-i2i-0-h` | Explicit binding |
| img2img | `MD-i2i-0-prompt` | Explicit binding |
| img2img | `MD-i2i-0-neg-prompt` | Explicit binding |
| img2img | `MD-i2i-0-seed` | Explicit binding |
| img2img | `MD-bbox-i2i-1-enable` | Explicit binding |
| img2img | `MD-i2i-1-blend-mode` | Explicit binding |
| img2img | `MD-i2i-1-feather` | Explicit binding |
| img2img | `MD-i2i-1-x` | Explicit binding |
| img2img | `MD-i2i-1-y` | Explicit binding |
| img2img | `MD-i2i-1-w` | Explicit binding |
| img2img | `MD-i2i-1-h` | Explicit binding |
| img2img | `MD-i2i-1-prompt` | Explicit binding |
| img2img | `MD-i2i-1-neg-prompt` | Explicit binding |
| img2img | `MD-i2i-1-seed` | Explicit binding |
| img2img | `MD-bbox-i2i-2-enable` | Explicit binding |
| img2img | `MD-i2i-2-blend-mode` | Explicit binding |
| img2img | `MD-i2i-2-feather` | Explicit binding |
| img2img | `MD-i2i-2-x` | Explicit binding |
| img2img | `MD-i2i-2-y` | Explicit binding |
| img2img | `MD-i2i-2-w` | Explicit binding |
| img2img | `MD-i2i-2-h` | Explicit binding |
| img2img | `MD-i2i-2-prompt` | Explicit binding |
| img2img | `MD-i2i-2-neg-prompt` | Explicit binding |
| img2img | `MD-i2i-2-seed` | Explicit binding |
| img2img | `MDV-i2i-enabled-checkbox` | Explicit binding |
| img2img | `MD-i2i-vae2gpu` | Explicit binding |
| img2img | `MD-i2i-enc-size` | Explicit binding |
| img2img | `MD-i2i-dec-size` | Explicit binding |
| img2img | `MD-i2i-fastenc` | Explicit binding |
| img2img | `MD-i2i-fastenc-colorfix` | Explicit binding |
| img2img | `MD-i2i-fastdec` | Explicit binding |
| img2img | `StableSR-scale` | Explicit binding |
| img2img | `StableSR-color-fix` | Explicit binding |
| img2img | `StableSR-save-original` | Explicit binding |
| img2img | `StableSR-pure-noise` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_main_accordion-checkbox` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_model` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_prompt` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_negative_prompt` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_confidence` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_mask_min_ratio` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_mask_max_ratio` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_x_offset` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_y_offset` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_dilate_erode` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_mask_merge_invert` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_mask_blur` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_denoising_strength` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_inpaint_full_res` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_inpaint_full_res_padding` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_use_inpaint_width_height` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_inpaint_width` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_inpaint_height` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_use_steps` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_steps` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_use_cfg_scale` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_cfg_scale` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_use_checkpoint` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_use_vae` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_use_sampler` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_sampler` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_scheduler` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_use_noise_multiplier` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_noise_multiplier` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_use_clip_skip` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_clip_skip` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_restore_face` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_controlnet_model` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_controlnet_weight` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_controlnet_guidance_start` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_controlnet_guidance_end` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_model_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_prompt_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_negative_prompt_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_confidence_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_mask_min_ratio_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_mask_max_ratio_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_x_offset_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_y_offset_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_dilate_erode_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_mask_merge_invert_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_mask_blur_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_denoising_strength_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_inpaint_full_res_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_inpaint_full_res_padding_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_use_inpaint_width_height_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_inpaint_width_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_inpaint_height_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_use_steps_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_steps_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_use_cfg_scale_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_cfg_scale_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_use_checkpoint_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_use_vae_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_use_sampler_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_sampler_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_scheduler_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_use_noise_multiplier_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_noise_multiplier_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_use_clip_skip_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_clip_skip_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_restore_face_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_controlnet_model_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_controlnet_weight_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_controlnet_guidance_start_2nd` | Explicit binding |
| img2img | `script_img2img_adetailer_ad_controlnet_guidance_end_2nd` | Explicit binding |
| img2img | `ultimateupscale_target_size_type` | Explicit binding |
| img2img | `ultimateupscale_custom_width` | Explicit binding |
| img2img | `ultimateupscale_custom_height` | Explicit binding |
| img2img | `ultimateupscale_custom_scale` | Explicit binding |
| img2img | `ultimateupscale_upscaler_index` | Explicit binding |
| img2img | `ultimateupscale_redraw_mode` | Explicit binding |
| img2img | `ultimateupscale_tile_width` | Explicit binding |
| img2img | `ultimateupscale_tile_height` | Explicit binding |
| img2img | `ultimateupscale_mask_blur` | Explicit binding |
| img2img | `ultimateupscale_padding` | Explicit binding |
| img2img | `ultimateupscale_seams_fix_type` | Explicit binding |
| img2img | `ultimateupscale_seams_fix_denoise` | Explicit binding |
| img2img | `ultimateupscale_seams_fix_width` | Explicit binding |
| img2img | `ultimateupscale_seams_fix_mask_blur` | Explicit binding |
| img2img | `ultimateupscale_seams_fix_padding` | Explicit binding |
| img2img | `ultimateupscale_save_upscaled_image` | Explicit binding |
| img2img | `ultimateupscale_save_seams_fix_image` | Explicit binding |
