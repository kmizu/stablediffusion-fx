# Grid image numbers implementation plan

**Goal:** Bundle index target `grid_numbers` as an opt-in native AlwaysOn plugin, decorating grid tiles with saved
image filename numbers and optional X/Y/Z information while preserving saved source images.

**Source/spec:** AlUlkesh/sd_grid_add_image_number, commit `693a8a710e1872d3ba9b63a41f69ee705f4ad54c`, inspected
README, `scripts/sd_grid_add_image_number.py`, LICENSE (Unlicense). Standalone Java2D reimplementation, no parent
reForge imports, dependencies or Python hook. Parent owns notices/ledger and final sbt/GPU/UI gates.

**Architecture:** Existing grids are client-side Java2D. A core `GridNumbers` tile renderer consumes filename and
persisted extra fields. `GridNumbersPlugin` writes only display configuration to generation params. Existing script
grid paths decorate copies of saved output tiles; the ordinary job path creates one numbered batch grid when
enabled and more than one image exists. The Python engine and source-image bytes stay unchanged.

**Approaches:** Engine-side Pillow decoration would introduce a new cross-process image tool for existing Java2D
grids; post-saving a composed grid loses individual cell geometry and saved filename correspondence. Decorating
saved tiles before the existing layout is the direct fit, and is the selected implementation approach.

## Authoritative behavior and native mapping

- Upstream has two settings: filename number (default true) and XYZ info (default false). Native panel additionally
  has opt-in Enabled; filename numbering starts enabled within the panel, XYZ info disabled.
- Use the saved filename's numeric prefix before its first hyphen, preserving leading zeros. Skip unnamed/non-numeric
  files and already-composed grid filenames. Native atomic output naming always includes the prefix.
- Number bottom-left. XYZ axis label/value X top-left, Y top-right, Z bottom-right; omit Nothing/absent axes.
  Use native axis definitions/selected values rather than upstream Python function-introspection guesses, supporting
  every native built-in and plugin-contributed axis. Attach per-cell captions to returned source descriptors after
  generation and retain them in the grid's source mapping; do not rewrite the already-saved source PNG metadata.
- White text on half-transparent black rectangles. Start at 24px logical SansSerif font, inset and inner padding10px;
  reduce to9px when needed, then remove the `Label: ` prefix and retry, and omit if still unfit. Preserve upstream's
  staggered right-corner offsets; native Java2D font metrics differ from Pillow's platform font metrics.
- Only copies used in grids are decorated. Final images, seed order, filenames and normal sampling are unchanged.
- Cover ordinary multi-image jobs, XYZ/Z subgrids, prompt matrix, sequence-script grids and loopback. Existing
  no-plugin behavior and grid creation stay unchanged; one-image ordinary jobs do not create an extra grid.
- Persist display options, source filename mapping, and axis metadata in native PNG parameters/full generation
  params so source identification is reviewable and replay retains display settings. Replay uses newly assigned
  saved filenames, rather than stale source-file numbers. No global per-job mutable label cache.

## Tasks / tests-first

- [x] Add core renderer munit tests: filename parsing, disabled no-op, four corners/opacity/white text, source pixel
  immutability, font shrinking/prefix fallback, tiny/long/unicode labels and RGB/RGBA input.
- [x] Parent runs RED compile/test before production implementation (worker cannot run sbt).
- [x] Implement renderer, native plugin controls/persistence and registration, shared script decoration, XYZ cell
  metadata, ordinary batch grid integration. Add plugin/engine tests for actual grid/source behavior and replay.
- [x] Parent runs targeted GREEN CPU suites; inspect renderer contact PNG from manual helper. Worker can run
  unrelated Python CPU regression suite if any Python path is changed (none expected).
- [x] Add manual AppServices helper for SD1.5/SDXL regular batch and XYZ grids, saved source hashes, numeric filename
  mapping, metadata/full-params replay and final/gallery grid counts. Parent runs models and native screenshots.

## Review focus / gates

Disabled state must create no new grids; batch scripts must retain their existing layouts, labels, partial-grid and
source-image behavior. Axis labels must follow displayed cells after expensive-axis loop reordering. Combined Z
grids must not re-label already-composed subgrids. Source metadata must survive the grid save path. Tiny images and
long strings must terminate without exceptions. No ledger promotion until parent sbt, real-model and UI checks pass.

## Verification evidence and parent acceptance

Parent-owned tests executed in this isolated worktree on 2026-10-26. RED first failed with the missing renderer
(`outputs/grid-numbers-red.log`). An additional real metadata assertion then reproduced each Z subgrid listing
all eight source images instead of its four visible tiles (`outputs/grid-numbers-mapping-red.log`). The fix passes
the visible-tile subset to each save operation, in row-major order, including only actually displayed batch heads.
Combined Z grids record all displayed subgrids; source finals remain separate and unchanged.

Final targeted native CPU evidence, supplied by the parent: core renderer **5 passed**, plugin/render-integration
**48 passed**, engine queue **15 passed**, `mcp/Test/compile` passed. The parent executed and visually inspected
the genuine Java2D helper contact sheet: four corners, translucent boxes, Unicode text and small/tiny fallback were
clear. Log: `outputs/grid-numbers-green-final.log`; contact: `outputs/grid-numbers-render-check/contact.png`.
These are ignored local evidence artifacts, not shipped product assets. Worker `git diff --check` passed.

```powershell
# Parent: CPU renderer, script grids, ordinary queued batches, metadata and replay
& "$env:LOCALAPPDATA/Coursier/data/bin/sbt.bat" -batch 'core/testOnly sdfx.core.GridNumbersSuite' 'plugins/testOnly sdfx.plugins.GridNumbersPluginSuite sdfx.plugins.XyzGridSuite sdfx.plugins.ScriptsSuite sdfx.plugins.PromptPluginsSuite' 'engine/testOnly sdfx.engine.JobQueueSuite' mcp/Test/compile
& "$env:LOCALAPPDATA/Coursier/data/bin/sbt.bat" -batch 'core/Test/runMain sdfx.core.GridNumbersRenderCheck outputs/grid-numbers-render-check'
# Parent: run twice with distinct output roots and SD1.5/SDXL checkpoint names
& "$env:LOCALAPPDATA/Coursier/data/bin/sbt.bat" -batch 'mcp/Test/runMain sdfx.mcp.RealGridNumbersCheck <python-exe> <checkpoint-dir> <checkpoint-name> <output-dir>'
```

Manual real-model helper checks disabled ordinary output, batch2/iterations2, exact final-pixel preservation, saved
filename mapping, full-params replay using newly saved numbers, unchanged existing source hashes, XYZ 2x2x2 with
three saved grids and all axis captions. Reports/full params are under `<output-dir>/grid-numbers-evidence/`.
Native PNG infotext import retains the two display settings; full params retain batch size/iteration count. As with
existing script metadata, selecting/restoring the XYZ script and axis values is handled by the host's normal UI.
Parent must inspect real-model grids and the native panel in txt2img/img2img (checkbox states, wrapping and overflow),
and run aggregate regression/assembly gates before promotion. No GPU, sbt, Python inference or native UI process
was launched by this implementation worker; all native test/model execution remains parent-owned.

### PNG restore follow-up

Independent review identified that a core infotext round trip did not prove the actual native restore path:
`GenerationVM.restoreInfotext` intentionally clears extras, so the two display settings never reached the panel.
`PluginsVM.restoreInfotext` now restores the real `grid_numbers` form properties, and MainWindow invokes it from
both Restore parameters and Send to img2img. Explicit false values are retained, including both options false;
the main toggle is enabled to represent the imported settings, while both false still produce no extra grid.
Absent/unrecognized grid fields leave the current panel alone; partial recognized metadata uses native defaults
for its missing counterpart. GenerationVM's existing extra-field clearing remains unchanged.

Tests exercise all four boolean combinations in txt2img and img2img, checking the actual form state and the next
JobBuilder/AlwaysOn configuration, rather than relying on hidden params extras. The parent's RED run failed with
the missing form-restore method. An initial GREEN exposed a malformed two-field partial test fixture (the existing
infotext parser needs at least three fields); valid PNG-style fixtures corrected that test setup. Final fresh
parent-owned run: `ui/testOnly sdfx.ui.GridNumbersRestoreSuite sdfx.ui.StateAndPluginsSuite sdfx.ui.GenerationVMSuite`
**28 passed**, zero failures, in `outputs/grid-numbers-restore-green.log`. Worker `git diff --check` passed.
The parent additionally owns actual JavaFX window.restore/window.sendTo checks and expanded-panel screenshots.

## Parent acceptance completed

Real SD1.5/SDXL `RealGridNumbersCheck` passed batch/XYZ, source pixel/hash preservation, visible source mapping and
full-param replay. Actual JavaFX Restore/Send-to replaced deliberately different grid/latent control values in
txt2img/img2img/Inpaint; expanded grid form inspected. Full Scala514 + assembly and torch Python731/1skip,
torch-free312/74skip passed. Unlicense notice collected; index88 promoted. Exact artifacts, checks and limits are
in the Grid Numbers/Latent Upscale section of `docs/VERIFICATION.md`. Overall extension-index work is still active.
