# Batch preview, gallery and ZIP verification — 2026-10-27

This scoped fix is independent of the paused v0.03 work.

## Findings and implementation

The normal generation path already performs real batched denoising. The misleading display came from slicing live latents to the first image and reporting only the batch's first image index. Progress now carries the actual latent batch size, previews render all members, and each generation has a gallery group with ZIP export. Single-image refinement progress remains single-image progress.

## Checks

- CPU real tiny diffusers SD1.5 and SDXL pipelines: five calls with batch four; ten UNet calls for two sampling steps with CFG batch eight; twenty unique image pixels, seeds, subseeds and paths. Text encoding alone is stubbed; no checkpoint download.
- Focused Python protocol, execution and preview suites: **15 passed**. Approximate previews, TAESD/fallback, extra latent channels, frame updates, aspect ratios and bounded large grids covered.
- Scala ProtocolSuite: **19 passed**. ProgressVMSuite and GalleryBatchSuite: **14 passed**. Includes legacy progress, four-image ranges, cross-job isolation, search-independent export, exact bytes, duplicate names, source-overwrite refusal and failure cleanup.
- Real JavaFX MainWindow submission on RTX 3090 with `animekawa_v10` (SD1.5) and `waiIllustriousSDXL_v170` (SDXL): **both passed**. Each used batch size four, count five, 256×256 and three steps, producing twenty files and ten 512×512 live preview frames. These are finite workflow checks, not image-quality evaluations.
- Both native checks showed the actual live 2×2 grid and final twenty-image gallery; screenshots inspected. ZIP entries were twenty unique names and exactly matched the generated PNG hashes. The test injects the destination and calls the production writer; the OS save dialog was not automated.
- `ui/assembly` passed with the integrated changes.

Native artifacts under `.worktrees/batch-workflow/outputs/`:

- `native-sd15/batch-native-1813660088003653444/`
- `native-sdxl/batch-native-18073572571797037878/`

Each contains `report.json`, `live-batch-preview.png`, `final-batch-gallery.png`, the twenty PNGs and `batch-twenty-images.zip`. Logs: `batch-scala.log`, `batch-ui-tests.log`, `batch-native.log`.

Groups are session-scoped. Export while generating includes only the count currently displayed; it does not wait for future results. No changes to the paused v0.03 work or its release tag are included.
