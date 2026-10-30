# VRAM Estimator — issue #136 / extension index 105

Implementation follows the documented four-mode calibration and regression behavior of
space-nuko/a1111-stable-diffusion-webui-vram-estimator, pinned revision
8ecde47f7aac1304058ab5af09702b703191c5f6 (MIT). The source notice is included in
THIRD_PARTY_NOTICES.md and licenses/. No new optional Python dependencies are needed.

## Behavior

The native VRAM Estimator tab explicitly runs txt2img, Lanczos hires, latent hires and
img2img workloads. Sizes advance by 64 through the selected maximum; batches are actual
simultaneously generated images. Each measurement clears inactive CUDA cache, resets
allocator peaks, records resident-model baseline and incremental reserved/allocated peak,
and retains before/after global usage separately. Global GPU peak is not claimed.
Vanilla workload images are not saved unless representative sampling is requested.

OOM/error terminates larger sizes of the current mode/batch, while subsequent groups
continue. Each row is atomically saved. Owner-specific interruption preserves completed
rows, honors Prevent interruption, leaves unrelated queued generation intact, and does
not arm a process restart. Temporary no-save flags and synthetic input are restored or
removed even on interruption/failure.

Four native charts show successful reserved-increment measurements by batch. History
retains distinct complete/partial runs, bounded to 32 records and 16 MiB; import validates
schema and identity fingerprint before remembering, and export saves the selected record.
Opening the tab does not start calibration.

Visible generation forms update estimates serially while the runtime is idle. Model,
runtime, GPU/driver and library identity must match the saved calibration. An integral
JSON number is canonicalized identically across Python and native JSON round trips.
Per-batch regression is cubic in sqrt(pixel count), reducing degree for sparse data;
unmeasured batches require a full-rank ten-term cubic surface. Unavailable or invalid
fits are reported without fabricated estimates. Displayed usage is current global used
memory plus 1.1 times predicted reserved increment. Extrapolation is labeled.

## Actual device and native acceptance

Windows / RTX 3090 24 GiB / torch 2.7.1+cu128, with animekawa_v10 (SD1.5) and
waiIllustriousSDXL_v170 (SDXL). Source engine process; real model files are read only.

- `outputs/verify-vram-holdout-{sd15,sdxl}/vram_holdout_*.json`: 40 cases per model,
  all four modes, 256..512 square and batches 1/2. Eight 448-square observations per
  model excluded from fitting before prediction. Absolute incremental-reserved errors
  were 5.5..106.5 MiB (SD1.5), 23.5..82.0 MiB (SDXL) in these runs.
- `outputs/vram-native-service-real{,-sdxl}.log`: native JSON history roundtrip and
  compatible estimate; protected stop refused; owner-specific cancellation accepted
  after protection disabled; separately owned queued generation saves its image;
  complete and interrupted histories both remain; owner releases.
- `outputs/vram-native-view-{sd15,sdxl}.log`: actual JavaFX calibration button runs
  eight measurements, four charts each contain the two expected points, history is
  loaded and compatible live estimate appears. SDXL additionally verifies the displayed
  dimensions update from 288 to 304 after the setting changes. View disposal completes.
  Screenshots `outputs/verify-vram-native-view-*/native-vram-view.png` visually inspected.
- `outputs/vram-bounded-oom-{sd15,sdxl}.log`: dedicated-process allocator cap just
  above resident model allocation induces actual CUDA OOM in each of four modes.
  Larger cases are skipped; cap and no-save flags restore; all four vanilla modes
  subsequently generate successfully with representative samples. Detailed JSON is in
  `outputs/verify-vram-bounded-oom-*/bounded-oom-report.json`. This is bounded recovery
  evidence, not a physical memory-capacity search.
- Representative one-step images from the initial four-mode runs were visually
  inspected. They are noisy/near-blank functional samples, not quality evidence.

## Regression and package acceptance

Whole Python regression exited 0 (`outputs/vram-full-python.log`, one skip). Whole Scala
passed 1292 tests with one skip (`outputs/vram-full-scala.log`). A subsequent focused
run passed four history/session tests, including empty-interrupted-history fallback;
UI compilation and assembly passed (`outputs/vram-final-build.log`).

`scripts/dev/check_vram_packaged.ps1` ran the updated service and JavaFX checks using
only test-harness classes plus the assembled jar, on both models. The service exports
and imports the same measurement profile, verifies no duplicate history entry, then
rejects a tampered identity before history changes. Native view tests verify actual
calibration, graphs, history and dimension update. Both logs
`outputs/vram-packaged-{sd15,sdxl}.log` passed, and jar SHA-256 stayed
`7e3c04326e90be9e1d1d2c0d82cf12233ac7f49b2b1de3a712022cf867589c68`.
These are jar/source-engine checks, not a newly published Windows ZIP release.

## Limits

The vanilla calibration excludes extensions, arbitrary samplers, input shapes/aspect
ratios and arbitrary hires scale/base-resolution relationships. Square pixel-area
approximation and extrapolation are not an OOM safety guarantee. Current cached
reservations may make the baseline conservative. Finite configured caps never prove
device capacity. CPU/dry-run backends explicitly reject calibration.

38 focused no-torch Python tests also pass. OS file chooser interaction itself was not
automated; the exact import/export methods used by those buttons were exercised with
real files and the real engine. Older development holdout reports predate integral-number
fingerprint canonicalization; their numeric measurements remain observations, while
the new native and packaged checks validate current canonical profile round trips.
