# Waifu2x (#137) — acceptance record

Seventh accepted wave of the twenty-issue goal; remote commit/closure is tracked in twenty-issue-release-progress.md.

## Provenance and implementation

The unlicensed extension wrapper is independently reimplemented. Eight UpConv7 JSON
models are byte-identical (Git blob identity) to pinned MIT nagadomi/waifu2x originals.
CARN V2 is byte-identical to the checkpoint inside the pinned GPL yu45020 archive.
Exact URLs, sizes and SHA256 are in python/sdfx/waifu2x_assets.json. Reference and
provenance helpers are scripts/dev/check_waifu_{model_provenance,reference}.py.

Native inference preserves the actual legacy graph (including CARN's alternating
block/three-compressor behavior), 64-pixel tile cores with three-pixel replicated
borders and remainder adjustment, and clipped/truncated uint8 conversion. It streams
one tile at a time, validates shapes/finiteness, verifies downloaded bytes atomically,
uses restricted weights-only loading, and retains only one model in cache. UpConv7
runs float32; CARN uses CUDA float16 or CPU float32. Output is RGB. A 64-megapixel
intermediate/output bound is explicit; scale is at most 16x. Weights download on
first use, with SDFX_WAIFU2X_CACHE override for offline provision.

Nine normal upscaler choices integrate with existing Extras/Upscale/Hires routes.
The native Script `waifu2x` supplies dedicated Style/Noise/2x–16x controls through
the existing ExtrasPlugin interface. The Windows packager now includes its JSON
weight manifest. No new Python dependency is required.

## Verified behavior and limits

- 48 focused Python tests passed, including interruption, atomic downloads, tile
  edges, scale iteration, catalog routing and actual tile progress. Final no-torch
  run: 36 passed, 10 skipped. Two Scala plugin tests passed.
- All nine CPU float32 models exactly match pinned licensed original outputs on a
  random 17x19 tensor; all nine tiled RGB results exactly match on 1x1, 58x135 and
  65x63 inputs. outputs/waifu2x-reference-report.json records zero errors.
- Actual RTX3090 engine tests: outputs/verify-waifu-sd15, verify-waifu-sdxl and
  verify-waifu-sdxl-768. Nine direct/Extras comparisons, CARN 4x/8x/16x and pixel
  Hires pass. CUDA comparisons allow at most one channel value and fewer than
  0.1% changed channels; actual same-route repeat variation is separately recorded.
  SD1.5 observed 0..3 changed channels/max 1, repeats up to 2 channels/max 1.
  CPU parity is exact; GPU parity is not universally bit-exact.
- SD1.5 source, Anime/Photo denoise endpoints, CARN and Hires were inspected.
  Small 256px SDXL and two-step Hires establish functionality, not quality.
  768px SDXL source, Anime high, Photo high and CARN 1536px results were inspected,
  as was the native MainWindow screenshot. No missing tiles or conspicuous seams
  were observed in those samples. This is bounded visual evidence.
- MainWindow Generate checks use real models and PNG dimensions/settings across
  Anime, Photo and CARN. Source-checkout repeated cancellation passed both models,
  but later packaged testing found a genuine race; the earlier passing runs alone
  were insufficient. See the regression and final-package evidence below.

## Regression and interruption race

Final whole Python regression exited 0 after tile-progress integration
(outputs/waifu-final-python.log). No-torch log: outputs/waifu-final-no-torch.log.
Whole Scala covered 1296 cases: 1293 passed, one skipped, two ledger tests failed.
The preceding VRAM Estimator wave incorrectly used a plugin target for a native
core tab. Corrected index 105 to core:vram-estimator; both affected suites then
passed all 24 tests. No test exemptions or product behavior changes for that
classification correction. Logs: outputs/waifu-full-scala-build.log and
outputs/waifu-ledger-recheck-build.log. Thus the previous wave's broad-pass claim
predated its final ledger change and did not establish final ledger consistency.

First packaged candidate 2c0382c1313c8ac8ca6537dd2a345c18130d98c231191dc193e7d3b3f022c3bc
FAILED SD1.5 cancellation: no partial images, but Finished with zero images instead
of Cancelled. A deterministic JobQueueSuite regression reproduces this without
user timing: a script catches a backend interrupted error and returns before the
interruption admission reply sets the queue flag. Generator's scoped interrupted
flag was set; the queue did not observe it. Red log: outputs/waifu-interrupt-race-red.log.

Generator now immediately notifies its engine-private listener when it observes
an authoritative backend interruption. The listener sets only its captured job's
flag before the script can return. The regression checks cancellation, retention
of already completed output, absence of Finished and following-job recovery.
Final engine/test passed 218 tests with one skip and assembly passed:
outputs/waifu-interrupt-race-green-build.log. Existing ownership/protection tests
are included. The change does not treat a blocked interrupt as admission.

## Windows package acceptance

Current fixed jar SHA256:
7cc26cc4d87145e87e5817dd6635b0173ce18bff2fcb9dc6a25821791e22160d.
Current fixed ZIP SHA256:
33d04c982c01f8deb9a9d7400bbd455c93ae286b61cba60069e32d78e796e596.
Build: outputs/waifu-package-fixed/build-2a351bfb0c2e410f8e6d01a09d433079.
Extracted package: outputs/waifu-fixed-extracted/package check d70e25afe0604a05aa4786a513f19193/SDNUI.
The ZIP is a local acceptance candidate, not a published release.

Extracted SDNUI.exe started with its bundled runtime and completed dry-run
(outputs/waifu-fixed-launch.log). The native GPU harness uses external JDK 21
with the extracted jar and Python source tree because the jpackage runtime omits
java.exe. scripts/dev/check_waifu_packaged.ps1 verifies all tracked Python module
hashes, the complete nine-model manifest and unchanged jar hash around acceptance.
The GPU Python venv and preverified model cache are external test fixtures.

Fixed-package SD1.5 and SDXL acceptance both exited 0:
outputs/waifu-fixed-packaged-sd15.log and outputs/waifu-fixed-packaged-sdxl.log.
Each checks three families, output dimensions/PNG settings, three consecutive
stops after actual tile progress with zero partial images, neural recovery and
ordinary diffusion recovery. The packaged SD1.5 CARN image and SDXL MainWindow
screenshot were visually inspected. Both retained the verified jar SHA256.

Independent read-only code review found no critical/important defects. It found
missing planned cache/device-transition and RGBA regression coverage; both tests
were added and all 48 focused tests passed (outputs/waifu-review-tests.log).
The cache test covers one-entry replacement, device changes, revalidation after
on-disk corruption, and unload; RGBA verifies RGB preservation without alpha
compositing. These additions change tests only, not the verified product bytes.
Notices cover 152 projects. Ledger: 112 done, 190 todo, 51 not applicable.