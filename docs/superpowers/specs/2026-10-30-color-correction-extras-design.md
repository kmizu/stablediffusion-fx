# Color Correction Extras (#15)

Implementation design, 2026-10-30. Not an acceptance record.

## Intent

Bring reference color correction into the actual native Extras workspace, retaining index 261's reference fallback and comparison behavior. Source investigation and pinned revisions are in `docs/color-correction-extras-investigation.md`.

## Architecture decision

The native ExtrasTab currently only upscales; it does not execute ExtrasPlugin transforms. Add a color correction section there and a generator service operation, rather than placing the feature solely in the generation Script dropdown. Also register an ExtrasPlugin with id `color_correction` for the existing script/MCP path. Both routes must use one Python implementation.

Alternatives considered: reuse ColorEnhance (incorrect algorithm); reuse ADetailer mean/std transfer (incorrect A1111 semantics); implement only a script (misses native Extras workflow). Shared engine processing with both entry points covers the requested behavior.

## Inputs and outputs

Native controls: Enable color correction, optional sample image, Method (A1111 / Wavelet / AdaIN), Strength, and Also output other methods. No sample means the original input captured before any upscale. Permit correction without choosing an upscaler. Snapshot all controls before scheduling asynchronous work. Disable overlapping submissions while a job runs and restore controls on failure.

Run each method against the same post-upscale target. Never apply comparison methods sequentially. Display the selected result as primary and retain each comparison output separately with method metadata. Preserve target dimensions. Preserve the existing input/result comparison and send-to actions. Clearly identify original input, reference and output roles.

The service owns temporary images and cleans them in finally blocks. Publish permanent results only after processing succeeds. Avoid engine work on the JavaFX thread. Pass the original source explicitly, not through mutable global state or a shared context field.

## Algorithms and licenses

A1111 mode must implement the pinned LAB histogram matching and luminosity blend semantics, with verified argument ordering and quantization. Wavelet and AdaIN need separately pinned licensed StableSR primary sources before implementation. Do not copy the unlicensed extension wrapper. Keep algorithm attribution and dependencies in the existing notice collector, with optional requirements where applicable. No model download is needed for these image algorithms.

Strength zero returns the source target byte-for-byte at pixel level; identity references should avoid unnecessary round-trip drift. Reject unknown methods, nonfinite or out-of-range strength, unreadable input and missing explicit reference paths. Specify RGB/alpha handling visibly and verify it rather than inheriting accidental conversion behavior.

## Verification and release gates

First write meaningful algorithm tests for distribution transfer, unequal dimensions, constant channels, exact bypass, invalid values and independence of comparison branches. Add service cleanup/error checks and native control/integration checks. Verify the native and script routes on real SD1.5 and SDXL outputs; inspect original/reference/results grids. Run relevant Python and Scala suites, assembly and packaged native acceptance. Record observed results and remaining limits before ledger promotion and issue closure. This is one issue in the twenty-issue release objective, not the release stopping point.
