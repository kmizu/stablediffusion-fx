# Twenty-issue release goal

Objective: resolve twenty issues, then publish and verify a release containing the fixes. This file tracks accepted issue resolutions, not extension-index rows or passing tests.

## Accepted and pushed

| Issue | Commit | Evidence |
|---|---|---|
| #15 color_correction | 35f402f3db16a7618d0d62834078cfd696caeaca | docs/color-correction-verification.md; closed 2026-10-29T23:18:32Z |
| #54 gpu_temperature_guard | 3663a6b8763b60f69bd64a371c8bd4d602a8fc59 | docs/temperature-guard-verification.md |
| #98 pnginfo_injection | 1f701d74e532309ad745b7baf4391ab992da28bd | docs/pnginfo-injection-verification.md; closed 2026-10-30T00:15:30Z |
| #120 steps_animation | 92bb873d64a0cfed7af7e86e0351050eef4614c2 | docs/steps-animation-verification.md; closed 2026-10-30T00:37:31Z |
| #122 stylepile | 3e579dd565e8c89a9bbd50abaec4cda324079f4b | docs/stylepile-verification.md; closed 2026-10-30T01:21:11Z |
| #136 vram_estimator | b9d113e426941424c0483530a9f8402c83f80307 | docs/vram-estimator-verification.md; closed 2026-10-30T02:27:15Z |
| #137 waifu2x | this wave commit | docs/waifu2x-verification.md |

Accepted count: **7 / 20**. Earlier closed #1, #7 and #141 predate this active goal and are not counted. No new release has been published for this goal.



## Next

The seventh wave is #137 waifu2x (this wave commit; docs/waifu2x-verification.md). Select and investigate the eighth distinct issue. The #54 AMD providers have fixture/schema checks, not AMD hardware validation; this limitation remains explicit in its acceptance record.

## Final release gates

Prove twenty distinct accepted issue resolutions on the remote branch, preserve each issue's behavior/license/verification conditions, build the actual distributable, test the extracted package, inspect real output images, publish the release, and verify remote tag/commit plus downloaded artifact hashes. A jar build alone does not satisfy publication.
