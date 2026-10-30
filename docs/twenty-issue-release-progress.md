# Twenty-issue release goal

Objective: resolve twenty issues, then publish and verify a release containing the fixes. This file tracks accepted issue resolutions, not extension-index rows or passing tests.

## Accepted and pushed

| Issue | Commit | Evidence |
|---|---|---|
| #15 color_correction | 35f402f3db16a7618d0d62834078cfd696caeaca | docs/color-correction-verification.md; closed 2026-10-29T23:18:32Z |
| #54 gpu_temperature_guard | 3663a6b8763b60f69bd64a371c8bd4d602a8fc59 | docs/temperature-guard-verification.md |
| #98 pnginfo_injection | 1f701d74e532309ad745b7baf4391ab992da28bd | docs/pnginfo-injection-verification.md; closed 2026-10-30T00:15:30Z |

Accepted count: **4 / 20** after the Steps Animation wave is pushed and #120 closed. Earlier closed #1, #7 and #141 predate this active goal and are not counted. No new release has been published for this goal.


| #120 steps_animation | Steps Animation wave, 2026-10-30 | docs/steps-animation-verification.md |

## In progress

Next issue selection follows the Steps Animation wave. AMD providers in that wave have fixture/schema checks, not AMD hardware validation; this limitation remains explicit in its acceptance record.

## Final release gates

Prove twenty distinct accepted issue resolutions on the remote branch, preserve each issue's behavior/license/verification conditions, build the actual distributable, test the extracted package, inspect real output images, publish the release, and verify remote tag/commit plus downloaded artifact hashes. A jar build alone does not satisfy publication.
