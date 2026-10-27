# Stable Diffusion NUI v0.03 — scope and evidence draft

This records the user-approved implementation scope and evidence available on
2026-10-27. It is **not a release-completion or publication claim**. Final native
generation, fresh-session PNG Info/Restore and packaged EXE outcomes must be
filled by the release coordinator from completed runs.

## Included scope

| Issue | v0.03 deliverable | Explicit remainder |
|---|---|---|
| #23 | Manual named per-mode generation settings: save, list/inspect, restore, delete; shared model settings, inactive native plugin/script drafts, eta and resource preflight; separate storage from selective presets | Automatic history/thumbnails, upstream IndexedDB/text import and storage compatibility, per-category upstream browser behavior, full window/session state |
| #24 | Local UTF-8 CSV tag/alias completion in positive/negative prompt editors; keyboard/mouse insertion, reload/disable and composition guards | Physical Japanese IME coverage; extra-network/trigger-word completion, wildcard trees, chants, translations, previews and full upstream configuration parity |
| #141, child of #140 | Each WD batch PNG contains effective prompts, actual per-image generation facts, WD pinned identity and input provenance before owned publication; existing PNG Info/Restore path reads the metadata | Other interrogators and LoRA/ControlNet/script/hires/mask compatibility remain under #140; source image pixels are not reconstructed from metadata |

User guides: [tag completion](tag-autocomplete.md), [named states](generation-states.md),
[WD PNG persistence](wd-batch-png.md), [Windows package](WINDOWS_RELEASE.md).
Source pins, original license hashes and distribution exclusions are recorded in
[the source/license audit](v003-source-license-audit.md).

## Completed evidence inspected

Paths below are relative to the integration checkout
`D:/stable-diffusion-webui-reForge/stablediffusion-fx/.worktrees/release-v003`.
This historical workspace spelling remains valid after the product rename.
Logs/artifacts are local verification outputs, not files promised inside the ZIP.

| Check | Recorded result | Evidence |
|---|---|---|
| Focused Scala core suites | 68 passed, 0 failures/errors | `outputs/v003-final-scala.log` |
| Focused Scala UI suites | 86 passed, 0 failures/errors | `outputs/v003-final-scala.log` |
| Focused Python selection | 38 passed | `outputs/v003-final-python.log` |
| Native tag editor helper | 10 checks passed; synthetic events only | `outputs/v003-tags.log` |
| Actual named-state window | 5 checks passed: Save, Restore, missing-checkpoint refusal, new MainWindow/AppServices reload, Delete | `outputs/v003-resumed-states.log`; the same log separately reports 6 state unit tests |
| WD worker with real SD1.5 | Passed; worker exit 0; persisted PNG text and source hashes checked | `outputs/v003-wd-sd15.log`, `outputs/v003-real-wd/wd-real-sd15-yi8jwl6b/report.json` |
| WD worker with real SDXL | Passed; worker exit 0; persisted PNG text and source hashes checked | `outputs/v003-wd-sdxl.log`, `outputs/v003-real-wd/wd-real-sdxl-twklii15/report.json` |

Each WD real-model check used one 256×256, two-step Euler img2img result, seed
12345, denoising strength 0.75, WD inference on CPU and generation on an RTX 3090.
The models were `animekawa_v10` (SD1.5) and `waiIllustriousSDXL_v170` (SDXL).
Both reports recorded **9.1 MB allocated / 22.0 MB reserved after unload** through
the worker memory endpoint. These are process allocator observations, not total
GPU usage or a guarantee of zero retained memory. Both reports explicitly leave
native PNG Info/Restore for a separate check. Two-step output is execution
evidence, not a quality evaluation or an exhaustive model matrix.

The tag helper uses actual JavaFX controls but injects keyboard and IME events.
It does not test a physical Japanese IME. The native state reopen check creates
a new window and services on the same app home; it is distinct from verifying a
fresh packaged EXE process.

## Release gates still to record

- Final integrated native SD1.5 and SDXL tag completion → named-state Restore →
  ordinary generation, including inspection of decoded final preview/screenshots.
- Native PNG Info/Restore of real WD outputs after a fresh application session,
  including effective prompts, actual seeds, model/VAE and retained image inputs.
- Final renamed assembly, package contents/notices and hashes; clean-path ZIP
  extraction; bundled-JVM `SDNUI.exe` startup and real generation.
- Final source/remote/tag agreement, uploaded assets and downloaded SHA-256.

No result is inferred from a started log or the existence of a screenshot. The
coordinator should replace these pending items with concrete artifact paths and
observed outcomes once the corresponding run finishes.

## Issue and ledger disposition

Keep #23 and #24 open as broader extension-target trackers while the excluded
upstream behaviors remain unresolved; report the accepted v0.03 subset in each.
Alternatively, explicitly split and link the remaining work before closing a
re-scoped issue. Neither approach should label all upstream parity as achieved.
Do not promote the broad extension ledger entries merely because this subset
ships. #141 may close after its bounded PNG/native-restart acceptance and final
release checks pass; #140 stays open for the broader batch-interrogation scope.
This document itself changes no issue or ledger status.
