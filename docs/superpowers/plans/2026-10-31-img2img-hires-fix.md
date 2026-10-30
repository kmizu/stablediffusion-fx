# img2img Hires Fix Implementation Plan

> For agentic workers: use superpowers:executing-plans to implement task-by-task.

**Goal:** Implement full pinned #61 post-image upscale/refine behavior and native restoration.
**Architecture:** An independent engine hook adds a job-local post-image pass; existing native latent Hires remains untouched. A registered AlwaysOn plugin owns controls and source metadata restoration.
**Tech Stack:** Scala3/JavaFX, Python/diffusers/Pillow, existing upscalers.
**Spec:** ../specs/2026-10-31-img2img-hires-fix-design.md

## Constraints and review focus

- Worktree4496 is based on b8a623f; never edit main or #85.
- No sbt/heavy/GPU tests until parent coordination. Parent subsequently authorized one local commit after acceptance; no push/closure from this worker.
- Preserve per-output seed order, native latent path, completed-output cancellation, adapter cleanup and disabled exactness.
- Test explicit dimensions/one-axis fallback, denoise0, prompt inheritance, overrides, malformed metadata and cancellation after one item.

## Tasks

- [x] Add failing pure normalization/target/pass-parameter tests in python/tests/test_img2img_hires_fix.py; run CPU-only focused suite; implement python/sdfx/img2img_hires_fix.py.
- [x] Add failing real generation orchestration tests for txt2img/img2img/inpaint/batching/cancel/disabled; integrate engine finish_batch, base validation, feature declarations and dryrun parity.
- [x] Add pass-specific adapter/path resolution and tests preserving manual/tag identities and cleanup; avoid shared checkpoint changes.
- [x] Add Img2ImgHiresFixPlugin with all source controls and bounded metadata replay, register it, connect restoration and notices; focused Scala tests and actual JavaFX controls passed.
- [x] Coordinate native/GPU/fresh-package acceptance, inspect images, record exact evidence. Both families passed actual MainWindow and extracted-package runs. Independent review's negative-LoRA interpretation was checked with seven upstream host/method traces and retracted; no product change.
- [x] Parent integration gate: independent review complete; whole Python1507/Scala1437 passed (one skip each). Parent authorized ledger classification, generated docs and one local commit; push/issue closure remain parent responsibilities.
