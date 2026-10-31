# Model Keyword Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans; do not commit during the parallel #48 wave.

**Goal:** Deliver the full native equivalent of #85 with per-image prompt selection.

**Architecture:** Pin upstream MIT source and investigate both databases' provenance; do not distribute the LoRA corpus without adequate rights. Isolate lookup, persistence and selection from UI and generation. Extend the actual generation path to use each image's effective prompt and preserve that prompt in per-image metadata. Native controls resolve catalog paths, expose choices and manage the user mapping files asynchronously.

**Tech Stack:** Scala 3/JavaFX, Python 3 diffusers, sbt, pytest.

**Spec:** `docs/superpowers/specs/2026-10-30-model-keyword-design.md`

## Global Constraints

- Preserve all original scope, LF, documented provenance/licences and original data.
- No sbt or heavy tests while #48 Scala suite is running; do not commit or push.
- Merge against completed #48 before final acceptance; GPU-test SD1.5 and SDXL, inspect images.

## Review Focus

- Old-hash collisions and equal filename scores: retain later matching row; fixture verifies.
- Empty, quoted, Unicode, malformed and changed mapping files: safe parsing and no data loss; fixture verifies.
- Different prompts within one batch: exact per-image seed/prompt and metadata; integration test verifies.
- Separate LoRA/TI/keyword order and user choice: all placement and choice modes; fixture verifies.
- Concurrent edits and cancellation: no lost user mapping or generation corruption; integration test verifies.

## Tasks

1. Pin source, license and full database bytes with hashes; verify provenance and identify data restrictions.
2. Test-first mapping lookup, legacy hash, CSV persistence, collision selection and per-image prompt composition.
3. Wire actual model/LoRA resolution, batch prompt encoding, metadata and script/hires integration; retain per-image semantics.
4. Add native UI controls and asynchronous mapping management, persistent settings and model refresh.
5. Run focused unit, full Scala/Python, actual SD1.5/SDXL and native/package acceptance after resource gate; visually inspect outputs and document all limits.

## Integration dependencies discovered while implementing

- Resolve the selected LoRA before `Generator.prepare` extracts adapters. Keyword values can themselves contain LoRA tags (checkpoint row `d84837a9`). The local reForge reference strips all per-image tags but activates the first prompt's adapter data per batch; reproduce that observed ordinary path or document a tested adaptation. Investigate Composable LoRA separately and preserve malformed-tag behavior without changing batch size.
- Build keyword plans once per request from the actually loaded checkpoint and selected LoRA; reserve independent iteration indices with preview semantics and preserve source/name/hash. Distinguish prompt style and Hires prompt transformations before deciding where to insert terms.
- Inside the worker, encode each effective prompt and combine batches per iteration, preserving schedule endpoints, SDXL pooled embeddings, NegPiP marker channels, different Neutral trees and Composable branch metadata. Keep ordinary paths unchanged when this plugin is off. Existing support primitives are not yet wired.
- Preserve per-image prompt/adapter state during Hires repeats, ADetailer, reference/ControlNet, callbacks and interruption. img2img Alternative inversion currently uses one batch-1 condition; a correct per-image inversion strategy is still required.
- Return the effective per-image text to Scala and write it into PNG metadata. Restore must avoid applying the keyword twice: record enough source/config/choice information to reproduce the same result or deliberately restore the effective prompt with generation hook off, with tests for the selected policy.
- Complete the native UI with dynamic matched keyword lists, selected TI/LoRA, all placement modes and mapping CRUD. The current editor and config plugin are intentionally unmounted/unregistered until generation behavior is ready.
- On integration, register `model_keyword` consistently in `hooks.KNOWN_HOOKS`, real `Engine.features()` and `dryrun.features()` and test actual wire dispatch. Do not advertise it while sampling remains unwired.
- Resolve the LoRA catalog provenance question and provide the complete original coverage through a substantiated approach; user-provided catalogs and an opt-in live lookup alone do not demonstrate offline parity.
- Upstream exposes `GET /model_keyword/get_keywords`; decide and implement an appropriate local native/MCP equivalent for inspection, preserving model/hash/source/keywords and testing access boundaries. A private worker request alone is not an externally usable replacement.
