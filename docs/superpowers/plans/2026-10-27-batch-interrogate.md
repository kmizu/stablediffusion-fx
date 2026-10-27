# Img2img Batch Interrogator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkboxes.

**Goal:** Interrogate each ordered Img2img input with explicitly available models and compose its own generation prompt without contaminating other images or changing original inputs.

**Architecture:** An immutable native job captures ordered inputs, prompts and interrogation options. A serialized worker produces bounded per-model text and a pure deterministic composer constructs each image's effective prompt; generation uses that exact per-item snapshot. Native UI exposes capability, preview and owned cancellation; optional models require independently reviewed assets.

**Tech Stack:** Scala 3 / JavaFX, Python 3.10–3.12, current Torch/Diffusers, JSON-lines worker, local interrogator adapters.

**Spec:** `docs/superpowers/specs/2026-10-27-batch-interrogate-design.md`.

## Global Constraints

- The original extension is MIT; archive exact license with source attribution before acceptance. External model/extension permissions are separate.
- Do not launch/import the parent Web UI, auto-download assets, substitute BLIP/WD for CLIP/Deepbooru, or claim synthetic outputs are real inference.
- Preserve source bytes, native prompts and all still-generation behavior when disabled. Each item derives from the immutable submitted prompts.
- Enforce limits, job ownership, targeted cancel/protection, explicit capability and no stale UI. Entry 217 remains pending until full scope and evidence are met.

## Review Focus

- Two images share prompt text or a repeated term: independent composition must not leak the prior interrogation; Task 1 and 3.
- Unavailable optional model or mid-batch failure: visible per-item failure, never silently use a substitute or previous caption; Tasks 2–4.
- Empty result, punctuation/emoji, escaped attention, empty replacement: retain user syntax and match documented rules; Task 1.
- Selected input or model changes while queued: snapshot identities and options and reject unsafe replacement; Tasks 2–3.
- Cancel/close during load, inference or generation: correct owner, no stale UI, cache/resource cleanup and no input mutation; Tasks 2–4.

### Task 1: Pure per-image prompt composition

**Files:** Create `python/sdfx/batch_interrogate.py` and `python/tests/test_batch_interrogate.py`.

**Interfaces:** `compose(original_positive: str, original_negative: str, ordered_results: list[str], options: BatchInterrogateOptions) -> ComposedPrompts` returns new positive/negative strings and diagnostics without mutable global state.

- [ ] Write failing independent fixtures for ordered interrogators, empty results, prepend/append, reverse, weighting, duplicate/exaggeration, positive/negative/custom filters, escaped attention, punctuation and emoji, ordered replacement including empty/unpaired pairs, WD tag `>` vs rating `>=`. Assert two items with different captions both start from the same untouched originals.
- [ ] Run `python -m pytest python/tests/test_batch_interrogate.py -q`, confirm expected failures, implement the spec's transformation order without importing the parent Web UI; explicitly record divergences from source bugs.
- [ ] Re-run tests green and review exact matches versus unlicensed dependencies; commit.

### Task 2: Capability and interrogator adapters

**Files:** Create `python/sdfx/batch_interrogate_backends.py`, `python/tests/test_batch_interrogate_backends.py`; inspect `python/sdfx/extras.py`, `python/sdfx/engine.py`, dependency/asset notices.

**Interfaces:** `available_backends() -> list[BackendStatus]`; `interrogate(image, selection, ctx) -> OrderedResults`; `unload(selection) -> UnloadResult`. Identities include explicit model/revision and supported options.

- [ ] Inspect actual installed APIs and immutable model sources/license terms; record per-model pins, hashes and preprocessing contracts. Explicit opt-in for any fetch; no trust in metadata as a grant. If no approved CLIP/Deepbooru/EXT assets exist, leave that choice unavailable.
- [ ] Write failing tests for backend/model order, unavailable dependency/invalid mode, strict thresholds/ratings, bounded image/text/VRAM, no implicit network, cancel between models and unload/failed load; use injected synthetic backend without claiming real inference.
- [ ] Implement adapters using independently permitted code and local assets with safe loading and limits. Run focused tests and review subprocess/offline behavior; commit.

### Task 3: Owned per-image Img2img job pipeline

**Files:** Modify `core/src/main/scala/sdfx/core/protocol/`, `engine/src/main/scala/sdfx/engine/EngineClient.scala`, `EngineSupervisor.scala`, `Generator.scala`, `python/sdfx/server.py`, `python/sdfx/engine.py`; add focused engine/Python tests.

**Interfaces:** Typed batch request contains ordered local image identities, frozen generation prompts/options and backend selections; owned handle exposes progress, targeted cancellation and typed per-item outcomes with effective prompt.

- [ ] Write failing tests for ordered input mapping, independent prompts/seeds/metadata, checkpoint or file replacement, selected model missing, individual failure and skipped item, disabled baseline, targeted pre-start/in-flight/commit cancellation and protection retry, input immutability and cleanup.
- [ ] Implement serialized per-image interrogation before generation using Tasks 1–2, bound snapshot/path handling, and truthful error/progress mapping. Preserve state/LoRA and do not mutate global UI prompt. Protect engine-owned operations and no-clobber outputs.
- [ ] Run focused Python/Scala engine and protocol suites; review owner and resource lifecycle. Commit.

### Task 4: Native controls and persistence

**Files:** Modify Img2img generation view/state in `ui/src/main/scala/sdfx/ui/`, appropriate settings and `core/` protocol types; add `ui/src/test/scala/sdfx/ui/BatchInterrogateStateSuite.scala` and a native UI helper.

**Interfaces:** One frozen request per submission; ordered model chooser, context-dependent settings, per-input effective prompt preview and status; bounded custom filter/replacement persistence with explicit overwrite confirmation.

- [ ] Write failing tests for disable/default, ordering, backend capability changes, multi-input admission, per-item output/error, filter/pair editing and explicit overwrite, close/stale progress, cancel/protection and minimum-window layout.
- [ ] Implement responsive JavaFX controls and state using Task 3 handles; never write user inputs. Run UI tests and real JavaFX helper; inspect normal/minimum screenshots and error/disabled states. Commit.

### Task 5: Full acceptance and notice

**Files:** `docs/VERIFICATION.md`, `docs/HANDOFF.md`, `scripts/third_party.py`, archived `licenses/Alvi-alvarez__sd-Img2img-batch-interrogator.txt`, generated notices/ledger only after full acceptance.

- [ ] Run exact-head targeted and full regressions, record commands/counts, failure recovery and offline/no-optional-dependency behavior.
- [ ] Run small real SD1.5 and SDXL Img2img batches using locally verified compatible interrogator assets; check actual composed prompts and metadata, no cross-item contamination, input hashes, disabled baseline and basic VRAM allocation/free/cancellation. Inspect native outputs/screenshots. This is bounded core evidence, not exhaustive model parity.
- [ ] Compare every model/control against spec inventory. Record missing optional CLIP EXT / WD EXT / Deepbooru and licensing as explicit unverified gaps; do not promote 217 or close issue if any required behavior lacks evidence. Archive and regenerate notices only after source review; commit.
