# Embedding Merge Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Execution delegated to this worker; no additional subagents.

**Goal:** Full pinned #47 expression, inspection, runtime and export behavior in native UI.
**Architecture:** Separate pure parser/numerics, model-vector adapter and scoped runtime installation. Typed queued inspection/export protocol and native tab; preserve raw-expression metadata.
**Tech Stack:** Python3.10/NumPy, existing torch/diffusers at integration, Scala3/JavaFX/JDK21.
**Spec:** ../specs/2026-10-31-embedding-merge-design.md

## Global constraints

No source runtime imports, no Python eval, no sbt/torch/GPU until coordinated. Preserve #46 TI ownership/refresh/offload contracts. Local wave commit is now authorized after final checks; parent handles push and closure. Store complete source provenance and exact license. Full SDXL branch scope includes conversions, grouping and infotext replay.

## Review focus

1. Unequal L/G row counts and tokenizer differences must not silently mix positional meanings.
2. Existing token/name/suffix collisions cannot overwrite persistent TI rows.
3. Cancellation or exceptions during either encoder installation must restore both streams and hooks.
4. Repeated expressions across positive/negative/Hires must restore original expressions in saved outputs.
5. File races, malformed/nonfinite expressions and large allocations must fail without partial output/state.

## Tasks

- [x] Parser/numerics: create python/tests/test_embedding_merge.py with source-derived hand-calculated fixtures, observe RED with CPU venv, implement python/sdfx/embedding_merge.py. Interfaces `parse_expression(text)->tuple[Action,...]`, `evaluate(actions, resolve_text, resolve_token, stream_names)->tuple[np.ndarray,...]`; enforce configured row/input bounds. Cover operator/result precedence, apostrophes, commas/special IDs, asymmetric streams and error cases. Targeted tests passed;14 SD1/16 SDXL numerical cases also matched the isolated source parser exactly.
- [x] Model adapter/inspection: create python/sdfx/embedding_merge_model.py and dedicated tests. `inspect(engine,text,grouping,ctx)->dict` supplies raw text/TI vectors and six grouped statistics, source-compatible literal/token handling and pasted-infotext reconstruction. Coordinate #46 integration before accessing model internals. Test exact rows against actual embedding tables, multi-token Unicode and live owned TI references.
- [x] Runtime: `expression_scope(engine,prompts,ctx)` yields transformed prompts/provenance and restores all transient rows/maps on exit. Add engine integration only atop agreed #46 loader contract. Test all success/failure/cancel paths, collisions, both-encoder rollback, schedules/base/Hires/negative and model switching. Model invalidation required if rollback fails.
- [x] Export/protocol: write source-compatible formats/conversions via owned publication with no-overwrite, integrate queued `embedding_merge` request in base/server/real/dry-run and typed Scala request/result/client. Tests prove formats/tensor equality, native preview request cannot publish, cancellation ownership, path/collision validation and failed activation reporting. Use #46 refresh rather than alternate loader.
- [x] Native tab and restoration: create EmbeddingMergeTab, MainWindow lifecycle, model-required state, grouping/inspect/save controls and tables; raw-expression PNG/native history restoration. Test actual control actions and metadata replay, queued snapshots and stale completion/disposal.
- [x] Acceptance: real SD15/SDXL runtime expression versus saved TI vector/RGB parity, base/negative/Hires, interruption and off recovery; fresh assembly/extracted package, screenshots/images, whole suites and independent review. Add pinned notice and only then ledger done; parent handles integration authorization.
