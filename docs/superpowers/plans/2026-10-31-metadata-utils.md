# Metadata Utils implementation plan

Base: 65b4ddc; managed worktree d912. Spec: ../specs/2026-10-31-metadata-utils-design.md. Execution delegated by parent; implement in this worker. No sbt/torch/GPU without coordinated slot. Parent authorized one local commit after independent re-review; no push/closure.

## Interfaces

- python/sdfx/metadata_utils.py: `read_metadata(path, *, check, progress)` returns `{path, metadata, editable, tensor_bytes, source_token}`; `write_metadata(path, metadata_json, new_name='', *, source_token, check, progress, publish)` returns `{path, tensor_bytes, tensor_sha256}`. stdlib only. `run(params, ctx)` dispatches read/write.
- Protocol `metadata_utils`: `{operation, path, metadata_json, new_name, source_token}`. Write requires the read-time identity/full-content token. Queued worker request; typed Scala request/result, owned cancel handle using existing requestExtras lifecycle.
- JavaFX MetadataUtilsTab: checkpoint/LoRA selection, refresh/load, JSON editor, output stem, Save Copy, Cancel, status. All worker I/O off FX thread; generation models remain unchanged.

## Tasks

- [x] RED: hand-built scalar/vector/zero-size tensor fixtures; read absent/nested metadata; write preserves source bytes, data bytes and tensor descriptors; default `_md`, Unicode and explicit stem. Run torch-free unittest/pytest. Implement pure helper then GREEN.
- [x] RED: invalid JSON/duplicate keys/nonfinite/size limits, truncated header/data, overlapping offsets, traversal/reserved stems, existing destination/source/link, failed write/cancel and concurrent destination creation; source mutation detection. Implement bounded validation and atomic cleanup; GREEN.
- [x] RED: worker dispatch/read/write with actual JobContext publication ownership; cancellation/recovery; typed wire default fields and errors. Implement base/server/features/client/protocol; focused tests passed.
- [x] RED: actual native load/save/race/cancel behavior; implement panel/MainWindow registration and scoped disposal, list refresh. Focused JDK21 tests and both real-model native runs passed.
- [x] Add pinned license/notice and source hashes; deterministic actual-model helper and MainWindow/extracted-package harness passed SD1.5/SDXL after GPU coordination; artifacts inspected. Latest packaged tests include saved metadata read-back and cancellation temporary cleanup.
- [x] Independent re-review confirmed no remaining concrete blocker. Current full Python1535 passed/1 skipped and Scala1441 passed/1 skipped; parent authorized ledger promotion and a local commit.
- [x] Review correction: read-time identity/content binding. Python RED/GREEN28 passed; current Scala/native and extracted-package stale editor rejection/reload, read cancellation, both model families and full regressions passed. Earlier failed timeout runs and their evidence limits remain recorded in the verification doc.

Review focus: exact raw string roundtrip despite object display; no metadata-driven tensor changes; no destination overwritten in races; cancellation owns only its copy; stale UI results cannot target a new selection.
