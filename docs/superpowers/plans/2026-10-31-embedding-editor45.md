# Embeddings Editor (#45) Implementation Plan

**Goal:** Independently implement the documented editor workflow: choose loaded embedding and vector, edit all coordinate weights with reference-color markers, save safely and preview unsaved edits, including SDXL's two independent encoders.

**Architecture:** Pure validated editor state owns unsaved values and preserves every unedited vector. A native JavaFX tab exposes model-dependent coordinate controls and guidance. A queued worker reads vectors/ranges using #46 ownership, publishes save through its transactional refresh, and temporarily substitutes preview vectors only within the serialized generation encoding scope. Coordinate the preview scope with #47; never nest offload placement guards, and restore on success, error or interruption.

**Tech:** Python/torch worker, Scala/JavaFX client, JSONL v1, pytest/munit; no upstream code or assets copied. Reference CodeExplode c8f290230f9360729abc3a3a57fe1befc2d43b23 has no license file.

**Review focus:** model/checkpoint changes while a draft is open; multislot SDXL vector identity and unequal widths; concurrent edits/file replacement; cancellation, offload hooks and preview rollback; saved output partial failure and interruption.

- [x] Test and implement pure finite bounded coordinate editing, selection/unsaved-state retention, guidance color parsing and marker math. Verify empty, invalid, flat range and both encoders without torch.
- [x] Test actual worker reading #46-owned vector rows and tokenizer guidance, with ranges based on original vocabulary per encoder; preserve active versus file snapshots and detect stale model/file.
- [x] Implement native editor: accessible paginated coordinate sliders/numeric fields, two encoder selectors, vectors, guidance, refresh, save and owned preview generation. Never silently discard unsaved values.
- [x] Build save publication with identity-bound overwrite consent, full-vector preservation, #46 live refresh and fail-closed recovery; meaningful interruption/collision tests.
- [x] Integrate temporary preview at serial generation encoding stage with #47 scope contract; isolate jobs, do not persist transient weights, restore on all exits and verify offload + model failure recovery.
- [ ] Verify source scope, real SD1.5/SDXL native preview/save, both offload modes, protected output policy, full regression and extracted package; inspect actual images. Do not mark ledger done or commit/push until coordinated.
