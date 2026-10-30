# Embedding Inspector implementation plan (#46)

**Goal:** Reimplement the pinned inspector workflow natively for SD1 and SDXL, preserving mixed/concatenated multivectors, inspection, save, tokenization and eval behavior.

**Architecture:** A Python worker module operates on explicitly paired text encoders and textual-inversion tensors. Worker endpoints expose bounded inspection and file operations to a JavaFX tab. Mixing and save logic are tested independently before protocol/UI integration. The upstream uses arbitrary Python eval; provide explicit opt-in tied to Allow custom code and execute only in the worker, with a clear warning, retaining tensor/item eval semantics. Do not advertise a restricted approximation as full eval.

**Constraints:** Upstream `w-e-w/embedding-inspector` at `5f3451ec0cb5c9359feb37c0b0f984f09d337509`, Unlicense; original author and reupload provenance retained. No code copied. No sbt or GPU work while #85 runs. User files never overwritten without explicit consent; save atomically. SDXL dual encoders must remain separate; never combine mismatched dimensions or silently discard an encoder.

**Review focus:** multi-vector uneven sizes and zeros; SDXL dual encoder missing or incompatible vectors; arbitrary eval authorization and error reporting; save collision and failure; tokenizer/mix output reloaded by actual SD1/SDXL.

- [ ] Test and implement weighted mixing, concatenation, combine-as-one and zero removal independently; validate SDXL pairs and eval per-vector and per-item semantics.
- [ ] Implement loaded/internal token lookup, deterministic cosine nearest, graphs/vector exports, tokenizer and source metadata. Test IDs, unknown names, multivectors, range and missing models.
- [ ] Implement atomic safetensors save and explicit overwrite, training step metadata, batch eval presets, error handling, and safe names; verify loader round trip on SD1 and SDXL.
- [ ] Add owned JSONL worker routing and JavaFX inspector, mixer, tokenizer and controls; no silent omitted upstream features. Add notes/license and protocol/dry-run tests.
- [ ] Coordinate sbt, real native SD1/SDXL model checks, actual reload/generation and visual inspection, packaging, independent review; do not claim completion until all pass.
