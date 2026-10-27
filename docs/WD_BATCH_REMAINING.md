# Remaining work: batch interrogation (#7)

The next release provides only local WD v3 interrogation before Img2Img. This is a draft for a future issue; no external issue has been created by this document.

- Implement and verify the original extension's other interrogator backends, including CLIP and Deepbooru, and the remaining WD variants. Keep model identities, license terms and download consent explicit.
- Complete original controls, backend selection, ordering, filtering, replacements and weighting where still absent from the native UI; compare against the pinned original and document deliberate differences, including alpha and preprocessing.
- Persist the actual per-image effective prompt and interrogator provenance in PNG metadata and support restoring it; currently the worker result and gallery carry the effective prompt but its output PNG does not.
- Expand real-model and native UI coverage beyond the single-image 256×256 two-step smoke cases, including multi-image batches, partial failures, cancellation and restart, and model combinations supported by the completed implementation.
- Revisit restrictions on scripts, extensions, LoRA, ControlNet, masks and hires only with explicit compatibility, resource and cancellation testing. Preserve input/output ownership and cleanup guarantees.
- Validate full upstream behavior and performance before describing #7 as complete; the v0.02 subset is not full parity.
