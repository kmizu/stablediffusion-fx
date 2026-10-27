# WD batch PNG persistence (#141)

Scope: child [#141](https://github.com/kmizu/stablediffusion-fx/issues/141) of
[#140](https://github.com/kmizu/stablediffusion-fx/issues/140), v0.03. This does
not complete the broader batch-interrogation extension or add other taggers.

Every successful registered batch output receives the existing A1111
`parameters` PNG text before the worker publishes its owned directory. It
records the effective per-input positive/negative prompts, actual per-image
seed and variation seed, dimensions, sampler/resolved scheduler, steps, CFG,
clip skip, denoise/resize, batch counts, eta, checkpoint/VAE names and hashes
returned by the engine, and optional hardware information. The WD extra field
uses JSON for the already defined provider provenance: model ID, full model
and tags SHA256, thresholds, formatting flags and preprocessing version.
Original input path and SHA256 are informational provenance, not replay paths.

The writer consumes the existing snake_case GenerationParams worker contract
using its typed parser. It does not introduce a parallel camelCase replay
schema. Numeric fields and JSON-quoted values use the native Infotext labels;
unknown WD fields remain readable in PNG Info as ordinary extension fields.
Core Restore now consumes batch count, batch size, eta and resize mode. The
form retains imported eta when submitting the next request. A selected init
image or painted mask is never replaced from PNG metadata.

All generated paths are checked against private item directories before any
metadata write. Text is bounded to 1 MiB per PNG, PNG processing to 128 MiB,
and the existing aggregate output budget is checked again after insertion.
Uncompressed Unicode iTXt is placed before IDAT for metadata-only readers.
Existing IDAT and all unrelated chunks remain byte-identical; existing
parameters chunks are replaced. Source images are never rewritten. Temporary
metadata files stay within the owned staging directory and are cleaned on
failure. Cancellation checks surround writes and the existing worker-owned
publication gate still decides cancellation versus commit atomically.

Restore needs an init image chosen by the user; a result PNG cannot recover
the original source pixels. Source paths and hashes are evidence of the
actual input, not an assurance the original file still exists. WD generation
still rejects native masks, hires, hooks, LoRA and enabled ControlNet. This
work does not relax those limits. Applied style content is captured in the
effective prompt; restoring it does not reapply a named style a second time.

Implementation follows existing native `Infotext.scala` and the repository's
WD provider contract. No upstream source was copied and no new dependency or
license is added; existing WD and batch-interrogation attribution remains in
the repository notices. Automated and real-model acceptance are recorded by
the release integrator in VERIFICATION.md; this document is not a claim that
those checks have already passed.
