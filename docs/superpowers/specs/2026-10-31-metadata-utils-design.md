# Metadata Utils (#82) design

Implement the full behavior of Tzigo/metadata_utils at a4cda9fe899f95de769496946497d20daedb516b (GPL-3.0), independently written. Retained originals: outputs/metadata-utils-source. No tensor deserialization or model modification is needed to edit metadata.

## Native behavior

A Metadata Utils panel provides Checkpoint and LoRA selection/refresh, editable JSON, optional output stem, Save Copy and owned Cancel. It explains that the file hash changes. The default output is the source stem plus `_md` in the same directory; explicit names remain simple stems in that directory. Existing destinations, including links and the source itself, are never overwritten. Successful copies become discoverable after catalog refresh. Empty metadata is editable as `{}` rather than the upstream non-JSON placeholder. Source has no metadata deletion command: replacing with `{}` clears entries.

The worker exposes `metadata_utils` read/write operations, available without loading a diffusion model, including dry-run. Read returns raw string-valued metadata plus an editable representation which expands valid JSON objects inside strings, matching upstream display. Write accepts an object; string values remain exact, object/array/scalar values become compact JSON strings to maintain the safetensors string map. Duplicate JSON keys, NaN/infinity, invalid shapes/offsets, oversized headers/metadata and invalid names fail before output publication. Limits: 100 MiB header, 4 MiB edited metadata. Native adaptations are documented rather than reproducing upstream invalid safetensors writes.

## Data integrity

Read and validate the bounded safetensors header and complete tensor layout; preserve all non-metadata header entries and exact data payload bytes. Reading also fingerprints the complete source with SHA256 and binds its filesystem identity into an opaque `source_token`; hashing reports progress and checks cancellation and identity each chunk. Every write must provide the read token. Before copying, verify the identity; during the existing copy stream, hash the original header and payload and compare against the read-time digest before publication. This rejects replacement and in-place same-size edits even with restored timestamps. The UI retains the token separately from editable metadata and requires reloading after stale-source rejection or cancelled read.

Stream data to a unique temporary file beside the destination; check cancellation between chunks, detect source mutation, fsync, then publish atomically with no-overwrite semantics. Never remove a preexisting destination. Clean owned temporary files on handled cancellation/failure. The publication boundary uses JobContext ownership so cancellation cannot report success after committing a file. Return output path, byte count and payload SHA256; do not claim cryptographic confidentiality or overwrite security.

## Acceptance

Pure stdlib tests use hand-built safetensors with independently specified data offsets/bytes, malformed/truncated fixtures, Unicode/object metadata, collision/link/source mutation and cancellation/failure injection. Protocol/native tests cover async stale selection, disable/save/cancel/dispose, empty metadata and refreshed catalog. Real SD1.5 and SDXL checkpoint and LoRA copies must preserve source hashes and payload bytes, load normally, and reproduce deterministic baseline images. Exercise actual MainWindow and fresh extracted Windows package, inspect images/screenshots. Hold sbt/torch/GPU until coordinated. No commit/push/closure until parent authorization.
