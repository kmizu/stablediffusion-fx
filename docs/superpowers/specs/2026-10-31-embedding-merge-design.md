# Embedding Merge (#47) design

Initial base777d128; integrated accepted main09202c7; upstream klimaleksus/stable-diffusion-webui-embedding-merge pinned40d439de3cda5965fc1f9cb4412ef329885d1d6d (default SDXL branch). Original README, complete1734-line script and license retained under outputs/embedding-merge-source with SHA256 manifest. License is Unlicense text prefixed `WTFPL, actually.`; preserve exact notice. Independent implementation, no A1111 runtime imports.

## Scope

Implement the entire pinned extension as a native Embedding Merge tab and automatic prompt-expression support. Inspect plain prompts, merge expressions or pasted generation infotext; six grouping choices (none/comma/parts/words/tokens/vectors), original text/token identity, row ranges/counts and min/max/sum/absolute sum/L2/sample standard deviation. Separate CLIP-L/G results on SDXL. Parse errors are actionable and never silently produce ordinary text.

Expression syntax: quoted literals with doubled-quote escapes and existing TI names; addition/subtraction with zero row padding; multiplication/division, SDXL L/G suffixes; row count crop/zero-padding and circular shifts; `=` applies modification to accumulated result. Commas append literals, token IDs, or special aliases0/00/000/0000 (ordinary0/pad/end/start). Preserve source operation order rather than Python arithmetic precedence. Angle and curly embedded expressions work in positive/negative/base/Hires prompts and surrounding native attention. Empty results remove that expression. Do not execute Python eval.

Native Save supports source-compatible PT and safetensors, including converted SD1 CLIP-L and SDXL CLIP-L/zero-G variants, safe output names, collision refusal and explicit current-model activation status. Output files live under managed embeddings/embedding_merge; never overwrite unrelated files. Saved expressions must reproduce runtime vectors and actual generated RGB. Prompt provenance preserves raw expressions across PNG save/restore; ephemeral names must never become unrecoverable saved prompts. Input/vector bounds and nonfinite/divide-by-zero failures are native safety adaptations, documented rather than silently normalized.

## Architecture and ownership

Pure parser/actions and NumPy float32 numerics are independent of torch and model state. Model adapter supplies ordered raw token/TI rows for each stream; frozen input-embedding rows are distinct from contextual/pooled encoder outputs. Runtime expressions receive collision-free scoped token names, installed only on the serial worker, then fully restored on success/error/interruption. No transient names enter persisted file ownership maps or catalog snapshots. Queued jobs capture raw expressions and merge settings.

#46 owner confirmed `embedding_token_owners[name][clip_l/clip_g]`, `loaded_embedding_files`, `embedding_snapshots`, `_with_embedding_encoder_access`, transactional embedding_refresh and fail-closed partial-load recovery. Preserve those contracts. Do not patch the older loader in this base; coordinate integration of #46 before runtime changes touch it. Temporary context snapshots both tokenizer maps and encoder weights, rejects all existing-token collisions, restores offload hooks and invalidates model if rollback fails.

## Acceptance

Source-derived grammar/numerical fixtures including uncommon tokens and L/G operations; no torch for initial tests. Real SD1.5/SDXL native and packaged runtime-versus-saved-vector RGB equality, positive/negative/Hires use, multiple expressions, original prompt replay, source immutability, switch/reload, cancellation and exact disabled recovery. Native grouping, table, export and infotext restoration controls exercised. No acceptance/ledger promotion before source parity review, whole regressions and visual inspection. Parent coordinates all sbt/torch/GPU slots. Local wave commit is now authorized after final checks; parent handles push and closure.
