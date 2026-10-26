# Tokenizer implementation plan

**Goal:** Reimplement extension-index entry 43 as a native CLIP token inspection tool.

**Architecture:** A queued `tokenize` JSON Lines request inspects only the loaded pipeline's real tokenizer(s). Typed Scala results feed a dedicated JavaFX tab; a tool capability plugin reserves id `tokenizer` without changing generation. SDXL returns separate tokenizer and tokenizer_2 results even when vocabularies match.

**Spec/source:** https://github.com/AUTOMATIC1111/stable-diffusion-webui-tokenizer at `ac6d541c7032e9f9c69c8ead2ed201302b06a4fe` (README and scripts/tokenizer.py inspected 2026-10-26). Documented UI supports text input, comma-separated ID input, colored decoded spans with IDs, and count/ID output. Root has no LICENSE; clean reimplementation, no copied source. No new dependencies.

**Constraints:** No parent reForge import/launch. No synthetic tokenizer in dry-run. No ledger/notice promotion before parent real-engine/app/UI validation. No GPU, sbt or install in this worker.

**Review focus:** Empty text; invalid/out-of-vocabulary IDs; multi-token UTF-8; long prompts without silent truncation; SDXL dual streams and dry-run/no-model errors.

## Tasks

- [x] Write failing CPU tests for real-tokenizer API adapter, byte pieces, ID decode, dual streams and validation (initial run: 11 failures due to absent method; added Unicode-vocabulary test failed before repair).
- [x] Implement pure tokenizer inspection module and queued BaseEngine request; dry-run reports unsupported rather than fake IDs.
- [x] Add typed protocol/client, native text/ID input, token chips, per-token table, decoded text and copy IDs.
- [x] Register tool capability plugin and Scala round-trip/plugin tests (execution deferred to integration owner).
- [x] Run targeted CPU tests and torch-free full suite; record limits.
- [x] Integration owner: Scala build/tests, real SD1.5/SDXL pipeline and actual app/native UI validation before ledger promotion.

Raw text is inspected literally (attention weights, schedules, BREAK, LoRA syntax and textual inversion expansion are not generation-conditioning simulation). Content IDs exclude BOS/EOS for text input; show special-token metadata separately. Counts and 75-token content capacity are clearly labeled, not claimed as actual generation chunks.

## Worker validation and integration handoff

Torch-free full suite, worktree `PYTHONPATH`, `CUDA_VISIBLE_DEVICES=-1`, main repository `.venv-test`: **274 passed, 38 skipped**. Targeted tests with main `.venv-smoke`, `USE_TF=0`, `SDFX_TEST_CACHED_TOKENIZERS=1` and cached-only real OpenAI CLIP-L + SDXL tokenizer_2 vocabularies: **15 passed**. Both runs include the final JSON Lines dry-run routing test. No model or tokenizer downloads and no GPU usage.

Real cached vocab check text `a cat, 日本語 café 🐈` returned IDs `[320,2368,267,39121,44353,34002,508,15304,2074,486]` in both streams, with CLIP-L PAD 49407 versus SDXL tokenizer_2 PAD 0. Each stream matched its direct tokenizer call and joined Unicode groups matched decoded text. Tests include empty text/IDs, malformed/out-of-vocabulary IDs, no truncation at 100 tokens, 8192-token bound, Unicode added vocabulary, no-model/dry-run errors. The fake vocabulary tests validate adapter logic, not model tokenization accuracy; the cached-vocabulary check is distinct.

### Integration commands/checks

1. Run `sbt.bat -batch test` and `sbt.bat -batch ui/assembly` on the integrated branch. New suites: `sdfx.core.TokenizerProtocolSuite`, `sdfx.plugins.TokenizerSuite`. This worker did not run Scala tooling.
2. CPU cached check: set `PYTHONPATH=<repo>/python`, `CUDA_VISIBLE_DEVICES=-1`, `USE_TF=0`, `SDFX_TEST_CACHED_TOKENIZERS=1`; run `.venv-smoke/Scripts/python.exe -m pytest python/tests/test_tokenizer_tool.py -o addopts='' -q`. Final expected count: 15.
3. Through actual engine JSON Lines, load each real checkpoint separately. Send `{"id":"t","type":"tokenize","params":{"text":"a cat, 日本語 café 🐈"}}`. Assert actual loaded model kind, stream count 1 for SD1.5 / 2 for SDXL, direct tokenizer equality, decode/re-encode of emitted IDs, no changed model/tensor state; repeat after ordinary generation.
4. In actual native app load SD1.5 then SDXL, open Tokenizer, import positive prompt, inspect text and copied IDs, switch tokenizer tabs, inspect Unicode group tooltip/table, copy all IDs, submit empty/malformed/unknown IDs and long text. Check resizing, readable scrolling, selection/copy behavior, status and usable recovery after errors. Dry-run inspection must show an explicit unsupported message. UI has a 512-group colored preview bound; the complete IDs/table remain available. Input limit is 65536 characters and 8192 tokens per stream.
5. Parent updates ledger and notices only after acceptance. No sampling hook was added, so `KNOWN_HOOKS` is unchanged. Real engine advertises `tokenizer`; dry-run deliberately omits it because it has no real vocabulary. No new dependency or third-party source inclusion.

## Parent acceptance completed

Actual `RealTokenizerCheck` passed on SD1.5 and SDXL through native controls, including both vocabularies, Unicode,
text/ID modes, copy/clipboard restoration, invalid-ID recovery, persisted tab selection and resized scrolling.
Screenshot inspection led to whole-page scrolling and stable help/input/table heights. Reports and screenshots:
`outputs/verify-tokenizer-{sd15,sdxl}-ui`. The final SDXL rerun explicitly waits for native resize pulses before
capturing compact/expanded scenes. Full Scala 491 passed + assembly; integrated torch Python707/1skip and no-torch
306/60skip. Ledger entry43 is done, notices identify clean reimplementation without copied unlicensed source.
See the parallel-wave section in `docs/VERIFICATION.md` for commands, evidence and scope.
