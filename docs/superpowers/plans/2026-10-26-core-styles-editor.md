# Native Styles Editor and Extended Style Saver

**Goal:** Complete both index163 StylesEditor and index248 ExtendedStyleSaver as the native `core:styles-editor`
workflow. Preserve user CSVs, support live prompt-style editing and separately identified extended presets.

## Authoritative sources / full comparison

Inspected source trees, READMEs, supplementary docs, source and recent history:

| Source | Pin / licence | Implemented behavior to preserve |
|---|---|---|
| chrisgoringe/Styles-Editor | `025f17f747975a1fc914f67806e09dcb80e6eb69`, Apache-2.0 | Editable name/positive/negative/sort/notes; native cell cut/copy/paste and multi-row deletion, duplicate, automatic saves and live styles refresh; name uniqueness by appending x; literal replace; all-column case-sensitive substring, insensitive substring and regex filtering with invalid-regex feedback; reorder/numeric autosort; notes; additional CSV categories using `category::name`, create/move/merge/remove empty categories; changed-only ten-minute backups, keep24, restore/import/export, optional AES-Crypt v2 `.aes` backups. |
| harukei-tech/sd-webui-extended-style-saver | `7750cc6b2dcba91951402a682819bcb4fcc43bea`, no licence in pinned tree | Save current positive/negative prompts, width/height, checkpoint and VAE only; UUID-distinct entries even with identical names/models; apply replaces prompts/size and loads model+VAE; exact current-model/VAE filtering, refresh, confirmed deletion, canceled save/delete performs no write. No other sampler/steps/guidance/ControlNet snapshot; ControlNet support was removed in history. |

No code is copied from either extension. Browser-specific Gradio/DOM workarounds become native controls and file
choosers. Extended presets retain a separate compatible CSV schema and UUID identity; they are not collapsed by
style name or silently applied as ordinary `{prompt}` templates.

## Architecture and native choices

- Native Styles tab, opened from every prompt toolbar. Editable spreadsheet and multiline detail editor, selection
  actions, sorting/filter/replace, category switch/move, backups and encryption. Existing prompt Apply retains
  A1111 merge/`{prompt}` semantics. Save current prompts asks create/overwrite explicitly; extended Save records
  only its documented six settings. Extended Apply validates/loads model then VAE sequentially and updates the
  target generation form, retaining unrelated settings and img2img inputs.
- `StyleDocument` preserves UTF-8/BOM, line endings, unchanged record bytes, field order/extra columns and quoting.
  Changed records use RFC4180 escaping; the legacy `text` prompt column remains usable. Invalid/foreign files are
  rejected for editing. Saving compares the loaded byte version immediately before atomic replacement; external
  changes fail visibly and require Reload, never overwrite from a stale UI cache.
- `StyleLibrary` owns AppHome config: master `styles.csv`, `style-files/<category>.csv`, `extended-styles.csv`, and
  `style-backups/`. Category paths are validated; external/import sources are read-only. The merged master remains
  the actual live generation source. Category writes merge immediately rather than waiting for a web tab switch.
- Notes use a CSV column while preserving unrelated fields. Standard style names are unique on edit/create using
  upstream's x suffix convention; existing duplicates are not silently merged. Extended rows always use stable UUIDs.
- AES-Crypt v2 compatibility uses JDK AES-CBC/HMAC/SHA256 and SecureRandom, based on official stream-format docs and
  the primary pyAesCrypt protocol description; no product Python crypto dependency. Authenticate before restore.
  Passwords remain transient in memory, never UI-state/config/logs. Plain active styles remain unencrypted, as upstream.
- File operations run off the FX thread. Publish dropdown changes only after successful disk persistence; reload
  reads the actual file. Backups are pending-change driven at600seconds and capped at24, with explicit Backup now.

## Test-first tasks and parent gates

- [x] Core document/library tests first: byte preservation, CRUD/duplicate/order/filter/replace, UUID distinction,
  template semantics, additional categories, external conflicts, atomic failure, backup retention/restore/cancel and
  AES-Crypt interoperability/tamper/password errors. Parent runs RED before production implementation.
- [x] Implement storage/backup/crypto and extended preset application without changing normal generation semantics.
- [x] Native editor/toolbar/live reload and explicit extended Apply; focused native acceptance helper.
- [x] Parent targeted GREEN native suites and real isolated-AppHome JavaFX controls/screenshots (CPU dry engine).
- [x] Parent SD1.5/SDXL real conditioning: edited named style equals explicit expanded prompt, immediate reload after
  writes, extended model/VAE/size/prompts restoration, unchanged unrelated settings, saved metadata/replay.

Parent owns all sbt/model/UI execution, final notice/ledger/generated-doc changes and promotion. Worker commits only
owned source/tests/manual acceptance/this plan, no push. No unsupported scope is inferred from unbuilt todo.md ideas.

## Frozen implementation evidence (2026-10-26)

Parent ran native RED before implementation (`outputs/styles-core-red.log` in the main checkout). Subsequent
observed REDs covered category-mirror removal/backup recovery, unsaved reordered category identity, foreign
`text` alongside canonical `prompt`, duplicate normalized headers, and unnamed draft preservation. Fixes retain
all row payloads, merge a category draft before identity-based moves, reject ambiguous schemas, and synchronize
master/category mirrors during restore. External byte-version conflicts remain errors rather than overwrites.

Parent's main-checkout `outputs/styles-core-ui-dry.log` completed at17:19:22 with:

```text
core/testOnly sdfx.core.StyleDocumentSuite sdfx.core.StyleLibrarySuite sdfx.core.StyleBackupCipherSuite sdfx.core.StylesSuite sdfx.core.CsvSuite
Passed: Total37, Failed0, Errors0
ui/Test/compile
success
ui/Test/runMain sdfx.ui.RealStylesEditorCheck <main-smoke-python> <checkpoint-directory> animekawa_v10 outputs/verify-styles-cpu --dry-run
success
```

The actual JavaFX helper exercised native autosave, multiline Unicode notes, byte preservation, live dropdowns,
clipboard cut/copy/paste with clipboard restoration, template Apply, duplicate/reorder/filter/replace, confirmed
and canceled deletion, creation from prompts, category moves, encrypted backup and full category recovery,
same-name UUID presets, exact model/VAE filtering and six-setting Apply with unrelated drafts preserved.
`outputs/verify-styles-cpu/styles-editor-report.json` records `status:passed`, `mode:cpu_dry_run` and
`password_persisted:false`; screenshots are `styles-editor.png`, `extended-presets.png`, and
`styles-minimum-viewport.png`. These are genuine native controls using a dry engine, not real-model conditioning
evidence. The later real-model gate is recorded below separately.

AES-Crypt interoperability is bidirectional. The committed Unicode-password fixture was encrypted by unchanged
external pyAesCrypt at `13a188f8799306ad9034167bf0bc0489319db274`; its provenance and hashes are beside it.
Parent's `core/Test/runMain sdfx.core.StyleCryptoInteropCheck outputs/styles-crypto-interop` emitted eight JDK
encryptions (empty, block-edge, partial and Unicode payloads). The worker ran:

```text
<main-smoke-python> scripts/dev/styles_crypto_reference_check.py outputs/styles-sources/pyAesCrypt/crypto.py outputs/styles-sources/crypto-reference-packages outputs/styles-crypto-interop
```

All eight were decrypted byte-for-byte by that unchanged external implementation; the helper asserts its source
SHA256 before import. Evidence: `outputs/styles-crypto-interop/reference-decrypt-report.json` (`status:passed`).
Test-only dependencies stayed in ignored `outputs/styles-sources/crypto-reference-packages`; no global install,
product runtime dependency, upstream source, or installed wheel is included in this change.

Intentional native differences: notes live in a preserved CSV column; native controls replace Gradio/browser DOM
workarounds; category changes merge immediately and on leaving a draft; active CSVs remain plaintext even when
backup encryption is enabled; extended preset files are separate from prompt-style/category backups. The UI
states that backup scope explicitly. The unlicensed ExtendedStyleSaver behavior is reimplemented independently.

## Final integrated real-model and cleanup evidence

The first SD1.5 native run completed its feature assertions and wrote its report, but then the acceptance helper
failed while restoring a clipboard format whose advertised content was null. This was a test-helper cleanup
failure, not a passing process or a product-conditioning failure; the original run is not counted as final GREEN.
Follow-up `2cd13edd0bca80cd4b14f28fe28717b77fcaa12f` preflights every advertised clipboard format before any
clipboard mutation, explicitly skips the mutating checks if preservation is unsafe, and attempts every cleanup
independently while retaining original/suppressed failures. The same unsafe pattern was fixed in RealTokenizerCheck.
Parent's `outputs/styles-cleanup-scala.log` records both NativeAcceptanceSafetySuite tests passing and UI test
compilation succeeding at19:49:06. The previously validated37 core tests remain the product regression evidence.

After integration, parent ran both actual native model gates to successful process exit in
`outputs/styles-native-models-clean.log` (main checkout):

| Model | Final run | Report |
|---|---|---|
| SD1.5 `animekawa_v10` |31seconds, exit0, completed19:50:48|`outputs/verify-styles-sd15/styles-editor-report.json`|
| SDXL `waiIllustriousSDXL_v170` |51seconds, exit0, completed19:51:39|`outputs/verify-styles-sdxl/styles-editor-report.json`|

Both reports say `status:passed`, `mode:real_model`, `skips:[]`, and `password_persisted:false`. Both executed
the clipboard checks and restored the clipboard without cleanup errors. Each uses its own isolated AppHome and
checks the complete native editing/category/backup/extended-preset workflow described above. For each model,
the named-style generation and the explicitly expanded positive/negative prompts produced exactly equal decoded
PNG pixels, and embedded prompt metadata matched the expansion. Settings were4steps,384×384,seed321: this is
a conditioning/persistence parity check, not an image-quality benchmark.

Both output directories retain `styles-editor.png`, `extended-presets.png`, and the actual whole native window
at1280×800 in `styles-minimum-viewport.png`; the earlier CPU screenshots are not substituted for model-run
artifacts. Parent owns final visual assessment, ledger/docs/notices and promotion. This evidence update changes
only this owned plan, with no further production or helper edits.

## Review follow-up: VAE consistency, single-flight Apply and rejected drafts

Independent review identified three reproducible integration issues after that initial integrated acceptance:
blank preset VAE reset the backend but retained a stale UI VAE selection; repeated Apply could overlap model/VAE
workers; a rejected autosave draft could disappear on row navigation. These are fixed together in the follow-up:

- Blank/whitespace-only preset VAE resolves to `Automatic` consistently in restore/request, native validation,
  worker comparison, committed UI selection and both current-model/VAE filters. Preserved CSV bytes are not
  rewritten merely by loading. Parent observed the new restore assertions fail before the fix
  (`outputs/styles-blank-vae-red.log`,9tests,1failure) and then StyleLibrarySuite pass all9.
- A shared FX-thread Apply flag remains held through model loading, VAE loading and form commit/failure. The
  actual native Apply button is disabled, and direct callback attempts are rejected before starting another
  worker. The first request therefore completes as one operation rather than competing with another Apply.
- Failed persistence writes the exact proposed document to a separate `style-drafts` recovery CSV before
  unlocking navigation and reports its path. If recovery also fails, row/scope/file navigation is blocked while
  detail fields and the pending document remain available for retry/close recovery. External bytes stay intact.

Parent ran `core/testOnly sdfx.core.StyleLibrarySuite`, passing all9 in `outputs/styles-review-green.log`
(worktree); that log then records a missing UI import. After its correction, `ui/Test/compile` passed and parent
ran the actual native helper with dry engine, completed20:01:41 in15seconds with clean exit
(`outputs/styles-review-native-cpu.log`).
`outputs/verify-styles-review-cpu/styles-editor-report.json` records all9checks, `status:passed`, `mode:cpu_dry_run`,
`skips:[]`. The additional checks induce a real external-edit conflict, inspect the recovery CSV after selecting
another row, assert external bytes unchanged, edit preset VAE blank through the native form while current VAE
selection is `None`, assert backend/UI/next-request default consistency, and attempt both disabled native Apply
and a reentrant direct callback while the worker is in flight. No external VAE weights are required.

These follow-up sources are frozen after that CPU/native gate. Parent will rerun actual SD1.5/SDXL acceptance
and aggregate tests against the integrated follow-up commit; the earlier real-model results above predate it.

### Double-failure recovery lock

Review of the exceptional fallback found that, when both active persistence and recovery persistence fail,
editable detail widgets could rebuild from the old document and replace the exact pending table-cell document.
The minimal follow-up disables only the details grid while `recoveryBlocked` is true; the Save/retry button stays
outside the grid and remains usable. No sampling, model or generation-request path changes.

Parent observed native RED in `outputs/styles-double-failure-red.log` (worktree,19seconds,exit1,20:05:52):
the new helper assertion failed because the detail widgets were not locked. After the three-line production
fix, the same actual native helper completed all10checks in25seconds with exit0 at20:07:31
(`outputs/styles-double-failure-green.log`; report `outputs/verify-styles-double-failure-green/styles-editor-report.json`).
The regression commits a real table-cell edit, forces an external
conflict and blocks the isolated recovery-directory path, waits for both persistence attempts to fail, checks
disabled details and enabled Retry, dispatches an attempted key edit, repairs the isolated paths and retries.
The resulting entire CSV must exactly equal the previously expected pending document bytes. The helper restores
its temporary path arrangement in finally. Parent owns final integrated aggregate verification and assembly.
