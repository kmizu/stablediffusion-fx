# Model Preset Manager (#87) implementation plan

Base: 4e0b32c; isolated worktree 6a08. Pinned behavioral reference rifeWithKaiju/model_preset_manager@4e25eebd4b24fb903d2815ac30b787f37a98923c has no license file. Independently implement behavior; no source code copied or imported. Parent selected full scope and authorized inline implementation.

## Design and complete scope

Use Scala core modelpresets for bounded interoperable JSON, per-checkpoint identity/storage, preset operations and Civitai retrieval. Expose through AppServices, a native Model Presets tab and MCP tools. Do not change Python inference or existing general/named presets. Preserve url/default_preset/trigger_words/presets external format and ###ModelPresets### sharing marker. Store internally by full SHA256 to avoid short-hash collisions; accept legacy short-hash named imports and export shareable documents. Keep infotext as editable text, using existing Infotext restoration for all three generation modes. No silent loss of unknown infotext text during storage/sharing.

Full operations: load local/current model, fetch missing info automatically, explicit refresh replacing URL/triggers and imported presets, choose/save/overwrite/rename/delete/set-default preset, append template, select/unselect trigger words, capture generated parameters, apply to txt2img/img2img/inpaint, set/open model URL, reveal/import/export preset JSON and sharing text, manual thumbnail and first remote thumbnail bounded to 300px preserving existing image. Native no-overwrite/conflict behavior must be explicit and reviewable; asynchronous work binds checkpoint identity and prevents stale results from editing another model.

Source discrepancy: README says explicit URL is used for retrieval, but source main.py:226 resolves from hash regardless. Implement documented manual URL precedence (Civitai model/model-version URLs), record adaptation. Open model page in system browser rather than embedding remote HTML. Literal trigger toggling avoids upstream unescaped regex deletion. Never execute remote preset text. Network reads bounded, timeout/cancellable, validate HTTP(S) and redirects, no local file endpoints; imported remote presets require explicit overwrite action. Existing preview never overwritten by download; manual replacement is explicit.

## Tasks, test-first

1. core ModelPresetDocument + ModelPresetStore: JSON roundtrip, bounded malformed inputs, exact infotext/Unicode, sharing extraction, CRUD/default fallback, identity keys, atomic revision checked writes and no overwrite export. RED then GREEN focused core suite.
2. CivitaiClient + service: API hash/version/model resolution, HTML/description sharing marker parse, timeout/status/oversize/cancel, image bounds and safe publication; local HTTP fixtures and immutable model context. Pure client independent from engine.
3. AppServices + MCP API: explicit checkpoint resolution, store/get/update/import/share/fetch/thumbnail actions, cancellable work, no arbitrary path bypass and no remote side effects without action. Focused protocol/service tests.
4. Native ModelPresetsTab: model selection/current binding, raw text, CRUD/default, triggers, URL, thumbnail, import/export/share and send/capture buttons. Reuse Infotext restoration; keep full original fields and model boundaries. Test UI state/revision/race behavior and actual MainWindow.
5. Coordinate #85 trigger behavior and later main integration. Real SD15/SDXL direct-vs-preset fixed-seed output equality in all generation modes; save/restart/import/share, PNG replay, cancellation/network errors, source model isolation and protected output. Fresh assembly/extracted Windows native acceptance, visual inspection, full Torch CPU(-1) and Scala. No heavy tests without slot.
6. Notices: no licensed source copied; pinned unlicensed reference notice. Ledger core:model-presets only after acceptance, generated docs, independent review. One wave commit only on parent instruction; no push/close.

Review focus: stale file/edit revisions; checkpoint switch during network response; full-vs-short model identity collision; existing thumbnail preservation; multiline/unrecognized infotext preservation; protected-session captured text confidentiality; HTML/JSON adversarial parsing and cancellation before writes.
