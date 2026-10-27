# Named generation states (#23)

Goal: Save/list/restore/delete complete named generation-settings snapshots per mode, separate from selected-field Config/Utilities presets.

1. Reuse typed NativePresetFields, PresetJson/Store and serialized NativePresetsSession; add opt-in model and WD batch bindings without changing existing presets.
2. Add generation-state collections in a dedicated config/generation-states directory and a native list/detail pane with explicit overwrite/delete confirmation, reload and backup recovery.
3. Preflight every captured field and referenced input before restoration; block the entire operation for unavailable choices/resources. Keep current init image and painted mask; explicitly exclude their pixels and window/session layout from this settings snapshot.
4. Add focused tests for full snapshot persistence, isolation, missing-model preflight and plugin/script recovery. Root performs sbt, GUI and model checks after integration.

Source: SenshiSentou/sd-webui-state-manager, commit 2af62ecd5287438bbc309f7e840d34aa5c8e3cb6 (MIT, checked via gh API 2026-10-27). README documents checkpoint/VAE/clip skip, generation, script/accordion controls, browser/inspector and manual saving. This is an independent native implementation of named settings, not browser-history/autosave/thumbnail or A1111 storage-format compatibility. No upstream implementation copied.
