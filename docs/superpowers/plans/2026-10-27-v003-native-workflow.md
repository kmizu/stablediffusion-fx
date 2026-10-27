# v0.03 native workflow implementation plan

> **For agentic workers:** Use subagent-driven development with isolated worktrees. Root integrates and verifies the final product.

**Goal:** Deliver the user-approved v0.03 scope: local tag completion (#24), named generation states (#23), and persistent WD batch PNG information (child of #140), with a verified JVM-bundled Windows release.

**Architecture:** Extend the existing JavaFX prompt editor and settings/preset infrastructure without replacing ordinary generation. Keep named full generation snapshots distinct from selective presets. Persist WD effective per-image settings and provenance before batch output publication, using metadata the existing PNG Info and Restore paths can read.

**Tech stack:** Scala 3, JavaFX, existing JSON/PNG codecs, Python worker, sbt, Windows jpackage.

## Independent implementation boundaries

- [ ] Tag completion: local CSV configuration, parsing/search/token replacement model, PromptEditor popup, keyboard and IME semantics. Tag agent owns PromptEditor and autocomplete/CSV settings. No downloads required.
- [ ] Named states: explicit snapshot coverage, atomic named store, list/save/restore/delete UI, missing resource warnings. State agent owns MainWindow/GenerationPanel integration. Reuse existing serializers and plugin control state rather than inventing conflicting presets.
- [ ] WD PNG persistence: per-image effective prompts, generation facts and pinned interrogator identity inside PNG before publication. WD agent owns batch worker/runner and metadata/Generator integration. Verify reopening PNG Info and restoring after a new app session.

## Integration and acceptance

- [ ] Pin original source behavior and licenses for #23/#24; retain existing WD notices. Do not copy unlicensed source.
- [ ] Resolve shared keyboard routing so Enter selects completion, Escape cancels, IME composition is preserved, and generation shortcuts remain deliberate.
- [ ] Verify ordinary prompt editing/generation, state round trips, missing-resource warnings, and WD PNG reopening with focused automated and native checks.
- [ ] Check SD1.5 and SDXL through actual native paths; representative generation/unload VRAM observations suffice. No exhaustive model matrix.
- [ ] Version v0.03, assemble and jpackage with required notices. Verify ZIP extraction, bundled-runtime EXE startup and real generation; publish tag/assets and verify downloaded SHA-256.
- [ ] Close only achieved issue scope; retain other #140 requirements as open. Update verification and handoff records with measured outcomes.

User authorization: the active goal explicitly requests implementation, verification and release of these three features. The trailing description of the prior planning-only turn is historical context, not a cancellation of the active implementation objective.
