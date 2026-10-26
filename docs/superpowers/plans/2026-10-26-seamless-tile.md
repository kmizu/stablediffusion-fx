# Seamless tile inpainting implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement extension-index target `seamless_tile` from brick2face/seamless-tile-inpainting's README.

**Architecture:** A Scala img2img Script circularly offsets the input by half its dimensions, builds a central
seam mask, and uses the existing transient inpainting generation path. It optionally reverses the offset before
saving final images. No Python hook or dependency is needed. This preserves the existing desktop/plugin design.

**Tech stack:** Scala 3, Java2D, existing PluginContext/diffusers img2img.

**Spec:** https://github.com/brick2face/seamless-tile-inpainting README retrieved 2026-10-26 (AGPL-3.0).
Reimplement documented behavior only. Controls: horizontal/vertical/both directions; masked area width in pixels;
split sides again after inpainting. Both-direction results can retain minor seams, as upstream documents.
The user's existing request authorizes implementing the index; these routine implementation choices follow the
handoff rather than requiring a new product/platform decision.

## Constraints and review focus

- Plugin id is `seamless_tile`; register in StandardPlugins.
- Resize source to requested generation dimensions, then mask/shift that coordinate space.
- Odd dimensions must round-trip exactly; never assume applying half shift twice is its inverse.
- Clamp mask width to the selected axis length; only selected axes are modified.
- Preserve batch size, output seeds, metadata, and cancellation behavior; do not publish shifted intermediate images.
- Clean temporary input/mask and transient outputs on success or error.
- Do not claim exact edge equality: neural inpainting is approximate, especially with both axes selected.
- Keep ledger todo until real SD1.5/SDXL app checks and visual inspection pass.

## Task 1: Plugin and geometry

Files: create `plugins/.../SeamlessTilePlugin.scala` and `plugins/src/test/.../SeamlessTileSuite.scala`;
modify `plugins/.../StandardPlugins.scala`.

- [x] Write tests for directional circular offset, odd-size inverse, central mask bounds, script request,
  restored batch outputs, interrupted job cleanup and registration. Run and observe failure.
- [x] Implement geometry and plugin. Run `plugins/testOnly sdfx.plugins.SeamlessTileSuite` until green.
- [x] Build jar; run real MCP plugin checks on SD1.5 and SDXL using a texture prompt and 2x2 repeated previews.
- [x] Inspect results; add verification and plugin docs, ledger target and upstream notice only after verification.
- [x] Run relevant full checks, independent review, commit/push a verified wave.

## Execution ledger

- Initial state: 6029602, clean worktree; index 83 done / 219 todo / 51 na. Image-tool verification wave complete.
- Geometry/registration/request tests first failed because implementation was absent. Batch test then detected seed
  5 repeated instead of 5/6; corrected per-output seed. Initial four tests passed and jar built.
- Real app runs: stone SD1.5 4/4; stone SDXL 4/4; grass SD1.5 4/4. Reviewed repeated previews and UI screenshot.
- Ruling: retain documented upstream model-dependent seam quality, not a promise of equal boundary pixels. SD1.5
  stone texture degraded in some cases; record it alongside the better SDXL and grass cases. A dedicated inpainting
  checkpoint was not available; no claim that one was tested. Bad model/texture combinations may still show seams.
- Review found lost model/variation metadata. Added regression, observed failure, retained original engine infotext;
  six tests now pass including error/cancellation cleanup. Final rebuilt app run 4/4 with live PNG model hashes and
  script metadata checked. Final full Scala suite and complete licence refresh pending at this ledger entry.
- Full suite after metadata fix: Scala 480 passed. Required licence refresh exposed false missing-licence labels
  after transient API errors; added retry/error/archive handling. Three failing tests became four passing tests.
  Independent collector review found no important defect; explicit timeout and existing-notice-byte preservation
  tests remain minor gaps. Full Python: GPU venv 261 passed; no-torch venv 225 passed / 15 skipped.
- Final live licence refresh succeeded for 109 projects, adding the upstream AGPL-3.0 text and restoring previously
  mislabelled MIT notices. Existing SuperMerger/close-confirmation licences were verified retained. Index is
  84 done / 218 todo (153 targets) / 51 na. No Python engine code changed in this wave. Review fix and full checks
  are complete; commit/push is the final action for this wave, not completion of the extension-index goal.
