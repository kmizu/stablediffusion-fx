# Inpaint Difference Implementation Plan

> For agentic workers: use superpowers:executing-plans for this wave. The user requested autonomous completion of twenty full issues and a release; do not narrow the original scope.

**Goal:** Implement #67, upstream index243, as a native Inpaint difference mode using two images and a computed mask.

**Architecture:** Reuse the native Inpaint panel and generation queue. A registered `inpaint_difference` plugin carries immutable operation settings and the base-image reference; the ordinary init image is the altered image. A shared Python CPU mask implementation serves both an asynchronous native preview endpoint and the actual inpaint generation path. The engine recomputes from request inputs for generation rather than trusting a potentially stale UI preview. Preserve the existing paint-mask mode and its state. Preview requests are debounced/coalesced and revision-guarded; stale results cannot overwrite newer settings. Preview artifacts and generation temporary images are owned and cleaned without touching source files.

**Tech stack:** Scala 3/JavaFX, JSONL protocol, Python/Pillow/NumPy/OpenCV; existing dependencies only.

**Source/spec:** `outputs/inpaint-difference-source` at MIT pin `656eed0a214186f4fb12fd6100fef582a01558bf`; issue #67, `docs/HANDOFF.md`, `docs/EXTENSIONS.md`, and `docs/PLUGINS.md`. The published `sdwi2iextender` 0.2.3 wheel was downloaded without installation and checked against PyPI SHA256; its component injector confirms the extension supplies the altered image as the actual generation image and the difference as mask.

## Invariants

- Source computes RGB absolute channel differences, integer mean, and strict `mean/255 > 1-threshold`; threshold default1. Resize base to altered dimensions. Optional one-pixel contour extraction precedes 3x3 dilation then erosion; each has 0–100 iterations. Match border semantics and ordering against pinned source fixtures. Standard Inpaint mask blur applies to generation; preview uses the same blur with configurable mask color and image-under-mask setting.
- Controls include original/altered images, swap, live recomputation, threshold, contour, dilation, erosion, blur, preview color and underlying-image visibility. Invalid or missing inputs must not leave a stale mask presented as current. Swap changes the generation image as upstream does.
- Independently normalize supported image formats and EXIF orientation; document any necessary differences from upstream RGB-only behavior. Bound dimensions, iterations and file sizes; errors must not mutate inputs or leave partial files. Generated image and mask dimensions must agree.
- Plugin id equals `inpaint_difference`. Any new engine hook appears in KNOWN_HOOKS and real/dryrun features. Preserve existing API behavior when disabled; no parent reForge imports or execution. Use LF. Retain MIT notice and dependency provenance.
- Queue requests must capture settings and both input references when submitted; no later form changes may change an already queued job. Temporary files must remain available through the job and be cleaned on completion/cancellation/error. Do not cancel an unrelated generation when replacing preview work.
- One commit for this wave; push private main and close #67 only after all original scope and gates pass. Goal remains 9/20 until then.

## Review focus

1. Stale preview / changed settings / swap while work is running: only the newest completed request is displayed and generation uses its captured inputs.
2. Alpha, grayscale, EXIF, unequal sizes, malformed/huge files: deterministic RGB handling, safe validation and visible errors.
3. Empty/full masks, threshold endpoints, image borders and holes: compare actual mask bytes with pinned source fixtures and preserve empty-mask semantics.
4. Queueing/cancellation and failures: no source overwrites, stale temp references, leaked files, or changes to ordinary painting and subsequent generation.
5. Actual native use, not just helper tests: source-derived fixtures, real UI input/swap/live controls, SD1.5 and SDXL image results, package executable and extracted-product checks.

## Tasks

- [x] Build pinned-source behavioral fixtures in an isolated reference harness without importing reForge. Verify border/contour/morphology/blur and size behavior; capture hashes and limitations. Write meaningful red tests first.
- [x] Implement CPU `python/sdfx/inpaint_difference.py`, typed bounded input validation and output artifacts; test pinned fixtures, formats, malformed inputs and no source writes. Add optional preview RPC to protocol, real engine and dry run with matching advertised features; exercise worker lifecycle and interruption.
- [x] Implement native plugin and generation integration, computing the mask at job execution from the captured settings and the original/altered references. Validate before model work and ensure owned temporaries are always removed. Test real and dry-run pipeline behavior, queued form changes, error/cancellation, ordinary paint and recovery.
- [x] Implement native Inpaint mode selector and responsive two-image UI using existing drop/paste patterns. Render color/underlay preview, swap and complete controls; debounce/coalesce worker operations, guard revisions and expose clear loading/errors. Preserve paint canvas state when switching. Test real JavaFX controls and settings persistence.
- [x] Verify actual application Generate/Interrupt and mask effects with real SD1.5 and SDXL; inspect images, compare generated mask bytes and independently prepared inpaint results. Run Python/Scala regressions and independent review. Assemble and test bundled-runtime executable plus extracted jar/Python native generation on both models.
- [ ] Add pinned MIT licence/notice and concise behavior/verification docs. Only after acceptance promote ledger index243, regenerate extensions docs, run ledger and licence gates, commit once, push and verify local/remote HEAD and clean tree, then verify #67 closed. Record 10/20; continue remaining ten issues before release.
