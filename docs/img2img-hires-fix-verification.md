# img2img Hires Fix (#61) verification

Implementation worktree: `C:/Users/Mizushima/.codex/worktrees/4496/stablediffusion-nui`, based on `b8a623fdad1d78317a10b37149192a59ccff3499`. Accepted into the extension ledger after independent review, full regressions and packaged acceptance; publication is a separate parent integration step.

## Source and scope

Full pinned README, script and MIT license from `Amadeus-AI/img2img-hires-fix` at `b8930527a8cc606dd79dd54ed23bc967314d1859` are retained in `outputs/img2img-hires-source`. SHA256:

- `img2img_hires_fix.py`: `23350ab43d5008c06d5dc156c9e612bd40ae91d651f9ebe04af1b048a84963ad`
- `README.md`: `097281fd8c9a15b1f680e53a4e830fd1e44d82ddee30a5d5d7f5741ecb273bcc`
- `LICENSE`: `ee4f8a871b2e9849c6aa3c64d8a160ae621c2d1982ba41007f124f3a47f17ec3`

The README/issue description suggests retaining a first-pass latent. The pinned script actually receives a PIL image in `postprocess_image`, upscales its pixels, VAE-encodes it and samples again. This implementation follows that actual behavior. Existing native latent Hires remains an independent earlier pass and retains its latent semantics. No upstream script code is imported or copied.

Full source controls are provided for txt2img/img2img/inpaint: scale or explicit pair of dimensions, pixel upscaler, steps/sampler/scheduler/CFG inheritance, denoise, independent prompt/negative overrides. PNG uses the source `img2img Hires Fix` JSON field; restored fields drive native forms and API generation. Pass-only adapters preserve structured manual refs separately from prompt tags, including LBW lookup.

Native adaptations: deterministic per-output seed rather than ambient random noise; immutable per-job configuration; dimensions aligned to 8/minimum64; denoise0 only resizes. Source default `R-ESRGAN 4x+` requires a matching installed model. Native refiner checkpoints remain unsupported by the existing engine, and this feature does not add a refiner architecture.

## Automated evidence

- Python focused helper/orchestration/dryrun tests: 30 passing; expanded regression set99 passing (`outputs/img2img-hires-python.log`). Includes all modes, batch seed order, completed-item cancellation, native latent tensors before pixel refinement, independent composable clock, adapter cleanup on failure, composed-size rejection before model loading and disabled exactness. Expensive model calls are substituted in orchestration tests; these are not GPU evidence.
- JDK21 focused Scala: metadata3, plugin1, Generator5 and UI3 passed (`outputs/img2img-hires-scala.log`). Source PNG restore/API replay, manual/tag pass separation/LBW, composed target JobLimits, absent/malformed metadata disabling prior form state, and all-mode form/settings JSON roundtrip included.
- Actual JavaFX controls selected sampler/scheduler/upscaler and edited prompts, persisted all form fields to disk, and restored them from PNG metadata. Screenshot inspected: `outputs/img2img-hires-native-controls/controls.png`; disk state adjacent. Fresh `ui/assembly` passed.
- Parent-coordinated full regressions exited0: Python1507 passed/1 skipped (`outputs/img2img-hires-python-full.log`); Scala1437 passed/1 skipped (`outputs/img2img-hires-scala-final.log`). The earlier full Scala run was aborted after a stale ProtocolSuite expected-key set and a missing isolated-worktree Python environment. The protocol expectation now checks omitted/null, empty and populated refinement adapter lists, and the worktree Python environment was linked to the existing test environment. ProtocolSuite/PixelScrambleSuite focused recheck passed34 tests before the successful full run. These fixes affect tests/environment only; packaged product evidence remains applicable.

## Real GPU evidence

Sequential local RTX3090 runs using `scripts/dev/img2img_hires_check.py`:

- SD1.5 `animekawa_v10.safetensors`,512 base; LoRA `20_yui.safetensors`.
- SDXL `waiIllustriousSDXL_v170.safetensors`,768 base; LoRA `alraune_v1_illustrious_b-000007.safetensors`.

Both exited0. Artifacts/logs: `outputs/img2img-hires-sd15/`, `outputs/img2img-hires-sdxl/`, and sibling `.log` files. Each directory retains `report.json`, exact JSONL request/response journal and stderr.

Both families generated txt2img, img2img, inpaint, explicit controls, native latent Hires followed by refinement, four-item batching and pass-only LoRA. Disabled output, deterministic replay, completed first item after cancellation of the second, and ordinary recovery after adapters/cancellation all match exact RGB pixels. Native latent coexistence produces800px SD1.5/1200px SDXL outputs. Cancellation tests confirm actual `img2img_hires` progress and one retained output.

Images inspected: SD1.5 baseline/refined/inpaint; SDXL refined/explicit-controls/native-latent-first. Coherent portraits without stripes or corruption. SD1.5 repeated VAE refinement is somewhat paler than its baseline. The explicit blue-hair prompt does not fully override red source hair at denoise.5; image-conditioned sampling is not a guarantee of semantic prompt replacement. Other inspected settings and source behavior are recorded in the protocol.

## Actual MainWindow

`RealImg2ImgHiresGenerationCheck` passed on both families against the fresh assembly: native Generate, structured manual plus same-name refinement tag, PNG Restore with exact RGB replay, second-item Interrupt retaining one exact completed image, feature-off recovery with exact RGB, native img2img, and native painted-mask Inpaint. Homes/reports/logs: `outputs/img2img-hires-mainwindow-final-sd15` and `outputs/img2img-hires-mainwindow-final-sdxl` (sibling `.log`, inner `img2img-hires-native-report.json`). Inpaint images inspected. The initial SD1.5 harness run was stopped because it clicked Generate before the asynchronous inpaint canvas was ready; final runs wait for a loaded canvas and actually paint it. Initial restored screenshots caught the asynchronous preview loading state, so the final package harness additionally waits for preview completion.

Assembly SHA256: `5a33150551c8eb373a4be97f348e83875c501ac7eb92c2cfba3c2626376fea18`.

## Fresh Windows package

ZIP: `outputs/img2img-hires-package/build-44eb1928f7664bf0a695f746874f20ab/SDNUI-v0.03-windows-x64.zip`, SHA256 `8995f99b0dc505b3fcaaec590bac1a009feb4e7ca4871fe7f03dc036fc1e50bc`.

Fresh extraction: `outputs/img2img-hires-package-extracted/package check c1471d3b08ab4c7d9022a6b739910738/SDNUI`. Bundled-runtime `SDNUI.exe` dry-run autogen exited0 with one PNG; screenshot inspected. `scripts/dev/check_img2img_hires_packaged.ps1` checked every packaged Python module against this worktree and the jar against the fresh assembly, then ran the same actual MainWindow acceptance on both SD1.5 and SDXL. Both exited0 with exact manual/tag PNG replay, exact retained partial output, exact off recovery, and all three native modes. Jar SHA256 was unchanged before/after. Only test harness classes came from the worktree; all product Scala/Python came from extraction.

Logs: `outputs/img2img-hires-package-build.log`, `img2img-hires-package-check.log`, `img2img-hires-packaged-sd15.log`, `img2img-hires-packaged-sdxl.log`. Each packaged-model home contains `img2img-hires-native-report.json` and `restored.png`; both loaded-preview screenshots inspected. This is an acceptance artifact, not a published release.

## Review correction: negative LoRA tags

An initial review interpreted the source's `extra_networks.deactivate(p, negative_data)` call as named-adapter removal. That interpretation was retracted after checking actual host behavior. In [A1111 v1.10.1 dispatch](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/82a973c04367123ae98bd9abdf80d9eda9b910e2/modules/extra_networks.py), deactivation dispatches by network type, with no adapter-name argument. [The LoRA handler](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/82a973c04367123ae98bd9abdf80d9eda9b910e2/extensions-builtin/Lora/extra_networks_lora.py) only reports/clears errors. The extension subsequently activates its positive prompt again before conditioning. No negative-name veto was added to the native product.

`scripts/dev/img2img_hires_source_trace.py` runs the exact retained source methods and actual host dispatch/LoRA code with expensive model calls substituted; `outputs/img2img-hires-source-trace.json` records seven passing traces: same-name positive/negative, negative-only name, inherited positive, default host LoRA (analogue of manual), malformed negative weight, unclosed tag, and lyco alias. All retain positive/default adapters at conditioning/sample. Source byte hashes are checked. The extension does not pin a host; this fixture specifically establishes behavior for host commit `82a973c04367123ae98bd9abdf80d9eda9b910e2`, not every possible fork. Host reference files are retained alongside the extension source.

Not yet claimed: actual ControlNet/ADetailer interaction for this new pass, or support for architectures already rejected by the engine. Existing source controls have not been silently removed to avoid these boundaries.
