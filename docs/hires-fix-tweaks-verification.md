# Hires Fix Tweaks (#56) verification

Status: implementation, complete regressions and extracted-package GPU acceptance passed. This is the ninth accepted wave; remote commit and issue closure are tracked in `twenty-issue-release-progress.md`.

## Source and scope

Behavior reference: `w-e-w/sd-webui-hires-fix-tweaks`, AGPL-3.0, pinned at `28902f742a795afc31d86641ad85466afdd280e6`. Thirteen source files retained under `outputs/hires-fix-tweaks-source`; prompt module SHA256 `31903bb40f960ae7a543997e917a3f3afe99379aa1e88e7512b9725e98bb7729`. Licence is bundled at `licenses/w-e-w__sd-webui-hires-fix-tweaks.txt`; notices collector covers 155 projects. Parent reForge code was never imported or launched; local model files were read only.

The binding scope is `superpowers/plans/2026-10-30-hires-fix-tweaks.md`. Native Hires controls cover independent CFG/XYZ, 1–16 refinements from one first-pass result, primary/variation seeds and resize noise, positive/negative prompt composition, network removal, separate ordered styles, template restoration, destination and display preferences.

## Implemented behavior and evidence

| Requirement | Implementation and verification |
|---|---|
| Prompt composition | Default, Append, Prepend and Prompt S/R; independent separators and escaped Unicode markers. Pure helpers match 47 pinned-source fixtures. Raw positive network removal precedes style expansion; negative handling stays independent. Resolved flags prevent double expansion. Explicit blank positive/negative prompts remain empty. |
| Styles and restoration | Ordered base/Hires styles survive native PNG Restore. Apply materializes selected styles in selection order and clears selection without changing enable state. Bounded source codec and versioned native raw-template codec validate exact recomposition and current styles before restoring; malformed/stale templates retain resolved text. Save/restore preferences are independent. |
| CFG and display | Native CFG slider preserves 0=inherited, axis validates 0–30. Both `Hires CFG Scale` and upstream `Hires CFG scale` restore, with native spelling preferred. Five checkpoint label modes preserve IDs and disambiguate duplicate basenames. Hide/show and radio/dropdown switching retain values. |
| Seed planning | Immutable plan resolves randomness once per job and preserves base-batch → repeat → item order. Variation holds the primary seed and advances subseeds. Per-pipeline noise injection reaches actual diffusers prepare_latents and restores methods on every exit. No global RNG/helper patch. Hires output facts and reuse history are separate from base seeds. |
| Exact large seeds | Decimal strings above 2^53−1 avoid IEEE754 rounding through hooks, Python response, PNG and UI history. Raw integer source metadata is preserved before floating parsing. Signed64 overflow from batch/repeat increments is rejected before sampling. Actual SDXL text/integer seed9007199254740993 generations match exactly. |
| Repeat execution | First pass and upscale occur once per base batch; latent path remains latent. Both models pass repeats100/101/102 versus independent single jobs exactly; disabled and recovery outputs are exact. Batch-size2 × n_iter2 × repeats2 yields eight correctly ordered outputs. Total output limit1600 is validated before model work. |
| LoRA separation | Optional hires_loras distinguishes legacy reuse from explicit empty. Manual refs survive network-tag removal and PNG replay, including separate TE/UNet/start/stop. Hires-only custom LBW presets are resolved by the host. Composable manual_loras independently preserves manual versus scoped/controller contributions, including same-name overlap and named/positional TE/UNet syntax. |
| Cancellation and outputs | Completed images are saved immediately; interrupt during a later Hires/ADetailer item retains earlier completed outputs and emits cancellation instead of success. Persisted Hires destination participates in gallery scanning; nondefault global output path takes precedence. Hooks, adapters and ordinary following generation recover. |

## Automated checks

- Whole Python after final production edits: 1340 passed, one skipped, exit0 (`outputs/hires-python-final-full.log`).
- Whole Scala before last review fixes: 1346 passed, one skipped, exit0 (`outputs/hires-scala-final-full.log`,441s). Windows4097-file scanner was confirmed live by stack dump and not restarted.
- Review fixes: core52 (infotext/seed codecs), Protocol28, plugin2, Generator27 including large-seed fake-engine→PNG roundtrip, UI history3 plus form/state checks passed. Assembly succeeded (`outputs/hires-assembly.log`). Final whole Scala rerun passed1351 tests with one skip, exit0,494s (`outputs/hires-scala-post-review-full.log`): core546, plugins160, engine228+1skip, UI371, MCP46.
- Independent review identified and confirmed fixes for Hires-only LBW preset resolution, structured manual PNG identity, Composable manual/tag provenance and named weights, per-item AD saving, upstream CFG spelling and large seed precision. No remaining concrete finding in the reviewed scope.

## Actual native and GPU acceptance

GPU: RTX3090; SD1.5 `animekawa_v10.safetensors`; SDXL `waiIllustriousSDXL_v170.safetensors`.
Manual adapters: Mitsumi Misato SHA256 `25c6c0c67489e141b4b42e2d478f8ba7300280e9e93f80736a5646dd9f383c92`; 20_il-shirobana SHA256 `c474f37ae784625772d12c866e14322765f568e5fe0fc2ad9826e4fff9991edc`.

- `outputs/hires-gpu-batches-sd15` and `hires-gpu-batches-sdxl`: latent repeats, pixel Lanczos variation.35/resize512, inherited/random seeds, multi-batch ordering, exact independent repeats and recovery. Random primary/subseed resolved once and replayed by effective values.
- `outputs/hires-native-manual-sd15` and `hires-native-manual-sdxl`: actual MainWindow Generate/Interrupt/Restore with base/Hires styles, persisted output/gallery, manual TE.25/UNet.45/start.1/stop.9 plus same-name tag.15. Removing the manual ref preserves the tag and changes pixels; restoring PNG restores exact pixels. Partial cancellation retains one image. Screenshots inspected.
- `outputs/hires-controls-complete/report.json`: actual style selection/reselection order, Apply, reuse, radio/dropdown, visibility/managed state, CFG slider binding/hide retention, five checkpoint modes with real popup selection retaining IDs, and Settings disk roundtrip.
- `outputs/hires-composable-final-sd15` and `hires-composable-final-sdxl`: manual+tag versus equivalent tag-only two-repeat output matches exactly, with_step false/true. Positive Hires trace retains manual UNet.3 after tag removal. Named weights/negative gates/controller scaling/zero TE are covered by actual-module CPU tests.
- `outputs/hires-ad-final-sd15` and `hires-ad-visual-sdxl`: real YOLO face detector plus inpaint on all four batch/repeat outputs. Interrupt in second item's inpaint preserves exactly the completed first image, identical to the full run. Ordinary recovery is exact. SDXL uses768 base/960 output,20 steps/CFG4 for an interpretable face; images inspected.
- `outputs/hires-integrations-sd15`: Hires-only custom LBW equals equivalent direct weights exactly, differs from no-LoRA; SD1.5 ControlNet canny repeats equal independent jobs exactly and differ from no-ControlNet. Recovery exact. Images inspected.
- `outputs/hires-integrations-final-sdxl`: same custom-LBW/recovery checks at768 base/960 output, plus large effective seed text/integer equality. Images inspected. SDXL ControlNet is not claimed by this run.

Visual limits: low-step512px SDXL at CFG6 can produce severe saturation/stripes even in the ordinary baseline or no-adapter Composable control. Single-prompt adapter controls are normal; reducing two-branch Composable CFG to3 reduces saturation but retains stylization (`hires-composable-balanced-final-sdxl`). AND sums conditional differences by design; these controls do not prove a new Hires or LoRA corruption defect. Numerical repeat equivalence is not claimed as a general image-quality guarantee. Earlier512px AD runs prove control flow but are superseded by the visually inspected768px SDXL AD run.

## Adaptations and compatibility

- Native controls live in Hires, with opt-in plugin registration, rather than a duplicate AlwaysOn form.
- Upstream subseed randomness accidentally checks the primary seed; native implementation checks subseed=-1.
- Native templates add bounded, validated raw inputs alongside source-compatible prompt-mode facts. Missing styles or mismatched resolved text never silently replace the displayed prompt.
- Manual LoRA metadata is independent of template-saving preferences. Absent metadata clears stale manual state.
- Optional hires_loras/manual_loras preserve old raw clients; explicit empty lists are distinct from omission.
- Large integer seeds use decimal-text transport where JSON doubles cannot represent them exactly. Ordinary seeds retain numeric representation.
- A hook-only destination is honored for that job; restart scanning uses the persisted native destination preference.

## Package evidence

Build: `outputs/hires-package/build-42ddc418d01f4409991c05e451065b2a/SDNUI-v0.03-windows-x64.zip`.
SHA256: `059c394b3dc687b3ad72b66277a4de0a575f6b36aad3df227888f606a4bbadbe`.
Extracted: `outputs/hires-extracted/package check 1488838b965f4a31851801f26636141c/SDNUI`.
Bundled-runtime SDNUI.exe dry-run startup/autogeneration passed; screenshot inspected (`outputs/hires-package-launch.log`).

`scripts/dev/check_hires_packaged.ps1` verifies every bundled Python module against the worktree, then loads only test harness classes plus the extracted product jar and Python. External JDK21 is used for the harness because jpackage omits java.exe; this is distinct from the separately verified bundled-runtime executable startup.

Both extracted-package native runs passed and exited0 (`outputs/hires-packaged-sd15.log`, `outputs/hires-packaged-sdxl.log`): Generate, two refinements, base/Hires styles, manual plus same-name tag, PNG Restore exact RGB, removal/re-Restore of manual LoRA, persisted destination/gallery, partial Interrupt and ordinary recovery. Reports are `hires-native-report.json` in the respective homes; restored screenshots inspected. Jar SHA256 remained `cb206ff0657e279db9a738f7e68717a7356664a13c5a753388ed4b133a597ba5` before/after each run. This is an acceptance artifact, not a published release.
