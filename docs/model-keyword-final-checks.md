# Model Keyword: pending final acceptance commands

These commands are a runbook, not results. Accepted #47, #16 and #87 through
main `5934d818` are integrated with verified backups; later main advances need the same treatment.
Use an exclusive GPU/sbt slot; run commands sequentially and stop
on failure. Use new output directories for every run. Inspect representative images,
retain logs and record hashes. The older package and GPU results in
`model-keyword-verification.md` predate the latest changes.

From the worktree root in PowerShell:

```powershell
$env:JAVA_HOME = 'C:\Program Files\Java\jdk-21'
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
$env:PYTHONPATH = (Resolve-Path python).Path
$py = 'C:\Users\Mizushima\dev\stablediffusion-nui\python\.venv-smoke\Scripts\python.exe'
$sbt = "$env:LOCALAPPDATA\Coursier\data\bin\sbt.bat"
$sd15 = 'D:\stable-diffusion-webui-reForge\models\Stable-diffusion\animekawa_v10.safetensors'
$sdxl = 'D:\stable-diffusion-webui-reForge\models\Stable-diffusion\waiIllustriousSDXL_v170.safetensors'
$l15 = 'D:\stable-diffusion-webui-reForge\models\Lora\20_yui.safetensors'
$lxl = 'D:\stable-diffusion-webui-reForge\models\Lora\alraune_v1_illustrious_b-000007.safetensors'
$e15 = (Resolve-Path outputs/keyword-embedding-fixtures/synthetic_keyword_sd15.safetensors).Path
$exl = (Resolve-Path outputs/keyword-embedding-fixtures/synthetic_keyword_sdxl.safetensors).Path
& $py -m pytest
if ($LASTEXITCODE -ne 0) { throw 'Python regression failed' }
& $sbt -batch test 'ui/Test/compile' 'ui/assembly'
if ($LASTEXITCODE -ne 0) { throw 'Scala regression or assembly failed' }
```

The synthetic embeddings are fixture-format checks, not trained embeddings.
For both ordinary model families, check replay and Hires, plus img2img, inpaint,
protected session, cancellation and recovery:

```powershell
./scripts/dev/check_model_keyword_native.ps1 -Python $py -Model $sd15 -Size 512 -Lora $l15 -LoraTagOptions '2:lbw=ALL:start=0.2:stop=0.8' -Embedding $e15 -OutputHome outputs/keyword-final-sd15 -ModesHome outputs/keyword-final-sd15-modes
./scripts/dev/check_model_keyword_native.ps1 -Python $py -Model $sdxl -Size 768 -Lora $lxl -LoraTagOptions '2:lbw=ALL:start=0.2:stop=0.8' -Embedding $exl -OutputHome outputs/keyword-final-sdxl -ModesHome outputs/keyword-final-sdxl-modes
./scripts/dev/check_model_keyword_native.ps1 -Python $py -Model $sdxl -Size 768 -Lora $lxl -LoraTagOptions '2:lbw=ALL:start=0.2:stop=0.8' -Embedding $exl -VramMode model_offload -CheckLoraReenable -OutputHome outputs/keyword-final-offload
```

Exercise the newly integrated Composable routes on both checkpoints; these
worker checks do not substitute for native/package acceptance:

```powershell
$env:SDFX_KEYWORD_CHECKPOINT = $sd15; $env:SDFX_KEYWORD_LORA = $l15; $env:SDFX_KEYWORD_SIZE = '512'
& $py scripts/dev/gpu_check.py $py outputs/keyword-final-negpip-sd15 model_keyword_composable_negpip
if ($LASTEXITCODE -ne 0) { throw 'SD15 NegPiP failed' }
& $py scripts/dev/gpu_check.py $py outputs/keyword-final-alt-sd15 model_keyword_composable_img2imgalt
if ($LASTEXITCODE -ne 0) { throw 'SD15 inversion failed' }
$env:SDFX_KEYWORD_CHECKPOINT = $sdxl; $env:SDFX_KEYWORD_LORA = $lxl; $env:SDFX_KEYWORD_SIZE = '768'
& $py scripts/dev/gpu_check.py $py outputs/keyword-final-negpip-sdxl model_keyword_composable_negpip
if ($LASTEXITCODE -ne 0) { throw 'SDXL NegPiP failed' }
& $py scripts/dev/gpu_check.py $py outputs/keyword-final-alt-sdxl model_keyword_composable_img2imgalt
if ($LASTEXITCODE -ne 0) { throw 'SDXL inversion failed' }
```

Rebuild the package using `scripts/package_windows.ps1 -OutputRoot
outputs/keyword-final-package`, extract its ZIP into a new directory, and set
`$package` to the extracted `SDNUI` directory. Ensure all required new sources
are tracked/staged first: the package script takes Python files from `git ls-files`.
Then run the extracted package against both checkpoints:

```powershell
./scripts/dev/check_model_keyword_packaged.ps1 -PackageRoot $package -Python $py -Model $sd15 -Size 512 -Lora $l15 -LoraTagOptions '2:lbw=ALL:start=0.2:stop=0.8' -Embedding $e15 -OutputHome outputs/keyword-final-package-sd15 -ModesHome outputs/keyword-final-package-sd15-modes
./scripts/dev/check_model_keyword_packaged.ps1 -PackageRoot $package -Python $py -Model $sdxl -Size 768 -Lora $lxl -LoraTagOptions '2:lbw=ALL:start=0.2:stop=0.8' -Embedding $exl -OutputHome outputs/keyword-final-package-sdxl -ModesHome outputs/keyword-final-package-sdxl-modes
```

The example output names above were used in earlier pre-#47 tests; change them to
fresh unique names before running. The wrapper checks packaged Python files and jar against this worktree. Inspect
the resulting images and review logs and hashes. Repeat genuine PhotoMaker v1,
synthetic DoRA, the #47 PhotoMaker on/off recovery, and their relevant offload
controls on the final integrated source; the earlier checks do not prove these
later combinations. The native generation harness now also checks the fresh
mounted enabled default and toggles off/on before generation; recompile its test
classes before invoking it. The same harness calls the real MCP lookup for the
checkpoint and selected LoRA, writes `mcp-model-keyword-lookup.json`, and generates
two MCP jobs to compare the default with an explicit false opt-out. An additional
assertion verifies that the accepted Model Presets tab and MCP tool coexist with
the keyword controls. These native-source checks passed on SD1.5 and SDXL after
#87 integration; fresh packaged behavior still needs rechecking. Fake-worker
`McpToolsSuite` lookup and default/opt-out tests also passed after #87 integration.

## Guarded PhotoMaker, Composable and Embedding Merge

Three production gates remain: `clora.runtime.Job` rejects PhotoMaker with loaded
Composable adapters; `neutral_runtime.validate` rejects PhotoMaker with a Neutral
plan; `reject_unverified_merge_composable` rejects the triple combination. Do not
remove them merely because a fake pipeline passes. Before an experimental guard
change, retain a fresh verified backup and use an isolated test worktree/copy.
Permit PhotoMaker at the first two gates only for the prepared Composable plan,
leaving unrelated Neutral conflicts and version/trigger validation in force. Lift
the triple gate for the candidate only after retaining raw expression metadata
through collation and restoring temporary encoder rows; do not alter behavior for
ordinary requests. Run the focused CPU suites and the actual `Engine._run` fake
pipeline tests, including cancellation and first-use/MCP tests, before scheduling GPU.

Then, in the coordinated GPU slot, configure the genuine official v1 source and
isolated dependencies (see `model-keyword-verification.md`) and run both CFGs in
fresh output homes. These commands use the previously verified local weights and
do not download them:

```powershell
$env:PYTHONPATH = "$(Resolve-Path python);$env:TEMP\sdfx-photomaker-v1-source;$env:TEMP\sdfx-photomaker-genuine-target-v2"
$env:SDFX_PHOTOMAKER_MODEL = (Resolve-Path outputs/keyword-photomaker-real-a/models/TencentARC__PhotoMaker/photomaker-v1.bin).Path
$env:SDFX_KEYWORD_LORA = $lxl
$env:SDFX_PHOTOMAKER_MERGE_EQ = '1'
$env:SDFX_PHOTOMAKER_CFG = '0.5'
& $py scripts/dev/gpu_check.py $py outputs/keyword-photo-triple-unguided model_keyword_photomaker_composable
if ($LASTEXITCODE -ne 0) { throw 'unguided PhotoMaker triple check failed' }
$env:SDFX_PHOTOMAKER_CFG = '6'
& $py scripts/dev/gpu_check.py $py outputs/keyword-photo-triple-guided model_keyword_photomaker_composable
if ($LASTEXITCODE -ne 0) { throw 'guided PhotoMaker triple check failed' }
```

Each worker run generates a reference face, two keyword-selected images with two
Composable branches and a scheduled edit, then an equivalent Embedding Merge
batch with the same seeds. It requires exact RGB parity for each pair and exact
ordinary generation recovery after adapters are off. Save stdout and worker
stderr, inspect representative images, and record model/adapter/source hashes.
This prepared case has not run and does not exercise interruption, protected
outputs or package execution. Add separately observed interruption/recovery,
offload and extracted-package cases before treating the guard change as accepted.

The pinned upstream LoRA catalog is bundled for offline first use under the
upstream MIT notice. Its Civitai-derived provenance and limits of individual
third-party rights verification are recorded in the scope audit.


## Representative signed and tiled PhotoMaker candidates

The 4-step diagnostic images contain severe artifacts and are not visual acceptance. After an explicit shared GPU slot is granted, use the checked local genuine PhotoMaker source and dependency target described above. The development harness below defaults to 768×768 and 20 steps; for MultiDiffusion it uses 512px tiles with 64px overlap, and asserts that more than one tile actually ran. Use distinct, nonexistent output names. It writes a verified source backup beside each output and restores exact source bytes in `finally`; forced termination cannot execute `finally`, so compare the printed hashes and restore from that backup before further work. It never changes the production guards permanently.

```powershell
$py = 'C:/Users/Mizushima/dev/stablediffusion-nui/python/.venv-smoke/Scripts/python.exe'
$env:SDFX_PHOTOMAKER_MODEL = (Resolve-Path outputs/keyword-photomaker-real-a/models/TencentARC__PhotoMaker/photomaker-v1.bin).Path
$env:SDFX_KEYWORD_LORA = 'D:/stable-diffusion-webui-reForge/models/Lora/alraune_v1_illustrious_b-000007.safetensors'
$env:CUDA_VISIBLE_DEVICES = '0'
& $py scripts/dev/run_guarded_photo_diagnostic.py signed $py outputs/keyword-photo-signed-visual-20 --size 768 --steps 20 --vram-mode model_offload
$code = $LASTEXITCODE; if ($code -ne 0) { throw "signed diagnostic exited $code" }
& $py scripts/dev/run_guarded_photo_diagnostic.py multidiffusion $py outputs/keyword-photo-multi-visual-20 --size 768 --steps 20 --tile 512 --overlap 64 --vram-mode model_offload
$code = $LASTEXITCODE; if ($code -ne 0) { throw "tiled diagnostic exited $code" }
```

The signed run saves `signed-effect.json`, verifies the signed/unsigned pixels differ and verifies plain versus equal-expression merge, interruption then merge recovery and PhotoMaker-off recovery. The tiled run records real `multi_tiles` events and checks tile count, plain/merge equality, interruption and recovery. Inspect representative signed, unsigned and tiled images at full resolution and record visual observations, exact model/LoRA/PhotoMaker hashes and logs; an image difference alone does not prove NegPiP's source semantics. Compare against an independent native NegPiP attention reference, including negative-prompt terms, multiple per-image terms, schedule variants and CFG boundaries, and validate signed PhotoMaker without Composable before considering the signed guard. Before considering the MultiDiffusion offload guard, verify tiled numeric equivalence/lifecycle at actual nontrivial overlap, interruption, sequential offload and both ordinary and compound usage. The Composable/PhotoMaker/Embedding Merge gates additionally require the full generation modes, SD15 where applicable, package/source integration, model/LoRA corpus provenance and broad regressions. All current guards remain active; this list is required evidence, not an automatic release rule.
