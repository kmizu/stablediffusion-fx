# Hardware Info in Metadata Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. No subagents. Parent owns sbt/GPU/native acceptance; stop production work until parent plan review and observed RED.

**Goal:** Independently implement index309 `hardware_info` in the standalone native app, including compact hardware facts, source batch timing, optional remark, Extras metadata and bounded pasted-image hardware-difference notices.

**Architecture:** Reuse System Info's actual hardware facts and native metadata settings, but emit the extension's own three infotext fields. Optional request settings and per-image typed result records carry provenance without changing GenerationParams or sampling. A small pure Python formatter/timer and a pure Scala infotext/paste-notice policy isolate the source contract from UI and device detection.

**Tech Stack:** Scala3/JavaFX, existing JSON Lines protocol/Python runtime, pytest and munit. No parent WebUI/Gradio imports, new model dependency or global package install.

**Spec:** The complete source behavior table and native mapping below; authoritative runtime/readme/history at `light-and-ray/sd-webui-hardware-info-in-metadata` commit `784a0dd8521108815aff007d7a4807ac6f98d251`.

## Global constraints

- Work only `.worktrees/hardware-info`, branch `agent/hardware-info-20261026`, base `533d2fb789bcc0b00415b59f3f4a17749c42042f`.
- No production edits until this plan is reviewed and parent observes initial Scala RED; worker may run CPU pytest using absolute main venv and worktree PYTHONPATH with CUDA_VISIBLE_DEVICES=-1.
- Full source scope is binding: compact normalization, per-image batch averaging/reset, remark, Extras and bounded infotext-paste differences all remain.
- Primary source has no license file/declaration in inspected tree/history. Independently rewrite behavior; do not copy its code. Parent owns licensing notices/ledger/generated docs/HANDOFF/promotion.
- Preserve pixel results, input PNG bytes, sampling/numerical paths, final-image counts, temp ownership, interruption and existing System Info opt-in behavior.
- Metadata starts disabled, using the existing native opt-in policy. Explain that controls affect future image operations. Source installation was always-on; this is an explicit native privacy/control adaptation, not removal of a feature.
- No unsupported-device/platform claims inferred from method names. Best-effort hardware detection has explicit unknown fallback; no arbitrary subprocess shell, network or model load for metadata.

## Authoritative source and complete behavior mapping

Primary [README](https://github.com/light-and-ray/sd-webui-hardware-info-in-metadata/blob/784a0dd8521108815aff007d7a4807ac6f98d251/Readme.md) and
[runtime](https://github.com/light-and-ray/sd-webui-hardware-info-in-metadata/blob/784a0dd8521108815aff007d7a4807ac6f98d251/scripts/hardware_info_in_metadata.py)
were read completely, with install.py and full runtime history. Remote HEAD was reverified at the pin.
The independent source-audit report, retained full clone and controlled original-source probe results live in
the sibling worktree's ignored `outputs/source-audit-309-336/`; the table here travels with implementation.
Runtime SHA256 `10996ee00ca008ff3df4d6b197a553e97bbd347c81e71a70b67b9cc6130be4f8`.

| Source contract | Native behavior and verification |
|---|---|
|Hardware string: CUDA device name + total VRAM rounded integer GiB with GB suffix; CPU brand; total RAM same units. No CUDA means CPU/RAM only.|Reuse System Info providers; obtain actual branded CPU via bounded OS facts when platform.processor is generic (Windows registry/Linux cpuinfo/macOS bounded sysctl with shell=false, otherwise explicit unknown). Reuse psutil RAM when available. Native uses the engine's actual device instead of assuming CUDA0. Source includes no other GPU brand API; uncollectable facts do not fail generation.|
|Remove case-insensitive vendor/product/trademark/CPU/GPU/@/gen tokens, ordinal generations, decimal GHz; collapse whitespace and spaces before commas.|Pure formatter with exact examples for NVIDIA/Intel and AMD, integer-GiB rounding and CPU-only operation. Latest code removes frequency even though an older README example includes it.|
|At import, collect once; exception logs and caches unknown.|Lazily cache compact hardware per engine lifetime, including unknown after provider failure. No full Git/library report per image. Engine restart refreshes the facts.|
|before_process resets monotonic start/generated count. Each postprocess image increments count, divides elapsed by batch_size, formats one decimal seconds and optional whole minutes; resets start after every batch_size-th image.|Per-request ImageTiming started at native generation entry before parse/setup; invoked immediately before saving each postprocessed final image. First iteration includes request setup, LoRA/embedding/control prep, base/hires/ADetailer/final processing; later iterations use source-style reset boundaries. Earlier final-image PNG serialization is included naturally between callback points. Scala model preloading/queue wait and final permanent-copy I/O are outside this engine interval; state this timing boundary clearly. Do not substitute whole-job timingS for per-image values.|
|Always-visible script applies txt2img/img2img; no custom generation controls.|Global native System Info metadata pane controls enablement and remark, captured per operation. Per-image records work for txt2img, img2img/Inpaint, batch×iterations and script subrequests. Grids must not receive invented timing or alter final gallery counts.|
|User remark textbox in global Infotext settings; emit Remark only if nonempty, read by postprocess.|Native persistent multiline remark in the same metadata card; literal Unicode/comma/newline handling and omission when empty. Frozen request settings apply to future operations, rather than changing in-flight provenance when UI fields change.|
|Extras firstpass starts timer; late postprocess writes same three fields, using whole-operation elapsed (not diffusion batch division).|Pass the same optional metadata options through UpscaleRequest; time Extras operation from image input processing through final upscale processing; attach typed record to result. Generator writes that record to the actual returned Extras PNG with clear postprocessing identity/size/upscaler fields. Preserve existing output metadata and input source bytes. Covers native Extras input/output and plugin upscale context; no new diffusion request.|
|Paste callback compares first comma segment of imported Hardware Info against cached local first segment; mismatch of two known values shows informational message, max twice/session.|Pure PasteNotices shared per MainWindow, used by actual Restore and Send-to paths. Factual provenance notice only; never load an image's hardware, replace local GPU, change prompts or enable metadata. CPU-only first segment still compares, described neutrally as recorded device rather than falsely asserting GPU. Unknown/equal/missing do not consume budget.|
|py-cpuinfo installer exists upstream.|No installer copied. Reuse existing System Info and bounded OS facts. If a new optional dependency is found necessary, report it to parent before changing requirements; parent owns notices.|

History intent: `9c08f37` fixes image count/batch averaging, `c3e1d8a` removes CPU frequency, `f64d09c` removes
generation wording, `decfc5f` paste callback, `bc2b0e0` CPU-only support, `495d25f` Extras, `a019f74` initialization
at before_process, `784a0dd` remark. No source feature is dropped because it was absent from the short README.

## Existing coverage and boundaries

`python/sdfx/system_info.py` already collects GPU name/total memory, RAM totals and platform CPU; existing
SystemInfoService/Tab whitelist omits RAM and VRAM totals and emits a separate System Info JSON field only on
generation. It has no source compact formatter, elapsed field, remark, Extras attachment or pasted mismatch
notice. GenerationResult.timingS is request-level and currently not written as Time taken. PngMetadata writes
new parameters and explicit extra chunks; copying pixels is not automatic merging of arbitrary source chunks.

This target is a native metadata feature, not a sampling hook. Do not invent a hook that changes conditioning
or require hook registration for a nonexistent sampling transform. Keep target name `hardware_info`; parent
decides final ledger mapping after acceptance. Existing raw System Info selections remain independent.

## File responsibilities and planned interfaces

| File | Responsibility / public interface |
|---|---|
|Create python/sdfx/hardware_info.py|`compact(*,gpu_name,vram_bytes,cpu_name,ram_bytes)->str`; `cached_profile(engine,supplier=None)->str`; `collect(engine)->str`; `ImageTiming(batch_size,clock=time.perf_counter).next_image()->str`; `record(hardware,time_taken,remark)->dict`. No Torch import or sampling work at module import.|
|Modify python/sdfx/system_info.py|Factor/reuse hardware fact providers; do not run full snapshot/Git/library inventory for each image. Expose compact cached profile through existing system_info response for native paste/UI preview.|
|Create core/.../infotext/HardwareInfo.scala|Nested `Options(enabled:Boolean=false,remark:String="")` and `Record(hardwareInfo:String,timeTaken:String,remark:Option[String])` with SnakePickle codecs; `fields(Option[Record]):ListMap[String,String]`; `PasteNotices.pasted(local:String,imported:Option[String]):Option[Difference]`, Difference(currentDevice,imageDevice).|
|Modify Settings.scala|`hardwareInfoEnabled:Boolean=false`, `hardwareInfoRemark:String=""`; backward-compatible absent fields and persistence independent of systemInfoMetadataFields.|
|Modify Protocol.scala / python params.py|Append `hardwareInfo:HardwareInfo.Options` to GenerationRequest/UpscaleRequest and Python equivalent; append optional per-image GeneratedImage.hardwareInfo and optional ImagePathResult.hardwareInfo. Existing payloads decode with absent fields. Validate option types; no GenerationParams numerical setting changes.|
|Modify python engine.py / dryrun.py|Initialize optional per-request profile/timer; attach record per final image at actual postprocess point. Enabled dry-run still records truthful CPU-only facts; labels remain dry-run and its timing is not a hardware performance claim. Off path does not collect hardware.|
|Modify Generator.scala|Inject trusted global options from current Settings into requests; merge typed per-image actual record into ImageFacts. Strip old Hardware Info/Remark/Time taken from newly generated provenance rather than copying imported machine/timing claims. Preserve unrelated extras. Extras PNG attachment is atomic and does not overwrite input source.|
|Modify engine SystemInfoService.scala / UI SystemInfoTab.scala|Native opt-in Hardware metadata card with enable toggle, multiline remark, compact read-only preview, timing-boundary/help text. Disabled remark is retained locally but not emitted. Keep existing per-field System Info selection intact.|
|Modify UI MainWindow.scala|Shared per-window PasteNotices on Restore and Send-to. Obtain local profile only from already-running engine/cached response; offline/unknown comparison silently suppresses notice and never starts or loads a model merely to paste metadata.|
|New focused tests/manual helper|Files described in tasks below; parent runs sbt and actual native/model helper. No notices/ledger/HANDOFF edits in worker branch.|

## Review focus

1. Imported provenance must never be relabeled as this machine's new generation facts; disabled metadata must not
   silently retain stale Hardware Info/Remark/Time taken via params.extra.
2. Per-image records must stay aligned with seeds and images across batch/iterations, scripts, hires and aborted
   requests; source timer state must not leak between requests.
3. Detection failure, CPU-only execution and offline PNG Restore must not trigger model loading or break useful
   operations; unknown values cannot generate misleading difference notices.
4. Extras result is an upscaled output, not newly sampled input-prompt conditioning; identify processing while
   preserving source bytes, exact pixels and unrelated output metadata.
5. Native opt-in and remark persistence must be clear, accessible at1280×800, and independent of existing System
   Info controls; no false hardware-performance claim from dry-run checks.

## Test-first tasks

### Task1: pin pure metadata/timing contracts and protocol records

**Files:** Create `python/tests/test_hardware_info.py`, `python/tests/test_hardware_info_protocol.py`,
`core/src/test/scala/sdfx/core/HardwareInfoSuite.scala`; production files are listed above, not edited yet.

- [x] Write pure formatter/cache/batch-reset/minutes/Extras/remark/invalid-batch tests and typed settings/record/paste tests.
- [x] Write actual dry-run stdio opt-in/per-image seed association/disabled-again and byte-identical output tests.
- [x] Worker run baseline existing CPU System Info tests and observe new CPU RED; retain logs in ignored outputs.
- [x] Parent review this plan, then run `core/testOnly sdfx.core.HardwareInfoSuite` and record expected missing API RED.
- [x] Only after parent authorization and observed RED, implement pure helpers/codecs/settings and optional per-image metadata.
- [x] Run targeted pytest and parent core GREEN, then commit owned files with exact evidence.

Expected initial failures: no sdfx.hardware_info Python module, no hardware_info per-image result on opted-in dry-run,
no Scala HardwareInfo/Settings/GeneratedImage fields. These are capability failures, not expected existing-suite failures.

### Task2: native generation and Extras integration

**Files:** GenerationRequest/GeneratedImage/UpscaleRequest/ImagePathResult codecs, Python request parsing,
engine/dryrun final-image/upscale methods, Generator.describe/prepare/upscaleImage; new focused
`engine/src/test/scala/sdfx/engine/HardwareInfoGeneratorSuite.scala` and protocol/Extras tests.

- [x] First add native request-option injection, absent-field compatibility, stale imported field stripping,
  per-image infotext association and Extras source-hash tests. Parent observes relevant RED before those edits.
- [x] Integrate native metadata through existing generation/upscale requests, no new standalone engine command.
- [ ] CPU tests use synthetic clocks/provider facts and actual dry-run images; enabled/disabled decoded pixels equal,
  all4 records align with batch2×iteration2 seeds, separate requests reset; no added gallery member or temp leak.
- [ ] Extras tests use tiny synthetic PNG, own output temp, source parameters/custom chunks, exact input hash and
  expected resized pixels. Unknown provider metadata remains nonfatal, source file never replaced.
- [ ] Parent reruns affected protocol/Generator/SystemInfo/PNG suites; worker runs bounded CPU tests only.

### Task3: native controls, pasted provenance and actual acceptance

**Files:** SystemInfoTab/Service, MainWindow restore/sendTo, new `ui/src/test/scala/sdfx/ui/RealHardwareInfoCheck.scala`.

- [ ] Add focused model tests for known/equal/unknown CPU/GPU paste comparison and two-mismatch session budget;
  baseline tests already cover the pure policy. Add public actual Restore/Send-to helper assertions before wiring.
- [ ] Implement native card and shared paste path, preserve ordinary metadata restore behavior. Async local profile
  uses FX-safe callbacks, session budget is shared, and profile request failure does not fail Restore/Send-to.
- [ ] Parent actual native dry-run helper in isolated AppHome: toggle/remark/settings reload, generated PNG fields,
  batch2×iterations2 association, Extras input/output metadata and source hash, actual Restore and Send-to differing/
  equal/unknown hardware notices capped at2; verify no model start on offline restore, no changes to sampling fields.
- [ ] Parent trained SD1.5 and SDXL helper: actual Hardware Info/Remark/Time taken are in saved txt2img/img2img PNGs;
  batch/iterations and hires timing association, Extras source hash, enabled/disabled exact output pixels. Timing
  numbers are measurements of declared phases, not a checkpoint/device speed leaderboard or image-quality benchmark.
- [ ] Parent inspect full native1280×800 and expanded screenshots, metadata/remark readability and controls; run final
  aggregate tests/assembly, independently review source fidelity, then handle notices/ledger/promotion.

## Exact initial commands (parent owns Scala)

```powershell
# From this worktree, child CPU only:
$env:PYTHONPATH = 'D:/stable-diffusion-webui-reForge/stablediffusion-fx/.worktrees/hardware-info/python'
$env:CUDA_VISIBLE_DEVICES = '-1'
& D:/stable-diffusion-webui-reForge/stablediffusion-fx/python/.venv-smoke/Scripts/python.exe -m pytest python/tests/test_hardware_info.py python/tests/test_hardware_info_protocol.py -q

# Parent, from this worktree:
sbt.bat -batch 'core/testOnly sdfx.core.HardwareInfoSuite'
```

At this plan/tests snapshot, no production feature is implemented and no model/UI acceptance claim is made.

## Observed test-first snapshot

Worker executed with absolute main smoke venv, this worktree PYTHONPATH and CUDA_VISIBLE_DEVICES=-1:

- Existing `test_system_info.py -k 'cpu_snapshot or snapshot_reports_actual or metadata_whitelist'`:3 selected
  tests passed,exit0 (`outputs/hardware-baseline-cpu.log`). This is a focused environment baseline, not a claim
  that the entire fresh worktree's Scala/GPU suite was run.
- New pure hardware tests: expected collection RED,exit2, missing `sdfx.hardware_info`
  (`outputs/hardware-pure-red.log`). No production module exists yet.
- New actual dry-run stdio attachment test: expected RED,exit1, `KeyError:hardware_info` on the opted-in result's
  per-image record (`outputs/hardware-protocol-red.log`). The baseline and opted-in calls returned all4 seeds
  correctly; failure is the missing feature, not process startup or a fake hardware assertion.

At that initial snapshot, parent Scala RED and plan review were still pending. Only this plan and three new test
files were included; no production changes, notice/ledger changes or model execution had been made.

## Task1 implementation evidence

Parent approved the complete plan and native adaptations, then observed expected missing HardwareInfo/settings/
protocol API RED (20compiler errors with inference cascades,30seconds,20:37:26,
`outputs/hardware-core-parent-red.log`). Only then were production helpers/codecs/settings and dry-run metadata
attachment implemented. Two additional provider/setup-clock tests first failed for their missing interfaces,
then passed (`outputs/hardware-provider-red.log`).

Worker CPU results:14 hardware cases passed, including actual stdio per-image records for batch2×iterations2,
seed association and enabled/off byte-identical PNGs (`outputs/hardware-task1-cpu-green.log`). The same14 passed
in the main no-Torch venv (`outputs/hardware-task1-no-torch-green.log`). Broader hardware/params/SystemInfo/actual
dry-run protocol regression passed71cases (`outputs/hardware-task1-python-regression.log`). All commands used
absolute main venv paths, this worktree PYTHONPATH and CUDA_VISIBLE_DEVICES=-1; no GPU/model execution.

Parent's first native GREEN exposed the intentional additive GenerationRequest.hardware_info field in an exact
JSON-key-set assertion:31passed/1failed. The test was updated to retain an exact key set including only the new
field, with configured/default nested option checks and legacy omitted-record compatibility. Parent reran
HardwareInfoSuite4 + ProtocolSuite18 + SettingsSuite11:33passed,0failed,11seconds,20:45:14,
`outputs/hardware-core-parent-green-fixed.log`. `SettingsSuite` is the class inside ConfigSuite.scala (the file
name is not the munit suite name).

This Task1 snapshot does not wire real-engine generation, Generator PNG embedding, Extras or native controls.
Those remain Tasks2/3, with independent RED gates. Task2 tests/stdlib fixture are separate from this product
commit; a new actual dry-run Extras test already fails for missing result.hardware_info
(`outputs/hardware-extras-protocol-red.log`), and parent owns the upcoming Generator RED.

## Task2 integration evidence

Test-only commit `d71ca836a724e25f42580243a2c05e92e2e60dca` freezes Generator/Extras assertions and a stdlib fake
fixture which supplies explicitly synthetic records only when requested. Parent observed all4 Generator tests
fail for the intended missing integration at20:48:41 (`outputs/hardware-generator-parent-red.log`): absent trusted
options, stale imported fields, absent saved per-image facts and absent Extras PNG parameters. The last failure
now has an explicit missing-parameters diagnostic rather than None.get. Production was changed only after RED.

Generator injects global options in its existing single prepare pass, strips only the three owned imported
provenance keys, and merges each actual per-image record into final infotext. Extras attaches facts atomically
to its actual output with upscaler/size processing identity, keeps existing output text, and refuses source
aliases. Python real/dry-run generation attach per-image records immediately before final image saving after
postprocessing; Extras records whole-operation elapsed immediately before output saving. No RNG/latent/image
kernel change or second prepare call is introduced.

Child CPU hardware/protocol suite passed15cases including real stdio Extras metadata, source bytes and exact
enabled/off output bytes (`outputs/hardware-task2-cpu-green.log`). Parent native GREEN at20:54:20 passed
HardwareInfoGeneratorSuite4 + GeneratorSuite14 and ProtocolSuite18 (`outputs/hardware-generator-parent-green.log`).
No real-model claim is made yet. Native card/paste/manual-helper tests are the next separate RED gate.

## Task3 real native acceptance

After the no-start ownership repair, parent actual native CPU helper passed all5 checks at21:37:32
(`outputs/hardware-status-native-final-green.log`, `outputs/verify-hardware-native-cpu-fixed/`).
Its earlier unknown profile was traced to the minimal no-Torch test environment missing the already declared
runtime dependency psutil==7.0.0; installing that exact pin produced actual CPU/RAM facts without product changes
or weakening detection assertions. The real-model environment already had a working psutil installation.

Actual SD1.5 and SDXL runs then passed all5 native checks at21:38:30 (32seconds) and21:39:19 (49seconds)
respectively: `outputs/hardware-real-parent.log`, `outputs/verify-hardware-native-{sd15,sdxl}/hardware-info-report.json`.
Each covers defaultOff/offline restoration, literal remark/settings persistence independent of SystemInfo field
selection, four batch/iteration records with exact seed association and enabled/off pixel parity, img2img/hires,
actual Extras button provenance and input-byte preservation, compact profile display, shared two-notice budget,
unchanged local model/device/settings, and ten-image final gallery isolation. The actual detected GPU profile is
RTX3090/24GB with CPU/RAM facts; no synthetic response is used in these native model gates.
Root inspected the native CPU and real SDXL metadata cards plus actual generated SD1.5 PNG. Four-step images test
metadata neutrality and routing, not image quality. Root still owns final main integration/regressions/notices
and ledger promotion; these worktree gates alone do not publish or complete index309.


## Independent scope review and passive status ownership repair

Independent review read the complete pinned309 source audit and current formatter/provider, per-image clocks,
protocol/Generator/Extras integration, native opt-in/remark settings, shared Restore/Send-to notices and disposal.
No additional source-scope omission was identified. Reviewed native privacy/timing/device adaptations remain
explicit above; real-model/native rendering acceptance and parent promotion are still separate gates.

The review found a real no-start race: HardwareProfileCache captured supervisor.pid, but its query then used
EngineClient.systemStatus through the ordinary ensureStarted request path. A process disappearing between
capture and dispatch could cause a passive pane/Restore lookup to launch another engine. Rejecting the stale
response afterwards would not undo that side effect.

Tests were added before the production repair. Parent's initial compile RED at21:33:45 also exposed an await
argument/inference error in the new test; all await calls were corrected to explicit durations before the clean
RED at21:34:55 (13seconds, outputs/hardware-status-ownership-red-fixed-test.log). That clean snapshot failed
for the missing systemStatusIfRunning API and captured-PID query signature, with compiler inference cascades.

The repair passes the captured PID into the cache query and uses EngineSupervisor.systemStatusIfRunning.
It snapshots the actual existing process, dispatches off the caller/FX thread, rechecks process identity before
stdio dispatch and on completion, and returns a bounded5second failure without ensureStarted/restart. A stale
response cannot fill the new process's cache; transport/timeout failures release the current cached promise
for retry. A late result after timeout cannot overwrite the already failed promise or a later retry.

The first GREEN attempt passed5/6 and found a test-only Windows identity assumption: Python's child PID need
not equal the supervised py-launcher PID. The test now compares the returned child PID with get_state from
the same current process, preserving the launcher PID as the supervisor ownership token. A fourth regression
uses a delayed250ms fixture status reply: a50ms first deadline must fail EngineTimeout; a2second retry then
succeeds on the same process without a second launch.

Parent final GREEN: HardwareStatusOwnershipSuite4 + HardwareProfileCacheSuite3,7passed/0failed,8seconds,
21:37:18. The same sbt run then completed the full native CPU/dry-run helper in14seconds at21:37:32, including
offline Restore/no engine start, persisted literal remark and independent System Info fields, four per-image
records/exact enabled-off pixels, img2img/hires, native Extras/source bytes and the two shared provenance notices.
Evidence: outputs/hardware-status-native-final-green.log and
outputs/verify-hardware-native-cpu-fixed/hardware-info-report.json. This is actual native/dry-run evidence,
not trained-model acceptance. Root owns subsequent SD1.5/SDXL runs, rendering inspection and promotion.
