# Native System Info plan

## Parent acceptance completed

Actual SD1.5/SDXL native checks passed loaded-model reporting, visible-only polling/charts, quick/turbo benchmark,
local history, actual interruption, preview without Submit, and opt-in metadata with exact RGB restoration.
Final SDXL helper follows native startup/checkpoint loading, waits Ready, refreshes the real report and produces
inspected overview/score-first benchmark screenshots. No cloud data was sent; server receipt and real high-batch
stress are not claimed. Full Scala522 + assembly, Python795/1skip and torch-free353/78skip passed. The retained
MIT benchmark-prompt notice was collected; index90 promoted. See `docs/VERIFICATION.md` for artifacts, exact scope
and adaptations. Worker validation notes below describe their earlier phase; the overall goal remains active.

**Target/spec:** extension index 90 `core:system-info`, vladmandic/sd-extension-system-info at `006f08f499bbe69c484f0f1cc332bbf0e75526c2`, MIT Copyright (c) 2023 Vladimir Mandic. README, scripts/system-info.py, benchmark.py, javascript/system-info.js, pages/README.md and history inspected. Current source JS polls every 5 seconds (README says 1 second); retain current 5-second visible-only polling.

## Binding source acceptance

- Current engine start time/version/model/VAE/state/queue/progress, RAM/VRAM allocation/reservation/peaks/OOM/retry/utilization; platform, device/dtype/backend, optimization flags, library/repository versions, samplers/extensions. Full refresh plus cheap live refresh while visible, interruption, bounded utilization/memory charts and inspectable JSON.
- Models and LoRAs from current source; README's embeddings/vector counts and hypernetwork inventory also exposed where safely readable. No unpickling arbitrary embedding files; report unavailable vector counts honestly. Full SHA256 model hashing is an explicit potentially expensive action; loaded hashes shown immediately.
- Standard benchmark presets quick/normal/extensive/ludicrous with batches 1;1,2,4;1,2,4,8,16;1,2,4,8,16,32,64. Turbo/default/long use 10/20/50 steps, optional warmup and dimensions. Real generation wall time and steps*batch/seconds, local deduplicated history. Dry-run is explicitly excluded as a performance result. Stop subsequent larger batches on errors/OOM, preserve failure details, support interrupt.
- Optional legacy submission: explicit preview then separate Submit action, exact destination/payload visible, optional typed identity and hostname choice. No network transmission in tests or automatic polling/benchmark actions. Legacy UDP cannot acknowledge server receipt; state this rather than claiming acceptance. Local JSON export and published leaderboard link are additional functions, not substitutes for submission.
- Latest source opt-in selectable generation metadata and read-only status API are included. Native JSON Lines replaces A1111 FastAPI; JavaFX replaces Gradio, and native app/plugin/catalog settings replace parent web-UI flags/repos. No A1111 imports/launch.

## Architecture and files

Python `system_info.py`: safe snapshot/inventory, benchmark normalization/runner, metadata whitelist, legacy packet preview/send. `BaseEngine` methods and server direct `system_status` route remain usable during generation; full info/benchmark/submission requests use single worker. JobContext records latest progress for status without invasive sampling changes. Scala client/service merges native app, plugin and queue facts and owns local history. Native System Info tool tab has Overview, Models & Networks, Benchmark and Info JSON sections, visibility polling, stable QA IDs, local export chooser and opt-in metadata selection.

No Plugin shim needed: target is core UI. Parent ledger/notices generated separately after acceptance. No new Python dependency; psutil is already runtime. Optional preinstalled `nvidia-smi` driver-version query is bounded and gracefully unavailable, never installed or bundled. No GPU/sbt execution by this worker. Parent owns real SD1.5/SDXL benchmarks, app integration and rendered UI acceptance.

- [x] RED/GREEN tests: initial four tests failed due to absent module; legacy metadata aliases and malformed submission requests separately failed before their repairs. Windows subprocess metadata generation reproduced a real JSON Lines hang before the bounded-output repair.
- [x] Python implementation, read-only capability/API and Scala client/service/history.
- [x] Native tool UI, state persistence and generation metadata pipeline.
- [x] Focused CPU and full torch-free tests, static checks, manual actual-engine/native UI helper prepared.
- [ ] Parent final native SD1.5/SDXL acceptance, review and ledger/notice promotion.

## Implemented native adaptations and source provenance

The fixed positive/negative benchmark prompts are retained verbatim from upstream `benchmark.py`, under its MIT licence (Copyright (c) 2023 Vladimir Mandic); classify the target **ported**, not exclusively clean reimplemented. The remaining implementation was independently written. Parent must collect the upstream MIT notice; no upstream JavaScript, Gradio/Python module or bundled third-party JS dependency is copied or imported.

Source benchmark batch presets and 10/20/50-step presets remain available, using Euler a, CFG7 and a fixed seed42 for reproducible native runs. Width/height are validated against native multiples-of-eight limits and source's minimum65536pixels. The selected preset is used for warmup too (source passes a non-string False, incidentally using default20steps); this typed native adaptation avoids that source UI bug. The native app reports backend/model/sampler/config explicitly; throughput across different backend implementations is not claimed numerically interchangeable. No arbitrary300it/s ceiling rejects a genuinely fast measured run. Successful output count/dimensions, finite wall time and positive duration determine valid measurements. PNG encoding/writing is skipped inside a scoped internal benchmark flag, matching source do_not_save_samples; finally restores ordinary generation even on interruption/failure. Benchmarks are not added to the native image gallery.

System start time is the standalone engine start; parent web-UI repository versions are replaced by the native app commit plus installed package versions. No Git remote URL, environment dump, username discovery or hostname enters normal snapshots. VAE/model file paths are excluded. Models inventory shows known loaded hashes immediately; expensive full SHA256 prefixes require Hash inventory. Hypernetwork inspection is inventory-only under the app's models/hypernetworks; inference remains its own unfinished target. Embedding vector counts come only from safe safetensors headers; pickle files are listed with count unavailable and never loaded. Image metadata retains legacy field names as aliases, and all choices are off by default. Unsupported accelerator fields remain unavailable rather than fabricated.

Legacy submission retains eleven pipe-delimited source columns plus digest, syslog priority14, SDBENCHMARK prefix, destination logs3.papertrailapp.com:32554/UDP and NUL termination. Preview binds a one-use token to exact packets for five minutes. The default identity is an explicitly typed username plus anonymous hostname; actual hostname is included only after its checkbox is chosen. No submission was sent in worker tests. UDP transport has no receipt acknowledgement; a successful local send is labelled sent_unconfirmed, and transport errors retain a local export option. Exported JSON and the published leaderboard link are additional conveniences, not substitutes for actual submission capability.

## Worker evidence and parent commands

- Full torch-free suite: **327 passed,75 skipped**. CUDA_VISIBLE_DEVICES=-1, worktree PYTHONPATH, main repository .venv-test.
- Focused real-torch environment CPU status/protocol/params plus tiny-engine suites: **52 passed**. No GPU visibility or model downloads.
- Actual tiny random VAE/UNet engine test: **1 passed**. It runs normal generation, a real CPU benchmark, then normal generation again. Benchmark makes zero PNG-save calls; the ordinary path resumes saving and produces identical RGB. Only CLIP encoding is supplied directly to avoid downloading an encoder; this is pipeline-contract evidence, not trained-model performance evidence.
- Parent independently ran engine SystemInfoSuite (**1 passed**) and ui/compile (**all37sources passed**). An initial parser failure from reserved Scala keyword export was repaired before that green build.
- Existing metadata protocol regression initially hung in Windows subprocess.communicate pipe-reader threads. Faulthandler evidence is retained in ignored outputs/system-info-debug/trace.txt; bounded temporary command output files repair this. The real subprocess regression now passes.
- Parent ran the actual SD1.5 RealSystemInfoCheck successfully (exit0), with report/snapshots in worktree `outputs/verify-system-info-sd15/` and main `outputs/system-info-native-sd15.log`. The report confirms visible-only polling/chart, quick/turbo benchmark/history, preview without Submit, interrupt, and explicit PNG metadata RGB equality/restoration. Model animekawa_v10, hash d4c1719023; benchmark batch1,10steps,256x256,0.394949s,25.319718it/s (one machine/run, not a general performance claim). This worker inspected the JSON and both rendered screenshots. Subsequently added read-only component dtypes/full JavaFX property reporting and score-first benchmark table formatting; no sampling change. Parent should compile the final SystemInfoViewSuite and validate SDXL before promotion.

Parent can run `sbt.bat -batch "ui/Test/runMain sdfx.ui.RealSystemInfoCheck <python> <checkpoint-dir> <checkpoint-name> <output-dir> <screenshot.png>"` separately for SD1.5 and SDXL. The helper creates isolated AppHome/persistSettings=false, loads the real model, exercises the actual MainWindow System Info tab, checks visible-only polling and actual GPU memory graph, runs quick/turbo256x256 benchmark and validates persisted history, previews legacy submission without pressing Submit, interrupts a long native benchmark, and checks explicit PNG metadata plus exact RGB equality when enabled/disabled. It writes system-info-report.json and overview/benchmark screenshots. The helper is manual, not auto-discovered by sbt test. The parent must inspect rendered screenshots and actual metrics before promotion.

Read-only protocol/API equivalents: system_status is serviced directly on the reader thread while generation runs; it includes request type/id, queued count, interruption and latest real progress. system_info performs queued full refresh/inventory; system_benchmark is an interruptible queued workload; system_submission_preview and system_submit are separate queued actions. All engine info actions avoid sampling-hook shims. Capabilities advertise system_info in both actual/dry-run engines, but dry-run explicitly refuses performance benchmarking.
