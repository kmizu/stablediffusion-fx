# Prevent Interruption: implementation plan and protocol proposal

**Status:** integrated and accepted on main; index320 promoted after final independent review,596Scala tests,867Python tests,397noTorch tests, assembly and actual CPU/SD1.5/SDXL native acceptance. Final evidence is in `docs/VERIFICATION.md` and `outputs/control-metadata-*`. Original worktree `.worktrees/core-prevent-interrupt`, branch `agent/core-prevent-interrupt-20261026`, base `d620d0f2d8ee5eb86a7a143c0040fb0dbb312ae1`; backend ownership corrections were integrated from `.worktrees/prevent-interrupt-targeting`. Root owns all Scala/native/GPU execution. Earlier pending sections below are chronological evidence, not current release blockers.

## Binding source scope

Index320 `core:prevent-interrupt`, source https://github.com/light-and-ray/sd-webui-prevent-interruption.git pin **ea22bc0de11875f3889f09c8e6ed7b879f321d05** (2024-06-26). README, complete `scripts/prevent_interruption.py`, `javascript/prevent_interruption.js`, `style.css`, all9history subjects and relevant complete synchronization/restart changes were inspected. Full audit and ignored pinned clone remain in sibling `core-image-compare/outputs/prevent-interruption-source-audit.md` and `source-prevent-interruption/`.

Original: one footer toggle, runtime-global per server, defaultOff, no persistent setting. Enabled blocks state interrupt, optional stop_generating, skip, user apply/restart/request_restart, and frontend restart/reload entry points. Backend protection covers button/API/CLI-extension callers; multiple frontends poll the same server value every5seconds. Disabled delegates arguments/results normally. Blocked calls warn without performing the original action. No LICENSE/grant in pinned tree or history filenames: independently reimplement behavior, copy no source code/CSS/assets.

No source close-confirmation, pending-only queue clear/cancel lock, new skip command, model/parameter lock or OS health-failure guarantee. README recommends a separate close-confirmation extension. Native has no skip/reload-UI primitive; cover existing active script/job cancellation and explicit user restart without inventing those operations. Do not guard teardown, automatic crash/timeout recovery or pending-only queue operations.

## Architecture and linearization decision

**Python server reader is the authoritative policy and admission point for an owned engine process.** State is `(enabled=false, revision=0)` at process creation, independent of model load/unload. Toggle/query are direct reader requests, never worker-queued; a blocked operation has no cancellation side effects. Shared GUI/MCP UI state is a display cache, never sufficient to decide whether a destructive operation is allowed. This prevents another frontend's just-enabled state being bypassed during a5second visual refresh window.

Use one atomic reader-thread admission for each interrupt/restart authorization. A request is allowed or blocked at that point; protection enabled after an accepted action cannot retroactively retract it. No reusable authorization token or security boundary is implied. A blocked operation returns a truthful decision and warning; no native job flag or supervisor escalation timer changes. Query/toggle can start the normal engine process if necessary but never load a checkpoint or run a model pipeline. A genuinely new process resetsOff, matching source; refresh the UI on supervisor epoch change instead of silently persisting/reapplying protection.

The critical native ordering is **admission → same-target validation → cancellation flag/timer**, not flag → engine request. Capture current job/request identity, await the bounded reader decision off the JavaFX thread, and mutate its flag only if that same job is still active. Never let a late accepted response cancel the next job. A request between script sub-generations can be admitted even with no active Python request; its native job flag then stops later iterations. An unknown/stale Python target can have `affected=false` while native script admission remains true.

### Proposed JSON Lines v1 additions (no production schema edits yet)

| Request | Input | Result |
|---|---|---|
| `interruption_protection` | `{}` query or `{"enabled":true/false}` set | `{"enabled":false,"revision":0}`; real bool only; increment revision only when value changes |
| existing `interrupt` | existing optional `target_id` | additive `{"accepted":bool,"blocked":bool,"affected":bool,"protection":{"enabled":bool,"revision":n}}` |
| `restart_authorization` | `{}` | `{"accepted":bool,"blocked":bool,"protection":{...}}`; admission only, no worker/model/shutdown side effect |

Malformed protection values return INVALID_PARAMS and leave state unchanged. Existing clients that ignore interrupt response fields remain compatible. `shutdown` retains normal teardown behavior. `restart_authorization` is invoked as the explicit restart action begins; after acceptance the owning supervisor restarts normally. Raw protocol clients receive the same gate on their existing interrupt action; they cannot arbitrarily restart the owning JVM. Add typed Scala `InterruptionProtection` / `InterruptionDecision` records and constants, and `interruption_blocked` for blocked synchronous explicit-restart API. Advertise a `prevent_interrupt` core capability in actual and dry-run engines, with no generation hook/plugin shim.

## Owned implementation tasks after parent approval/RED

### 1. Reader policy and actual Python contract

Files: `python/sdfx/server.py` (policy state, validation and direct request dispatch), `python/sdfx/base.py` / actual+dryrun feature lists (capability only), `core/.../protocol/Protocol.scala` and `engine/.../EngineClient.scala` (typed schema/client).

Tests already written: `python/tests/test_prevent_interruption.py`9cases use a real Server and a blocked worker thread, no mocked response objects. Verify defaultOff/new-server reset, idempotent shared query/set, strict bool validation, active and target-pending raw interrupt side effects absent while protected, responsive query/toggle/interrupt while worker remains busy, completion after blocked interrupt, explicit restart admission and unchanged shutdown.

Run CPU RED with main absolute `.venv-test` and worktree PYTHONPATH/CUDA_VISIBLE_DEVICES=-1. Implement minimum dispatch/state only after parent plan review/RED. Then run9cases GREEN plus existing server/status tests. Never import or launch parent reForge.

### 2. Native queue/supervisor active cancellation and user restart

Files: `engine/.../EngineSupervisor.scala`, `JobQueue.scala`, `AppServices.scala`, and a focused `InterruptionProtectionService.scala` if needed to own live snapshots and presentation. Avoid a second independent enforcement flag. Keep existing callback compatibility where required; expose structured action results for UI/MCP. All blocking admission I/O runs off FX, uses bounded waits, and fails closed for the requested user action if the policy cannot be read. No raw policy error is interpreted as permission.

`JobQueue.interruptCurrent` / cancel(running): obtain admission before setting AtomicBoolean or emitting terminal cancellation. Preserve pending cancel/clear unchanged. `EngineSupervisor.interrupt`: blocked response arms no timer. Allowed response retains normal grace escalation and target identity checks. `restart()` is the explicit guarded route; `forceRestart`/health recovery and `close`/stop/internal shutdown remain unguarded. Distinguish allowed=false from nothing-running; do not report a blocked job as cancelled. Native queue shutdown must use its internal teardown route rather than the guarded user operation.

Tests already written: `engine/.../PreventInterruptionSuite.scala`4cases: active double cancellation cannot drop a two-image job or mark it cancelled; pending-only cancel/clear unchanged;100ms supervisor escalation does not affect a protected >100ms generation; explicit restart blocked with same PID while forceRestart succeeds and new server resetsOff. Existing stdlib-only fake engine fixture must add the proposed reader contract during implementation; until then this suite fails unsupported at setup. The Python suite independently validates the actual implementation, so fake fixture behavior is not the only oracle. Before editing each integration path add any additional identity-race/defaultOff test needed by the chosen API shape.

### 3. MCP routes and truthful combined outcomes

Files: `mcp/.../McpTools.scala` and narrowly related cancellation context handling if needed. Add `interruption_protection` tool accepting optional enabled and returning effective state. Existing interrupt gets additive `blocked` + protection fields while retaining interrupted/cleared. `interrupt(clear_queue=true)` may still clear separately requested pending jobs; active interruption stays blocked and outputs continue. Running cancel_job/transport cancellation uses the same admission before **generation/job** flags. Transport cancellation can stop waiting for a disconnected client, but must not relabel or terminate the protected job; retain truthful pollable completion. Pending waiting-for-engine cancellation stays unchanged. Trace every existing `aj.cancelRequested`, queue.cancel and delayed retry path to ensure no earlier generation flag bypasses the gate.

Test already written: `mcp/.../PreventInterruptionMcpSuite.scala` asserts combined blocked/false active outcome plus cleared1 and the running job's successful result, followed by shared query/unlock. Implementation must additionally add the running async cancel_job + notification tests before modifying those paths, including the dequeue/start race and no accidental deferred cancellation after unlocking. No outbound messages/network are needed; use in-process MCP server/fake engine.

### 4. Native controls and state convergence

Files: `StatusBar.scala`, `MainWindow.scala`, `SystemInfoTab.scala` interrupt route, possibly `UiContext.scala` for shared action callback. Accessible defaultOff footer toggle with visible effective state and tooltip; styling follows existing Theme. Source says prevent, not confirm: blocked controls show reason and do not offer an implicit bypass. Explicit user can turn toggleOff normally. Keep toggle usable during model/generation work. All interrupt controls (Generate→Interrupt, Esc, SystemInfo) and explicit Restart/Settings restart go through central admission, not a local enabled checkbox. No new Settings persistence or UiState protection field.

Poll direct status at most every5seconds while native window exists and refresh on local operation + engine epoch changes; GUI/MCP frontend views converge within5seconds. Stop timer/listeners on dispose. Control updates remain on FX, no blocking driver/model work. MainWindow integration must preserve parent Styles/VAE/Comparison work when semantically merged from this older base.

Prepare `RealPreventInterruptionCheck` manual helper (parent only): isolated AppServices, actual controls and in-process MCP, bounded dry-run job/script, protect→blocked Interrupt/Esc/SystemInfo/Restart→successful completion→unlock→normal interruption. Verify two frontend views converge, pending clear remains separate, toggles resetOff for new server, shutdown/health recovery still complete. Root requires actual protected in-progress generation on both SD15 and SDXL before promotion, even though this feature changes no sampling kernel. The helper supports real checkpoint arguments; retain modest30step/256SD15/512SDXL workloads with an actual first-progress latch and assert the protected output completes. This is control acceptance, not a numerical quality/baseline claim. Add UI state/callback tests before editing production controls.

## Verification and handoff gates

- First snapshot: this plan + new tests only. Python RED recorded: **9failed in0.24s**, each expected UNSUPPORTED unknown `interruption_protection`; no import/setup errors. No production files changed. A captured repeat is `outputs/prevent-interruption-red.log`:9failed in0.30s with the same expected reason. Scala suites remain unrun because parent owns sbt.
- Parent approved the plan and independently reproduced Python9RED (0.26s), engine4RED and meaningful MCP unknown-tool RED before authorizing implementation. Parent then reproduced additional MCP notification/deferred-cancel RED before that repair and missing-native-toggle UI RED before UI implementation.
- After approval: targeted Python real reader/server tests; parent engine4/MCP1 plus additional race/notification/UI tests; ordinary Off behavior and existing queue/supervisor/MCP regression suites. Parent alone runs sbt/native helper and broad integration checks. No GPU or sbt by child.
- No universal OS interruption guarantee, new skip/queue/close feature, automatic lock persistence or source-code copying. No ledger/license/generated-doc edits in this branch. Final handoff includes exact source pin, all modified paths, CPU/runtime evidence and precise remaining limitations.


## Incremental implementation evidence and current gates

- Initial plan/tests-only commit `bbf5e084068ecdbfc59fa78a20aa815a6250f4b2`; reader/protocol implementation `d59513a15ccf170c7b56128dc5784689aaeda9bd`. New Python control tests increased to12 including capability and real JSON Lines subprocess restart/reset. Worker smoke-venv CPU:12passed in16.03s; no-torch combined new reader/existing server/status/protocol:42passed in6.63s. Existing Off interrupt test now asserts the exact additive accepted/blocked/affected/protection response. Child ran no GPU, sbt or native app.
- Parent `outputs/prevent-interruption-engine-green.log`: core typed protocol2passed, engine36passed (Prevent6, existing queue15, supervisor15), completed20:20:56. This predates the subsequent actual backend-handover regression and does not validate its repair. Startup-ordering fix `4e051e65df4595ba6f9012081c2c043cf2028371` was carried into this branch separately from cancellation methods and already exists in published main533d2fb.
- MCP test/protocol fixture `a80e9a575ee3449fdc8e8f33811d769fba562466` preceded production `103255521636cefa3357b192d927d73a794d00df`. Parent independently confirmed unknown-tool and actual deferred-notification retry RED. Structured native cancellation distinguishes blocked from dequeue/start miss; only the latter is retryable. Waiting-job cancellation and queue publication are synchronized so a pending cancel does not become an unguarded active flag.
- Independent review found the initial post-response native identity check insufficient: an unbound delayed callback could reselect backend requestB after jobA completed. `3aab53dc1c490b95a711631692602f06000f6a2e` adds an actual supervisor dispatch gate and requires B to complete; root owns RED. A separate assigned worker owns the private backend owner/request binding repair. Do not call this fixed until its commit is integrated and root validates real backend behavior. It is an ownership correction within the existing cancellation contract, not a new protection feature.
- Parent native UI RED: missing Prevent interruption toggle,35seconds, completed20:29:38 in `outputs/prevent-interruption-ui-red.log`. UI implementation `87718708028a5beb4a031668d2e787da13c09e44` adds runtime-only acknowledged checkbox/status,5second polling with process/revision guards, disposal, asynchronous native outcome feedback, shared SystemInfo route and guarded explicit restart instead of health-forceRestart. Child has not compiled or run it.

Manual parent commands from this worktree:

```
sbt.bat -batch "core/testOnly sdfx.core.InterruptionProtocolSuite" "engine/testOnly sdfx.engine.PreventInterruptionSuite sdfx.engine.JobQueueSuite sdfx.engine.EngineSupervisorSuite" "mcp/testOnly sdfx.mcp.PreventInterruptionMcpSuite" "ui/Test/compile"
sbt.bat -batch "ui/Test/runMain sdfx.ui.RealPreventInterruptionCheck <python> outputs/verify-prevent-interrupt-cpu"
sbt.bat -batch "ui/Test/runMain sdfx.ui.RealPreventInterruptionCheck <python> outputs/verify-prevent-interrupt-sd15 D:/stable-diffusion-webui-reForge/models/Stable-diffusion animekawa_v10"
sbt.bat -batch "ui/Test/runMain sdfx.ui.RealPreventInterruptionCheck <python> outputs/verify-prevent-interrupt-sdxl D:/stable-diffusion-webui-reForge/models/Stable-diffusion waiIllustriousSDXL_v170"
```

Use the main absolute smoke Python for actual model modes. CPU mode creates only a tiny inspectable fake checkpoint header and uses the actual `--dry-run --dry-run-delay-ms35` subprocess, never a trained model. Real mode uses existing actual models; parent alone runs it. Helper waits for first actual sampling progress and verifies workload still pending, footer actually in enabled Interrupt mode and SystemInfo action enabled, then exercises footer/Esc/SystemInfo/MCP and explicit restart while On; final image completes with same process. Off must restore JobCancelled outcome. Shared other-frontend polling convergence and CPU health-restart/new-serverOff are distinct checks. It writes `protected.png` and `prevent-interruption-report.json`; do not claim results before inspecting root artifacts. Normal close/teardown and pending-only clearing retain their separate semantics.


### UI process-identity follow-up and measured CPU native acceptance

Independent review found that a first toggle request captured with PID=None could later publish old processB's On/revision1 response after health replacement processC was already Off/revision0, incorrectly assigning revision1 to C and ignoring its subsequent revision0 snapshots. The error rollback used the same unsafe publication path. Tests-only `9d640499b891cb359ce0c7c50592cfceeae9c9e6` deterministically delivers that actual private presentation callback after the CPU helper observes newC0. Parent RED reproduced the exact stale-unbound assertion in26seconds (20:40:29, `outputs/prevent-interruption-unbound-red.log`).

UI repair `3456bd5b15a5fcb8accbad29b253174159d7bc9f`: reject all unbound or obsolete-PID real replies, perform a fresh PID-bound query, separate synthetic stopped reset, and never republish cached rollback state into another process's revision namespace. Independent UI review approved this minimal patch with same-PID revision ordering and disposed callback guards retained.

## Integrated acceptance, final review still open

Main Python regression passed **852 tests, 1 skip** (`outputs/prevent-integrated-python.log`);
the independent no-Torch run passed **382 tests, 82 skips** (`outputs/prevent-no-torch.log`).
Main MCP protection suite passed all4 cases and the actual dry-run native helper passed all5 checks
(`outputs/prevent-integrated-native.log`, `outputs/verify-prevent-cpu/prevent-interruption-report.json`).

The initial real-model run's handle disappeared with no remaining owned Java/Python processes and no success
report; it is incomplete evidence, not a pass. Resumed real acceptance completed normally on26October2026:
SD1.5 in89seconds at21:31:17 and SDXL in49seconds at21:32:06 (`outputs/prevent-real-resumed.log`).
Both reports (`outputs/verify-prevent-sd15-retry/`, `outputs/verify-prevent-sdxl/`) confirm actual sampling-progress
before protected footer/Esc/SystemInfo/MCP actions and explicit restart; finished output with unchanged process;
ordinary cancellation after unlock; and the SystemInfo native Interrupt stopping its own actual benchmark outside
JobQueue. Root inspected both generated lighthouse PNGs and the native footer. The screenshot immediately after
completion precedes asynchronous image presentation and retains stacked blocked-action toasts; it proves the
control state, not final preview rendering or image quality. No sampling-kernel or numerical-parity claim is made.

Independent final review identified two further ordering cases to reproduce before promotion: explicit Off restart
during a remembered-model startup reload, and concurrent request allocation order differing from successful wire
order. Ledger/notices/release remain pending those regression gates.

Parent GREEN `outputs/prevent-interruption-unbound-green.log` completed20:43:42 in23seconds. Read actual `outputs/verify-prevent-unbound-green/prevent-interruption-report.json`: `success=true`, `real_model=false`, **5 checks passed** covering acknowledged defaultOff/other-frontend convergence; actual dry-run progress latch and enabled footer/Esc/SystemInfo/MCP plus protected user restart with completed output; Off cancellation outcome; CPU health restart/new processOff; and stale unbound callback rejection. The `control-test` fake inspectable header is labelled kindSD15 by dry-run only: this is **not trained SD15/SDXL inference evidence**. Actual model gates remain pending backend owner-binding integration and parent real-model runs. The root native screenshot was reported readable at the1280-class viewport; no additional visual redesign was performed during this correction.
