# Prevent Interruption: backend ownership and lifecycle follow-up

Bound to the approved core-prevent-interrupt source plan. Clean standalone transport/lifecycle corrections; no upstream code copied, dependency, model kernel, GenerationParams, replay preparation, RNG, settings persistence, or metadata changes.

## Observed regressions and correction

1. Actual delayed dispatch after queue A completes and B reports sampling progress cancelled B (parent RED 0.806s). The queue now carries A's immutable job owner through Generator model/VAE/generation/script transport requests. EngineProcess stores the owner separately from params. Supervisor restricts candidate request IDs by owner and sends owner plus target to the actual Python reader; the reader checks policy, target and owner under the same lock before cancellation. Unknown/finished owners have no cancellation side effect. Unowned raw callers retain their prior global/target behavior. Both engine and MCP fake transports implement current/queued owner checks rather than unconditional interrupt events.
2. Typed query/toggle during a real live new PID's two-second startup reload timed out at 300ms (parent RED 3.103s). The typed method now uses an asynchronous live-reader route, independently bounded at five seconds, instead of awaiting the worker/model reload. Failed controls never grant cancellation permission.
3. Authorized explicit restart detached the old process and waited for close; concurrent supervisor close followed by release incorrectly started a replacement (parent gated RED 0.543s). Restart establishes its start intent before close, retains the captured stop epoch, and invokes ensureStarted with that epoch rather than unconditional start(). A later stop wins atomically. Explicit restart from Stopped remains supported. Start-attempt epochs are captured when allocating the promise under the lifecycle lock. Existing startup terminal-state-before-promise ordering is retained.

## Verification

- New actual Python reader ownership tests: initial 9 failures (expected missing owner enforcement), then focused real reader/server policy/dry-run/system-info tests **46 passed, 1 skipped in 12.34s** using main absolute .venv-test, worktree PYTHONPATH and CUDA_VISIBLE_DEVICES=-1.
- The Python tests include current and queued foreign owner rejection, targeted queued cancellation, invalid owner inputs, and an actual blocked worker proving owner transport does not enter generation params and clears after completion.
- Parent owns Scala execution: `engine/testOnly sdfx.engine.PreventInterruptionSuite sdfx.engine.JobQueueSuite sdfx.engine.EngineSupervisorSuite`; MCP regressions should also run with the updated fixture. Child has run no sbt or GPU work. Scala GREEN pending at freeze.
- Targeted native suite includes actual delayed backend handover, typed query and toggle during reload, gated concurrent close, and positive manual restart from Stopped. No external submission/clipboard/network action.

## Remaining contract boundary

Public unowned raw interrupt intentionally retains global/target semantics. Follow-up cbfa387 test-only snapshot exposed three additional ownership edge failures, all observed RED by parent (idle late-dispatch cancellation, stale standalone handle cancellation, queued cancellation grace restarting unrelated work). Queue entries now have private UUID transport identities independent of public Job.id; idle captures also have an unmatched private identity. Public GenerationHandle requests get a fresh private owner when outside a queue. Reader interrupt response adds active_id only for actually active admitted work (null for queued/noactive); supervisor arms grace only for that matching active identity. Queued cancellation remains queued and cannot cause unrelated active work to be force-restarted. Both fixtures implement this distinction. Two actual Python reader active/queued response tests initially failed with missing active_id, then passed. Typed UI/MCP result schema and generation parameters remain unchanged.

Parent executed first frozen7dad0bd GREEN:40 engine tests including all three original regressions and Queue/Supervisor;42 MCP tests. Follow-up Scala GREEN pending at the second freeze. Full torch-free Python baseline at first freeze:373 passed,79 skipped in17.46s. No child sbt/GPU execution.

Second frozen snapshot Python full torch-free suite: **375 passed,79 skipped in19.90s**, exit0; outputs/prevent-targeting-python-final.log. New response identity tests RED2failed0.12s before implementation. Existing exact dry-run raw interrupt response test now also requires active_id equal to the actually interrupted generation ID.


## Global runtime controls follow-up

Parent final snapshot0a81651 verified43 engine and42 MCP tests GREEN. Independent review then found queue-specific idle safety had narrowed the global UI/MCP interrupt path: nonqueue System Info benchmark work no longer matched an owner. New test-only eb3c1ac demonstrates actual standalone generation while queue idle: protected action blocked, Off action must interrupt; plus delayed empty snapshot and delayed completed-A snapshot must leave later B intact. Parent observed all3 RED with corrected `*runtime*` filter (the initial `runtime*` selected0 tests and is not evidence).

New public `AppServices.interruptRuntime(): Future[QueueInterruption]` is the common UI/MCP route. It captures the queue entry and fallback actual process/request before dispatching admission I/O on a worker. Queue-only APIs remain queue-only. Nonqueue fallback is an immutable process+request-ID snapshot; an empty snapshot sends an impossible target instead of becoming a global late interrupt, and a replacement process is never selected. Supervisor selects oldest pending worker request, excluding reader controls, so an earlier load is not skipped for a queued generation. The authoritative reader still controls policy and actual active identity, and only matching active admission arms a grace timer. Native entry flags still require the captured entry to be current after admission.

PF owns all UI/MCP caller updates and the real System Info benchmark helper; child changed no UI/MCP callers in this follow-up. No Python/model/kernel changes, no sbt/GPU execution. New Scala GREEN pending parent execution at freeze.


Final global concurrency regression321953e: independent nonqueue B active while native A is current/waiting behind it; parent observed one actual RED (B completed instead of Interrupted,21:09:41). Parent declared global semantics explicitly: capture actual backend request and native current entry independently; accepted admission stops only that backend request and sets only that still-current captured native entry flag. Blocked admission does neither. Queue-specific cancellation remains owner-bound, and pending-only entries remain unchanged. The global fallback is therefore used regardless of whether a native entry exists. This closes the System Info benchmark plus waiting native generation combination; public AppServices API and PF UI/MCP routing are unchanged. Prior7df snapshot GREEN46 engine tests; final concurrency GREEN pending at freeze.
