package sdfx.engine

import munit.FunSuite
import sdfx.core.model.{GenerationParams, LoraRef}
import sdfx.core.protocol.*
import java.util.concurrent.{CountDownLatch, TimeUnit}
import scala.concurrent.{Await, ExecutionContext, Future, Promise}
import scala.concurrent.duration.*

class WdBatchSafetySuite extends FunSuite:
  given ExecutionContext = ExecutionContext.global
  test("resolved prompt LoRA is rejected before worker dispatch") {
    val resolved = GenerationParams(prompt = "clean prompt", loras = Seq(LoraRef("resolved", 1.0)))
    intercept[EngineError](WdBatchSafety.validateResolved(resolved))
  }

  test("all-error and malformed completed rows cannot report success") {
    val failed = WdBatchResult(Seq(WdBatchItem("error", "/bad", error = Some("unreadable"))), true)
    val error = intercept[EngineError](WdBatchSafety.validateResult(failed))
    assert(error.message.contains("unreadable"))
    val missing = WdBatchResult(Seq(WdBatchItem("complete", "/input")), true)
    intercept[EngineError](WdBatchSafety.validateResult(missing))
    val empty = WdBatchResult(Seq(WdBatchItem("complete", "/input", positive = Some("p"), negative = Some("n"),
      result = Some(GenerationResult(Seq.empty)))), true)
    intercept[EngineError](WdBatchSafety.validateResult(empty))
  }

  test("held publication leaves cancellation responsive and reports true protection") {
    val entered = new CountDownLatch(1); val release = new CountDownLatch(1)
    val control = new ExtrasBatchControl(() => Future.successful(InterruptionDecision(true, false, InterruptionProtection())),
      () => Future.successful(InterruptionProtection(enabled = true)))
    val published = Future(control.publishSafely { entered.countDown(); release.await(5, TimeUnit.SECONDS) })
    try
      assert(entered.await(2, TimeUnit.SECONDS))
      val decision = Await.result(control.cancel(), 1.second)
      assert(decision.blocked && decision.protection.enabled)
    finally release.countDown()
    Await.result(published, 3.seconds)
  }

  test("inactivity timeout requests targeted cancellation and reports unknown outcome") {
    val pending = Promise[Int]()
    val activity = new java.util.concurrent.atomic.AtomicLong(System.nanoTime())
    var cancelled = false
    val error = intercept[EngineError] {
      WdBatchSafety.awaitActive(pending.future, activity, 20.millis, () => cancelled = true)
    }
    assert(cancelled && error.message.contains("unknown"))
  }

  test("all published paths are validated before any gallery event") {
    val parent = java.nio.file.Files.createTempDirectory("wd-safety-paths")
    val item = java.nio.file.Files.createDirectory(parent.resolve("item-0000"))
    val valid = java.nio.file.Files.writeString(item.resolve("good.png"), "fake")
    val row = WdBatchItem("complete", "/source", Some("p"), Some("n"),
      Some(GenerationResult(Seq(GeneratedImage(valid.toString, 1), GeneratedImage(parent.resolve("missing.png").toString, 2)))))
    intercept[EngineError](WdBatchSafety.validatePaths(WdBatchResult(Seq(row), true), parent))
    val duplicate = row.copy(result = Some(GenerationResult(Seq(GeneratedImage(valid.toString, 1), GeneratedImage(valid.toString, 2)))))
    intercept[EngineError](WdBatchSafety.validatePaths(WdBatchResult(Seq(duplicate), true), parent))
  }

  test("stale timeout cannot restart a replacement supervisor process") {
    import FakeEngine.*
    val sup = new EngineSupervisor(() => launch())
    try
      Await.result(sup.startAsync(), 20.seconds)
      val old = sup.processSnapshot.get
      Await.result(sup.forceRestart(), 20.seconds)
      val replacement = sup.pid
      assert(replacement.nonEmpty)
      assert(!Await.result(sup.forceRestartIfCurrent(old, EngineError("engine_timeout", "stale")), 20.seconds))
      assertEquals(sup.pid, replacement)
      Await.result(sup.request("get_state"), 20.seconds)
    finally sup.close()
  }

  test("finished WD control cannot turn committed success into queue cancellation") {
    val control = new ExtrasBatchControl(
      () => Future.successful(InterruptionDecision(true, false, InterruptionProtection(), affected = false)),
      () => Future.successful(InterruptionProtection()))
    control.finish()
    val late = Await.result(control.cancel(), 2.seconds)
    assert(late.accepted && !late.affected)
    assert(!WdBatchSafety.markQueueCancelled(late, Some(control)))
    val prior = new ExtrasBatchControl(
      () => Future.successful(InterruptionDecision(true, false, InterruptionProtection(), affected = false)),
      () => Future.successful(InterruptionProtection()))
    val accepted = Await.result(prior.cancel(), 2.seconds)
    prior.finish()
    assert(WdBatchSafety.markQueueCancelled(accepted, Some(prior)))
  }

  test("terminal final-stage gate unaffected cancel does not mark batch cancelled") {
    val control = new ExtrasBatchControl(
      () => Future.successful(InterruptionDecision(true, false, InterruptionProtection(), affected = false)),
      () => Future.successful(InterruptionProtection()))
    val gate = new TemporalLabCancelGate("owner", _ => () => Future.successful(
      InterruptionDecision(true, false, InterruptionProtection(), affected = false)))
    assert(Await.result(control.attach(gate, finalStage = true), 2.seconds))
    gate.dispatched(); gate.completed()
    val decision = Await.result(control.cancel(), 2.seconds)
    assert(decision.accepted && !decision.affected && !control.cancellationAccepted)
  }

  test("WD timeout while queued does not kill other owner on same worker") {
    import FakeEngine.*
    import sdfx.core.model.GenerationParams
    val sup = new EngineSupervisor(() => launch())
    try
      Await.result(sup.startAsync(), 20.seconds)
      val client = new EngineClient(sup)
      val long = client.generate(GenerationRequest(GenerationParams(steps = 20, width = 64, height = 64,
        extra = Map("step_delay" -> "0.1", "allow_no_model" -> "true")), java.nio.file.Files.createTempDirectory("other-owner").toString), GenerationMode.Txt2Img)
      assert(FakeEngine.eventually(5.seconds)(Await.result(sup.request("get_state"), 5.seconds)("requests").arr.exists(_.str == "txt2img")))
      val pid = sup.pid
      val request = WdBatchPrepareRequest("/images", GenerationRequest(GenerationParams(), ""), "/out", "result", "token")
      val wd = client.wdBatch(request)
      assert(FakeEngine.eventually(5.seconds)(sup.processSnapshot.exists(_.pendingTypes.contains("batch_interrogate_prepare"))))
      val activity = new java.util.concurrent.atomic.AtomicLong(System.nanoTime())
      intercept[EngineError] {
        WdBatchSafety.awaitActive(wd.result, activity, 20.millis, () => WdBatchSafety.timeoutAction(wd, wd.control, sup))
      }
      assert(Await.result(long.result, 20.seconds).images.nonEmpty)
      assertEquals(sup.pid, pid)
      intercept[EngineError](Await.result(wd.result, 20.seconds))
    finally sup.close()
  }

  Seq("active-gate", "startup-idle").foreach { mode =>
    test(s"cooperative WD $mode cancellation does not restart a still-active worker") {
      import FakeEngine.*
      import sdfx.core.model.GenerationParams
      val sup = new EngineSupervisor(() => launch(), interruptGrace = 100.millis)
      try
        Await.result(sup.startAsync(), 20.seconds)
        val req = GenerationRequest(GenerationParams(steps = 30, width = 64, height = 64,
          extra = Map("step_delay" -> "0.08", "ignore_interrupt" -> "true", "allow_no_model" -> "true")),
          java.nio.file.Files.createTempDirectory("wd-cooperative").toString)
        val active = sup.requestExtras(Protocol.Types.Txt2Img, Json.toJs(req), _ => (), "wd-owner",
          () => Future.successful(true), () => ())
        assert(eventually(5.seconds)(Await.result(sup.request("get_state"), 5.seconds)("requests").arr.exists(_.str == "txt2img")))
        val pid = sup.pid
        val other = sup.requestExtras("echo", ujson.Obj("value" -> "other"), _ => (), "other-owner",
          () => Future.successful(true), () => ())
        val decision = if mode == "active-gate" then
          Await.result(sup.captureInterrupt(Some("wd-owner"), restartOnTimeout = false)(), 5.seconds)
        else Await.result(sup.interruptAfterStartupOwned("wd-owner", restartOnTimeout = false), 5.seconds)
        assert(decision.accepted && decision.affected)
        Thread.sleep(450) // exceed the 100ms forced-restart grace while WD remains active
        assertEquals(sup.pid, pid)
        assert(!active.isCompleted, "fixture must remain pending beyond the restart grace")
        Await.result(active, 20.seconds)
        Await.result(other, 20.seconds)
        assertEquals(sup.pid, pid)
      finally sup.close()
    }
  }
