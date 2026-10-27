package sdfx.engine

import munit.FunSuite
import sdfx.core.protocol.{EngineError, InterruptionDecision, InterruptionProtection, TemporalLabDeflicker}
import java.util.concurrent.{CountDownLatch, TimeUnit}
import scala.concurrent.{Await, ExecutionContext, Future, Promise}
import scala.concurrent.duration.*

class TemporalLabCancellationSuite extends FunSuite:
  given ExecutionContext = ExecutionContext.global
  private val enabled = InterruptionProtection(enabled = true)
  private val disabled = InterruptionProtection(enabled = false)
  private val accepted = InterruptionDecision(accepted = true, blocked = false, protection = disabled, affected = true)
  private val blocked = InterruptionDecision(accepted = false, blocked = true, protection = enabled, affected = false)

  test("blocking writer is dispatched away from the caller thread") {
    val entered = new CountDownLatch(1)
    val release = new CountDownLatch(1)
    val gate = new TemporalLabCancelGate("owner", _ => () => {
      entered.countDown()
      if !release.await(5, TimeUnit.SECONDS) then throw new IllegalStateException("writer was not released")
      Future.successful(accepted)
    })
    gate.dispatched()
    try
      val result = gate.cancel()
      assert(entered.await(2, TimeUnit.SECONDS))
      assert(!result.isCompleted)
      release.countDown()
      assertEquals(Await.result(result, 3.seconds), accepted)
    finally release.countDown()
  }

  test("blocked admission is surfaced and later retry may be accepted") {
    var count = 0
    val gate = new TemporalLabCancelGate("owner", _ => () => {
      count += 1
      Future.successful(if count == 1 then blocked else accepted)
    })
    gate.dispatched()
    assertEquals(Await.result(gate.cancel(), 3.seconds), blocked)
    assertEquals(Await.result(gate.cancel(), 3.seconds), accepted)
    assertEquals(count, 2)
  }

  test("pre-start cancellation is admitted before the owned envelope is sent") {
    val captured = Promise[String]()
    val gate = new TemporalLabCancelGate("this-owner", owner => () => {
      captured.trySuccess(owner)
      Future.successful(accepted)
    })
    val result = gate.cancel()
    assert(!result.isCompleted)
    assert(!captured.isCompleted)
    val admission = gate.beforeDispatch()
    assertEquals(Await.result(captured.future, 3.seconds), "this-owner")
    assert(Await.result(result, 3.seconds).accepted)
    assert(!Await.result(admission, 3.seconds))
  }

  test("cancellation during envelope write cannot overtake the envelope") {
    val writes = new java.util.concurrent.ConcurrentLinkedQueue[String]()
    val gate = new TemporalLabCancelGate("owner", _ => () => {
      writes.add("interrupt")
      Future.successful(accepted)
    })
    assert(Await.result(gate.beforeDispatch(), 3.seconds))
    writes.add("write-begin")
    val result = gate.cancel()
    assert(!result.isCompleted)
    writes.add("write-end")
    gate.dispatched()
    Await.result(result, 3.seconds)
    assertEquals(writes.toArray.toSeq.map(_.toString), Seq("write-begin", "write-end", "interrupt"))
  }

  test("terminal request never interrupts a subsequent owner") {
    var captures = 0
    val gate = new TemporalLabCancelGate("old-owner", _ => () => {
      captures += 1
      Future.successful(accepted)
    })
    gate.dispatched()
    gate.completed()
    val result = Await.result(gate.cancel(), 3.seconds)
    assert(!result.affected)
    assertEquals(captures, 0)
  }

  test("capture always receives this request's unique owner") {
    val owners = new java.util.concurrent.ConcurrentLinkedQueue[String]()
    val gate = new TemporalLabCancelGate("mine", owner => () => {
      owners.add(owner)
      Future.successful(accepted)
    })
    gate.dispatched()
    Await.result(gate.cancel(), 3.seconds)
    assertEquals(owners.toArray.toSeq.map(_.toString), Seq("mine"))
  }

  test("blocked pre-start cancellation dispatches work and a later retry can succeed") {
    var count = 0
    val gate = new TemporalLabCancelGate("owner", _ => () => {
      count += 1
      Future.successful(if count == 1 then blocked else accepted)
    })
    val first = gate.cancel()
    assert(Await.result(gate.beforeDispatch(), 3.seconds))
    assertEquals(Await.result(first, 3.seconds), blocked)
    gate.dispatched()
    assertEquals(Await.result(gate.cancel(), 3.seconds), accepted)
    assertEquals(count, 2)
  }

  test("cancel while actual supervisor startup is held prevents work reaching worker") {
    import FakeEngine.*
    val started = new CountDownLatch(1)
    val release = new CountDownLatch(1)
    val sup = new EngineSupervisor(() => {
      started.countDown()
      if !release.await(10, TimeUnit.SECONDS) then throw new IllegalStateException("startup not released")
      launch()
    })
    try
      val startup = sup.startAsync()
      assert(started.await(3, TimeUnit.SECONDS))
      val handle = new EngineClient(sup).temporalLab(TemporalLabDeflicker("frames", "output", "result"))
      val decision = handle.interrupt()
      assert(!decision.isCompleted)
      release.countDown()
      Await.result(startup, 20.seconds)
      val admitted = Await.result(decision, 20.seconds)
      assert(admitted.accepted && !admitted.blocked && admitted.affected)
      val error = intercept[EngineError](Await.result(handle.result, 20.seconds))
      assert(error.isInterrupted)
      val state = Await.result(sup.request("get_state"), 20.seconds)
      assert(!state("requests").arr.exists(_.str == "temporal_lab"))
    finally
      release.countDown()
      sup.close()
  }

  test("too-late publication leaves the request in flight without reporting protection") {
    val late = InterruptionDecision(accepted = false, blocked = false, protection = disabled,
      affected = false, message = Some("Output publication is in progress"), tooLate = true)
    val gate = new TemporalLabCancelGate("owner", _ => () => Future.successful(late))
    gate.dispatched()
    val decision = Await.result(gate.cancel(), 3.seconds)
    assert(decision.tooLate && !decision.accepted && !decision.blocked && !decision.affected)
  }
