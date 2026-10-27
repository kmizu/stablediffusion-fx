package sdfx.engine

import munit.FunSuite
import sdfx.core.protocol.{InterruptionDecision, InterruptionProtection}
import scala.concurrent.{Await, Future, Promise}
import scala.concurrent.duration.*
import scala.collection.mutable.ArrayBuffer
import java.util.concurrent.{CountDownLatch, TimeUnit}
import sdfx.core.model.GenerationParams
import sdfx.core.protocol.{GenerationRequest, WdBatchPrepareRequest, EngineError}

class WdBatchFlowSuite extends FunSuite:
  private val accepted = InterruptionDecision(true, false, InterruptionProtection(), affected = true)

  test("prepare token flows to execute and response remains partial") {
    val calls = ArrayBuffer.empty[(String, ujson.Value)]
    val control = new ExtrasBatchControl(() => Future.successful(accepted))
    val transport = new ExtrasBatchHandle(control, (kind, value) =>
      calls += ((kind, value))
      if kind == "batch_interrogate_prepare" then Future.successful(ujson.Obj("token" -> ("a" * 32), "image_count" -> 1, "partial" -> true))
      else Future.successful(ujson.Obj("rows" -> ujson.Arr(), "partial" -> true))
    )
    val result = Await.result(WdBatchFlow.run(transport, ujson.Obj("images_root" -> "original")), 2.seconds)
    assertEquals(calls.map(_._1).toSeq, Seq("batch_interrogate_prepare", "batch_interrogate_execute"))
    assertEquals(calls(1)._2("token").str, "a" * 32)
    assert(result("partial").bool)
  }

  test("accepted cancel between prepare and execute prevents execute") {
    val pending = Promise[ujson.Value]()
    val calls = ArrayBuffer.empty[String]
    val control = new ExtrasBatchControl(() => Future.successful(accepted))
    val transport = new ExtrasBatchHandle(control, (kind, value) =>
      calls += kind
      if kind == "batch_interrogate_prepare" then pending.future else Future.successful(ujson.Obj())
    )
    val result = WdBatchFlow.run(transport, ujson.Obj())
    Await.result(transport.cancel(), 2.seconds)
    pending.success(ujson.Obj("token" -> ("a" * 32), "image_count" -> 1, "partial" -> true))
    intercept[Throwable](Await.result(result, 2.seconds))
    assertEquals(calls.toSeq, Seq("batch_interrogate_prepare"))
  }

  test("actual supervisor startup cancellation prevents any WD worker request") {
    import FakeEngine.*
    val started = new CountDownLatch(1); val release = new CountDownLatch(1)
    val sup = new EngineSupervisor(() => {
      started.countDown()
      if !release.await(10, TimeUnit.SECONDS) then throw new IllegalStateException("startup not released")
      launch()
    })
    try
      val starting = sup.startAsync()
      assert(started.await(3, TimeUnit.SECONDS))
      val request = WdBatchPrepareRequest("/images", GenerationRequest(GenerationParams(), ""), "/out", "result", "token")
      val handle = new EngineClient(sup).wdBatch(request)
      val cancel = handle.interrupt()
      assert(!cancel.isCompleted)
      release.countDown()
      Await.result(starting, 20.seconds)
      assert(Await.result(cancel, 20.seconds).accepted)
      assert(intercept[EngineError](Await.result(handle.result, 20.seconds)).isInterrupted)
      val state = Await.result(sup.request("get_state"), 20.seconds)
      assert(!state("requests").arr.exists(x => x.str.startsWith("batch_interrogate_")))
    finally
      release.countDown()
      sup.close()
  }

  test("queue may supply its owner and pre-cancelled shared gate") {
    import FakeEngine.*
    val sup = new EngineSupervisor(() => launch())
    try
      Await.result(sup.startAsync(), 20.seconds)
      val control = new ExtrasBatchControl(() => Future.successful(accepted))
      Await.result(control.cancel(), 3.seconds)
      val request = WdBatchPrepareRequest("/images", GenerationRequest(GenerationParams(), ""), "/out", "result", "token")
      val handle = new EngineClient(sup).wdBatchOwned(request, "queue-owner", control)
      assert(intercept[EngineError](Await.result(handle.result, 20.seconds)).isInterrupted)
      val state = Await.result(sup.request("get_state"), 20.seconds)
      assert(!state("requests").arr.exists(x => x.str.startsWith("batch_interrogate_")))
    finally sup.close()
  }
