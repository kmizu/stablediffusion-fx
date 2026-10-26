package sdfx.mcp

import sdfx.core.model.GenerationParams
import sdfx.engine.{Job, JobCancelledException}
import sdfx.engine.JobEvent
import scala.concurrent.Await
import scala.concurrent.{ExecutionContext, Future}
import scala.concurrent.duration.*
import java.util.concurrent.ConcurrentLinkedQueue
import scala.jdk.CollectionConverters.*

class PreventInterruptionMcpSuite extends munit.FunSuite:
  import TestEnv.*
  private given ExecutionContext = ExecutionContext.global
  override val munitTimeout: Duration = 90.seconds
  test("combined MCP interrupt truthfully reports blocked active job and separately cleared pending jobs") {
    val home = tempHome()
    val s = services(home)
    try
      s.start()
      val server = new McpServer(new McpTools(s).all)
      def invoke(name: String, args: ujson.Obj): ujson.Value =
        val envelope = server.handle(callReq(name, args), _ => ()).getOrElse(fail(s"No reply for $name"))
        assert(!envelope.obj.contains("error") && envelope.obj.contains("result"), s"$name protocol error: $envelope")
        val reply = envelope("result")
        assert(!reply("isError").bool, reply.toString)
        reply("structuredContent")
      assertEquals(invoke("interruption_protection", ujson.Obj("enabled" -> true))("enabled").bool, true)
      val first = Job(GenerationParams(steps = 25, width = 64, height = 64, extra = Map("step_delay" -> ".03")))
      val complete = s.queue.submit(first)
      assert(eventually()(s.queue.current.exists(_.id == first.id)))
      val waiting = s.queue.submit(Job(first.params))
      val result = invoke("interrupt", ujson.Obj("clear_queue" -> true))
      assertEquals(result("blocked").bool, true)
      assertEquals(result("interrupted").bool, false)
      assertEquals(result("cleared").num.toInt, 1)
      intercept[JobCancelledException](Await.result(waiting, 10.seconds))
      assertEquals(Await.result(complete, 10.seconds).images.size, 1)
      assertEquals(invoke("interruption_protection", ujson.Obj())("enabled").bool, true)
      assertEquals(invoke("interruption_protection", ujson.Obj("enabled" -> false))("enabled").bool, false)
    finally
      s.shutdown()
      deleteTree(home.root)
  }
  test("running async cancel_job reports blocked and completion remains pollable") {
    val home = tempHome()
    val s = services(home)
    try
      s.start()
      val server = new McpServer(new McpTools(s).all)
      def invoke(name: String, args: ujson.Obj): ujson.Value =
        val reply = server.handle(callReq(name, args), _ => ()).get
        assert(!reply.obj.contains("error"), s"$name: $reply")
        val result = reply("result")
        assert(!result("isError").bool, result.toString)
        result("structuredContent")
      invoke("interruption_protection", ujson.Obj("enabled" -> true))
      val added = invoke("txt2img_async", ujson.Obj("prompt" -> "protected", "steps" -> 40, "width" -> 64,
        "height" -> 64, "extra" -> ujson.Obj("step_delay" -> ".02")))
      val id = added("job_id").str
      assert(eventually()(s.queue.current.exists(_.id == id)))
      val blocked = invoke("cancel_job", ujson.Obj("job_id" -> id))
      assert(blocked.obj.get("blocked").contains(ujson.True), blocked.toString)
      assertEquals(blocked("cancelled").bool, false)
      val done = invoke("get_job", ujson.Obj("job_id" -> id, "wait_seconds" -> 10))
      assertEquals(done("status").str, "done")
      assertEquals(done("images").arr.size, 1)
    finally
      s.shutdown()
      deleteTree(home.root)
  }
  test("blocked transport cancellation is not deferred until the user unlocks") {
    val home = tempHome()
    val s = services(home)
    val events = new ConcurrentLinkedQueue[JobEvent]()
    val logs = new ConcurrentLinkedQueue[String]()
    val remove = s.queue.addListener(events.add(_))
    val removeLog = s.addLogListener(logs.add(_))
    try
      s.start()
      val server = new McpServer(new McpTools(s).all)
      val enabled = Await.result(s.client.raw("interruption_protection", ujson.Obj("enabled" -> true)), 10.seconds)
      assert(enabled("enabled").bool)
      val request = callReq("txt2img", ujson.Obj("prompt" -> "notification protected", "steps" -> 40,
        "width" -> 64, "height" -> 64, "extra" -> ujson.Obj("step_delay" -> ".02")), id = ujson.Str("active"))
      val accepted = server.accept(request, _ => ())
      assert(accepted.isRight, accepted.toString)
      val reply = Future(accepted.toOption.get.apply())
      assert(eventually()(events.asScala.exists(_.isInstanceOf[JobEvent.Progress])))
      server.handle(ujson.Obj("jsonrpc" -> "2.0", "method" -> "notifications/cancelled",
        "params" -> ujson.Obj("requestId" -> "active")), _ => ())
      assert(eventually()(logs.asScala.exists(_.contains("Interruptions are blocked"))))
      Await.result(s.client.raw("interruption_protection", ujson.Obj("enabled" -> false)), 10.seconds)
      assertEquals(Await.result(reply, 10.seconds), None) // transport stopped waiting; generation still completes
      assert(eventually()(events.asScala.exists(_.isInstanceOf[JobEvent.JobFinished])))
      assert(!events.asScala.exists(_.isInstanceOf[JobEvent.JobCancelled]))
      assert(!events.asScala.exists(_.isInstanceOf[JobEvent.JobFailed]))
    finally
      remove()
      removeLog()
      s.shutdown()
      deleteTree(home.root)
  }
  test("global MCP interrupt still stops existing non-queue work after unlocking") {
    val home = tempHome()
    val s = services(home)
    try
      s.start()
      val server = new McpServer(new McpTools(s).all)
      def invoke(name: String, args: ujson.Obj): ujson.Value =
        val envelope = server.handle(callReq(name, args), _ => ()).get
        assert(!envelope.obj.contains("error"), envelope.toString)
        val result = envelope("result")
        assert(!result("isError").bool, result.toString)
        result("structuredContent")
      invoke("interruption_protection", ujson.Obj("enabled" -> true))
      val progress = new java.util.concurrent.atomic.AtomicBoolean(false)
      val request = sdfx.core.protocol.GenerationRequest(GenerationParams(seed = 81, steps = 40, width = 64, height = 64,
        extra = Map("step_delay" -> ".02")), home.tmpDir(s.settings).toString)
      val handle = s.client.generate(request, sdfx.engine.GenerationMode.Txt2Img, {
        case _: sdfx.core.protocol.EngineEvent.Progress => progress.set(true)
        case _ => ()
      })
      assert(eventually()(progress.get()))
      assert(s.queue.isIdle && !handle.result.isCompleted)
      val blocked = invoke("interrupt", ujson.Obj())
      assert(blocked("blocked").bool && !blocked("interrupted").bool)
      invoke("interruption_protection", ujson.Obj("enabled" -> false))
      val admitted = invoke("interrupt", ujson.Obj())
      assert(admitted("interrupted").bool && !admitted("blocked").bool,
        s"Global Off control lost existing non-queue work: $admitted")
      assertEquals(intercept[sdfx.core.protocol.EngineError](Await.result(handle.result, 10.seconds)).code, "interrupted")
    finally
      s.shutdown()
      deleteTree(home.root)
  }
