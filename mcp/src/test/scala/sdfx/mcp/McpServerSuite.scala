package sdfx.mcp

import java.util.concurrent.{CountDownLatch, TimeUnit}
import scala.concurrent.duration.*
import scala.concurrent.{Await, ExecutionContext, Future}

/** Protocol level tests with in-memory tools (no engine). */
class McpServerSuite extends munit.FunSuite:
  import TestEnv.*
  given ExecutionContext = ExecutionContext.global

  private val cancelled = new CountDownLatch(1)

  private val echo = new McpTool:
    def name = "echo"
    def description = "echo"
    def inputSchema = Schema.obj(Seq("text" -> Schema.str()), Seq("text"))
    def call(args: ujson.Obj, ctx: ToolContext) =
      val a = Args(args, Seq("text"))
      ToolResult.text(a.reqStr("text"))

  private val slow = new McpTool:
    def name = "slow"
    def description = "reports progress then waits for cancellation"
    def inputSchema = Schema.empty
    def call(args: ujson.Obj, ctx: ToolContext) =
      ctx.progress(1, Some(3), "one")
      ctx.progress(1, Some(3), "dup dropped")
      ctx.progress(2, Some(3), "two")
      ctx.onCancel(() => cancelled.countDown())
      val end = System.nanoTime() + 10.seconds.toNanos
      while !ctx.isCancelled && System.nanoTime() < end do Thread.sleep(10)
      ToolResult.text(if ctx.isCancelled then "cancelled" else "finished")

  private val boom = new McpTool:
    def name = "boom"
    def description = "throws"
    def inputSchema = Schema.empty
    def call(args: ujson.Obj, ctx: ToolContext) = throw new RuntimeException("kaboom")

  private def server = new McpServer(Seq(echo, slow, boom))

  test("initialize echoes a supported protocol version, otherwise offers the latest") {
    val s = server
    for v <- McpServer.SupportedVersions do
      val r = s.handle(request("initialize", ujson.Obj("protocolVersion" -> v, "capabilities" -> ujson.Obj(), "clientInfo" -> ujson.Obj("name" -> "t", "version" -> "1"))), _ => ()).get
      assertEquals(r("result")("protocolVersion").str, v)
    val r = s.handle(request("initialize", ujson.Obj("protocolVersion" -> "1999-01-01")), _ => ()).get("result")
    assertEquals(r("protocolVersion").str, "2025-06-18")
    assertEquals(r("capabilities")("tools")("listChanged").bool, false)
    assertEquals(r("serverInfo")("name").str, "stablediffusion-fx")
    assert(r("instructions").str.contains("txt2img"))
    assertEquals(s.handle(ujson.Obj("jsonrpc" -> "2.0", "method" -> "notifications/initialized"), _ => ()), None)
    assert(s.initialized)
  }

  test("ping, unknown method, invalid request, unknown tool") {
    val s = server
    assertEquals(s.handle(request("ping", id = ujson.Str("p1")), _ => ()).get("result"), ujson.Obj())
    val nf = s.handle(request("nope/nothing"), _ => ()).get
    assertEquals(nf("error")("code").num.toInt, -32601)
    val bad = s.handle(ujson.Obj("jsonrpc" -> "1.0", "id" -> 3, "method" -> "ping"), _ => ()).get
    assertEquals(bad("error")("code").num.toInt, -32600)
    assertEquals(bad("id").num.toInt, 3)
    assertEquals(s.handle(ujson.Str("x"), _ => ()).get("error")("code").num.toInt, -32600)
    val unknownTool = s.handle(callReq("missing", ujson.Obj()), _ => ()).get
    assertEquals(unknownTool("error")("code").num.toInt, -32602)
    val badArgs = s.handle(request("tools/call", ujson.Obj("name" -> "echo", "arguments" -> ujson.Arr())), _ => ()).get
    assertEquals(badArgs("error")("code").num.toInt, -32602)
  }

  test("tool errors become isError results") {
    val s = server
    val missing = s.handle(callReq("echo", ujson.Obj()), _ => ()).get("result")
    assert(missing("isError").bool)
    assert(textOf(missing).contains("text"))
    val unknownArg = s.handle(callReq("echo", ujson.Obj("text" -> "a", "bogus" -> 1)), _ => ()).get("result")
    assert(unknownArg("isError").bool)
    assert(textOf(unknownArg).contains("bogus"))
    val thrown = s.handle(callReq("boom", ujson.Obj()), _ => ()).get("result")
    assert(thrown("isError").bool)
    assert(textOf(thrown).contains("kaboom"))
    val ok = s.handle(callReq("echo", ujson.Obj("text" -> "hi")), _ => ()).get("result")
    assertEquals(ok("isError").bool, false)
    assertEquals(textOf(ok), "hi")
  }

  test("batches: responses for requests only, empty batch rejected") {
    val s = server
    val r = s.handle(ujson.Arr(request("ping", id = 1), ujson.Obj("jsonrpc" -> "2.0", "method" -> "notifications/initialized"), request("nope", id = 2)), _ => ()).get
    assertEquals(r.arr.map(_("id").num.toInt).toSeq, Seq(1, 2))
    assert(r.arr(1).obj.contains("error"))
    assertEquals(s.handle(ujson.Arr(), _ => ()).get("error")("code").num.toInt, -32600)
    assertEquals(s.handle(ujson.Arr(ujson.Obj("jsonrpc" -> "2.0", "method" -> "notifications/initialized")), _ => ()), None)
  }

  test("progress notifications are strictly increasing and cancellation suppresses the reply") {
    val s = server
    val rec = new Recorder
    val f = Future(s.handle(callReq("slow", ujson.Obj(), id = ujson.Str("req-9"), progressToken = Some(ujson.Str("tok"))), rec))
    assert(eventually()(rec.all.size >= 2))
    s.handle(ujson.Obj("jsonrpc" -> "2.0", "method" -> "notifications/cancelled", "params" -> ujson.Obj("requestId" -> "req-9", "reason" -> "user")), _ => ())
    assert(cancelled.await(5, TimeUnit.SECONDS))
    assertEquals(Await.result(f, 15.seconds), None)
    val ps = rec.all.map(_("params"))
    assert(rec.all.forall(_("method").str == "notifications/progress"))
    assertEquals(ps.map(_("progress").num), Seq(1.0, 2.0))
    assert(ps.forall(_("progressToken").str == "tok"))
    assertEquals(ps.head("total").num, 3.0)
  }

  test("duplicate in-flight request ids are rejected (1 and 1.0 are the same id)") {
    val s = server
    val first = s.accept(callReq("echo", ujson.Obj("text" -> "a"), id = ujson.Num(7)), _ => ())
    assert(first.isRight)
    val dup = s.accept(callReq("echo", ujson.Obj("text" -> "b"), id = ujson.Num(7.0)), _ => ())
    assertEquals(dup.left.toOption.flatten.get("error")("code").num.toInt, -32600)
    assertEquals(first.toOption.get().get("result")("content")(0)("text").str, "a")
    // after completion the id is free again
    assertEquals(s.handle(callReq("echo", ujson.Obj("text" -> "c"), id = ujson.Num(7)), _ => ()).get("result")("content")(0)("text").str, "c")
  }

  test("a cancel that arrives before the call is registered still cancels it") {
    val s = server
    s.handle(ujson.Obj("jsonrpc" -> "2.0", "method" -> "notifications/cancelled", "params" -> ujson.Obj("requestId" -> "early")), _ => ())
    val t0 = System.nanoTime()
    assertEquals(s.handle(callReq("slow", ujson.Obj(), id = ujson.Str("early")), _ => ()), None)
    assert((System.nanoTime() - t0) < 5.seconds.toNanos)
  }

  test("no progress after the call finished; fatal errors still get a reply; params null accepted") {
    @volatile var leaked: ToolContext = null
    val leaky = new McpTool:
      def name = "leaky"
      def description = "keeps its context"
      def inputSchema = Schema.empty
      def call(args: ujson.Obj, ctx: ToolContext) =
        leaked = ctx
        ctx.progress(1, None, "")
        ToolResult.text("done")
    val fatal = new McpTool:
      def name = "fatal"
      def description = "stack overflow"
      def inputSchema = Schema.empty
      def call(args: ujson.Obj, ctx: ToolContext) = throw new StackOverflowError()
    val s = new McpServer(Seq(leaky, fatal))
    val rec = new Recorder
    s.handle(callReq("leaky", ujson.Obj(), progressToken = Some(ujson.Str("p"))), rec)
    assertEquals(rec.all.size, 1)
    leaked.progress(2, None, "late")
    assertEquals(rec.all.size, 1)
    val r = s.handle(callReq("fatal", ujson.Obj()), _ => ()).get
    assertEquals(r("error")("code").num.toInt, -32603)
    assert(s.handle(ujson.Obj("jsonrpc" -> "2.0", "id" -> 1, "method" -> "ping", "params" -> ujson.Null), _ => ()).get.obj.contains("result"))
  }

  test("deeply nested JSON is rejected without a stack overflow") {
    assert(JsonRpc.parseText("[" * 100000 + "]" * 100000).isLeft)
    assert(JsonRpc.parseText("""{"a":"[[[[[[[["}""").isRight)
  }

  test("tools/list descriptors are valid JSON schema objects") {
    val tools = server.handle(request("tools/list"), _ => ()).get("result")("tools").arr
    assertEquals(tools.map(_("name").str).toSeq, Seq("echo", "slow", "boom"))
    tools.foreach(t => assertEquals(t("inputSchema")("type").str, "object"))
  }
