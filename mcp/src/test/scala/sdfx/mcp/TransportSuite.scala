package sdfx.mcp

import sdfx.engine.AppServices

import java.io.{BufferedReader, InputStreamReader, PipedInputStream, PipedOutputStream}
import java.net.URI
import java.net.http.{HttpClient, HttpRequest, HttpResponse}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import java.util.concurrent.{ConcurrentHashMap, LinkedBlockingQueue, TimeUnit}
import scala.concurrent.duration.*

class TransportSuite extends munit.FunSuite:
  import TestEnv.*

  override val munitTimeout: Duration = 180.seconds

  private lazy val home = tempHome()
  private lazy val services: AppServices =
    val s = TestEnv.services(home)
    s.start()
    s
  private lazy val tools = new McpTools(services).all

  override def afterAll(): Unit =
    services.shutdown()
    deleteTree(home.root)

  // ---- stdio ---------------------------------------------------------------------------------------------------

  test("stdio end to end: concurrent requests, ping during a long call, parse errors, ordered lines") {
    val toServer = new PipedOutputStream()
    val serverIn = new PipedInputStream(toServer, 1 << 20)
    val fromServer = new PipedInputStream(1 << 22)
    val serverOut = new PipedOutputStream(fromServer)
    val transport = new StdioTransport(serverIn, serverOut, new McpServer(tools))
    val t = new Thread(() => transport.run(), "stdio-test")
    t.setDaemon(true)
    t.start()

    val lines = new LinkedBlockingQueue[ujson.Value]()
    val reader = new Thread(() => {
      val r = new BufferedReader(new InputStreamReader(fromServer, StandardCharsets.UTF_8))
      try
        var l = r.readLine()
        while l != null do
          lines.put(ujson.read(l))
          l = r.readLine()
      catch case _: Exception => ()
    })
    reader.setDaemon(true)
    reader.start()

    def send(s: String): Unit =
      toServer.write((s + "\n").getBytes(StandardCharsets.UTF_8))
      toServer.flush()
    val byId = new ConcurrentHashMap[String, ujson.Value]()
    def awaitId(id: String, timeout: FiniteDuration = 60.seconds): ujson.Value =
      val end = System.nanoTime() + timeout.toNanos
      while !byId.containsKey(id) && System.nanoTime() < end do
        Option(lines.poll(100, TimeUnit.MILLISECONDS)).foreach { v =>
          v.obj.get("id").filterNot(_ == ujson.Null).foreach(i => byId.put(ujson.write(i), v))
          if v.obj.get("id").contains(ujson.Null) then byId.put("null", v)
        }
      Option(byId.get(id)).getOrElse(fail(s"no response for $id"))

    send(ujson.write(request("initialize", ujson.Obj("protocolVersion" -> "2025-03-26", "capabilities" -> ujson.Obj(), "clientInfo" -> ujson.Obj("name" -> "t", "version" -> "0")), ujson.Num(1))))
    assertEquals(awaitId("1")("result")("protocolVersion").str, "2025-03-26")
    send("""{"jsonrpc":"2.0","method":"notifications/initialized"}""")
    send("{not json")
    assertEquals(awaitId("null")("error")("code").num.toInt, -32700)
    send(ujson.write(request("tools/list", id = ujson.Num(2))))
    assert(awaitId("2")("result")("tools").arr.size >= 15)
    // a long generation, then a ping that must be answered first
    send(ujson.write(callReq("txt2img", ujson.Obj("prompt" -> "日本語 prompt", "steps" -> 20, "width" -> 64, "height" -> 64,
      "extra" -> ujson.Obj("step_delay" -> "0.05")), ujson.Str("gen"))))
    Thread.sleep(200)
    send(ujson.write(request("ping", id = ujson.Str("ping"))))
    awaitId("\"ping\"", 5.seconds)
    assert(!byId.containsKey("\"gen\""), "ping must not wait for the generation")
    val gen = awaitId("\"gen\"")
    assert(!gen("result")("isError").bool, gen.toString)
    val p = Paths.get(gen("result")("structuredContent")("images")(0)("path").str)
    assert(Files.isRegularFile(p))
    assert(gen("result")("structuredContent")("images")(0)("infotext").str.startsWith("日本語 prompt"))
    send(ujson.write(request("no/such", id = ujson.Num(5))))
    assertEquals(awaitId("5")("error")("code").num.toInt, -32601)
    toServer.close()
    t.join(10000)
    assert(!t.isAlive, "transport stops at EOF")
  }

  // ---- HTTP ----------------------------------------------------------------------------------------------------

  private val client = HttpClient.newBuilder().version(HttpClient.Version.HTTP_1_1).build()

  private def post(url: String, body: String, headers: (String, String)*): HttpResponse[String] =
    val b = HttpRequest.newBuilder(URI.create(url)).POST(HttpRequest.BodyPublishers.ofString(body))
      .header("Content-Type", "application/json").header("Accept", "application/json, text/event-stream")
    headers.foreach((k, v) => b.header(k, v))
    client.send(b.build(), HttpResponse.BodyHandlers.ofString())

  private def initBody = ujson.write(request("initialize", ujson.Obj("protocolVersion" -> "2025-06-18", "capabilities" -> ujson.Obj(),
    "clientInfo" -> ujson.Obj("name" -> "http-test", "version" -> "0")), ujson.Num(1)))

  test("http: session flow, 400/404 without valid session, 202 notifications, 405 GET, 403 bad origin, DELETE") {
    val t = new HttpTransport(() => new McpServer(tools), 0, None)
    try
      val url = t.url
      val init = post(url, initBody)
      assertEquals(init.statusCode, 200)
      assert(init.headers.firstValue("Content-Type").get.startsWith("application/json"))
      val sid = init.headers.firstValue("Mcp-Session-Id").orElseThrow()
      assertEquals(ujson.read(init.body)("result")("protocolVersion").str, "2025-06-18")

      assertEquals(post(url, ujson.write(request("ping", id = ujson.Num(2)))).statusCode, 400)
      assertEquals(post(url, ujson.write(request("ping", id = ujson.Num(2))), "Mcp-Session-Id" -> "bogus").statusCode, 404)
      val note = post(url, """{"jsonrpc":"2.0","method":"notifications/initialized"}""", "Mcp-Session-Id" -> sid, "MCP-Protocol-Version" -> "2025-06-18")
      assertEquals(note.statusCode, 202)
      assertEquals(note.body, "")
      val ping = post(url, ujson.write(request("ping", id = ujson.Num(3))), "Mcp-Session-Id" -> sid)
      assertEquals(ping.statusCode, 200)
      assertEquals(ujson.read(ping.body)("id").num.toInt, 3)
      assertEquals(post(url, "{oops", "Mcp-Session-Id" -> sid).statusCode, 400)
      assertEquals(post(url, ujson.write(request("ping")), "Mcp-Session-Id" -> sid, "MCP-Protocol-Version" -> "1900-01-01").statusCode, 400)

      val gen = post(url, ujson.write(callReq("txt2img", ujson.Obj("prompt" -> "over http", "steps" -> 2, "width" -> 64, "height" -> 64), ujson.Num(4))), "Mcp-Session-Id" -> sid)
      assertEquals(gen.statusCode, 200)
      val res = ujson.read(gen.body)("result")
      assert(!res("isError").bool, res.toString)
      assert(res("content").arr.exists(_("type").str == "image"))
      // the job went through the shared queue -> visible in the gallery
      assert(services.gallery.search("over http").nonEmpty)

      val get = client.send(HttpRequest.newBuilder(URI.create(url)).GET().header("Mcp-Session-Id", sid).build(), HttpResponse.BodyHandlers.ofString())
      assertEquals(get.statusCode, 405)
      assertEquals(post(url, initBody, "Origin" -> "https://evil.example.com").statusCode, 403)
      assertEquals(post(url, initBody, "Origin" -> "http://localhost:3000").statusCode, 200)

      val del = client.send(HttpRequest.newBuilder(URI.create(url)).DELETE().header("Mcp-Session-Id", sid).build(), HttpResponse.BodyHandlers.ofString())
      assertEquals(del.statusCode, 200)
      assertEquals(post(url, ujson.write(request("ping")), "Mcp-Session-Id" -> sid).statusCode, 404)
    finally t.close()
  }

  test("http: SSE stream with progress for tools/call with a progressToken") {
    val t = new HttpTransport(() => new McpServer(tools), 0, None)
    try
      val init = post(t.url, initBody)
      val sid = init.headers.firstValue("Mcp-Session-Id").orElseThrow()
      val r = post(t.url, ujson.write(callReq("txt2img", ujson.Obj("prompt" -> "sse", "steps" -> 4, "width" -> 64, "height" -> 64,
        "return_images" -> false), ujson.Num(9), Some(ujson.Str("pt")))), "Mcp-Session-Id" -> sid)
      assertEquals(r.statusCode, 200)
      assert(r.headers.firstValue("Content-Type").get.startsWith("text/event-stream"))
      val events = r.body.split("\n\n").toSeq.filter(_.nonEmpty).map(e => ujson.read(e.linesIterator.find(_.startsWith("data: ")).get.drop(6)))
      assert(events.init.forall(_("method").str == "notifications/progress"), events.toString)
      assert(events.size >= 3)
      val values = events.init.map(_("params")("progress").num)
      assertEquals(values, values.sorted.distinct)
      assertEquals(events.last("id").num.toInt, 9)
      assert(!events.last("result")("isError").bool)
    finally t.close()
  }

  test("http: bearer token and McpHttp.start") {
    val c = McpHttp.start(services, 0, Some("s3cret"))
    val t = c.asInstanceOf[HttpTransport]
    try
      assertEquals(post(t.url, initBody).statusCode, 401)
      assertEquals(post(t.url, initBody, "Authorization" -> "Bearer wrong").statusCode, 401)
      val ok = post(t.url, initBody, "Authorization" -> "Bearer s3cret")
      assertEquals(ok.statusCode, 200)
      assert(ok.headers.firstValue("Mcp-Session-Id").isPresent)
    finally c.close()
  }

  private def initSession(url: String): String =
    val r = post(url, initBody)
    assertEquals(r.statusCode, 200, r.body)
    r.headers.firstValue("Mcp-Session-Id").orElseThrow()

  test("http: body limit, invalid message -> 400, idle session expiry") {
    val t = new HttpTransport(() => new McpServer(tools), 0, None, idleTtlMillis = 300, maxBodyBytes = 4096)
    try
      val sid = initSession(t.url)
      assertEquals(post(t.url, "[" + ("1," * 5000) + "1]", "Mcp-Session-Id" -> sid).statusCode, 413)
      val bad = post(t.url, """{"jsonrpc":"1.0","id":1}""", "Mcp-Session-Id" -> sid)
      assertEquals(bad.statusCode, 400)
      assertEquals(ujson.read(bad.body)("error")("code").num.toInt, -32600)
      assertEquals(post(t.url, ujson.write(request("ping")), "Mcp-Session-Id" -> sid, "MCP-Protocol-Version" -> "2025-03-26").statusCode, 400)
      Thread.sleep(700)
      assertEquals(post(t.url, ujson.write(request("ping")), "Mcp-Session-Id" -> sid).statusCode, 404)
    finally t.close()
  }

  test("http: eviction never drops a session with a call in flight; cancel across two POSTs") {
    val t = new HttpTransport(() => new McpServer(tools), 0, None, maxSessions = 1)
    try
      val a = initSession(t.url)
      val slow = ujson.write(callReq("txt2img", ujson.Obj("prompt" -> "slow http", "steps" -> 100, "width" -> 64, "height" -> 64,
        "extra" -> ujson.Obj("step_delay" -> "0.05")), ujson.Num(50)))
      val inflight = new Thread(() => {
        val r = post(t.url, slow, "Mcp-Session-Id" -> a)
        lastSlow = Some(r)
      })
      inflight.start()
      assert(eventually()(services.queue.current.exists(_.origin == "mcp")))
      assertEquals(post(t.url, initBody).statusCode, 503) // the only session is busy
      val cancel = post(t.url, """{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":50}}""", "Mcp-Session-Id" -> a)
      assertEquals(cancel.statusCode, 202)
      inflight.join(30000)
      val r = lastSlow.get
      assertEquals(r.statusCode, 200)
      assertEquals(ujson.read(r.body)("error")("code").num.toInt, -32800)
      assert(eventually()(services.queue.isIdle))
      // now idle: a new session may evict it
      initSession(t.url)
      assertEquals(post(t.url, ujson.write(request("ping")), "Mcp-Session-Id" -> a).statusCode, 404)
    finally t.close()
  }
  @volatile private var lastSlow: Option[HttpResponse[String]] = None

  test("stdio: oversized line is rejected and a cancel sent right after the call wins") {
    val toServer = new PipedOutputStream()
    val serverIn = new PipedInputStream(toServer, 1 << 20)
    val fromServer = new PipedInputStream(1 << 20)
    val serverOut = new PipedOutputStream(fromServer)
    val transport = new StdioTransport(serverIn, serverOut, new McpServer(tools), maxLineBytes = 2048)
    val t = new Thread(() => transport.run(), "stdio-limit")
    t.setDaemon(true)
    t.start()
    val lines = new LinkedBlockingQueue[ujson.Value]()
    val reader = new Thread(() => {
      val r = new BufferedReader(new InputStreamReader(fromServer, StandardCharsets.UTF_8))
      try
        var l = r.readLine()
        while l != null do
          lines.put(ujson.read(l))
          l = r.readLine()
      catch case _: Exception => ()
    })
    reader.setDaemon(true)
    reader.start()
    def send(s: String): Unit =
      toServer.write((s + "\n").getBytes(StandardCharsets.UTF_8))
      toServer.flush()
    send("x" * 10000)
    val tooBig = lines.poll(10, TimeUnit.SECONDS)
    assertEquals(tooBig("error")("code").num.toInt, -32600)
    send(ujson.write(callReq("txt2img", ujson.Obj("prompt" -> "never", "steps" -> 100, "width" -> 64, "height" -> 64,
      "extra" -> ujson.Obj("step_delay" -> "0.05")), ujson.Str("race"))))
    send("""{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":"race"}}""")
    send(ujson.write(request("ping", id = ujson.Str("after"))))
    val got = Iterator.continually(lines.poll(10, TimeUnit.SECONDS)).takeWhile(_ != null).find(_("id") == ujson.Str("after"))
    assert(got.isDefined)
    assert(eventually()(services.queue.isIdle))
    Thread.sleep(500)
    assert(!lines.toArray.exists(_.asInstanceOf[ujson.Value]("id") == ujson.Str("race")), "cancelled call must not be answered")
    toServer.close()
    t.join(10000)
  }

  test("headless McpMain process: every stdout line is a JSON-RPC message") {
    val home2 = tempHome()
    try
      val pkg = home2.root.resolve("python/sdfx")
      Files.createDirectories(pkg)
      Files.writeString(pkg.resolve("__init__.py"), "")
      Files.copy(script, pkg.resolve("__main__.py"))
      Files.writeString(home2.root.resolve("config/settings.json"), """{"last_model": "foo"}""")
      val java = Paths.get(System.getProperty("java.home"), "bin", "java").toString
      val pb = new ProcessBuilder(java, "-cp", System.getProperty("java.class.path"), "sdfx.mcp.McpMain", "--mcp", "--dry-run", "--home", home2.root.toString)
      pb.redirectError(ProcessBuilder.Redirect.DISCARD)
      val p = pb.start()
      val w = p.getOutputStream
      def send(s: String): Unit =
        w.write((s + "\n").getBytes(StandardCharsets.UTF_8))
        w.flush()
      send(initBody)
      send("""{"jsonrpc":"2.0","method":"notifications/initialized"}""")
      send(ujson.write(request("tools/list", id = ujson.Num(2))))
      send(ujson.write(callReq("txt2img", ujson.Obj("prompt" -> "proc", "steps" -> 2, "width" -> 64, "height" -> 64), ujson.Num(3), Some(ujson.Str("pt")))))
      val r = new BufferedReader(new InputStreamReader(p.getInputStream, StandardCharsets.UTF_8))
      val seen = scala.collection.mutable.ArrayBuffer.empty[ujson.Value]
      var done = false
      while !done do
        val l = r.readLine()
        if l == null then done = true
        else
          val v = ujson.read(l) // throws on any non-JSON line
          assertEquals(v("jsonrpc").str, "2.0")
          seen += v
          if v.obj.get("id").contains(ujson.Num(3)) then
            done = true
            w.close()
      assert(p.waitFor(60, TimeUnit.SECONDS))
      Iterator.continually(r.readLine()).takeWhile(_ != null).foreach(l => assertEquals(ujson.read(l)("jsonrpc").str, "2.0"))
      val res = seen.find(_.obj.get("id").contains(ujson.Num(3))).get("result")
      assert(!res("isError").bool, res.toString)
      assert(seen.exists(_.obj.get("method").contains(ujson.Str("notifications/progress"))))
    finally deleteTree(home2.root)
  }

  test("origin / host validation helpers") {
    assert(HttpTransport.originAllowed(None))
    assert(HttpTransport.originAllowed(Some("http://127.0.0.1:7861")))
    assert(HttpTransport.originAllowed(Some("http://[::1]:80")))
    assert(!HttpTransport.originAllowed(Some("null")))
    assert(!HttpTransport.originAllowed(Some("http://localhost.evil.com")))
    assert(!HttpTransport.originAllowed(Some("file://localhost")))
    assert(!HttpTransport.originAllowed(Some("http://user@localhost")))
    assert(HttpTransport.hostAllowed(Some("127.0.0.1:7861")))
    assert(HttpTransport.hostAllowed(Some("localhost")))
    assert(!HttpTransport.hostAllowed(Some("attacker.com:7861")))
  }
