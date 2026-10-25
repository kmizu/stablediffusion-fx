package sdfx.mcp

import sdfx.core.png.PngMetadata
import sdfx.engine.AppServices

import java.nio.file.{Files, Paths}
import java.util.Base64
import scala.concurrent.duration.*
import scala.concurrent.{Await, ExecutionContext, Future}

class McpToolsSuite extends munit.FunSuite:
  import TestEnv.*
  given ExecutionContext = ExecutionContext.global

  override val munitTimeout: Duration = 180.seconds

  private lazy val home = tempHome()
  private lazy val services: AppServices =
    val s = TestEnv.services(home)
    s.start()
    s
  private lazy val tools = new McpTools(services)
  private lazy val server = new McpServer(tools.all)

  override def afterAll(): Unit =
    services.shutdown()
    deleteTree(home.root)

  private def call(tool: String, args: ujson.Value, rec: ujson.Value => Unit = _ => (), id: ujson.Value = ujson.Null, token: Option[ujson.Value] = None): ujson.Value =
    val r = server.handle(callReq(tool, args, id, token), rec).getOrElse(fail("no response"))
    assert(!r.obj.contains("error"), s"JSON-RPC error: $r")
    r("result")

  private def ok(tool: String, args: ujson.Value): ujson.Value =
    val r = call(tool, args)
    assert(!r("isError").bool, s"$tool failed: ${textOf(r)}")
    r

  test("tools/list: every tool has a JSON schema object with valid required fields") {
    val ts = server.handle(request("tools/list"), _ => ()).get("result")("tools").arr
    val names = ts.map(_("name").str).toSet
    for n <- Seq("txt2img", "img2img", "txt2img_async", "img2img_async", "get_job", "list_models", "set_model", "list_loras",
        "list_samplers", "list_recent", "get_image", "get_status", "interrupt", "list_scripts", "run_script")
    do assert(names.contains(n), n)
    ts.foreach { t =>
      val s = t("inputSchema")
      assertEquals(s("type").str, "object", t("name").str)
      val props = s("properties").obj
      s.obj.get("required").foreach(_.arr.foreach(r => assert(props.contains(r.str), s"${t("name").str}: required ${r.str} not in properties")))
      props.values.foreach(p => assert(p.obj.contains("type") || p.obj.contains("oneOf"), s"${t("name").str}: untyped property $p"))
      assert(t("description").str.nonEmpty)
      t.obj.get("outputSchema").foreach(o => assertEquals(o("type").str, "object"))
    }
    val txt = ts.find(_("name").str == "txt2img").get
    assertEquals(txt("inputSchema")("required").arr.map(_.str).toSeq, Seq("prompt"))
    assert(txt.obj.contains("outputSchema"))
    val i2i = ts.find(_("name").str == "img2img").get
    assertEquals(i2i("inputSchema")("required").arr.map(_.str).toSet, Set("prompt", "init_image"))
  }

  test("txt2img returns image + text + structuredContent and saves a PNG with parameters") {
    val r = ok("txt2img", ujson.Obj("prompt" -> "a red fox", "negative_prompt" -> "blurry", "seed" -> 42, "steps" -> 3, "width" -> 64, "height" -> 64,
      "loras" -> ujson.Arr(ujson.Obj("name" -> "My Lora", "weight" -> 0.7)), "max_image_side" -> 32))
    val content = r("content").arr
    val img = content.find(_("type").str == "image").get
    assertEquals(img("mimeType").str, "image/png")
    val bytes = Base64.getDecoder.decode(img("data").str)
    val preview = javax.imageio.ImageIO.read(new java.io.ByteArrayInputStream(bytes))
    assertEquals(preview.getWidth, 32) // downscaled from 64
    val sc = r("structuredContent")("images").arr
    assertEquals(sc.size, 1)
    val path = Paths.get(sc(0)("path").str)
    assert(Files.isRegularFile(path))
    assert(path.startsWith(home.root.resolve("outputs")))
    assertEquals(sc(0)("seed").num.toLong, 42L)
    val params = PngMetadata.readParameters(path).get
    assert(params.startsWith("a red fox, <lora:My Lora:0.7>"), params)
    assert(params.contains("Negative prompt: blurry"))
    assert(params.contains("Seed: 42"))
    assert(textOf(r).contains(path.toString))
    assertEquals(sc(0)("infotext").str, params)
    assertEquals(javax.imageio.ImageIO.read(path.toFile).getWidth, 64) // full size on disk
  }

  test("txt2img return_images=false and batch") {
    val r = ok("txt2img", ujson.Obj("prompt" -> "x", "steps" -> 1, "width" -> 64, "height" -> 64, "batch_size" -> 2, "n_iter" -> 2, "return_images" -> false))
    assert(!r("content").arr.exists(_("type").str == "image"))
    assertEquals(r("structuredContent")("images").arr.size, 4)
  }

  test("invalid params produce isError results") {
    def err(args: ujson.Obj, needle: String) =
      val r = call("txt2img", args)
      assert(r("isError").bool, s"expected error for $args")
      assert(textOf(r).toLowerCase.contains(needle.toLowerCase), s"'${textOf(r)}' lacks '$needle'")
    err(ujson.Obj(), "prompt")
    err(ujson.Obj("prompt" -> "x", "steps" -> 0), "steps")
    err(ujson.Obj("prompt" -> "x", "width" -> 70), "width")
    err(ujson.Obj("prompt" -> "x", "steps" -> "many"), "steps")
    err(ujson.Obj("prompt" -> "x", "batch_size" -> 8, "n_iter" -> 4), "16")
    err(ujson.Obj("prompt" -> "x", "sampler_name" -> "Nope"), "Euler a")
    err(ujson.Obj("prompt" -> "x", "checkpoint" -> "missing_model"), "checkpoint not found")
    err(ujson.Obj("prompt" -> "x", "loras" -> ujson.Arr(ujson.Obj("name" -> "ghost"))), "LoRA not found")
    err(ujson.Obj("prompt" -> "x", "colour" -> "red"), "colour")
    val i2i = call("img2img", ujson.Obj("prompt" -> "x", "init_image" -> "not an image at all"))
    assert(i2i("isError").bool)
  }

  test("img2img with base64 input (and data URI mask) defaults to the init image size") {
    val b64 = pngBase64(96, 72)
    val r = ok("img2img", ujson.Obj("prompt" -> "paint it", "init_image" -> b64, "mask" -> s"data:image/png;base64,${pngBase64(96, 72)}",
      "denoising_strength" -> 0.4, "inpainting_fill" -> "original", "steps" -> 2, "seed" -> 7))
    val p = Paths.get(r("structuredContent")("images")(0)("path").str)
    assert(p.toString.contains("img2img"), p.toString)
    val params = PngMetadata.readParameters(p).get
    assert(params.contains("Size: 96x72"), params)
    assert(params.contains("Denoising strength: 0.4"), params)
    assert(r("content").arr.exists(_("type").str == "image"))
    // the temp input files were cleaned up
    assert(eventually()(!Files.list(home.root.resolve("outputs/.tmp")).toList.toString.contains("sdfx-")))
  }

  test("progress notifications are emitted when a progressToken is given") {
    val rec = new Recorder
    val r = call("txt2img", ujson.Obj("prompt" -> "p", "steps" -> 5, "width" -> 64, "height" -> 64, "extra" -> ujson.Obj("step_delay" -> "0.02")),
      rec, token = Some(ujson.Num(77)))
    assert(!r("isError").bool)
    val ps = rec.all.filter(_("method").str == "notifications/progress").map(_("params"))
    assert(ps.size >= 3, ps.toString)
    assert(ps.forall(_("progressToken").num == 77))
    val values = ps.map(_("progress").num)
    assertEquals(values, values.sorted.distinct)
    assertEquals(values.last, 100.0)
    assert(ps.exists(_.obj.get("message").exists(_.str.contains("step"))))
  }

  test("notifications/cancelled interrupts the job tied to the request") {
    val rec = new Recorder
    val f = Future(server.handle(callReq("txt2img", ujson.Obj("prompt" -> "slow", "steps" -> 100, "width" -> 64, "height" -> 64,
      "extra" -> ujson.Obj("step_delay" -> "0.05")), ujson.Str("cancel-me"), Some(ujson.Str("t"))), rec))
    assert(eventually()(rec.all.exists(_("params")("progress").num > 1)))
    assert(services.queue.current.exists(_.origin == "mcp"))
    server.handle(ujson.Obj("jsonrpc" -> "2.0", "method" -> "notifications/cancelled", "params" -> ujson.Obj("requestId" -> "cancel-me")), _ => ())
    assertEquals(Await.result(f, 30.seconds), None) // no reply for a cancelled request
    assert(eventually()(services.queue.isIdle))
  }

  test("async job flow: submit, poll with wait, images when done") {
    val sub = ok("txt2img_async", ujson.Obj("prompt" -> "async", "steps" -> 4, "width" -> 64, "height" -> 64, "extra" -> ujson.Obj("step_delay" -> "0.02")))
    val id = sub("structuredContent")("job_id").str
    val r = ok("get_job", ujson.Obj("job_id" -> id, "wait_seconds" -> 30))
    val s = r("structuredContent")
    assertEquals(s("status").str, "done")
    assertEquals(s("progress").num, 1.0)
    assertEquals(s("images").arr.size, 1)
    assert(Files.isRegularFile(Paths.get(s("images")(0)("path").str)))
    assert(r("content").arr.exists(_("type").str == "image"))
    val unknown = call("get_job", ujson.Obj("job_id" -> "nope"))
    assert(unknown("isError").bool)
  }

  test("async job can be cancelled") {
    val sub = ok("txt2img_async", ujson.Obj("prompt" -> "long", "steps" -> 100, "width" -> 64, "height" -> 64, "extra" -> ujson.Obj("step_delay" -> "0.05")))
    val id = sub("structuredContent")("job_id").str
    assert(eventually()(services.queue.current.exists(_.id == id)))
    ok("cancel_job", ujson.Obj("job_id" -> id))
    val r = call("get_job", ujson.Obj("job_id" -> id, "wait_seconds" -> 30))
    assertEquals(r("structuredContent")("status").str, "cancelled")
  }

  test("interrupt stops the running job") {
    val sub = ok("txt2img_async", ujson.Obj("prompt" -> "long", "steps" -> 100, "width" -> 64, "height" -> 64, "extra" -> ujson.Obj("step_delay" -> "0.05")))
    val id = sub("structuredContent")("job_id").str
    assert(eventually()(services.queue.current.exists(_.id == id)))
    val r = ok("interrupt", ujson.Obj())
    assert(r("structuredContent")("interrupted").bool)
    assertEquals(call("get_job", ujson.Obj("job_id" -> id, "wait_seconds" -> 30))("structuredContent")("status").str, "cancelled")
  }

  test("list_recent and get_image; path traversal is rejected") {
    ok("txt2img", ujson.Obj("prompt" -> "gallery cat", "steps" -> 1, "width" -> 64, "height" -> 64))
    val recent = ok("list_recent", ujson.Obj("limit" -> 3, "query" -> "gallery cat"))
    val items = recent("structuredContent")("images").arr
    assert(items.nonEmpty)
    assert(items(0)("infotext").str.contains("gallery cat"))
    val path = items(0)("path").str
    val img = ok("get_image", ujson.Obj("path" -> path, "max_side" -> 16))
    assert(img("content").arr.exists(_("type").str == "image"))
    val rel = home.root.resolve("outputs").relativize(Paths.get(path)).toString
    ok("get_image", ujson.Obj("path" -> rel))
    val secret = home.root.resolve("config/secret.png")
    Files.write(secret, Base64.getDecoder.decode(pngBase64()))
    for bad <- Seq(secret.toString, "../config/secret.png", home.root.resolve("outputs/../config/secret.png").toString, "C:\\Windows\\win.ini", "")
    do
      val r = call("get_image", ujson.Obj("path" -> bad))
      assert(r("isError").bool, s"$bad must be rejected")
  }

  test("model, LoRA, sampler, script listings and set_model") {
    val models = ok("list_models", ujson.Obj())
    val cks = models("structuredContent")("checkpoints").arr
    assertEquals(cks.map(_("name").str).toSet, Set("foo", "sdxl/Bar_XL"))
    assert(cks.find(_("name").str == "foo").get("loaded").bool)
    val set = ok("set_model", ujson.Obj("name" -> "Bar_XL"))
    assertEquals(set("structuredContent")("loaded")("name").str, "Bar_XL")
    assert(call("set_model", ujson.Obj("name" -> "nope"))("isError").bool)
    ok("set_model", ujson.Obj("name" -> "foo"))
    val loras = ok("list_loras", ujson.Obj())
    assertEquals(loras("structuredContent")("loras")(0)("tag").str, "<lora:My Lora:1.0>")
    val samplers = ok("list_samplers", ujson.Obj())("structuredContent")
    assert(samplers("samplers").arr.exists(_("name").str == "Euler a"))
    assert(!samplers("samplers").arr.exists(_("name").str == "Weird"))
    assertEquals(samplers("styles").arr.map(_.str).toSeq, Seq("Pretty"))
    val scripts = ok("list_scripts", ujson.Obj())("structuredContent")("scripts").arr
    val xyz = scripts.find(_("id").str == "xyz_grid").get
    assert(xyz("controls").arr.exists(_("key").str == "x_type"))
    val status = ok("get_status", ujson.Obj())("structuredContent")
    assertEquals(status("engine_state").str, "ready")
    assertEquals(status("loaded_model")("name").str, "foo")
    assertEquals(status("engine")("device").str, "cpu")
  }

  test("structuredContent of txt2img / img2img conforms to the declared outputSchema") {
    val ts = server.handle(request("tools/list"), _ => ()).get("result")("tools").arr
    def schemaOf(n: String) = ts.find(_("name").str == n).get("outputSchema")
    val r = ok("txt2img", ujson.Obj("prompt" -> "schema", "steps" -> 1, "width" -> 64, "height" -> 64))
    assertEquals(validate(r("structuredContent"), schemaOf("txt2img")), Nil)
    // the JSON text block mirrors structuredContent for older clients
    assert(r("content").arr.exists(c => c("type").str == "text" && c("text").str.startsWith("{") && ujson.read(c("text").str) == r("structuredContent")))
    val i = ok("img2img", ujson.Obj("prompt" -> "schema", "init_image" -> pngBase64(64, 64), "steps" -> 1))
    assertEquals(validate(i("structuredContent"), schemaOf("img2img")), Nil)
  }

  test("init_image paths: outputs folder and allowed dirs only, UNC rejected, uniform errors") {
    val gen = ok("txt2img", ujson.Obj("prompt" -> "source", "steps" -> 1, "width" -> 64, "height" -> 64))
    val inOutputs = gen("structuredContent")("images")(0)("path").str
    ok("img2img", ujson.Obj("prompt" -> "from outputs", "init_image" -> inOutputs, "steps" -> 1))
    val rel = home.root.resolve("outputs").relativize(Paths.get(inOutputs)).toString
    ok("img2img", ujson.Obj("prompt" -> "relative", "init_image" -> rel, "steps" -> 1))

    val outside = Files.createTempDirectory("sdfx-outside-")
    try
      val secret = outside.resolve("secret.png")
      Files.write(secret, Base64.getDecoder.decode(pngBase64()))
      def denied(spec: String, needle: String) =
        val r = call("img2img", ujson.Obj("prompt" -> "x", "init_image" -> spec, "steps" -> 1))
        assert(r("isError").bool, s"$spec must be rejected")
        assert(textOf(r).contains(needle), textOf(r))
      denied(secret.toString, "allowed folders")
      denied(outside.resolve("missing.png").toString, "allowed folders") // same message: no existence oracle
      denied("../config/styles.csv", "allowed folders")
      denied("\\\\attacker.invalid\\share\\x.png", "UNC")
      denied("//attacker.invalid/share/x.png", "UNC")
      // an explicitly allowed folder works
      val extra = new McpServer(new McpTools(services, McpTools.waitForEngine(services), allowedDirs = Seq(outside)).all)
      val r = extra.handle(callReq("img2img", ujson.Obj("prompt" -> "allowed", "init_image" -> secret.toString, "steps" -> 1)), _ => ()).get("result")
      assert(!r("isError").bool, textOf(r))
      // get_image follows the same policy
      assert(call("get_image", ujson.Obj("path" -> secret.toString))("isError").bool)
      assert(!extra.handle(callReq("get_image", ujson.Obj("path" -> secret.toString)), _ => ()).get("result")("isError").bool)
    finally deleteTree(outside)
  }

  test("size caps: input image dimensions, hires target, pixels per job") {
    val big = call("img2img", ujson.Obj("prompt" -> "x", "init_image" -> pngBase64(4104, 8), "width" -> 64, "height" -> 64))
    assert(big("isError").bool && textOf(big).contains("too large"), textOf(big))
    val hires = call("txt2img", ujson.Obj("prompt" -> "x", "width" -> 2048, "height" -> 2048, "hires" -> ujson.Obj("scale" -> 4)))
    assert(hires("isError").bool && textOf(hires).contains("hires target"), textOf(hires))
    val batch = call("txt2img", ujson.Obj("prompt" -> "x", "width" -> 4096, "height" -> 4096, "batch_size" -> 2))
    assert(batch("isError").bool && textOf(batch).contains("too large"), textOf(batch))
    val total = call("txt2img", ujson.Obj("prompt" -> "x", "width" -> 2048, "height" -> 2112, "batch_size" -> 2, "n_iter" -> 8))
    assert(total("isError").bool && textOf(total).contains("in total"), textOf(total))
    assert(call("txt2img", ujson.Obj("prompt" -> "x", "max_image_side" -> 8192))("isError").bool)
  }

  test("pending MCP jobs are capped") {
    val slowArgs = ujson.Obj("prompt" -> "fill", "steps" -> 100, "width" -> 64, "height" -> 64, "return_images" -> false, "extra" -> ujson.Obj("step_delay" -> "0.05"))
    try
      val first = ok("txt2img_async", slowArgs)("structuredContent")("job_id").str
      assert(eventually()(services.queue.current.exists(_.id == first)))
      (1 to McpTools.MaxPendingJobs).foreach(_ => ok("txt2img_async", slowArgs))
      val over = call("txt2img_async", slowArgs)
      assert(over("isError").bool && textOf(over).contains("too many queued"), textOf(over))
    finally
      ok("interrupt", ujson.Obj("clear_queue" -> true))
      assert(eventually(30.seconds)(services.queue.isIdle))
  }

  test("async submit returns immediately while the engine is still starting") {
    val home2 = tempHome()
    val s2 = TestEnv.services(home2)
    val gate = scala.concurrent.Promise[Any]()
    try
      val srv = new McpServer(new McpTools(s2, McpTools.awaitStart(gate.future)).all)
      val t0 = System.nanoTime()
      val sub = srv.handle(callReq("txt2img_async", ujson.Obj("prompt" -> "early bird", "steps" -> 1, "width" -> 64, "height" -> 64)), _ => ()).get("result")
      assert((System.nanoTime() - t0) < 5.seconds.toNanos)
      assertEquals(sub("structuredContent")("status").str, "waiting_for_engine")
      val id = sub("structuredContent")("job_id").str
      val st = srv.handle(callReq("get_job", ujson.Obj("job_id" -> id)), _ => ()).get("result")("structuredContent")
      assertEquals(st("status").str, "waiting_for_engine")
      gate.completeWith(s2.startAsync())
      val done = srv.handle(callReq("get_job", ujson.Obj("job_id" -> id, "wait_seconds" -> 60)), _ => ()).get("result")("structuredContent")
      assertEquals(done("status").str, "done", done.toString)
    finally
      s2.shutdown()
      deleteTree(home2.root)
  }

  test("run_script obeys the strict job limits (X/Y/Z Size axis, cell count)") {
    val base = ujson.Obj("prompt" -> "limits", "steps" -> 1, "width" -> 64, "height" -> 64)
    val huge = call("run_script", ujson.Obj("script_id" -> "xyz_grid", "values" -> ujson.Obj("x_type" -> "Size", "x_values" -> "64x64, 8192x8192"),
      "params" -> base, "return_images" -> false))
    assert(huge("isError").bool, textOf(huge))
    assert(textOf(huge).contains("job limit exceeded"), textOf(huge))
    val many = call("run_script", ujson.Obj("script_id" -> "xyz_grid", "values" -> ujson.Obj("x_type" -> "Seed", "x_values" -> "1-70"),
      "params" -> base, "return_images" -> false))
    assert(many("isError").bool, textOf(many))
    assert(textOf(many).contains("64 generations"), textOf(many))
    // async path reports the same failure
    val sub = ok("run_script", ujson.Obj("script_id" -> "xyz_grid", "async" -> true, "values" -> ujson.Obj("x_type" -> "Size", "x_values" -> "8192x8192"),
      "params" -> base))
    val st = call("get_job", ujson.Obj("job_id" -> sub("structuredContent")("job_id").str, "wait_seconds" -> 30))("structuredContent")
    assertEquals(st("status").str, "failed")
    assert(st("error").str.contains("job limit exceeded"), st.toString)
  }

  test("run_script never runs custom code and checks image control paths against the allowed dirs") {
    val cc = call("run_script", ujson.Obj("script_id" -> "custom_code", "values" -> ujson.Obj("code" -> "result = 1")))
    assert(cc("isError").bool)
    assert(cc("content")(0)("text").str.contains("custom_code cannot be run over MCP"))
    val outside = java.nio.file.Files.createTempFile("outside", ".png")
    val r = call("run_script", ujson.Obj("script_id" -> "img2img_alt", "values" -> ujson.Obj("override_prompt" -> "x"),
      "params" -> ujson.Obj("init_image" -> outside.toString)))
    assert(r("isError").bool, "init_image outside the allowed dirs must be refused")
    val scripts = ok("list_scripts", ujson.Obj())("structuredContent")("scripts").arr
    assert(scripts.exists(_("id").str == "svd"))
    assert(scripts.exists(_("id").str == "focal_crop"))
  }

  test("run_script xyz_grid produces a grid and the images") {
    val r = ok("run_script", ujson.Obj("script_id" -> "xyz_grid", "values" -> ujson.Obj("x_type" -> "Seed", "x_values" -> "1,2", "include_sub_images" -> true),
      "params" -> ujson.Obj("prompt" -> "grid", "steps" -> 1, "width" -> 64, "height" -> 64), "max_image_side" -> 64))
    val imgs = r("structuredContent")("images").arr
    assert(imgs.exists(_("grid").bool), imgs.toString)
    assert(imgs.count(i => !i("grid").bool) >= 2)
    assert(call("run_script", ujson.Obj("script_id" -> "nope"))("isError").bool)
    assert(call("run_script", ujson.Obj("script_id" -> "xyz_grid", "values" -> ujson.Obj("bogus" -> 1)))("isError").bool)
  }
