package sdfx.mcp

import java.nio.file.Files
import scala.concurrent.duration.*

class AlwaysOnMcpSuite extends munit.FunSuite:
  import TestEnv.*
  override val munitTimeout: Duration = 180.seconds
  private lazy val home = tempHome()
  private lazy val host =
    val s = services(home)
    s.start()
    s
  private lazy val server = new McpServer(new McpTools(host).all)
  override def afterAll(): Unit = { host.shutdown(); deleteTree(home.root) }
  private def call(name: String, args: ujson.Obj): ujson.Value =
    server.handle(callReq(name, args), _ => ()).get("result")
  private def base = ujson.Obj("prompt" -> "a teapot", "width" -> 64, "height" -> 64, "steps" -> 1,
    "return_images" -> false, "seed" -> 1)
  private def withFreeU(p: ujson.Obj): ujson.Obj =
    p("always_on") = ujson.Obj("freeu" -> ujson.Obj("enabled" -> true, "b1" -> 1.25))
    p

  test("discover AlwaysOn controls and typed input schema separately from scripts") {
    val r = call("list_always_on", ujson.Obj("tab" -> "txt2img"))
    assert(!r("isError").bool, textOf(r))
    val plugins = r("structuredContent")("plugins").arr
    val freeu = plugins.find(_("id").str == "freeu").get
    assertEquals(freeu("values_schema")("properties")("b1")("type").str, "number")
    assertEquals(freeu("values_schema")("additionalProperties").bool, false)
    assert(!plugins.exists(_("id").str == "xyz_grid"))
  }

  test("AlwaysOn reaches sync, async and script sub-generations") {
    val sync = call("txt2img", withFreeU(base))
    assert(!sync("isError").bool, textOf(sync))
    assert(sync("structuredContent")("images")(0)("infotext").str.contains("FreeU Stages"))
    val async = call("txt2img_async", withFreeU(base))
    assert(!async("isError").bool, textOf(async))
    val id = async("structuredContent")("job_id").str
    val result = call("get_job", ujson.Obj("job_id" -> id, "wait_seconds" -> 10, "return_images" -> false))
    assertEquals(result("structuredContent")("status").str, "done")
    assert(result("structuredContent")("images")(0)("infotext").str.contains("FreeU Stages"))
    val scriptParams = withFreeU(base)
    scriptParams.obj.remove("return_images")
    val script = call("run_script", ujson.Obj("script_id" -> "prompt_matrix", "params" -> scriptParams, "return_images" -> false))
    assert(!script("isError").bool, textOf(script))
    assert(script("structuredContent")("images").arr.exists(_("infotext").str.contains("FreeU Stages")))
  }

  test("AlwaysOn image inputs reject outside paths and do not inherit saved image values") {
    val outside = Files.createTempFile("mcp-always-on-outside-", ".png")
    try
      val p = base
      p("always_on") = ujson.Obj("ip_adapter" -> ujson.Obj("enabled" -> false, "image" -> outside.toString))
      val denied = call("txt2img", p)
      assert(denied("isError").bool)
      assert(textOf(denied).contains("allowed folders"), textOf(denied))
      host.updateSettings(s => s.copy(pluginValues = s.pluginValues.updated("ip_adapter",
        Map("enabled" -> "true", "image" -> outside.toString))))
      val safe = base
      safe("always_on") = ujson.Obj("ip_adapter" -> ujson.Obj("enabled" -> false))
      val response = call("txt2img", safe)
      assert(!response("isError").bool, textOf(response))
    finally Files.deleteIfExists(outside)
  }

  test("run_script cannot inherit a private wildcard directory from desktop values") {
    val outside = Files.createTempDirectory("mcp-private-wildcards-")
    val secret = outside.resolve("mcp_private_probe.txt")
    Files.writeString(secret, "private-wildcard-sentinel")
    try
      host.updateSettings(s => s.copy(pluginValues = s.pluginValues.updated("dynamic_prompts", Map("wildcard_dirs" -> outside.toString))))
      val p = base
      p.obj.remove("return_images")
      p("prompt") = "__mcp_private_probe__"
      val response = call("run_script", ujson.Obj("script_id" -> "dynamic_prompts", "params" -> p, "return_images" -> false))
      assert(!textOf(response).contains("private-wildcard-sentinel"), textOf(response))
      val rejected = call("run_script", ujson.Obj("script_id" -> "dynamic_prompts", "params" -> p,
        "values" -> ujson.Obj("wildcard_dirs" -> outside.toString)))
      assert(rejected("isError").bool)
    finally
      host.updateSettings(s => s.copy(pluginValues = s.pluginValues.removed("dynamic_prompts")))
      Files.deleteIfExists(secret); Files.deleteIfExists(outside)
  }

  test("malformed plugin configuration fails before generating") {
    val cases = Seq(
      ujson.Obj("missing" -> ujson.Obj("enabled" -> true)),
      ujson.Obj("freeu" -> ujson.Obj("missing" -> 1)),
      ujson.Obj("freeu" -> ujson.Obj("b1" -> 999)),
      ujson.Obj("freeu" -> ujson.Obj("enabled" -> "not-bool")),
      ujson.Obj("xyz_grid" -> ujson.Obj())
    )
    for value <- cases do
      val p = base
      p("always_on") = value
      val response = call("txt2img", p)
      assert(response("isError").bool, response.toString)
    val disabled = withFreeU(base)
    disabled("always_on")("freeu")("enabled") = false
    val result = call("txt2img", disabled)
    assert(!result("isError").bool, textOf(result))
    assert(!result("structuredContent")("images")(0)("infotext").str.contains("FreeU Stages"))
  }

  private def assertRejectedWithoutQueue(tool: String, args: ujson.Obj, message: String): Unit =
    val queued = new java.util.concurrent.ConcurrentLinkedQueue[String]()
    val remove = host.queue.addListener {
      case sdfx.engine.JobEvent.JobQueued(job, _) => queued.add(job.id)
      case _ => ()
    }
    try
      val response = call(tool, args)
      assert(response("isError").bool, response.toString)
      assert(textOf(response).toLowerCase.contains(message.toLowerCase), textOf(response))
      assert(queued.isEmpty, s"$tool queued ${queued.toArray.toSeq.mkString(", ")}")
      assert(host.queue.isIdle)
    finally remove()

  private def withHiresSeed(p: ujson.Obj): ujson.Obj =
    p("always_on") = ujson.Obj("hires_fix_tweaks" -> ujson.Obj("enabled" -> true, "seed" -> "bad"))
    p

  private def withoutControlNetImage(p: ujson.Obj): ujson.Obj =
    p("controlnet") = ujson.Arr(ujson.Obj("enabled" -> true, "model" -> "test-control"))
    p

  private def scriptArgs(configure: ujson.Obj => ujson.Obj, async: Boolean = false): ujson.Obj =
    val p = configure(base)
    p.obj.remove("return_images")
    ujson.Obj("script_id" -> "prompt_matrix", "params" -> p, "return_images" -> false, "async" -> async)

  test("semantic AlwaysOn errors are rejected before sync, async and script queueing") {
    assertRejectedWithoutQueue("txt2img", withHiresSeed(base), "Invalid Hires seed")
    assertRejectedWithoutQueue("txt2img_async", withHiresSeed(base), "Invalid Hires seed")
    assertRejectedWithoutQueue("run_script", scriptArgs(withHiresSeed), "Invalid Hires seed")
    assertRejectedWithoutQueue("run_script", scriptArgs(withHiresSeed, async = true), "Invalid Hires seed")
  }

  test("txt2img ControlNet requires an image before queueing while img2img keeps init-image fallback") {
    val model = home.root.resolve("models/controlnet/test-control.safetensors")
    Files.createDirectories(model.getParent)
    Files.write(model, Array[Byte](1, 2, 3))
    host.catalog.refresh()
    assertRejectedWithoutQueue("txt2img", withoutControlNetImage(base), "ControlNet image required")
    assertRejectedWithoutQueue("txt2img_async", withoutControlNetImage(base), "ControlNet image required")
    assertRejectedWithoutQueue("run_script", scriptArgs(withoutControlNetImage), "ControlNet image required")
    assertRejectedWithoutQueue("run_script", scriptArgs(withoutControlNetImage, async = true), "ControlNet image required")
    val fallback = withoutControlNetImage(base)
    fallback("init_image") = pngBase64(64, 64)
    val result = call("img2img", fallback)
    assert(!result("isError").bool, textOf(result))
  }
