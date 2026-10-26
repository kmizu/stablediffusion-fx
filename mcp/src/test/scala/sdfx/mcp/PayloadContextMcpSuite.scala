package sdfx.mcp

import sdfx.core.payload.PayloadState
import sdfx.engine.AppServices
import scala.concurrent.duration.*

class PayloadContextMcpSuite extends munit.FunSuite:
  import TestEnv.*
  override val munitTimeout: Duration = 180.seconds

  private def withServer(test: (AppServices, McpServer) => Unit): Unit =
    val home = tempHome()
    val services = TestEnv.services(home)
    try
      services.start()
      test(services, new McpServer(new McpTools(services).all))
    finally
      services.shutdown()
      deleteTree(home.root)

  private def call(server: McpServer, name: String, args: ujson.Value): ujson.Value =
    val r = server.handle(callReq(name, args), _ => ()).get("result")
    assert(!r("isError").bool, textOf(r))
    r

  test("sync, async and script MCP routes retain only actual invocation context and native bodies") {
    withServer { (s, server) =>
      val gen = ujson.Obj("prompt" -> "MCP 猫", "seed" -> 5, "steps" -> 1, "width" -> 64, "height" -> 64)
      call(server, "txt2img", gen)
      val sync = s.payloads.snapshots.last
      val submitted = ujson.read(sync.contextJson.get)("submitted")
      assertEquals(submitted("origin").str, "mcp")
      assertEquals(submitted("source_mode").str, "txt2img")
      assertEquals(submitted("configuration")("ui_forms").str, "not_available_for_origin")
      assertEquals(submitted("configuration")("selected_script"), ujson.Null)
      assertEquals(sync.state, PayloadState.Succeeded)
      assertEquals(ujson.read(sync.bodyJson.get)("params")("prompt").str, "MCP 猫")

      val a = call(server, "img2img_async", ujson.Obj("prompt" -> "masked", "init_image" -> pngBase64(64, 64),
        "mask" -> pngBase64(64, 64), "width" -> 64, "height" -> 64, "steps" -> 1, "seed" -> 6))
      val id = a("structuredContent")("job_id").str
      call(server, "get_job", ujson.Obj("job_id" -> id, "wait_seconds" -> 30))
      val image = s.payloads.snapshots.find(_.key.jobId == id).get
      val input = ujson.read(image.contextJson.get)("submitted")
      assertEquals(input("source_mode").str, "img2img")
      assertEquals(input("configuration")("has_mask").bool, true)
      assertEquals(image.requestType, "img2img")
      assertEquals(image.state, PayloadState.Succeeded)
      assert(ujson.read(image.bodyJson.get)("params")("img2img")("init_image").str.endsWith(".png"))
      assert(!image.bodyJson.get.contains("base64"))

      call(server, "run_script", ujson.Obj("script_id" -> "xyz_grid", "values" -> ujson.Obj("x_type" -> "Seed", "x_values" -> "1,2"),
        "params" -> gen))
      val last = s.payloads.snapshots.last
      val scriptContext = ujson.read(last.contextJson.get)("submitted")("configuration")
      assertEquals(scriptContext("selected_script")("id").str, "xyz_grid")
      assertEquals(scriptContext("selected_script")("values")("x_values").str, "1,2")
      // Effective Values include registry defaults absent from the sparse MCP caller.
      assert(scriptContext("selected_script")("values").obj.contains("y_type"))
      assertEquals(scriptContext("ui_forms").str, "not_available_for_origin")
      assertEquals(s.payloads.snapshots.count(_.key.jobId == last.key.jobId), 2)
    }
  }
