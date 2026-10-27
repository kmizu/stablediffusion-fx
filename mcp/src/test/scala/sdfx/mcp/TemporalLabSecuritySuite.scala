package sdfx.mcp

import munit.FunSuite

class TemporalLabSecuritySuite extends FunSuite:
  import TestEnv.*

  test("temporal local filesystem operations are not exposed as MCP tools") {
    val home = tempHome()
    val services = TestEnv.services(home)
    try
      val server = new McpServer(new McpTools(services).all)
      val tools = server.handle(request("tools/list"), _ => ()).get("result")("tools").arr
      assert(!tools.exists(_("name").str == "temporal_lab"))
      val response = server.handle(callReq("temporal_lab", ujson.Obj(
        "operation" -> "extract", "video_path" -> "C:/private/video.avi",
        "output_parent" -> "C:/private", "output_name" -> "result")), _ => ()).get
      assert(response.obj.contains("error"), s"unknown temporal tool unexpectedly accepted: $response")
    finally
      services.shutdown()
      deleteTree(home.root)
  }
