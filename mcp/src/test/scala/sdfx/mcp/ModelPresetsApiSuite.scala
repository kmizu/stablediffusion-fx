package sdfx.mcp

import sdfx.core.modelpresets.ModelPresetDocument

class ModelPresetsApiSuite extends munit.FunSuite:
  test("remote sharing presets require explicit overwrite even on first fetch") {
    val home = TestEnv.tempHome()
    val services = TestEnv.services(home)
    try
      services.start()
      val remote = ModelPresetDocument.empty.save("remote", "remote prompt").setDefault("remote")
      val version = ujson.Obj("id" -> 99, "modelId" -> 12, "trainedWords" -> ujson.Arr("actual trigger"), "images" -> ujson.Arr())
      val client = new sdfx.core.modelpresets.CivitaiPresetClient((uri, _, _) => {
        val result = if uri.getPath.contains("by-hash") then version else
          ujson.Obj("id" -> 12, "description" -> remote.share, "modelVersions" -> ujson.Arr(version))
        result.render().getBytes("UTF-8")
      })
      val service = new sdfx.core.modelpresets.ModelPresetService(home.configDir.resolve("model-presets"), client)
      val tool = new ModelPresetsTool(services, service)
      def call(args: ujson.Obj) = tool.call(args, ToolContext.none).structured.get
      val first = call(ujson.Obj("action" -> "fetch", "checkpoint" -> "foo", "expected_revision" -> ujson.Null))
      assertEquals(first("document")("presets").obj.keySet.toSet, Set("default"))
      assertEquals(first("document")("trigger_words").arr.map(_.str).toVector, Vector("actual trigger"))
      val local = ModelPresetDocument.parse(first("document").render()).save("mine", "local prompt").setDefault("mine")
      val saved = call(ujson.Obj("action" -> "save", "checkpoint" -> "foo", "expected_revision" -> first("revision"), "document" -> ujson.read(local.json)))
      val kept = call(ujson.Obj("action" -> "fetch", "checkpoint" -> "foo", "expected_revision" -> saved("revision")))
      assertEquals(kept("document")("presets")("mine").str, "local prompt")
      assert(!kept("document")("presets").obj.contains("remote"))
      val replaced = call(ujson.Obj("action" -> "fetch", "checkpoint" -> "foo", "expected_revision" -> kept("revision"), "overwrite" -> true))
      assertEquals(ModelPresetDocument.parse(replaced("document").render()), remote)
    finally
      services.shutdown(); TestEnv.deleteTree(home.root)
  }
  test("MCP model preset library saves with revision and preserves raw text without generation") {
    val home = TestEnv.tempHome()
    val services = TestEnv.services(home)
    try
      services.start()
      val tool = new McpTools(services).all.find(_.name == "model_presets").getOrElse(fail("model_presets tool missing"))
      def call(args: ujson.Obj) = tool.call(args, ToolContext.none).structured.get
      val first = call(ujson.Obj("action" -> "get", "checkpoint" -> "foo"))
      val text = "猫\nNegative prompt: fog\nSteps: 20, Seed: 42, Future field: test"
      val document = ModelPresetDocument.empty.save("mine", text).setDefault("mine")
      val saved = call(ujson.Obj("action" -> "save", "checkpoint" -> "foo", "expected_revision" -> first("revision"),
        "document" -> ujson.read(document.json)))
      assertEquals(saved("document")("presets")("mine").str, text)
      val shared = call(ujson.Obj("action" -> "share", "checkpoint" -> "foo"))
      assertEquals(ModelPresetDocument.parseShared(shared("sharing_text").str), document)
      val stale = intercept[Exception](call(ujson.Obj("action" -> "save", "checkpoint" -> "foo",
        "expected_revision" -> ujson.Null, "document" -> ujson.read(document.json))))
      assert(stale.getMessage.contains("changed"))
      assert(services.queue.isIdle)
    finally
      services.shutdown(); TestEnv.deleteTree(home.root)
  }

  test("API fetch publishes first thumbnail without replacing an existing one; manual replacement is explicit") {
    val home = TestEnv.tempHome()
    val services = TestEnv.services(home)
    try
      services.start()
      val png = java.util.Base64.getDecoder.decode(TestEnv.pngBase64(600, 300))
      val client = new sdfx.core.modelpresets.CivitaiPresetClient((uri, limit, check) => {
        check()
        if uri.getPath.contains("by-hash") then
          """{"modelId":12,"trainedWords":[],"images":[{"url":"https://image.civitai.com/a.png"}]}""".getBytes("UTF-8")
        else if uri.getPath.endsWith("/12") then """{"description":""}""".getBytes("UTF-8")
        else png
      })
      val service = new sdfx.core.modelpresets.ModelPresetService(home.configDir.resolve("model-presets"), client)
      val tool = new ModelPresetsTool(services, service)
      def call(args: ujson.Obj) = tool.call(args, ToolContext.none).structured.get
      val fetched = call(ujson.Obj("action" -> "fetch", "checkpoint" -> "foo", "expected_revision" -> ujson.Null))
      val target = java.nio.file.Path.of(fetched("thumbnail").str)
      val image = javax.imageio.ImageIO.read(target.toFile)
      assertEquals((image.getWidth, image.getHeight), (300, 150))
      val original = java.nio.file.Files.readAllBytes(target).toVector
      call(ujson.Obj("action" -> "thumbnail", "checkpoint" -> "foo", "image_base64" -> TestEnv.pngBase64(16, 16)))
      assertEquals(java.nio.file.Files.readAllBytes(target).toVector, original)
      call(ujson.Obj("action" -> "thumbnail", "checkpoint" -> "foo", "image_base64" -> TestEnv.pngBase64(16, 16), "replace" -> true))
      assertEquals(javax.imageio.ImageIO.read(target.toFile).getWidth, 16)
      assert(services.queue.isIdle)
    finally
      services.shutdown(); TestEnv.deleteTree(home.root)
  }
