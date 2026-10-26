package sdfx.core

import sdfx.core.json.SnakePickle
import sdfx.core.model.*
import sdfx.core.protocol.*

class ProtocolSuite extends munit.FunSuite:
  import EngineEvent.*

  private def ev(s: String) = EngineEvent.parse(ujson.read(s))

  test("EngineEvent.parse: progress (full and defaults)") {
    assertEquals(ev("""{"id":"r1","type":"event","event":"progress","step":3,"total":28,"pass":"hires","eta_s":1.5,"image":1,"images":4}"""),
      Some(Progress(3, 28, "hires", 1.5, 1, 4)))
    assertEquals(ev("""{"event":"progress"}"""), Some(Progress(0, 0, "base", 0.0, 0, 1)))
    assertEquals(ev("""{"event":"progress","step":2.9,"total":10.0}"""), Some(Progress(2, 10)))
  }

  test("EngineEvent.parse: preview") {
    assertEquals(ev("""{"event":"preview","jpeg_b64":"/9j/4AAQ","step":5}"""), Some(Preview("/9j/4AAQ", 5)))
    assertEquals(ev("""{"event":"preview","jpeg_b64":"abc"}"""), Some(Preview("abc", 0)))
    assertEquals(ev("""{"event":"preview","step":5}"""), None)
    assertEquals(ev("""{"event":"preview","jpeg_b64":null}"""), None)
  }

  test("EngineEvent.parse: log") {
    assertEquals(ev("""{"event":"log","level":"warning","message":"VRAM low"}"""), Some(Log("warning", "VRAM low")))
    assertEquals(ev("""{"event":"log"}"""), Some(Log("info", "")))
    assertEquals(ev("""{"event":"log","message":"日本語ログ"}"""), Some(Log("info", "日本語ログ")))
  }

  test("EngineEvent.parse: unknown / missing / malformed never throws") {
    assertEquals(ev("""{"event":"telemetry","x":1}"""), None)
    assertEquals(ev("""{"type":"event"}"""), None)
    assertEquals(ev("""{"event":null}"""), None)
    assertEquals(ev("""{"event":42}"""), None)
    assertEquals(ev("""[1,2]"""), None)
    assertEquals(ev("""null"""), None)
    assertEquals(ev(""""progress""""), None)
    assertEquals(ev("""{"event":"progress","step":"3","pass":null,"eta_s":"x"}"""), Some(Progress(0, 0)))
    assertEquals(ev("""{"event":"log","level":null,"message":123}"""), Some(Log("info", "")))
  }

  test("EngineHello: full") {
    val js = """{"id":null,"type":"hello","protocol":1,"engine_version":"0.1.0","python":"3.10.11","torch":"2.3.1+cu121",
               |"diffusers":"0.30.0","cuda":true,"device":"cuda:0","vram_mb":24576,"dry_run":false,"features":["controlnet","freeu"]}""".stripMargin
    val h = SnakePickle.read[EngineHello](js)
    assertEquals(h, EngineHello(1, "0.1.0", "3.10.11", Some("2.3.1+cu121"), Some("0.30.0"), true, "cuda:0", 24576, false,
      Seq("controlnet", "freeu")))
  }

  test("EngineHello: lenient — only required fields, optional ones missing or null") {
    assertEquals(SnakePickle.read[EngineHello]("""{"protocol":1,"engine_version":"x"}"""), EngineHello(1, "x"))
    val h = SnakePickle.read[EngineHello]("""{"protocol":2,"engine_version":"y","torch":null,"diffusers":null,"extra_future":[1]}""")
    assertEquals(h, EngineHello(2, "y"))
  }

  test("EngineHello: missing required field fails") {
    intercept[Exception](SnakePickle.read[EngineHello]("""{"engine_version":"x"}"""))
  }

  test("EngineHello round trip and snake keys") {
    val h = EngineHello(1, "v", torch = Some("t"), vramMb = 8, dryRun = true)
    val js = SnakePickle.writeJs(h)
    assertEquals(js.obj.keys.toSet, Set("protocol", "engine_version", "python", "torch", "diffusers", "cuda", "device",
      "vram_mb", "dry_run", "features"))
    assertEquals(SnakePickle.read[EngineHello](js), h)
  }

  test("GenerationRequest JSON has snake keys and round trips") {
    val req = GenerationRequest(
      params = GenerationParams(prompt = "p", negativePrompt = "n", cfgScale = 6, loras = Seq(LoraRef("foo", 0.5))),
      outputDir = "D:/out/.tmp",
      loraPaths = Map("foo" -> "D:/models/Lora/foo.safetensors"),
      embeddingPaths = Seq("D:/emb/easyneg.pt"),
      controlnetPaths = Map("canny" -> "D:/cn/canny.pth"),
      upscalerPaths = Map("4x-UltraSharp" -> "D:/up/4x.pth"),
      previewEvery = 3,
      previewMethod = "full"
    )
    val js = Json.toJs(req)
    assertEquals(js.obj.keys.toSet, Set("params", "output_dir", "lora_paths", "embedding_paths", "controlnet_paths",
      "upscaler_paths", "preview_every", "preview_method", "lbw_presets"))
    assertEquals(js("params")("negative_prompt").str, "n")
    assertEquals(js("params")("cfg_scale").num, 6.0)
    assertEquals(js("lora_paths").obj.keys.toSeq, Seq("foo"))
    assertEquals(js("upscaler_paths").obj.keys.toSeq, Seq("4x-UltraSharp"))
    assertEquals(Json.fromJs[GenerationRequest](js), req)
    assertEquals(Json.fromJs[GenerationRequest](ujson.read(ujson.write(js))), req)
  }

  test("GenerationRequest minimal") {
    val r = Json.fromJs[GenerationRequest](ujson.read("""{"params":{},"output_dir":"o"}"""))
    assertEquals(r, GenerationRequest(GenerationParams(), "o"))
  }

  test("GenerationResult: snake keys, defaults when reading engine output") {
    val js = ujson.read(
      """{"images":[{"path":"D:/o/1.png","seed":4294967296,"subseed":7,"width":832,"height":1216},{"path":"b.png","seed":1}],
        |"timing_s":12.5,"model_name":"animagineXL","model_hash":"4496b36d48","lora_hashes":{"foo":"abcdef123456"},
        |"scheduler":"Karras","warnings":["LoRA not found: bar"],"unknown":true}""".stripMargin)
    val r = Json.fromJs[GenerationResult](js)
    assertEquals(r.images, Seq(GeneratedImage("D:/o/1.png", 4294967296L, 7, 832, 1216), GeneratedImage("b.png", 1)))
    assertEquals(r.timingS, 12.5)
    assertEquals(r.modelName, Some("animagineXL"))
    assertEquals(r.vaeName, None)
    assertEquals(r.loraHashes, Map("foo" -> "abcdef123456"))
    assertEquals(r.scheduler, Some("Karras"))
    assertEquals(r.warnings, Seq("LoRA not found: bar"))
    val out = Json.toJs(r)
    Seq("timing_s", "model_name", "model_hash", "vae_name", "vae_hash", "lora_hashes").foreach(k => assert(out.obj.contains(k), k))
    assertEquals(Json.fromJs[GenerationResult](out), r)
    assertEquals(Json.fromJs[GenerationResult](ujson.read("""{"images":[]}""")), GenerationResult(Nil))
  }

  test("other messages round trip with snake keys") {
    def rt[T: SnakePickle.ReadWriter](t: T, keys: String*) =
      val js = Json.toJs(t)
      keys.foreach(k => assert(js.obj.contains(k), s"$k in $js"))
      assertEquals(Json.fromJs[T](js), t)
    rt(ModelEntry("m", "p", "sdxl", 6617.5), "size_mb")
    rt(FileEntry("f", "p"), "size_mb")
    rt(LoadModelRequest("p", vaePath = Some("v"), vaeTiling = true), "vae_path", "vae_tiling", "vram_mode")
    rt(LoadedModel("n", "p", "sdxl", "h", 1234, Some("vae"), Some("vh")), "loaded_ms", "vae_name", "vae_hash")
    rt(SamplerList(Seq(SamplerInfo("Euler a"), SamplerInfo("DDIM", false, "Uniform")), Seq("Automatic", "Karras")), "samplers", "schedulers")
    rt(SamplerInfo("x"), "default_scheduler")
    rt(UpscaleRequest("i.png", "R-ESRGAN 4x+", 2.0, "o", Map("R-ESRGAN 4x+" -> "p")), "image_path", "output_dir", "upscaler_paths")
    rt(PreprocessRequest("canny", "i.png", "o", Map("low_threshold" -> 100.0)), "image_path", "output_dir")
    rt(ImagePathResult("x.png"), "path")
    rt(MemoryInfo(1, 2, 3, 4), "allocated_mb", "reserved_mb", "total_mb", "ram_used_mb")
  }

  test("LoadedModel lenient read") {
    assertEquals(Json.fromJs[LoadedModel](ujson.read("""{"name":"n","path":"p","kind":"sd15","hash":"h"}""")),
      LoadedModel("n", "p", "sd15", "h"))
  }

  test("EngineError") {
    val e = EngineError(Protocol.ErrorCodes.Interrupted, "stopped")
    assert(e.isInterrupted)
    assertEquals(e.getMessage, "[interrupted] stopped")
    assert(!EngineError(Protocol.ErrorCodes.Oom, "x", "detail").isInterrupted)
  }

  test("protocol constants") {
    assertEquals(Protocol.Types.Txt2Img, "txt2img")
    assertEquals(Protocol.Types.ListModels, "list_models")
    assertEquals(Protocol.ErrorCodes.InvalidParams, "invalid_params")
    assertEquals(Version.protocol, 1)
    assert(Version.label.contains(Version.app))
  }
