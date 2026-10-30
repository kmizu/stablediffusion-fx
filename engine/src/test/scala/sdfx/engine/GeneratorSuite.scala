package sdfx.engine

import sdfx.core.config.Settings
import sdfx.core.infotext.Infotext
import sdfx.core.model.{GenerationParams, Img2ImgParams, HiresParams, LoraRef}
import sdfx.core.png.PngMetadata
import sdfx.core.protocol.*

import java.nio.file.Files
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

class GeneratorSuite extends munit.FunSuite:
  import FakeEngine.*

  override val munitTimeout: Duration = 120.seconds

  test("img2img refinement target obeys job limits including preceding native Hires") {
    val params = GenerationParams(width = 512, height = 512,
      hires = HiresParams(enabled = true, scale = 2),
      hooks = Map("img2img_hires_fix" -> ujson.Obj("ratio" -> 4, "upscaler" -> "Lanczos")))
    intercept[JobLimitExceeded](JobLimits(maxSide = 2048).check(params, 1))
  }

  test("img2img refinement discovers override adapters without enabling them in the base pass") {
    val p = GenerationParams(prompt = "cat", loras = Seq(LoraRef("My Lora", .2)), hooks = Map(
      "img2img_hires_fix" -> ujson.Obj("prompt" -> "dog <lora:detail:.6>", "upscaler" -> "Lanczos")))
    val prep = services.generator.prepare(p)
    assertEquals(prep.request.params.loras.map(_.name), Seq("My Lora"))
    assertEquals(prep.request.img2imgHiresLoras.get.map(_.name), Seq("My Lora", "detail"))
    assertEquals(prep.request.params.hooks("img2img_hires_fix")("prompt").str, "dog")
    assert(prep.recorded.extra.contains("img2img Hires Fix"))
  }

  test("img2img refinement preserves manual and tagged identity and resolves custom LBW presets") {
    Files.writeString(home.configDir.resolve("lbwpresets.txt"), "DETAIL:0,0,1\n")
    try {
      val p = GenerationParams(prompt = "cat <lora:My Lora:.7>", loras = Seq(LoraRef("My Lora", .2)), hooks = Map(
        "img2img_hires_fix" -> ujson.Obj("prompt" -> "dog <lora:detail:.6:lbw=DETAIL>", "upscaler" -> "Lanczos")))
      val prep = services.generator.prepare(p)
      assertEquals(prep.request.params.loras.map(_.weight), Seq(.2, .7))
      assertEquals(prep.request.img2imgHiresLoras.get.map(_.name), Seq("My Lora", "detail"))
      assertEquals(prep.request.lbwPresets("DETAIL"), "0,0,1")
      val restored = Infotext.toParams(Infotext.parse(Infotext.write(prep.recorded,
        sdfx.core.infotext.ImageFacts(seed = prep.recorded.seed))))
      val replay = services.generator.prepare(restored)
      assertEquals(replay.request.img2imgHiresLoras, prep.request.img2imgHiresLoras)
      val disabled = services.generator.prepare(p.copy(hooks = Map.empty, extra = prep.recorded.extra))
      assert(!disabled.recorded.extra.contains("img2img Hires Fix"))
    } finally Files.deleteIfExists(home.configDir.resolve("lbwpresets.txt"))
  }

  test("video policy follows the job snapshot and blocks img2vid before dispatch") {
    val session = new sdfx.core.png.ImageScrambleSession
    session.unlock("secret")
    val denied = services.generator.forImageSession(session.snapshot)
    assert(!denied.plaintextVideoAllowed)
    val error = intercept[IllegalStateException](denied.engineCall("img2vid", ujson.Obj()))
    assert(error.getMessage.contains("plaintext video"))
    session.setPlaintextVideoAllowed(true)
    val allowed = services.generator.forImageSession(session.snapshot)
    assert(allowed.plaintextVideoAllowed && !denied.plaintextVideoAllowed)
    session.lock()
    assert(services.generator.forImageSession(session.snapshot).plaintextVideoAllowed)
  }

  test("intermediate captures use a persistent destination without recording machine paths") {
    val p = GenerationParams(seed = 5, hooks = Map("save_intermediate" -> ujson.Obj("every_n" -> 2,
      "output_dir" -> "../../untrusted")))
    val prep = services.generator.prepare(p)
    val capture = prep.request.params.hooks("save_intermediate")
    assertEquals(capture("output_dir").str, home.outputs(services.settings).toAbsolutePath.toString)
    assertEquals(prep.recorded.hooks("save_intermediate")("output_dir").str, "../../untrusted")
    val normal = services.generator.prepare(p.copy(hooks = Map("save_intermediate" -> ujson.Obj("every_n" -> 2))))
    assert(!normal.recorded.hooks("save_intermediate").obj.contains("output_dir"))
    assertEquals(prep.request.outputDir, home.tmpDir(services.settings).toAbsolutePath.toString)
  }

  test("Composable trace uses a persistent destination without recording machine paths") {
    val p = GenerationParams(hooks = Map("composable_lora" -> ujson.Obj("enabled" -> true, "plot_weights" -> true)))
    val prep = services.generator.prepare(p)
    assertEquals(prep.request.params.hooks("composable_lora")("output_dir").str,
      home.outputs(services.settings).toAbsolutePath.toString)
    assert(!prep.recorded.hooks("composable_lora").obj.contains("output_dir"))
  }

  test("Composable trace survives image cleanup and is logged rather than registered as temporary") {
    val traceHome = tempHome()
    val traceServices = new AppServices(traceHome, Settings(lastModel = Some("foo")), dryRun = true, builtins = Nil,
      launchOverride = Some(() => launch()), persistSettings = false, installShutdownHook = false)
    val logs = scala.collection.mutable.ArrayBuffer.empty[String]
    val temps = scala.collection.mutable.ArrayBuffer.empty[java.nio.file.Path]
    val listener = new GenerationListener:
      override def onLog(s: String): Unit = logs += s
      override def onTempFile(p: java.nio.file.Path): Unit = temps += p
    try {
      traceServices.start()
      val images = traceServices.generator.generateWith(GenerationParams(seed = 9, steps = 1, width = 64, height = 64,
        hooks = Map("composable_lora" -> ujson.Obj("enabled" -> true, "plot_weights" -> true))), listener)
      val trace = java.nio.file.Paths.get(logs.find(_.startsWith("Composable LoRA trace: ")).get.stripPrefix("Composable LoRA trace: "))
      assert(Files.isRegularFile(trace))
      assert(trace.startsWith(traceHome.outputs(traceServices.settings)))
      assert(!temps.contains(trace))
      assert(temps.forall(p => !Files.exists(p)))
      assert(images.forall(i => Files.isRegularFile(i.path)))
    } finally {
      traceServices.shutdown()
      deleteTree(traceHome.root)
    }
  }

  private lazy val home = tempHome()
  private lazy val services =
    val s = new AppServices(home, Settings(lastModel = Some("foo")), dryRun = true, builtins = Nil,
      launchOverride = Some(() => launch()), persistSettings = false)
    val stale = Files.writeString(home.tmpDir(s.settings).resolve("stale-leftover.png"), "x")
    Files.setLastModifiedTime(stale, java.nio.file.attribute.FileTime.fromMillis(System.currentTimeMillis() - 7L * 3600 * 1000))
    // A fresh file may belong to another process sharing this home (GUI + MCP) and must survive.
    Files.writeString(home.tmpDir(s.settings).resolve("fresh-other-process.png"), "x")
    s.start()
    s

  override def afterAll(): Unit =
    services.shutdown()
    deleteTree(home.root)

  private def engineState = await(services.client.raw("get_state"))

  test("Composable LoRA keeps branch and step tags while resolving repeated adapters only once") {
    val prompt = "red <lora:My Lora:.5> AND blue [<lora:My Lora:.8>:2]"
    val enabled = services.generator.prepare(GenerationParams(prompt = prompt, seed = 7,
      hooks = Map("composable_lora" -> ujson.Obj("with_step" -> true))))
    assertEquals(enabled.request.params.prompt, prompt)
    assertEquals(enabled.request.params.loras.map(_.name), Seq("My Lora"))
    assertEquals(enabled.request.params.loras.head.weight, .5)
    assertEquals(enabled.request.loraPaths.keySet, Set("My Lora"))
    assertEquals(enabled.request.manualLoras, Some(Nil))
    val manual = LoraRef("My Lora", weight = .6, unet = Some(.3))
    val overlap = services.generator.prepare(GenerationParams(prompt = prompt, loras = Seq(manual),
      hires = HiresParams(enabled = true, prompt = "sky"),
      hooks = Map("composable_lora" -> ujson.Obj("with_step" -> true))))
    assertEquals(overlap.request.manualLoras, Some(Seq(manual)))
    assertEquals(overlap.request.params.loras.map(_.name), Seq("My Lora"))
    val disabled = services.generator.prepare(GenerationParams(prompt = prompt, seed = 7,
      hooks = Map("composable_lora" -> ujson.Obj("enabled" -> false))))
    assert(!disabled.request.params.prompt.contains("<lora:"))
    assertEquals(disabled.request.manualLoras, None)
    assertEquals(disabled.request.params.loras.head.weight, 1.3)
    val textDisabled = services.generator.prepare(GenerationParams(prompt = prompt, seed = 7,
      hooks = Map("composable_lora" -> ujson.Obj("enabled" -> "false"))))
    assertEquals(textDisabled.request.params.prompt, disabled.request.params.prompt)
    assertEquals(textDisabled.request.params.loras, disabled.request.params.loras)
  }

  test("start refreshes the catalog and loads the last model") {
    val c = services.catalog
    assertEquals(c.checkpoints.map(_.name).toSet, Set("foo", "sdxl/Bar_XL"))
    assertEquals(c.loras.map(_.name).toSet, Set("My Lora", "sub/detail"))
    assertEquals(c.samplers.schedulers, Seq("Automatic", "Karras"))
    assert(engineState("loaded").str.endsWith("foo.safetensors"))
    assertEquals(services.supervisor.loadedModel.map(_.name), Some("foo"))
  }

  test("txt2img end to end: seeds, LoRA extraction, infotext, naming") {
    val p = GenerationParams(prompt = "a cat <lora:My Lora:0.7>, best", negativePrompt = "ugly", seed = -1, steps = 3,
      width = 64, height = 72, batchSize = 2, styles = Seq("Pretty"))
    val out = services.generator.generate(p)
    assertEquals(out.size, 2)
    val seed0 = out.head.seed
    assert(seed0 >= 0 && seed0 < 4294967295L)
    assertEquals(out(1).seed, seed0 + 1)

    // What the engine received: stripped prompt + resolved LoRA path.
    val req = engineState("last_generation")("request")
    assertEquals(req("params")("prompt").str, "a cat, best, masterpiece")
    assertEquals(req("params")("seed").num.toLong, seed0)
    assertEquals(req("params")("loras").arr.map(l => (l("name").str, l("weight").num)).toSeq, Seq(("My Lora", 0.7)))
    assert(req("lora_paths")("My Lora").str.endsWith("My Lora.safetensors"))
    assertEquals(engineState("last_generation")("type").str, "txt2img")

    out.zipWithIndex.foreach { (img, i) =>
      assert(Files.isRegularFile(img.path))
      assertEquals(img.path.getFileName.toString, f"0000$i-${seed0 + i}.png")
      assertEquals(img.path.getParent.getParent.getFileName.toString, "txt2img")
      val text = PngMetadata.readParameters(img.path).getOrElse(fail("no parameters"))
      assertEquals(text, img.infotext)
      val pi = Infotext.parse(text)
      assertEquals(pi.prompt, "a cat <lora:My Lora:0.7>, best, masterpiece")
      assertEquals(pi.negativePrompt, "ugly, lowres")
      assertEquals(pi.get("Seed"), Some((seed0 + i).toString))
      assertEquals(pi.get("Size"), Some("64x72"))
      assertEquals(pi.get("Model"), Some("foo"))
      assertEquals(pi.get("Model hash"), Some("abcdef1234"))
      assertEquals(pi.get("Schedule type"), Some("Karras"))
      assertEquals(pi.get("Lora hashes"), Some("My Lora: 1234abcd"))
      val back = Infotext.toParams(pi)
      assertEquals((back.width, back.height, back.steps, back.seed), (64, 72, 3, seed0 + i))
      assertEquals(img.params.styles, Nil)
      assertEquals(PngMetadata.readImage(img.path).getWidth, 64)
    }
    // temp files from the engine are removed
    val tmp = home.tmpDir(services.settings)
    assert(Files.list(tmp).iterator().asScala.forall(!_.getFileName.toString.startsWith("tmp-")))

    // the counter continues
    val next = services.generator.generate(p.copy(seed = 42, batchSize = 1, styles = Nil))
    assertEquals(next.head.path.getFileName.toString, "00002-42.png")
  }

  test("scrambled img2img input is decoded for worker and cleaned after generation") {
    val src = home.root.resolve("protected-input.png")
    val original = new java.awt.image.BufferedImage(8, 8, java.awt.image.BufferedImage.TYPE_INT_RGB)
    for y <- 0 until 8; x <- 0 until 8 do original.setRGB(x, y, (x << 16) | (y << 8) | 7)
    sdfx.core.png.ScrambledPng.write(original, src, Nil, Some("secret"))
    val session = new sdfx.core.png.ImageScrambleSession
    session.unlock("secret")
    var observed: Option[java.nio.file.Path] = None
    var pixels: Array[Int] = Array.emptyIntArray
    var assets = ""
    val listener = new GenerationListener:
      override private[engine] def imageSecret = session.snapshot
      override private[engine] def payloadCaptureOptions = Some(CaptureOptions(includeImageContents = true))
      override def onPreparedRequest(snapshot: sdfx.core.payload.PayloadSnapshot): Unit =
        assets = snapshot.assetsJson.getOrElse("")
        val body = ujson.read(snapshot.bodyJson.get)
        val path = java.nio.file.Paths.get(body("params")("img2img")("init_image").str)
        observed = Some(path)
        pixels = PngMetadata.readImage(path).getRGB(0, 0, 8, 8, null, 0, 8)
    try
      val p = GenerationParams(seed = 4, steps = 1, width = 64, height = 64, img2img = Some(Img2ImgParams(src.toString)))
      val out = services.generator.generateWith(p, listener)
      assertEquals(out.size, 1)
      assert(observed.nonEmpty && observed.get != src)
      assert(original.getRGB(0, 0, 8, 8, null, 0, 8).sameElements(pixels))
      assert(!Files.exists(observed.get))
      assert(!assets.contains("png_data_url"))
      observed = None
      intercept[EngineError](services.generator.generateWith(p.copy(extra = Map("force_error" -> "internal")), listener))
      assert(observed.nonEmpty && !Files.exists(observed.get))
      assertEquals(PngMetadata.readTexts(src).get("Encrypt"), Some("pixel_shuffle_3"))
    finally Files.deleteIfExists(src)
  }

  test("worker input paths for mask, ControlNet and image hooks are decoded together") {
    val src = home.root.resolve("protected-input-many.png")
    val image = new java.awt.image.BufferedImage(8, 8, java.awt.image.BufferedImage.TYPE_INT_RGB)
    for y <- 0 until 8; x <- 0 until 8 do image.setRGB(x, y, (x << 16) | (y << 8) | 7)
    sdfx.core.png.ScrambledPng.write(image, src, Nil, Some("secret"))
    val session = new sdfx.core.png.ImageScrambleSession; session.unlock("secret")
    var paths = Seq.empty[java.nio.file.Path]
    var images = Seq.empty[Array[Int]]
    val listener = new GenerationListener:
      override private[engine] def imageSecret = session.snapshot
      override def onPreparedRequest(snapshot: sdfx.core.payload.PayloadSnapshot): Unit =
        val params = ujson.read(snapshot.bodyJson.get)("params")
        val values = Seq(params("img2img")("init_image").str, params("img2img")("mask").str,
          params("controlnet")(0)("image").str, params("hooks")("inpaint_difference")("base_path").str,
          params("hooks")("reference")("image").str, params("hooks")("photomaker")("images")(0).str)
        paths = values.map(java.nio.file.Paths.get(_))
        images = paths.map(p => PngMetadata.readImage(p).getRGB(0, 0, 8, 8, null, 0, 8))
    try
      val path = src.toString
      val p = GenerationParams(seed = 4, steps = 1, width = 64, height = 64,
        img2img = Some(Img2ImgParams(path, mask = Some(path))),
        controlnet = Seq(sdfx.core.model.ControlNetUnit(image = Some(path))),
        hooks = Map("inpaint_difference" -> ujson.Obj("base_path" -> path),
          "reference" -> ujson.Obj("image" -> path), "photomaker" -> ujson.Obj("images" -> ujson.Arr(path))))
      services.generator.generateWith(p, listener)
      assertEquals(paths.size, 6)
      assert(paths.forall(_ != src))
      assert(images.forall(_.sameElements(image.getRGB(0, 0, 8, 8, null, 0, 8))))
      assert(paths.forall(p => !Files.exists(p)))
      assertEquals(PngMetadata.readTexts(src).get("Encrypt"), Some("pixel_shuffle_3"))
    finally Files.deleteIfExists(src)
  }

  test("direct image tools use a captured session secret and clean worker inputs") {
    val src = home.root.resolve("protected-tool.png")
    val image = new java.awt.image.BufferedImage(8, 8, java.awt.image.BufferedImage.TYPE_INT_RGB)
    sdfx.core.png.ScrambledPng.write(image, src, Nil, Some("direct secret"))
    val session = new sdfx.core.png.ImageScrambleSession; session.unlock("direct secret")
    val snapshot = session.snapshot
    session.lock()
    try
      services.generator.upscaleImage(src, "Lanczos", 2, secret = snapshot)
      val upscale = java.nio.file.Paths.get(await(services.client.raw("get_state"))("last_extra")("params")("image_path").str)
      assertNotEquals(upscale, src); assert(!Files.exists(upscale))
      services.generator.preprocess(src, "canny", secret = snapshot)
      val pre = java.nio.file.Paths.get(await(services.client.raw("get_state"))("last_extra")("params")("image_path").str)
      assertNotEquals(pre, src); assert(!Files.exists(pre))
      val context = services.generator.forImageSession(snapshot)
      val decoded = context.readImage(src)
      assertEquals(decoded.getWidth, 8)
      val out = context.saveImage(decoded, "direct-test", GenerationParams(seed = 8), "test")
      assertEquals(PngMetadata.readTexts(out.path).get("Encrypt"), Some("pixel_shuffle_3"))
      assert(!out.toString.contains("direct secret"))
      intercept[IllegalArgumentException](services.generator.upscaleImage(src, "Lanczos", 2))
    finally Files.deleteIfExists(src)
  }

  test("checkpoint and VAE switching follow params") {
    services.generator.generate(GenerationParams(prompt = "x", seed = 1, steps = 1, width = 64, height = 64,
      checkpoint = Some("Bar_XL"), vae = Some("vae-ft-mse")))
    val st = engineState
    assert(st("loaded").str.endsWith("Bar_XL.safetensors"), st.toString)
    assert(st("vae").str.endsWith("vae-ft-mse.safetensors"), st.toString)
    val loads = st("load_count").num.toInt
    // same checkpoint again -> no reload
    val out = services.generator.generate(GenerationParams(prompt = "x", seed = 2, steps = 1, width = 64, height = 64,
      checkpoint = Some("sdxl/Bar_XL"), vae = Some("vae-ft-mse")))
    assertEquals(engineState("load_count").num.toInt, loads)
    assertEquals(Infotext.parse(out.head.infotext).get("VAE"), Some("vae-ft-mse.safetensors"))
    intercept[EngineError](services.generator.generate(GenerationParams(seed = 1, checkpoint = Some("nope"))))
    services.loadCheckpoint("foo")
  }

  test("img2img uses the img2img request type and output folder") {
    val init = services.generator.generate(GenerationParams(prompt = "src", seed = 7, steps = 1, width = 64, height = 64)).head.path
    val out = services.generator.generate(GenerationParams(prompt = "edit", seed = 8, steps = 1, width = 64, height = 64,
      img2img = Some(Img2ImgParams(init.toString, denoisingStrength = 0.4))))
    assertEquals(engineState("last_generation")("type").str, "img2img")
    assertEquals(out.head.path.getParent.getParent.getFileName.toString, "img2img")
    assertEquals(Infotext.parse(out.head.infotext).get("Denoising strength"), Some("0.4"))
  }

  test("invalid params and engine errors surface as EngineError") {
    val e = intercept[EngineError](services.generator.generate(GenerationParams(width = 63)))
    assertEquals(e.code, Protocol.ErrorCodes.InvalidParams)
  }

  test("unknown LoRA is skipped with a warning") {
    val logs = scala.collection.mutable.ArrayBuffer.empty[String]
    val l = new GenerationListener:
      override def onLog(m: String): Unit = logs += m
    services.generator.generateWith(GenerationParams(prompt = "p <lora:ghost:1> <lora:detail:0.5>", seed = 3, steps = 1, width = 64, height = 64), l)
    assert(logs.exists(_.contains("LoRA not found: ghost")), logs.toString)
    val req = engineState("last_generation")("request")
    assertEquals(req("params")("loras").arr.map(_("name").str).toSeq, Seq("detail"))
    assert(req("lora_paths")("detail").str.replace('\\', '/').endsWith("sub/detail.safetensors"))
  }

  test("stale temp files are removed at startup") {
    services.catalog
    assert(!Files.exists(home.tmpDir(services.settings).resolve("stale-leftover.png")))
    assert(Files.exists(home.tmpDir(services.settings).resolve("fresh-other-process.png")))
  }

  test("LoRA / checkpoint paths outside the model dirs are not accepted") {
    Files.write(home.root.resolve("outside.safetensors"), Array[Byte](1))
    val logs = scala.collection.mutable.ArrayBuffer.empty[String]
    val l = new GenerationListener:
      override def onLog(m: String): Unit = logs += m
    val prompt = "p <lora:outside.safetensors:1> <lora:models/lora/My Lora.safetensors:1>"
    services.generator.generateWith(GenerationParams(prompt = prompt, seed = 3, steps = 1, width = 64, height = 64), l)
    assert(logs.exists(_.contains("LoRA not found: outside.safetensors")), logs.toString)
    val req = engineState("last_generation")("request")
    assertEquals(req("params")("loras").arr.map(_("name").str).toSeq, Seq("models/lora/My Lora.safetensors"))
    intercept[EngineError](services.generator.generate(GenerationParams(seed = 1, checkpoint = Some("outside.safetensors"))))
    intercept[EngineError](services.loadCheckpoint("../outside.safetensors"))
  }

  test("engine interrupted / oom errors propagate with their code") {
    val base = GenerationParams(prompt = "e", seed = 1, steps = 1, width = 64, height = 64)
    val i = intercept[EngineError](services.generator.generate(base.copy(extra = Map("force_error" -> "interrupted"))))
    assert(i.isInterrupted)
    val o = intercept[EngineError](services.generator.generate(base.copy(extra = Map("force_error" -> "oom"))))
    assertEquals(o.code, Protocol.ErrorCodes.Oom)
  }

  test("hires prompt composition reaches prepared request and avoids replaying append") {
    val p = GenerationParams(prompt = "cat", negativePrompt = "blur", seed = 1,
      hires = sdfx.core.model.HiresParams(enabled = true, prompt = "detail"),
      hooks = Map("hires_fix_tweaks" -> ujson.Obj("prompt_mode" -> "Append", "separator" -> ", ")))
    val prepared = services.generator.prepare(p)
    assertEquals(prepared.request.params.hires.prompt, "cat, detail")
    assertEquals(prepared.request.params.hires.negativePrompt, "blur")
    assertEquals(services.generator.prepare(prepared.recorded).request.params.hires.prompt, "cat, detail")
  }

  test("hires network removal separates pass adapters while preserving manual references") {
    val p = GenerationParams(prompt = "cat <lora:detail:0.5>", seed = 1,
      hires = sdfx.core.model.HiresParams(enabled = true),
      hooks = Map("hires_fix_tweaks" -> ujson.Obj("remove_networks" -> true)))
    val prepared = services.generator.prepare(p).request
    assertEquals(prepared.params.loras.map(_.name), Seq("detail"))
    assertEquals(prepared.hiresLoras, Some(Nil))
    assert(prepared.loraPaths.contains("detail"))
    val manual = services.generator.prepare(p.copy(prompt = "cat", loras = Seq(sdfx.core.model.LoraRef("detail")))).request
    assertEquals(manual.hiresLoras.map(_.map(_.name)), Some(Seq("detail")))
    assertEquals(services.generator.prepare(p.copy(hooks = Map.empty)).request.hiresLoras, None)
  }

  test("hires-only block weights carry custom presets with the separate pass adapters") {
    val file = home.configDir.resolve("lbwpresets.txt")
    val original = Option.when(Files.exists(file))(Files.readString(file))
    try
      Files.writeString(file, "HIRES_CUSTOM:1,0,1\n")
      val p = GenerationParams(prompt = "cat", seed = 1,
        hires = sdfx.core.model.HiresParams(enabled = true, prompt = "cat <lora:detail:1:lbw=HIRES_CUSTOM>"),
        hooks = Map("hires_fix_tweaks" -> ujson.Obj()))
      val request = services.generator.prepare(p).request
      assertEquals(request.params.loras, Nil)
      assertEquals(request.hiresLoras.get.head.lbw, Some("HIRES_CUSTOM"))
      assertEquals(request.lbwPresets.get("HIRES_CUSTOM"), Some("1,0,1"))
    finally original match
      case Some(text) => Files.writeString(file, text)
      case None => Files.deleteIfExists(file)
  }

  test("manual and same-name tagged adapters retain separate pass behavior across PNG restore") {
    val original = services.settings
    val manual = sdfx.core.model.LoraRef("detail", .6, Some(.4), Some("ALL"), Some(.1), Some(.9))
    val p = GenerationParams(prompt = "cat <lora:detail:0.2>", seed = 7, steps = 1, width = 64, height = 64,
      loras = Seq(manual), hires = sdfx.core.model.HiresParams(enabled = true),
      hooks = Map("hires_fix_tweaks" -> ujson.Obj("remove_networks" -> true)))
    try
      for save <- Seq(true, false) do
        services.updateSettings(_ => original.copy(hiresFixTweaks = original.hiresFixTweaks.copy(saveTemplate = save)))
        val before = services.generator.prepare(p).request
        val image = services.generator.generate(p).head
        val parsed = Infotext.parse(PngMetadata.readParameters(image.path).get)
        val restored = sdfx.core.infotext.HiresTemplateRestore.restore(Infotext.toParams(parsed), parsed.fields, Nil)
        val after = services.generator.prepare(restored).request
        assertEquals(restored.loras, Seq(manual))
        assertEquals(after.params.loras, before.params.loras)
        assertEquals(after.hiresLoras, before.hiresLoras)
        assertEquals(after.hiresLoras.get, Seq(manual))
    finally services.updateSettings(_ => original)
  }

  test("hires final directory preserves engine temp paths, global precedence and collision safety") {
    val p = GenerationParams(seed = 55, steps = 1, width = 64, height = 64,
      hires = sdfx.core.model.HiresParams(enabled = true),
      hooks = Map("hires_fix_tweaks" -> ujson.Obj("output_dir" -> "hires-results")))
    val prepared = services.generator.prepare(p)
    assertEquals(prepared.request.outputDir, home.tmpDir(services.settings).toAbsolutePath.toString)
    val a = services.generator.generate(p).head
    val b = services.generator.generate(p).head
    assert(a.path.startsWith(home.resolve("hires-results")))
    assert(b.path.startsWith(home.resolve("hires-results")))
    assert(a.path != b.path)
    assert(Files.isRegularFile(a.path) && Files.isRegularFile(b.path))
    assertEquals(services.settings.outputDir, "outputs")
    val old = services.settings
    try
      services.updateSettings(_.copy(outputDir = "global-results"))
      val global = services.generator.generate(p).head
      assert(global.path.startsWith(home.resolve("global-results")))
    finally services.updateSettings(_ => old)
    try
      val movingSettings = new GenerationListener:
        override def onProgress(p: EngineEvent.Progress): Unit =
          services.updateSettings(_.copy(outputDir = "changed-during-job"))
      val stable = services.generator.generateWith(p, movingSettings).head
      assert(stable.path.startsWith(home.resolve("hires-results")))
    finally services.updateSettings(_ => old)
    val ordinary = services.generator.generate(p.copy(hires = p.hires.copy(enabled = false))).head
    assert(ordinary.path.startsWith(home.outputs(services.settings)))
  }

  test("persistent hires output preference survives settings reload and gallery rebuild") {
    val original = services.settings
    try
      val configured = original.copy(hiresFixTweaks = sdfx.core.config.HiresFixTweaksPreferences("persisted-hires"))
      val file = home.configDir.resolve("hires-settings-test.json")
      Settings.save(file, configured)
      services.updateSettings(_ => Settings.load(file))
      val p = GenerationParams(seed = 71, steps = 1, width = 64, height = 64,
        hires = sdfx.core.model.HiresParams(enabled = true))
      val image = services.generator.generate(p).head
      assert(image.path.startsWith(home.resolve("persisted-hires")))
      assert(services.gallery.scan().exists(_.path == image.path))
      assert(services.resetGallery().scan().exists(_.path == image.path))
      services.updateSettings(_.copy(outputDir = "global-with-old-hires"))
      assert(services.resetGallery().scan().exists(_.path == image.path))
    finally
      services.updateSettings(_ => original)
      services.resetGallery()
  }

  test("partial interrupted results save gallery images before propagating cancellation") {
    for transient <- Seq(false, true) do
      val saved = scala.collection.mutable.ArrayBuffer.empty[sdfx.core.plugins.api.OutputImage]
      var cancelled = false
      val listener = new GenerationListener:
        override def onImageSaved(image: sdfx.core.plugins.api.OutputImage, grid: Boolean): Unit = saved += image
        override def onEngineInterrupted(): Unit = cancelled = true
      val p = GenerationParams(seed = 23, steps = 1, width = 64, height = 64,
        extra = Map("partial_interrupted" -> "true"))
      val error = intercept[EngineError] {
        if transient then services.generator.generateTransientWith(p, listener)
        else services.generator.generateWith(p, listener)
      }
      assert(error.isInterrupted)
      assert(cancelled)
      assertEquals(saved.size, 1)
      assert(Files.isRegularFile(saved.head.path))
      assert(PngMetadata.readParameters(saved.head.path).nonEmpty)
  }

  test("dedicated hires seed facts survive typed transport and PNG saving") {
    val image = services.generator.generate(GenerationParams(seed = 23, steps = 1, width = 64, height = 64,
      extra = Map("hires_metadata" -> "true"))).head
    val text = PngMetadata.readParameters(image.path).get
    assertEquals(image.seed, 23L)
    assert(text.contains("Hires seed:"))
    assert(text.contains("'Seed':100"))
    assert(text.contains("'Subseed':200"))
    assert(text.contains("'Strength':0.4"))
    assert(text.contains("'Resize':[512,768]"))
    val large = services.generator.generate(GenerationParams(seed = 23, steps = 1, width = 64, height = 64,
      extra = Map("hires_metadata_large" -> "true"))).head
    val restored = Infotext.toParams(Infotext.parse(PngMetadata.readParameters(large.path).get))
    assertEquals(restored.hooks("hires_fix_tweaks")("seed"), ujson.Str("9007199254740993"))
    assertEquals(restored.hooks("hires_fix_tweaks")("subseed"), ujson.Str(Long.MaxValue.toString))
  }

  test("PNG stores raw templates separately and save preference suppresses stale metadata") {
    val original = services.settings
    val p = GenerationParams(prompt = "cat", negativePrompt = "bad", seed = 23, steps = 1, width = 64, height = 64,
      hires = sdfx.core.model.HiresParams(enabled = true, prompt = "detail"),
      hooks = Map("hires_fix_tweaks" -> ujson.Obj("prompt_mode" -> "Append", "append_separator" -> " + ")))
    try {
      services.updateSettings(_ => original.copy(hiresFixTweaks = original.hiresFixTweaks.copy(saveTemplate = true)))
      val image = services.generator.generate(p).head
      val fields = Infotext.parse(PngMetadata.readParameters(image.path).get).fields
      assertEquals(fields("Hires prompt"), "cat + detail")
      val template = ujson.read(fields("NUI Hires template"))
      assertEquals(template("hires_prompt").str, "detail")
      assertEquals(template("prompt").str, "cat")
      assertEquals(sdfx.core.infotext.HiresPromptMetadata.parse(fields("Hires prompt mode")).get.positive.get.prompt, "detail")
      services.updateSettings(_ => original.copy(hiresFixTweaks = original.hiresFixTweaks.copy(saveTemplate = false)))
      val replay = services.generator.generate(p.copy(extra = fields)).head
      val omitted = Infotext.parse(PngMetadata.readParameters(replay.path).get).fields
      assert(!omitted.contains("NUI Hires template"))
      assert(!omitted.contains("Hires prompt mode"))
    } finally services.updateSettings(_ => original)
  }

  test("upscale, memory, choices") {
    val img = services.generator.generate(GenerationParams(prompt = "u", seed = 9, steps = 1, width = 64, height = 64)).head.path
    val up = services.generator.upscale(img, "4x-Ultra", 2)
    assert(Files.isRegularFile(up))
    assertEquals(await(services.client.memory()).totalMb, 8192.0)
    assertEquals(services.generator.choices(sdfx.core.plugins.api.ChoiceSource.Samplers), Seq("Euler a", "DPM++ 2M"))
    assert(services.generator.choices(sdfx.core.plugins.api.ChoiceSource.Upscalers).contains("4x-Ultra"))
    assert(services.generator.tempFile(".png").getParent == home.tmpDir(services.settings))
  }
