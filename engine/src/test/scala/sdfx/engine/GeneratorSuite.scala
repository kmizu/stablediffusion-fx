package sdfx.engine

import sdfx.core.config.Settings
import sdfx.core.infotext.Infotext
import sdfx.core.model.{GenerationParams, Img2ImgParams}
import sdfx.core.png.PngMetadata
import sdfx.core.protocol.*

import java.nio.file.Files
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

class GeneratorSuite extends munit.FunSuite:
  import FakeEngine.*

  override val munitTimeout: Duration = 120.seconds

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
    val disabled = services.generator.prepare(GenerationParams(prompt = prompt, seed = 7,
      hooks = Map("composable_lora" -> ujson.Obj("enabled" -> false))))
    assert(!disabled.request.params.prompt.contains("<lora:"))
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

  test("upscale, memory, choices") {
    val img = services.generator.generate(GenerationParams(prompt = "u", seed = 9, steps = 1, width = 64, height = 64)).head.path
    val up = services.generator.upscale(img, "4x-Ultra", 2)
    assert(Files.isRegularFile(up))
    assertEquals(await(services.client.memory()).totalMb, 8192.0)
    assertEquals(services.generator.choices(sdfx.core.plugins.api.ChoiceSource.Samplers), Seq("Euler a", "DPM++ 2M"))
    assert(services.generator.choices(sdfx.core.plugins.api.ChoiceSource.Upscalers).contains("4x-Ultra"))
    assert(services.generator.tempFile(".png").getParent == home.tmpDir(services.settings))
  }
