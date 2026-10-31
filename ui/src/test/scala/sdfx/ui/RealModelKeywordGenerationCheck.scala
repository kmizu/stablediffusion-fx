package sdfx.ui

import java.nio.file.{Files, Path}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, LinkedBlockingQueue, TimeUnit}
import javafx.application.Platform
import javafx.scene.{Node, Parent}
import javafx.scene.control.{Button, CheckBox, ComboBox, TabPane, TextField, TitledPane}
import javafx.stage.Stage
import scala.jdk.CollectionConverters.*
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.infotext.{Infotext, Img2ImgHiresMetadata}
import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.OutputImage
import sdfx.core.png.PngMetadata
import sdfx.engine.{AppServices, JobEvent}
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{Mode, UiState}

/** Real MainWindow mapping controls and generation, with optional LoRA tag and textual inversion. */
object RealModelKeywordGenerationCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A] { def call(): A = body })
    Platform.runLater(task); task.get(30, TimeUnit.SECONDS)
  def main(args: Array[String]): Unit =
    val options = args.filter(_.startsWith("--vram-mode="))
    val checkReenable = args.contains("--check-lora-reenable")
    require(options.length <= 1, "only one vram mode may be specified")
    val positional = args.filterNot(value => value.startsWith("--vram-mode=") || value == "--check-lora-reenable")
    require(positional.length >= 4 && positional.length <= 7, "isolated-home python checkpoint size [lora [tag-options [embedding]]] [--vram-mode=mode]")
    val home = AppHome(Path.of(positional(0)).toAbsolutePath)
    if Files.exists(home.root) then
      val entries = Files.list(home.root)
      try require(!entries.findAny().isPresent, "acceptance home must be empty")
      finally entries.close()
    val model = Path.of(positional(2)).toAbsolutePath
    val size = positional(3).toInt
    val lora = positional.lift(4).map(p => Path.of(p).toAbsolutePath)
    val loraName = lora.map(p => p.getFileName.toString.replaceFirst("\\.[^.]+$", ""))
    require(!checkReenable || lora.isDefined, "LoRA re-enable check needs a LoRA")
    val tagOptions = positional.lift(5).filter(_.nonEmpty)
    val embedding = positional.lift(6).filter(_.nonEmpty).map(p => Path.of(p).toAbsolutePath)
    val embeddingName = embedding.map(p => p.getFileName.toString.replaceFirst("\\.[^.]+$", ""))
    val vramMode = options.headOption.map(_.stripPrefix("--vram-mode=")).getOrElse("gpu")
    require(Set("gpu", "model_offload", "sequential_offload")(vramMode), "invalid vram mode")
    require(tagOptions.forall(value => !value.contains('>') && !value.contains('\n')), "invalid LoRA tag options")
    Files.createDirectories(home.pythonDir)
    val services = new AppServices(home, Settings(pythonExe = Some(Path.of(positional(1)).toAbsolutePath.toString), vramMode = vramMode,
      modelDirs = ModelDirs(checkpoints = Seq(model.getParent.toString), lora = lora.toSeq.map(_.getParent.toString),
        embeddings = embedding.toSeq.map(_.getParent.toString))), dryRun = false,
      builtins = StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false)
    val results = new LinkedBlockingQueue[Either[String, Seq[OutputImage]]]()
    services.queue.addListener {
      case e: JobEvent.JobFinished => results.offer(Right(e.output.images)); ()
      case e: JobEvent.JobFailed =>
        val detail = e.error match
          case err: sdfx.core.protocol.EngineError => err.detail
          case other => other.toString
        Files.writeString(home.root.resolve("worker-error.txt"), detail)
        results.offer(Left(e.message)); ()
      case _ => ()
    }
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    def nodes(node: Node): Seq[Node] = Seq(node) ++ (node match
      case p: Parent => p.getChildrenUnmodifiable.asScala.toSeq.flatMap(nodes)
      case _ => Nil)
    def find(id: String): Node = nodes(window.root).find(_.getId == id).getOrElse(sys.error(s"Missing $id"))
    def findGeneration(id: String): Node = nodes(window.panels(Mode.Txt2Img).node).find(_.getId == id)
      .getOrElse(sys.error(s"Missing txt2img control $id"))
    def waitFor(label: String)(f: => Boolean): Unit =
      val end = System.nanoTime() + TimeUnit.SECONDS.toNanos(45)
      while !f && System.nanoTime() < end do Thread.sleep(50)
      require(f, s"Timed out: $label")
    def generate(): Seq[OutputImage] =
      waitFor("queue idle")(services.queue.isIdle && fx(!window.progress.get.running))
      fx(window.statusBar.generateBtn.fire())
      val result = results.poll(300, TimeUnit.SECONDS)
      require(result != null, "generation timed out")
      val captured = services.payloads.snapshots.flatMap(_.bodyJson).map(ujson.read(_))
      Files.writeString(home.root.resolve("generation-requests.json"), ujson.Arr.from(captured).render(indent = 2))
      result.fold(e => throw new IllegalStateException(e), identity)
    try
      services.start(); val loaded = services.loadCheckpoint(model.toString, remember = false)
      fx {
        window = new MainWindow(new Stage(), services, UiState(params = Map("txt2img" -> GenerationParams(
          prompt = "a ceramic teapot", seed = 71, width = size, height = size, steps = 5, batchSize = 2))), CliOptions(), _ => ())
        window.show(); window.model.checkpoint.set(loaded.name)
        val tabs = find("main-tabs").asInstanceOf[TabPane]
        val presets = tabs.getTabs.asScala.find(_.getId == "model-presets-tab")
          .getOrElse(sys.error("Accepted Model Presets tab is missing after integration"))
        tabs.getSelectionModel.select(presets)
        require(presets.getContent.getId == "model-presets-pane", "Model Presets content is not mounted")
        tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getId == "model-keyword-tab").get)
      }
      waitFor("mapping editor")(fx(nodes(window.root).exists(_.getId == "model-keyword-save")))
      waitFor("mapping check")(fx(!find("model-keyword-save").asInstanceOf[Button].isDisabled))
      fx {
        find("model-keyword-mapping").asInstanceOf[TextField].setText("red|blue")
        find("model-keyword-save").asInstanceOf[Button].fire()
      }
      waitFor("keyword choices")(fx(window.ctx.modelKeywordChoices.asScala.contains("blue")))
      val mcp = new sdfx.mcp.McpTools(services)
      require(mcp.all.exists(_.name == "model_presets"), "Model Presets MCP tool missing after integration")
      val mcpLookup = mcp.all.find(_.name == "model_keyword_lookup").get
      val checkpointLookup = mcpLookup.call(ujson.Obj(), sdfx.mcp.ToolContext.none).structured.get
      require(checkpointLookup("model").str == model.getFileName.toString &&
        checkpointLookup("match_source").str == "custom-mappings.txt" &&
        checkpointLookup("keywords").arr.map(_.str).toSeq == Seq("red", "blue") &&
        checkpointLookup("hash").str.matches("[0-9a-f]{8}"), "real MCP checkpoint lookup mismatch")
      loraName.foreach { name =>
        fx {
          find("model-keyword-kind").asInstanceOf[ComboBox[String]].setValue("LoRA")
          find("model-keyword-lora").asInstanceOf[ComboBox[String]].setValue(name)
        }
        waitFor("LoRA mapping check")(fx(!find("model-keyword-save").asInstanceOf[Button].isDisabled))
        fx {
          find("model-keyword-mapping").asInstanceOf[TextField].setText("texture-word|style-word")
          find("model-keyword-save").asInstanceOf[Button].fire()
        }
        waitFor("LoRA keyword choices")(fx(window.ctx.loraKeywordChoices.asScala.contains("texture-word")))
      }
      val loraLookup = loraName.map { name =>
        val result = mcpLookup.call(ujson.Obj("kind" -> "lora", "name" -> name), sdfx.mcp.ToolContext.none).structured.get
        require(result("match_source").str == "lora-keyword-user.txt" &&
          result("keywords").arr.map(_.str).toSeq == Seq("texture-word", "style-word"), "real MCP LoRA lookup mismatch")
        result
      }
      Files.writeString(home.root.resolve("mcp-model-keyword-lookup.json"), ujson.Obj(
        "checkpoint" -> checkpointLookup, "lora" -> loraLookup.getOrElse(ujson.Null)).render(indent = 2))
      val mcpGenerate = mcp.all.find(_.name == "txt2img").get
      def mcpPrompt(enabled: Option[Boolean]): String =
        val args = ujson.Obj("prompt" -> "a ceramic teapot", "seed" -> 101, "width" -> size,
          "height" -> size, "steps" -> 3, "return_images" -> false)
        enabled.foreach(value => args("model_keyword") = value)
        val result = mcpGenerate.call(args, sdfx.mcp.ToolContext.none).structured.get
        require(results.poll(30, TimeUnit.SECONDS) != null, "MCP job completion event missing")
        Infotext.parse(result("images")(0)("infotext").str).prompt
      val mcpDefaultPrompt = mcpPrompt(None)
      val mcpDisabledPrompt = mcpPrompt(Some(false))
      require(mcpDefaultPrompt.contains("red") && mcpDefaultPrompt.contains("blue"), "MCP default skipped keywords")
      require(mcpDisabledPrompt == "a ceramic teapot", "MCP explicit opt-out rewrote the prompt")
      fx(window.select(Mode.Txt2Img))
      waitFor("mounted keyword section")(fx(nodes(window.panels(Mode.Txt2Img).node).exists(_.getId == "plugin-model_keyword-section")))
      fx(findGeneration("plugin-model_keyword-section").asInstanceOf[TitledPane].setExpanded(true))
      waitFor("mounted generation controls")(fx(nodes(window.panels(Mode.Txt2Img).node).exists(_.getId == "plugin-model_keyword-enabled")))
      embeddingName.foreach(name => waitFor("embedding catalog choice")(
        fx(findGeneration("plugin-model_keyword-embedding").asInstanceOf[ComboBox[String]].getItems.contains(name))))
      fx {
        val enabled = findGeneration("plugin-model_keyword-enabled").asInstanceOf[CheckBox]
        require(enabled.isSelected && window.plugins(Mode.Txt2Img).form("model_keyword").get.isActive,
          "fresh mounted Model Keyword must start enabled")
        enabled.setSelected(false)
        require(!window.plugins(Mode.Txt2Img).form("model_keyword").get.isActive,
          "mounted Model Keyword opt-out did not disable it")
        enabled.setSelected(true)
        findGeneration("plugin-model_keyword-model_choice").asInstanceOf[ComboBox[String]].setValue("iterate")
        loraName.foreach(_ => findGeneration("plugin-model_keyword-lora_choice").asInstanceOf[ComboBox[String]].setValue("texture-word"))
        embeddingName.foreach(name => findGeneration("plugin-model_keyword-embedding").asInstanceOf[ComboBox[String]].setValue(name))
        require(window.activeMode == Mode.Txt2Img && window.plugins(Mode.Txt2Img).form("model_keyword").get.isActive,
          "mounted keyword controls did not enable the txt2img job")
        tagOptions.foreach(options => window.vms(Mode.Txt2Img).prompt.set(
          s"<lora:${loraName.get}:0.7:0.6:$options> a ceramic teapot"))
      }
      val images = generate(); require(images.size == 2)
      val infos = images.map(image => Infotext.parse(PngMetadata.readParameters(image.path).get))
      val inserted = if lora.isDefined then "texture-word " else ""
      def withoutTags(value: String): String = value.replaceAll("<lora:[^>]+>", "").replaceAll("\\s+", " ").trim
      require(infos.map(_.prompt).zip(Seq("red", "blue")).forall { (prompt, color) =>
        withoutTags(prompt).endsWith(s"${inserted}$color a ceramic teapot")
      })
      tagOptions.foreach(options => require(infos.forall(_.prompt.contains(s"<lora:${loraName.get}:0.7:0.6:$options>")),
        "LoRA tag options missing from replay metadata"))
      embeddingName.foreach(name => require(infos.forall(_.prompt.contains(name)), "embedding missing from effective prompt"))
      embeddingName.foreach { name =>
        val listed = scala.concurrent.Await.result(services.client.raw("embedding_inspector",
          ujson.Obj("operation" -> "list", "embedding_paths" -> ujson.Arr.from(embedding.toSeq.map(p => ujson.Str(p.toString))))),
          scala.concurrent.duration.Duration(60, TimeUnit.SECONDS))
        require(listed("loaded").arr.exists(item => item("name").str == name),
          "keyword embedding was not loaded into the text encoder")
      }
      waitFor("mounted refinement section")(fx(nodes(window.panels(Mode.Txt2Img).node).exists(_.getId == "plugin-img2img_hires_fix-section")))
      fx(findGeneration("plugin-img2img_hires_fix-section").asInstanceOf[TitledPane].setExpanded(true))
      waitFor("mounted refinement controls")(fx(nodes(window.panels(Mode.Txt2Img).node).exists(_.getId == "plugin-img2img_hires_fix-upscaler")))
      fx {
        findGeneration("plugin-img2img_hires_fix-enabled").asInstanceOf[CheckBox].setSelected(true)
        findGeneration("plugin-img2img_hires_fix-upscaler").asInstanceOf[ComboBox[String]].setValue("Lanczos")
        val form = window.plugins(Mode.Txt2Img).form("img2img_hires_fix").get
        form.prop("ratio").get.set("1")
        form.prop("steps").get.set("2")
      }
      val refined = generate(); require(refined.size == 2)
      val refinedText = PngMetadata.readParameters(refined.head.path).get
      val refinedInfo = Infotext.parse(refinedText)
      val config = Img2ImgHiresMetadata.parse(refinedInfo.fields(Img2ImgHiresMetadata.key)).get
      require(withoutTags(config("prompt").str).endsWith(s"${inserted}red a ceramic teapot"), "refinement preview was not recorded")
      fx(window.restore(refinedText))
      require(fx(!window.plugins(Mode.Txt2Img).form("model_keyword").get.isActive), "restore would apply keyword twice")
      require(fx(window.plugins(Mode.Txt2Img).form("img2img_hires_fix").get.isActive), "refinement restore was lost")
      val replay = generate().head
      def pixels(image: OutputImage): Array[Int] =
        val loaded = javax.imageio.ImageIO.read(image.path.toFile)
        loaded.getRGB(0, 0, loaded.getWidth, loaded.getHeight, null, 0, loaded.getWidth)
      require(pixels(refined.head).sameElements(pixels(replay)), "restored keyword/refinement image differs")
      if checkReenable then
        fx {
          findGeneration("plugin-img2img_hires_fix-enabled").asInstanceOf[CheckBox].setSelected(false)
          window.vms(Mode.Txt2Img).prompt.set("a ceramic teapot")
        }
        generate()
        val plainRequest = ujson.read(services.payloads.snapshots.last.bodyJson.get)
        require(plainRequest("params")("loras").arr.isEmpty, "intervening job still has LoRA")
        fx(window.restore(refinedText))
        val reenabled = generate().head
        require(pixels(refined.head).sameElements(pixels(reenabled)), "LoRA re-enable after a plain job differs")
      println(s"PASS mounted Model Keyword generation: ${images.map(_.path).mkString(", ")}")
    finally
      if window != null then fx { window.dispose(); window.stage.close() }
      services.shutdown(); Platform.exit()
