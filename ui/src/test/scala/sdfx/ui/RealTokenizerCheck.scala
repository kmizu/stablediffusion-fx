package sdfx.ui

import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.scene.Node
import javafx.scene.control.*
import javafx.scene.input.{Clipboard, DataFormat}
import javafx.stage.Stage
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.infotext.{ImageFacts, Infotext}
import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.Values
import sdfx.core.protocol.{Json, TokenPiece}
import sdfx.engine.{AppServices, EngineLaunch, EngineTimeouts}
import sdfx.plugins.{ComposableLoraPlugin, StandardPlugins}
import sdfx.ui.vm.{Mode, UiState}

import java.nio.file.{Files, Path, Paths}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import javax.imageio.ImageIO
import scala.concurrent.Await
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

/** MANUAL real-engine/native-UI acceptance entrypoint; deliberately not a normal test suite.
  * Args: python checkpoint-directory checkpoint-name output-directory screenshot.png
  * Runs one loaded model per process. Never changes the user's app settings or stores clipboard contents.
  */
object RealTokenizerCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body
    )
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)

  private def eventually(label: String, timeout: FiniteDuration = 60.seconds)(test: => Boolean): Unit =
    val end = System.nanoTime() + timeout.toNanos
    while !test && System.nanoTime() < end do Thread.sleep(50)
    require(test, s"Timed out: $label")

  def main(args: Array[String]): Unit =
    require(args.length == 5, "python checkpoint-directory checkpoint-name output-directory screenshot.png")
    val python = Paths.get(args(0)).toAbsolutePath.normalize()
    val checkpointDir = Paths.get(args(1)).toAbsolutePath.normalize()
    val output = Paths.get(args(3)).toAbsolutePath.normalize()
    val screenshot = Paths.get(args(4)).toAbsolutePath.normalize()
    Files.createDirectories(output)
    Option(screenshot.getParent).foreach(Files.createDirectories(_))
    val engineDir = AppHome.detect().pythonDir
    require(Files.isDirectory(engineDir.resolve("sdfx")), s"Run from the integrated app repository: $engineDir")
    val home = AppHome(Files.createTempDirectory(output, "tokenizer-app-home-"))
    val settings = Settings(pythonExe = Some(python.toString), modelDirs = ModelDirs(checkpoints = Seq(checkpointDir.toString)),
      outputDir = output.resolve("images").toString, mcpHttpEnabled = false)
    val services = new AppServices(home, settings, dryRun = false, StandardPlugins.withOwnAxes,
      launchOverride = Some(() => EngineLaunch(Seq(python.toString, "-m", "sdfx"), engineDir,
        Map("PYTHONPATH" -> engineDir.toString, "USE_TF" -> "0"))),
      persistSettings = false, timeouts = EngineTimeouts(load = 10.minutes), installShutdownHook = false)
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS), "JavaFX startup timed out")
    var window: MainWindow = null
    var clipboardBackup: Option[java.util.HashMap[DataFormat, Object]] = None
    var primaryFailure: Throwable = null
    var skips = Vector.empty[String]
    val text = "a cat, 日本語 café 🐈"
    val expectedIds = Seq(320, 2368, 267, 39121, 44353, 34002, 508, 15304, 2074, 486)
    var checks = Vector.empty[String]
    def checked(label: String): Unit = checks :+= label
    def find[A <: Node](id: String): A =
      val n = window.scene.lookup(s"#$id")
      require(n != null, s"Missing control #$id")
      n.asInstanceOf[A]
    def inspect(value: String, ids: Boolean): Unit =
      fx {
        find[ComboBox[String]]("tokenizer-mode").getSelectionModel.select(if ids then "Token IDs" else "Text")
        find[TextArea]("tokenizer-input").setText(value)
        find[Button]("tokenizer-inspect").fire()
        require(find[Button]("tokenizer-inspect").isDisabled, "Inspect handler did not start")
      }
      eventually("tokenizer Future callback") { fx { !find[Button]("tokenizer-inspect").isDisabled } }
    def snapshot(path: Path): Unit = fx {
      window.root.applyCss()
      window.root.layout()
      require(ImageIO.write(SwingFXUtils.fromFXImage(window.scene.snapshot(null), null), "png", path.toFile))
    }
    try
      window = fx {
        val w = new MainWindow(new Stage(), services, UiState(selectedTab = "tokenizer"), CliOptions())
        w.vms(Mode.Txt2Img).prompt.set(text)
        w.show()
        w
      }
      val hello = services.start()
      require(!hello.dryRun && hello.features.contains("tokenizer"), "Must use the actual tokenizer engine")
      val loaded = services.loadCheckpoint(args(2), remember = false)
      val streamCount = if loaded.kind == "sdxl" then 2 else 1
      require(Set("sd15", "sdxl").contains(loaded.kind), s"Expected SD1.5/SDXL, got ${loaded.kind}")
      val direct = Await.result(services.client.tokenize(text), 60.seconds)
      require(direct.streams.size == streamCount)
      require(direct.streams.forall(_.ids == expectedIds), "Actual CLIP vocabulary returned unexpected Unicode IDs")
      checked("actual AppServices engine and loaded model tokenizer IDs")
      fx {
        window.model.checkpoint.set(loaded.name)
        require(window.state.selectedTab == "tokenizer", "Tokenizer selection was not restored")
        val tabs = find[TabPane]("main-tabs")
        tabs.getSelectionModel.select(0)
        tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getId == "tokenizer-tab").get)
        require(window.state.selectedTab == "tokenizer", "Tokenizer selection was not captured")
        UiState.save(UiState.file(home), window.state)
        require(UiState.load(UiState.file(home)).selectedTab == "tokenizer")
        find[Button]("tokenizer-from-prompt").fire()
        require(find[TextArea]("tokenizer-input").getText == text, "Prompt import handler failed")
        find[Button]("tokenizer-inspect").fire()
      }
      eventually("imported prompt result") { fx { !find[Button]("tokenizer-inspect").isDisabled } }
      checked("native tab selection restore/save and current prompt button")
      fx {
        val results = find[TabPane]("tokenizer-results")
        require(results.getTabs.size == streamCount)
        direct.streams.zipWithIndex.foreach { (stream, index) =>
          results.getSelectionModel.select(index)
          window.root.applyCss()
          window.root.layout()
          val table = find[TableView[TokenPiece]](s"tokenizer-${stream.name}-tokens")
          require(table.getItems.asScala.map(_.id).toSeq == expectedIds)
          require(find[TextArea](s"tokenizer-${stream.name}-decoded").getText == stream.decoded)
        }
        results.getSelectionModel.selectFirst()
        window.root.applyCss()
        window.root.layout()
        NativeAcceptanceSafety.captureClipboard() match
          case Right(saved) =>
            clipboardBackup = Some(saved)
            find[Button]("tokenizer-tokenizer-copy-ids").fire()
            require(Clipboard.getSystemClipboard.getString == expectedIds.mkString(", "), "Copy IDs handler failed")
            checked("native copy IDs button; clipboard restoration attempted on exit")
          case Left(reason) =>
            val skip = "native copy IDs button: " + reason + "; original clipboard left unchanged"
            skips :+= skip
            println("TOKENIZER SKIP: " + skip)
      }
      checked("native stream tabs, token table and decoded Unicode")
      inspect(expectedIds.mkString(", "), ids = true)
      fx {
        require(find[TabPane]("tokenizer-results").getTabs.size == streamCount)
        require(find[TextArea]("tokenizer-tokenizer-decoded").getText == direct.streams.head.decoded)
        require(find[Label]("tokenizer-status").getText.contains("IDs decoded as supplied"))
      }
      checked("native ID mode decoding through button and Future callback")
      inspect("999999999", ids = true)
      fx {
        require(find[TabPane]("tokenizer-results").getTabs.isEmpty)
        require(find[Label]("tokenizer-status").getText.contains("not in"))
      }
      inspect(text, ids = false)
      fx {
        require(find[TextArea]("tokenizer-tokenizer-ids").getText == expectedIds.mkString(", "))
        require(find[Label]("tokenizer-status").getText.contains("literal special tokens retained"))
      }
      checked("invalid ID error and successful native recovery")
      fx { window.stage.setWidth(1280); window.stage.setHeight(800) }
      eventually("compact native resize pulse") { fx { window.scene.getWidth < 1300 && window.scene.getHeight < 810 } }
      eventually("minimum-size layout") { fx { window.tokenizer.node.getWidth > 100 && window.tokenizer.node.getHeight > 300 } }
      fx {
        val scroll = find[ScrollPane]("tokenizer-tokenizer-scroll")
        require(scroll.getViewportBounds.getWidth > 100 && scroll.getViewportBounds.getHeight > 100)
        scroll.setVvalue(1)
        find[ScrollPane]("tokenizer-page-scroll").setVvalue(1)
      }
      snapshot(screenshot.resolveSibling(screenshot.getFileName.toString.replaceFirst("\\.png$", "") + "-compact.png"))
      fx {
        window.stage.setWidth(1600)
        window.stage.setHeight(1000)
      }
      eventually("expanded native resize pulse") { fx { window.scene.getWidth > 1500 && window.scene.getHeight > 900 } }
      fx {
        find[ScrollPane]("tokenizer-tokenizer-scroll").setVvalue(0)
        find[ScrollPane]("tokenizer-page-scroll").setVvalue(0)
      }
      snapshot(screenshot)
      checked("minimum and expanded viewport, scrolling and native screenshots")
      fx {
        find[TabPane]("main-tabs").getSelectionModel.selectFirst()
        val header = "a cat\nSteps: 7, Sampler: Euler, CFG scale: 3, Seed: 321, Size: 512x512"
        window.restore(header + ", Grid image numbers: False, Grid XYZ info: True")
        val txtGrid = window.plugins(Mode.Txt2Img).form("grid_numbers").get
        require(txtGrid.values.bool("enabled") && !txtGrid.values.bool("image_numbers") && txtGrid.values.bool("xyz_info"))
        val source = output.resolve("native-replay-source.png")
        ImageIO.write(new java.awt.image.BufferedImage(64, 64, java.awt.image.BufferedImage.TYPE_INT_RGB), "png", source.toFile)
        val replay = header + ", Grid image numbers: True, Grid XYZ info: False, Latent upscale method: bilinear, Latent upscale scheduler: karras"
        def resetImageMode(mode: Mode): Unit =
          val forms = window.plugins(mode)
          forms.selectedScript.set("")
          forms.form("latent_upscale").get.props("method").set("nearest-exact")
          forms.form("latent_upscale").get.props("scheduler").set("simple")
          val grid = forms.form("grid_numbers").get
          grid.props("enabled").set("false")
          grid.props("image_numbers").set("false")
          grid.props("xyz_info").set("true")
        resetImageMode(Mode.Img2Img)
        window.sendTo(Mode.Img2Img, source, Some(replay))
        def verifyImageMode(mode: Mode): Unit =
          val forms = window.plugins(mode)
          require(window.activeMode == mode)
          require(forms.selectedScript.get == "latent_upscale")
          require(forms.form("latent_upscale").get.values.str("method") == "bilinear")
          require(forms.form("latent_upscale").get.values.str("scheduler") == "karras")
          require(forms.form("grid_numbers").get.values.bool("enabled"))
          require(forms.form("grid_numbers").get.values.bool("image_numbers"))
          require(!forms.form("grid_numbers").get.values.bool("xyz_info"))
        verifyImageMode(Mode.Img2Img)
        resetImageMode(Mode.Inpaint)
        window.sendTo(Mode.Inpaint, source, Some(replay))
        verifyImageMode(Mode.Inpaint)
        resetImageMode(Mode.Inpaint)
        window.restore(replay)
        verifyImageMode(Mode.Inpaint)
        find[TabPane]("main-tabs").getSelectionModel.selectFirst()
        resetImageMode(Mode.Img2Img)
        window.restore(replay)
        verifyImageMode(Mode.Img2Img)
        window.root.applyCss()
        window.root.layout()
        val script = window.panels(Mode.Img2Img).content.lookup(".script-section")
        require(script != null)
        require(ImageIO.write(SwingFXUtils.fromFXImage(script.getParent.snapshot(null, null), null), "png", output.resolve("latent_upscale-form.png").toFile))
      }
      checked("actual Restore and Send to paths restore grid flags and latent script in img2img/inpaint")
      fx {
        val plugin = new ComposableLoraPlugin
        val keys = Seq("with_step", "uc_text_model_encoder", "uc_diffusion_model", "plot_weights", "single_no_uc", "hires_global_step")
        def text(value: Boolean): String =
          val values = Values(Map("enabled" -> "true") ++ keys.map(_ -> value.toString))
          Infotext.write(plugin.process(GenerationParams(prompt = "a cat"), values), ImageFacts(seed = 7))
        find[TabPane]("main-tabs").getSelectionModel.selectFirst()
        val txtForm = window.plugins(Mode.Txt2Img).form("composable_lora").get
        keys.foreach(k => txtForm.props(k).set("false"))
        txtForm.props("enabled").set("false")
        window.restore(text(true))
        require(txtForm.values.bool("enabled") && keys.forall(txtForm.values.bool(_)))
        val imageForm = window.plugins(Mode.Img2Img).form("composable_lora").get
        keys.foreach(k => imageForm.props(k).set("true"))
        imageForm.props("enabled").set("false")
        window.sendTo(Mode.Img2Img, output.resolve("native-replay-source.png"), Some(text(false)))
        require(imageForm.values.bool("enabled") && keys.forall(k => !imageForm.values.bool(k)))
      }
      checked("native Composable PNG Restore and Send to replace all six true/false source controls")
      fx {
        find[TabPane]("main-tabs").getSelectionModel.selectFirst()
        val form = window.panels(Mode.Txt2Img).content
        window.root.applyCss()
        window.root.layout()
        Seq("prompt_fusion", "save_intermediate", "grid_numbers", "composable_lora").foreach { id =>
          val vm = window.plugins(Mode.Txt2Img).alwaysOnForms.find(_.plugin.id == id).get
          vm.props("enabled").set("true")
          if id == "save_intermediate" then vm.props("mode").set("Video")
          val pane = form.lookupAll(".titled-pane").asScala.collect {
            case p: TitledPane if p.getText == vm.plugin.name => p
          }.head
          pane.setExpanded(true)
          pane.setAnimated(false)
          if id == "save_intermediate" then
            pane.lookupAll(".titled-pane").asScala.collect {
              case p: TitledPane if p.getText == "Video export" => p
            }.foreach { p => p.setAnimated(false); p.setExpanded(true) }
          if id == "composable_lora" then
            pane.lookupAll(".titled-pane").asScala.collect { case p: TitledPane => p }
              .foreach { p => p.setAnimated(false); p.setExpanded(true) }
          window.root.applyCss()
          window.root.layout()
          val path = output.resolve(s"$id-form.png")
          require(ImageIO.write(SwingFXUtils.fromFXImage(pane.snapshot(null, null), null), "png", path.toFile))
        }
      }
      checked("native expanded Prompt Fusion and intermediate video forms")
      val report = ujson.Obj("success" -> true, "model" -> Json.toJs(loaded), "tokenization" -> Json.toJs(direct),
        "checks" -> ujson.Arr.from(checks), "skips" -> ujson.Arr.from(skips), "screenshot" -> screenshot.toString,
        "home" -> home.root.toString, "clipboard_saved_to_disk" -> false)
      Files.writeString(output.resolve("tokenizer-ui-report.json"), ujson.write(report, indent = 2))
      println(s"Real tokenizer UI acceptance passed: ${loaded.kind}; report ${output.resolve("tokenizer-ui-report.json")}")
    catch
      case error: Throwable =>
        primaryFailure = error
        if window != null then
          try snapshot(output.resolve("tokenizer-ui-failure.png")) catch case _: Throwable => ()
        throw error
    finally
      NativeAcceptanceSafety.cleanup(primaryFailure)(
        () => fx { clipboardBackup.foreach(NativeAcceptanceSafety.restore) },
        () => { if window != null then fx { window.dispose() } },
        () => { if window != null then fx { window.stage.close() } },
        () => services.shutdown(),
        () => Platform.exit())
