package sdfx.ui

import javafx.application.Platform
import javafx.stage.Stage
import javafx.embed.swing.SwingFXUtils
import java.nio.file.{Files, Paths}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit, LinkedBlockingQueue}
import javax.imageio.ImageIO
import sdfx.core.config.{AppHome, Settings, ModelDirs, HiresFixTweaksPreferences}
import sdfx.core.model.{GenerationParams, HiresParams, PromptStyle, Styles, LoraRef, LoraSyntax}
import sdfx.core.plugins.api.OutputImage
import sdfx.core.png.PngMetadata
import sdfx.core.infotext.Infotext
import sdfx.engine.{AppServices, JobEvent}
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{UiState, Mode}

/** Real MainWindow Generate/Interrupt/Restore; isolated home, actual GPU backend. */
object RealHiresGenerationCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body)
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)
  def main(args: Array[String]): Unit =
    require(args.length == 3 || args.length == 4, "home python checkpoint [manual-lora]")
    val home = AppHome(Paths.get(args(0)).toAbsolutePath)
    val model = Paths.get(args(2)).toAbsolutePath
    val loraPath = args.lift(3).orElse(sys.env.get("SDFX_CHECK_LORA")).map(p => Paths.get(p).toAbsolutePath)
    val manual = loraPath.map(p => LoraRef(p.getFileName.toString.stripSuffix(".safetensors"), .25,
      unet = Some(.45), start = Some(.1), stop = Some(.9)))
    Files.createDirectories(home.pythonDir)
    Files.createDirectories(home.configDir)
    val styles = Seq(PromptStyle("Hires test", "{prompt}, watercolor illustration", "{prompt}, watermark"),
      PromptStyle("Base test", "{prompt}, soft shadows", "{prompt}, blurry"))
    Files.writeString(home.stylesFile, Styles.toCsv(styles))
    val services = new AppServices(home, Settings(pythonExe = Some(Paths.get(args(1)).toAbsolutePath.toString),
      modelDirs = ModelDirs(checkpoints = Seq(model.getParent.toString), lora = loraPath.toSeq.map(_.getParent.toString)),
      hiresFixTweaks = HiresFixTweaksPreferences(outputDir = "hires-images")), dryRun = false,
      builtins = StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false)
    val finished = new LinkedBlockingQueue[Either[String, Seq[OutputImage]]]()
    val cancelled = new LinkedBlockingQueue[Seq[OutputImage]]()
    val reached = new java.util.concurrent.atomic.AtomicReference[CountDownLatch]()
    services.queue.addListener {
      case e: JobEvent.JobFinished => finished.offer(Right(e.output.images)); ()
      case e: JobEvent.JobFailed => finished.offer(Left(e.message)); ()
      case e: JobEvent.JobCancelled => cancelled.offer(e.partial.images); ()
      case e: JobEvent.Progress if e.progress.pass == "hires" && e.progress.image >= 1 && e.progress.step >= 2 =>
        Option(reached.get()).foreach(_.countDown())
      case _ => ()
    }
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    def idle(): Unit =
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(20)
      while (!services.queue.isIdle || fx(window.progress.get.running)) && System.nanoTime() < deadline do Thread.sleep(20)
      require(services.queue.isIdle && fx(!window.progress.get.running))
    def generate(): Seq[OutputImage] =
      idle(); fx(window.statusBar.generateBtn.fire())
      val result = finished.poll(180, TimeUnit.SECONDS)
      require(result != null, "native generation timed out")
      result.fold(message => throw new IllegalStateException(message), identity)
    def rgb(output: OutputImage): Array[Int] =
      val im = ImageIO.read(output.path.toFile)
      im.getRGB(0, 0, im.getWidth, im.getHeight, null, 0, im.getWidth)
    try
      services.start(); services.loadCheckpoint(model.toString, remember = false)
      fx {
        window = new MainWindow(new Stage(), services, UiState(params = Map("txt2img" -> GenerationParams(
          prompt = "blue ceramic teapot, flowers, wooden table, sunlight", negativePrompt = "lowres",
          seed = 7, width = 512, height = 512, steps = 10,
          hires = HiresParams(enabled = true, scale = 1.5, secondPassSteps = 8)))), CliOptions(), _ => ())
        window.show(); window.model.checkpoint.set(model.toString)
      }
      val baseline = generate().head
      fx {
        val f = window.plugins(Mode.Txt2Img).form("hires_fix_tweaks").get
        Map("enabled" -> "true", "batch_count" -> "2", "seed_enabled" -> "true", "seed" -> "100",
          "prompt_mode" -> "Append", "append_separator" -> ", ", "styles_enabled" -> "true",
          "styles" -> "[\"Hires test\"]").foreach((k, v) => f.props(k).set(v))
        window.vms(Mode.Txt2Img).hiresPrompt.set("soft light")
        window.vms(Mode.Txt2Img).promptStyles.setAll("Base test")
        manual.foreach { ref =>
          window.vms(Mode.Txt2Img).manualLoras.setAll(ref)
          window.vms(Mode.Txt2Img).prompt.set(window.vms(Mode.Txt2Img).prompt.get + " " + LoraSyntax.tag(ref.name, .15))
          f.props("remove_networks").set("true")
        }
      }
      val images = generate()
      require(images.size == 2, s"expected two Hires images, got ${images.size}")
      require(images.forall(_.path.startsWith(home.resolve("hires-images"))))
      require(images.forall(i => services.resetGallery().scan().exists(_.path == i.path)))
      val info = PngMetadata.readParameters(images.head.path).get
      require(Infotext.parse(info).fields("Hires prompt").contains("watercolor illustration"))
      fx {
        window.plugins(Mode.Txt2Img).form("hires_fix_tweaks").get.reset()
        window.restore(info)
        require(window.vms(Mode.Txt2Img).hiresPrompt.get == "soft light", "native Restore lost raw template")
        require(window.vms(Mode.Txt2Img).toParams.styles == Seq("Base test"), "native Restore lost base styles")
        require(window.panels(Mode.Txt2Img).prompts.view.lookup("#txt2img-prompt-style-order")
          .asInstanceOf[javafx.scene.control.Label].getText.contains("Base test"), "restored style not visible")
        val f = window.plugins(Mode.Txt2Img).form("hires_fix_tweaks").get
        require(f.props("prompt_mode").get == "Append" && f.props("styles").get == "[\"Hires test\"]")
        require(window.vms(Mode.Txt2Img).toParams.loras == manual.toSeq, "manual LoRA details lost on native Restore")
      }
      val restored = generate()
      require(restored.size == 1 && java.util.Arrays.equals(rgb(restored.head), rgb(images.head)), "native Restore changed RGB")
      fx {
        window.stage.getScene.getRoot.applyCss(); window.stage.getScene.getRoot.layout()
        ImageIO.write(SwingFXUtils.fromFXImage(window.stage.getScene.snapshot(null), null), "png", home.root.resolve("hires-restored.png").toFile)
        manual.foreach { ref =>
          val pane = window.panels(Mode.Txt2Img).prompts.view.lookup("#txt2img-manual-loras").asInstanceOf[javafx.scene.Parent]
          require(pane.isVisible, "manual LoRAs not visible")
          import scala.jdk.CollectionConverters.*
          pane.lookupAll(".button").asScala.collectFirst { case b: javafx.scene.control.Button if b.getText == "Remove" => b }.get.fire()
          require(window.vms(Mode.Txt2Img).toParams.loras.isEmpty, "manual removal failed")
          require(window.vms(Mode.Txt2Img).prompt.get.contains(LoraSyntax.tag(ref.name, .15)), "manual removal changed prompt tag")
        }
      }
      if manual.nonEmpty then
        require(!java.util.Arrays.equals(rgb(generate().head), rgb(images.head)), "removing manual LoRA made no pixel change")
        fx(window.restore(info))
        require(java.util.Arrays.equals(rgb(generate().head), rgb(images.head)), "manual LoRA re-restore changed RGB")
      idle()
      val latch = new CountDownLatch(1)
      reached.set(latch)
      fx {
        window.plugins(Mode.Txt2Img).form("hires_fix_tweaks").get.props("batch_count").set("3")
        window.vms(Mode.Txt2Img).hiresSteps.set(24)
        window.statusBar.generateBtn.fire()
      }
      require(latch.await(120, TimeUnit.SECONDS), "second Hires pass did not start")
      fx(window.statusBar.generateBtn.fire())
      val partial = cancelled.poll(60, TimeUnit.SECONDS)
      require(partial != null && partial.size == 1, "native cancellation did not retain exactly one completed image")
      reached.set(null)
      idle()
      require(finished.isEmpty, "cancelled job also emitted finished")
      fx(window.restore(PngMetadata.readParameters(baseline.path).get))
      require(java.util.Arrays.equals(rgb(generate().head), rgb(baseline)), "native cancellation recovery changed RGB")
      fx {
        ImageIO.write(SwingFXUtils.fromFXImage(window.stage.getScene.snapshot(null), null), "png", home.root.resolve("hires-native.png").toFile)
      }
      Files.writeString(home.root.resolve("hires-native-report.json"), ujson.Obj("status" -> "passed",
        "images" -> ujson.Arr.from(images.map(_.path.toString)), "partial" -> partial.head.path.toString,
        "restore_rgb_exact" -> true, "recovery_rgb_exact" -> true,
        "manual_lora" -> loraPath.map(_.toString).getOrElse("")).render(2))
      println("PASS native Hires generation, styles/template Restore exact, output/gallery, partial cancellation and exact recovery")
    finally
      fx { if window != null then { window.dispose(); window.stage.close() } }
      services.shutdown(); Platform.exit()
