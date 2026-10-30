package sdfx.ui

import javafx.application.Platform
import javafx.stage.Stage
import javafx.scene.control.TabPane
import javafx.embed.swing.SwingFXUtils
import java.nio.file.{Files, Paths}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit, LinkedBlockingQueue}
import javax.imageio.ImageIO
import sdfx.core.config.{AppHome, Settings, ModelDirs}
import sdfx.core.model.{GenerationParams, Img2ImgParams, LoraRef, LoraSyntax}
import sdfx.core.plugins.api.OutputImage
import sdfx.core.png.PngMetadata
import sdfx.core.infotext.Infotext
import sdfx.engine.{AppServices, JobEvent}
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{UiState, Mode}

/** Actual MainWindow Generate/Restore/Interrupt in an isolated home, real backend. */
object RealImg2ImgHiresGenerationCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body)
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)
  def main(args: Array[String]): Unit =
    require(args.length == 4 || args.length == 5, "home python checkpoint lora [sdxl]")
    val home = AppHome(Paths.get(args(0)).toAbsolutePath)
    val model = Paths.get(args(2)).toAbsolutePath
    val lora = Paths.get(args(3)).toAbsolutePath
    val sdxl = args.lift(4).contains("sdxl")
    val side = if sdxl then 768 else 512
    val ref = LoraRef(lora.getFileName.toString.stripSuffix(".safetensors"), .2)
    Files.createDirectories(home.pythonDir)
    Files.createDirectories(home.configDir)
    val services = new AppServices(home, Settings(pythonExe = Some(Paths.get(args(1)).toAbsolutePath.toString),
      modelDirs = ModelDirs(checkpoints = Seq(model.getParent.toString), lora = Seq(lora.getParent.toString))),
      dryRun = false, builtins = StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false)
    val finished = new LinkedBlockingQueue[Either[String, Seq[OutputImage]]]()
    val cancelled = new LinkedBlockingQueue[Seq[OutputImage]]()
    val reached = new java.util.concurrent.atomic.AtomicReference[CountDownLatch]()
    services.queue.addListener {
      case e: JobEvent.JobFinished => finished.offer(Right(e.output.images)); ()
      case e: JobEvent.JobFailed => finished.offer(Left(e.message)); ()
      case e: JobEvent.JobCancelled => cancelled.offer(e.partial.images); ()
      case e: JobEvent.Progress if e.progress.pass == "img2img_hires" && e.progress.image == 1 && e.progress.step >= 1 =>
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
      val result = finished.poll(240, TimeUnit.SECONDS)
      require(result != null, "native generation timed out")
      result.fold(message => throw new IllegalStateException(message), identity)
    def rgb(output: OutputImage): Array[Int] =
      val im = ImageIO.read(output.path.toFile)
      im.getRGB(0, 0, im.getWidth, im.getHeight, null, 0, im.getWidth)
    val base = GenerationParams(prompt = "a woman with red hair in a garden, portrait, detailed illustration",
      negativePrompt = "low quality, blurry", seed = 173, width = side, height = side, steps = 16,
      cfgScale = if sdxl then 4 else 6)
    val config = Map("enabled" -> "true", "ratio" -> "1.25", "upscaler" -> "Lanczos", "steps" -> "12",
      "denoise" -> "0.5", "prompt" -> (base.prompt + " " + LoraSyntax.tag(ref.name, .1)))
    try
      services.start(); services.loadCheckpoint(model.toString, remember = false)
      fx {
        window = new MainWindow(new Stage(), services, UiState(params = Map("txt2img" -> base)), CliOptions(), _ => ())
        window.show(); window.model.checkpoint.set(model.toString)
      }
      val baseline = generate().head
      fx {
        window.plugins(Mode.Txt2Img).form("img2img_hires_fix").get.load(sdfx.core.plugins.api.Values(config))
        window.vms(Mode.Txt2Img).manualLoras.setAll(ref)
      }
      val active = generate().head
      require(ImageIO.read(active.path.toFile).getWidth == side * 5 / 4)
      require(services.resetGallery().scan().exists(_.path == active.path))
      val info = PngMetadata.readParameters(active.path).get
      require(Infotext.parse(info).fields.contains("img2img Hires Fix"))
      fx {
        window.plugins(Mode.Txt2Img).form("img2img_hires_fix").get.reset()
        window.restore(info)
        val form = window.plugins(Mode.Txt2Img).form("img2img_hires_fix").get
        require(form.isActive && form.values.str("prompt") == config("prompt"))
        require(window.vms(Mode.Txt2Img).toParams.loras == Seq(ref))
      }
      val replay = generate().head
      require(java.util.Arrays.equals(rgb(active), rgb(replay)), "native PNG replay changed RGB")
      val previewDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(20)
      while !fx(window.preview.currentPath.contains(replay.path) && window.preview.zoom.image != null) &&
          System.nanoTime() < previewDeadline do Thread.sleep(30)
      require(fx(window.preview.currentPath.contains(replay.path) && window.preview.zoom.image != null), "native preview did not load")
      fx {
        window.stage.getScene.getRoot.applyCss(); window.stage.getScene.getRoot.layout()
        ImageIO.write(SwingFXUtils.fromFXImage(window.stage.getScene.snapshot(null), null), "png", home.root.resolve("restored.png").toFile)
      }
      idle()
      val latch = new CountDownLatch(1)
      reached.set(latch)
      fx { window.vms(Mode.Txt2Img).batchCount.set(3); window.statusBar.generateBtn.fire() }
      require(latch.await(180, TimeUnit.SECONDS), "second refinement did not start")
      fx(window.statusBar.generateBtn.fire())
      val partial = cancelled.poll(60, TimeUnit.SECONDS)
      require(partial != null && partial.size == 1, "native cancellation must retain one completed image")
      require(java.util.Arrays.equals(rgb(active), rgb(partial.head)), "completed first image changed on cancellation")
      reached.set(null); idle()
      fx {
        window.restore(PngMetadata.readParameters(baseline.path).get)
        window.vms(Mode.Txt2Img).batchCount.set(1)
      }
      require(!fx(window.plugins(Mode.Txt2Img).form("img2img_hires_fix").get.isActive))
      require(java.util.Arrays.equals(rgb(baseline), rgb(generate().head)), "off recovery changed RGB")
      val modeOutputs = Seq(Mode.Img2Img, Mode.Inpaint).map { mode =>
        idle()
        fx {
          window.stage.getScene.lookup("#main-tabs").asInstanceOf[TabPane].getSelectionModel.select(mode.ordinal)
          window.vms(mode).load(base.copy(img2img = Some(Img2ImgParams(initImage = baseline.path.toString,
            denoisingStrength = .5))))
          window.plugins(mode).form("img2img_hires_fix").get.load(sdfx.core.plugins.api.Values(config.updated("prompt", base.prompt)))
        }
        if mode == Mode.Inpaint then
          // Native Generate exports the canvas, so wait for its asynchronous source image and paint there.
          val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(20)
          while !fx(window.vms(mode).hasMask.get) && System.nanoTime() < deadline do
            fx(window.panels(mode).mask.get.paintDemo())
            Thread.sleep(30)
          require(fx(window.vms(mode).hasMask.get), "native inpaint canvas did not become ready")
        val image = generate().head
        require(ImageIO.read(image.path.toFile).getWidth == side * 5 / 4)
        mode.key -> image.path.toString
      }
      Files.writeString(home.root.resolve("img2img-hires-native-report.json"), ujson.Obj("status" -> "passed",
        "active" -> active.path.toString, "partial" -> partial.head.path.toString,
        "restore_rgb_exact" -> true, "recovery_rgb_exact" -> true, "partial_rgb_exact" -> true,
        "modes" -> ujson.Obj.from(modeOutputs.map((k, v) => k -> ujson.Str(v)))).render(2))
      println("PASS MainWindow img2img Hires modes, manual/tag PNG replay exact, partial cancellation exact and recovery exact")
    finally
      fx { if window != null then { window.dispose(); window.stage.close() } }
      services.shutdown(); Platform.exit()
