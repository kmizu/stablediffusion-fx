package sdfx.ui

import javafx.application.Platform
import javafx.stage.Stage
import javafx.embed.swing.SwingFXUtils
import java.nio.file.{Files, Paths}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit, LinkedBlockingQueue}
import javax.imageio.ImageIO
import sdfx.core.config.{AppHome, Settings, ModelDirs}
import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.OutputImage
import sdfx.core.png.PngMetadata
import sdfx.core.infotext.Infotext
import sdfx.engine.{AppServices, JobEvent}
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{UiState, Mode}

object RealTokenDownsamplingCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body)
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)
  def main(args: Array[String]): Unit =
    require(args.length == 3, "home python checkpoint")
    val home = AppHome(Paths.get(args(0)).toAbsolutePath)
    val model = Paths.get(args(2)).toAbsolutePath
    Files.createDirectories(home.pythonDir)
    val services = new AppServices(home, Settings(pythonExe = Some(Paths.get(args(1)).toAbsolutePath.toString),
      modelDirs = ModelDirs(checkpoints = Seq(model.getParent.toString))), dryRun = false,
      builtins = StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false)
    val finished = new LinkedBlockingQueue[Either[String, OutputImage]]()
    val cancelled = new LinkedBlockingQueue[java.lang.Integer]()
    val reachedStep = new java.util.concurrent.atomic.AtomicReference[CountDownLatch]()
    services.queue.addListener {
      case event: JobEvent.JobFinished => finished.offer(event.output.images.headOption.toRight("no image")); ()
      case event: JobEvent.JobFailed => finished.offer(Left(event.message)); ()
      case event: JobEvent.JobCancelled => cancelled.offer(event.partial.images.size); ()
      case event: JobEvent.Progress if event.progress.step >= 2 => Option(reachedStep.get()).foreach(_.countDown())
      case _ => ()
    }
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    def idle(): Unit =
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10)
      while (!services.queue.isIdle || fx(window.progress.get.running)) && System.nanoTime() < deadline do Thread.sleep(20)
      require(services.queue.isIdle && fx(!window.progress.get.running))
    def generate(): OutputImage =
      idle()
      fx(window.statusBar.generateBtn.fire())
      val result = finished.poll(120, TimeUnit.SECONDS)
      require(result != null, "native generation timed out")
      result.fold(message => throw new IllegalStateException(message), identity)
    def rgb(output: OutputImage): Array[Int] =
      val image = ImageIO.read(output.path.toFile)
      image.getRGB(0, 0, image.getWidth, image.getHeight, null, 0, image.getWidth)
    try
      services.start(); services.loadCheckpoint(model.toString, remember = false)
      fx {
        val size = if model.getFileName.toString.toLowerCase.contains("sdxl") then 768 else 512
        window = new MainWindow(new Stage(), services, UiState(params = Map("txt2img" -> GenerationParams(
          prompt = "blue and white ceramic teapot, flowers, wooden table, detailed illustration",
          seed = 7, width = size, height = size, steps = 8))), CliOptions(), _ => ())
        window.show(); window.model.checkpoint.set(model.toString)
      }
      val baseline = generate()
      fx {
        val form = window.plugins(Mode.Txt2Img).form("token_downsampling").get
        form.props("enabled").set("true"); form.props("factor").set("2")
        form.props("factor2").set("2"); form.props("disable_after").set("0.75")
      }
      val active = generate()
      require(!java.util.Arrays.equals(rgb(active), rgb(baseline)), "active hook made no pixel change")
      val info = PngMetadata.readTexts(active.path)(PngMetadata.ParametersKey)
      val fields = Infotext.parse(info).fields
      require(fields("ToDo factor") == "2" && fields("ToDo factor2") == "2" && fields("ToDo disable after") == "0.75")
      fx {
        window.plugins(Mode.Txt2Img).form("token_downsampling").get.reset()
        window.plugins(Mode.Txt2Img).restoreInfotext(info)
        require(window.plugins(Mode.Txt2Img).form("token_downsampling").get.isActive)
        ImageIO.write(SwingFXUtils.fromFXImage(window.stage.getScene.snapshot(null), null), "png", home.root.resolve("todo-native.png").toFile)
      }
      for attempt <- 1 to 3 do
        idle()
        val latch = new CountDownLatch(1)
        reachedStep.set(latch)
        fx { window.vms(Mode.Txt2Img).steps.set(40); window.statusBar.generateBtn.fire() }
        require(latch.await(90, TimeUnit.SECONDS), "no diffusion progress")
        fx(window.statusBar.generateBtn.fire())
        val count = cancelled.poll(30, TimeUnit.SECONDS)
        require(count != null && count.intValue == 0, s"native cancellation $attempt did not discard unfinished image")
        reachedStep.set(null)
      idle()
      fx {
        window.plugins(Mode.Txt2Img).restoreInfotext(PngMetadata.readTexts(baseline.path)(PngMetadata.ParametersKey))
        require(!window.plugins(Mode.Txt2Img).form("token_downsampling").get.isActive)
        window.vms(Mode.Txt2Img).steps.set(8)
      }
      require(java.util.Arrays.equals(rgb(generate()), rgb(baseline)), "native cancellation leaked hook state")
      println(s"PASS native Token Downsampling: active PNG/replay, three cancellations, exact off recovery; ${active.path}")
    finally
      fx { if window != null then { window.dispose(); window.stage.close() } }
      services.shutdown(); Platform.exit()
