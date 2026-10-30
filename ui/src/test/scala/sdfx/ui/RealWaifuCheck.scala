package sdfx.ui

import javafx.application.Platform
import javafx.stage.Stage
import javafx.embed.swing.SwingFXUtils
import java.nio.file.{Files, Paths}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit, LinkedBlockingQueue}
import javax.imageio.ImageIO
import sdfx.core.config.{AppHome, Settings, ModelDirs}
import sdfx.core.model.{GenerationParams, Img2ImgParams}
import sdfx.core.plugins.api.OutputImage
import sdfx.core.png.PngMetadata
import sdfx.core.infotext.Infotext
import sdfx.engine.{AppServices, JobEvent}
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{UiState, Mode}

/** MainWindow's real Generate button -> native plugin -> actual GPU upscaling. */
object RealWaifuCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body)
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)
  def main(args: Array[String]): Unit =
    require(args.length == 4, "home python checkpoint generated-input")
    val home = AppHome(Paths.get(args(0)).toAbsolutePath)
    val model = Paths.get(args(2)).toAbsolutePath
    val input = Paths.get(args(3)).toAbsolutePath
    val sourceImage = ImageIO.read(input.toFile)
    Files.createDirectories(home.pythonDir)
    val services = new AppServices(home, Settings(pythonExe = Some(Paths.get(args(1)).toAbsolutePath.toString),
      modelDirs = ModelDirs(checkpoints = Seq(model.getParent.toString))), dryRun = false,
      builtins = StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false)
    val finished = new LinkedBlockingQueue[Either[String, OutputImage]]()
    val cancelled = new LinkedBlockingQueue[Seq[OutputImage]]()
    val tileReady = new java.util.concurrent.atomic.AtomicReference[CountDownLatch]()
    val tileOwner = new java.util.concurrent.atomic.AtomicReference[String]()
    services.queue.addListener {
      case event: JobEvent.JobFinished => finished.offer(event.output.images.headOption.toRight("no image")); ()
      case event: JobEvent.JobFailed => println(s"NATIVE FAILURE ${event.jobId}: ${event.message}"); finished.offer(Left(event.message)); ()
      case event: JobEvent.JobCancelled => println(s"NATIVE CANCEL ${event.jobId}: ${event.partial.images.size} partial"); cancelled.offer(event.partial.images); ()
      case event: JobEvent.Progress if event.progress.pass == "waifu2x" =>
        Option(tileReady.get()).foreach { latch => tileOwner.set(event.jobId); latch.countDown() }
      case _ => ()
    }
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    def waitIdle(): Unit =
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10)
      while (!services.queue.isIdle || fx(window.progress.get.running)) && System.nanoTime() < deadline do Thread.sleep(20)
      require(services.queue.isIdle && fx(!window.progress.get.running), "queue and rendered UI did not settle idle")
    try
      services.start(); services.loadCheckpoint(model.toString, remember = false)
      fx {
        window = new MainWindow(new Stage(), services, UiState(params = Map("img2img" -> GenerationParams(
          seed = 7, width = 256, height = 256, img2img = Some(Img2ImgParams(input.toString))))), CliOptions(), _ => ())
        window.show(); window.model.checkpoint.set(model.toString); window.select(Mode.Img2Img)
        window.plugins(Mode.Img2Img).selectedScript.set("waifu2x")
      }
      val results = Seq("Anime" -> "High", "Photo" -> "Low", "CarnV2" -> "None").map { (style, noise) =>
        waitIdle()
        fx {
          val form = window.plugins(Mode.Img2Img).form("waifu2x").get
          form.props("style").set(style); form.props("noise").set(noise); form.props("scale").set("2x")
          window.statusBar.generateBtn.fire()
        }
        val result = finished.poll(120, TimeUnit.SECONDS)
        require(result != null, "native Waifu2x timed out")
        val output = result.fold(message => throw new IllegalStateException(message), identity)
        val image = ImageIO.read(output.path.toFile)
        require(image.getWidth == sourceImage.getWidth * 2 && image.getHeight == sourceImage.getHeight * 2)
        val fields = Infotext.parse(PngMetadata.readTexts(output.path)(PngMetadata.ParametersKey)).fields
        require(fields("Waifu2x style") == style && fields("Waifu2x scale") == "2x", fields.toString)
        output.path
      }
      fx {
        ImageIO.write(SwingFXUtils.fromFXImage(window.stage.getScene.snapshot(null), null), "png", home.root.resolve("waifu-native-window.png").toFile)
      }
      val cancelInput = home.root.resolve("cancel-input.png")
      ImageIO.write(sourceImage.getSubimage(0, 0, 256, 256), "png", cancelInput.toFile)
      for attempt <- 1 to 3 do
        waitIdle()
        val firstTile = new CountDownLatch(1)
        tileReady.set(firstTile)
        fx {
          window.vms(Mode.Img2Img).initImage.set(cancelInput.toString)
          window.plugins(Mode.Img2Img).form("waifu2x").get.props("scale").set("16x")
          window.statusBar.generateBtn.fire()
        }
        require(firstTile.await(60, TimeUnit.SECONDS), "no actual neural tile progress before native cancellation")
        fx {
          println(s"NATIVE STOP $attempt current=${window.progress.get.current} owner=${tileOwner.get()}")
          require(window.progress.get.current.contains(tileOwner.get()), "tile progress and rendered UI belong to different jobs")
          window.statusBar.generateBtn.fire()
        }
        val partial = cancelled.poll(30, TimeUnit.SECONDS)
        require(partial != null && partial.isEmpty, s"cancelled neural upscale missing or published partial image: partial=$partial, completion=${finished.poll()}, UI=${fx(window.progress.get)}")
        tileReady.set(null)
      waitIdle()
      fx {
        window.plugins(Mode.Img2Img).form("waifu2x").get.props("scale").set("2x")
        window.statusBar.generateBtn.fire()
      }
      val neuralRecovery = finished.poll(120, TimeUnit.SECONDS)
      require(neuralRecovery != null && neuralRecovery.isRight, s"neural recovery failed: $neuralRecovery")
      waitIdle()
      fx {
        window.select(Mode.Txt2Img)
        window.plugins(Mode.Txt2Img).selectedScript.set("")
        window.vms(Mode.Txt2Img).width.set(256); window.vms(Mode.Txt2Img).height.set(256)
        window.vms(Mode.Txt2Img).steps.set(2)
        window.statusBar.generateBtn.fire()
      }
      val generationRecovery = finished.poll(120, TimeUnit.SECONDS)
      require(generationRecovery != null && generationRecovery.isRight, s"diffusion recovery failed: $generationRecovery")
      println(s"PASS native Waifu2x: three families, GPU dimensions/settings, actual tile cancellation without partial output, neural/diffusion recovery: ${results.mkString(", ")}")
    finally
      fx { if window != null then { window.dispose(); window.stage.close() } }
      services.shutdown(); Platform.exit()
