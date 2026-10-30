package sdfx.ui

import javafx.application.Platform
import javafx.stage.Stage
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.png.{PngMetadata, ScrambledPng}
import sdfx.core.plugins.api.OutputImage
import sdfx.engine.{AppServices, JobEvent}
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{Mode, UiState}
import java.nio.file.{Files, Path}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, LinkedBlockingQueue, TimeUnit}

/** Actual MainWindow Generate button and Python dry-run worker, not a diffusion-model test. */
object RealScrambledGenerationCheck:
  def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A] { def call(): A = body })
    Platform.runLater(task); task.get(30, TimeUnit.SECONDS)
  def main(args: Array[String]): Unit =
    require(args.length == 2)
    val root = Path.of(args(0)).toAbsolutePath; Files.createDirectories(root)
    val home = AppHome(Files.createTempDirectory(root, "home-")); Files.createDirectories(home.pythonDir)
    val models = Files.createDirectories(home.root.resolve("test-models"))
    val checkpoint = models.resolve("fixture.safetensors")
    Files.write(checkpoint, Array[Byte](2, 0, 0, 0, 0, 0, 0, 0, 123, 125))
    val service = new AppServices(home, Settings(pythonExe = Some(Path.of(args(1)).toAbsolutePath.toString),
      modelDirs = ModelDirs(checkpoints = Seq(models.toString))), dryRun = true,
      builtins = StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false)
    val events = new LinkedBlockingQueue[Either[String, OutputImage]]()
    service.queue.addListener {
      case e: JobEvent.JobFinished => events.offer(Right(e.output.images.head)); ()
      case e: JobEvent.JobFailed => events.offer(Left(e.message)); ()
      case _ => ()
    }
    val ready = new CountDownLatch(1); Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    def next(): OutputImage =
      val result = events.poll(90, TimeUnit.SECONDS)
      require(result != null, "generation timed out")
      result.fold(e => throw new IllegalStateException(e), identity)
    def idle(): Unit =
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while (!service.queue.isIdle || fx(window.progress.get.running)) && System.nanoTime() < deadline do Thread.sleep(50)
      require(service.queue.isIdle && fx(!window.progress.get.running))
    try
      service.start(); service.loadCheckpoint(checkpoint.toString, remember = false)
      fx {
        window = new MainWindow(new Stage(), service, UiState(), CliOptions(), _ => ())
        window.show(); window.select(Mode.Txt2Img); window.model.checkpoint.set(checkpoint.toString)
        val vm = window.vms(Mode.Txt2Img)
        vm.prompt.set("test portrait"); vm.seed.set(173); vm.steps.set(5); vm.width.set(64); vm.height.set(64)
        window.statusBar.generateBtn.fire()
      }
      val plain = next(); idle()
      fx {
        service.imageScrambleSession.unlock("secret"); window.ctx.imageAccessChanged()
        window.statusBar.generateBtn.fire()
        service.imageScrambleSession.lock(); window.ctx.imageAccessChanged()
      }
      val protectedImage = next(); idle()
      require(PngMetadata.readTexts(protectedImage.path).get("Encrypt").contains("pixel_shuffle_3"), "queued secret not captured")
      val expected = PngMetadata.readImage(plain.path); val actual = ScrambledPng.read(protectedImage.path, Some("secret")).image
      require(expected.getRGB(0, 0, 64, 64, null, 0, 64).sameElements(actual.getRGB(0, 0, 64, 64, null, 0, 64)), "generated pixels differ")
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while !fx(window.preview.imageStatus.startsWith("Cannot")) && System.nanoTime() < deadline do Thread.sleep(50)
      require(fx(window.preview.zoom.image == null && window.preview.imageStatus.startsWith("Cannot")), "locked image remained visible")
      println("PASS MainWindow dry-run: queued secret snapshot, image parity and locked preview")
    finally
      if window != null then fx { window.dispose(); window.stage.close() }
      service.shutdown(); Platform.exit()
