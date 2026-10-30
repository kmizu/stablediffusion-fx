package sdfx.ui

import javafx.application.Platform
import javafx.stage.Stage
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

/** Actual native Generate path and persisted PNG metadata, with a real model. */
object RealStepsAnimationCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body)
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)

  def main(args: Array[String]): Unit =
    require(args.length == 3, "home python checkpoint")
    val home = AppHome(Paths.get(args(0)).toAbsolutePath)
    val checkpoint = Paths.get(args(2)).toAbsolutePath
    Files.createDirectories(home.pythonDir)
    val services = new AppServices(home, Settings(
      pythonExe = Some(Paths.get(args(1)).toAbsolutePath.toString),
      modelDirs = ModelDirs(checkpoints = Seq(checkpoint.getParent.toString))), dryRun = false,
      builtins = StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false)
    val cancelled = new CountDownLatch(1)
    val finished = new LinkedBlockingQueue[Either[String, OutputImage]]()
    services.queue.addListener {
      case event: JobEvent.JobFinished => finished.offer(event.output.images.headOption.toRight("no image")); ()
      case _: JobEvent.JobCancelled => cancelled.countDown()
      case event: JobEvent.JobFailed => finished.offer(Left(event.message)); ()
      case _ => ()
    }
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    try
      services.start()
      services.loadCheckpoint(checkpoint.toString, remember = false)
      fx {
        val size = if checkpoint.getFileName.toString.toLowerCase.contains("sdxl") then 768 else 512
        window = new MainWindow(new Stage(), services, UiState(params = Map("txt2img" ->
          GenerationParams(prompt = "a ceramic teapot on a wooden table", seed = 7, steps = 4, width = size, height = size))), CliOptions(), _ => ())
        window.show()
        window.model.checkpoint.set(checkpoint.toString)
      }
      val outputs = Seq(false, true).map { enabled =>
        fx {
          window.plugins(Mode.Txt2Img).selectedScript.set(if enabled then "steps_animation" else "")
          window.plugins(Mode.Txt2Img).form("steps_animation").get.props("duration").set("1")
          window.plugins(Mode.Txt2Img).form("steps_animation").get.props("fps").set("10")
          window.statusBar.generateBtn.fire()
        }
        val result = finished.poll(120, TimeUnit.SECONDS)
        require(result != null, "native generation timed out")
        val image = result.fold(message => throw new IllegalStateException(message), identity)
        val fields = Infotext.parse(PngMetadata.readTexts(image.path)(PngMetadata.ParametersKey)).fields
        if enabled then require(fields.contains("Steps animation"), "saved animation settings missing")
        image.path
      }
      val images = outputs.map(path => ImageIO.read(path.toFile))
      val arrays = images.map(image => image.getRGB(0, 0, image.getWidth, image.getHeight, null, 0, image.getWidth))
      require(java.util.Arrays.equals(arrays.head, arrays(1)), "animation recording changed pixels")
      val files = Files.walk(home.root)
      try require(files.anyMatch(path => path.toString.endsWith(".mp4")), "native animation video missing")
      finally files.close()
      def countFiles(suffix: String): Long =
        val stream = Files.walk(home.root)
        try stream.filter(path => path.toString.endsWith(suffix)).count()
        finally stream.close()
      val beforeFrames = countFiles(".png")
      val beforeVideos = countFiles(".mp4")
      fx {
        window.vms(Mode.Txt2Img).steps.set(40)
        window.plugins(Mode.Txt2Img).form("steps_animation").get.props("run_incomplete").set("true")
        window.statusBar.generateBtn.fire()
      }
      val frameDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(120)
      while countFiles(".png") < beforeFrames + 2 && System.nanoTime() < frameDeadline do Thread.sleep(50)
      require(countFiles(".png") >= beforeFrames + 2, "no native captured frames before interrupt")
      fx { window.statusBar.generateBtn.fire() }
      require(cancelled.await(30, TimeUnit.SECONDS), "native animation must end cancelled")
      require(countFiles(".mp4") == beforeVideos + 1, "native partial video missing")
      val idleDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5)
      while !services.queue.isIdle && System.nanoTime() < idleDeadline do Thread.sleep(20)
      require(services.queue.isIdle)
      fx {
        window.vms(Mode.Txt2Img).steps.set(4)
        window.plugins(Mode.Txt2Img).selectedScript.set("")
        window.statusBar.generateBtn.fire()
      }
      val recovery = finished.poll(120, TimeUnit.SECONDS)
      require(recovery != null && recovery.isRight, s"native recovery failed: $recovery")
      println(s"PASS native Steps Animation: video saved, exact RGB, native partial interrupt/recovery; ${outputs.mkString(", ")}")
    finally
      fx { if window != null then { window.dispose(); window.stage.close() } }
      services.shutdown()
      Platform.exit()
