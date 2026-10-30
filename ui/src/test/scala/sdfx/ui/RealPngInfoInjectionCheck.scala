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
object RealPngInfoInjectionCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body)
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)

  def main(args: Array[String]): Unit =
    require(args.length >= 3, "home python checkpoint [wildcards]")
    val wildcards = args.lift(3).contains("wildcards")
    val home = AppHome(Paths.get(args(0)).toAbsolutePath)
    val checkpoint = Paths.get(args(2)).toAbsolutePath
    Files.createDirectories(home.pythonDir)
    val wildcardDir = home.pythonDir.resolve("pnginfo-wildcards")
    if wildcards then
      Files.createDirectories(wildcardDir)
      Files.writeString(wildcardDir.resolve("positive.txt"), "a ceramic teapot on a wooden table\n")
      Files.writeString(wildcardDir.resolve("negative.txt"), "blurry\n")
    val services = new AppServices(home, Settings(
      pythonExe = Some(Paths.get(args(1)).toAbsolutePath.toString),
      modelDirs = ModelDirs(checkpoints = Seq(checkpoint.getParent.toString))), dryRun = false,
      builtins = StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false)
    val finished = new LinkedBlockingQueue[Either[String, OutputImage]]()
    services.queue.addListener {
      case event: JobEvent.JobFinished => finished.offer(event.output.images.headOption.toRight("no image")); ()
      case event: JobEvent.JobFailed => finished.offer(Left(event.message)); ()
      case _ => ()
    }
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    try
      services.start()
      val loaded = services.loadCheckpoint(checkpoint.toString, remember = false)
      fx {
        val size = if checkpoint.getFileName.toString.toLowerCase.contains("sdxl") then 768 else 512
        window = new MainWindow(new Stage(), services, UiState(params = Map("txt2img" ->
          GenerationParams(prompt = if wildcards then "__positive__" else "a ceramic teapot on a wooden table", negativePrompt = if wildcards then "__negative__" else "", seed = 7, steps = 8, width = size, height = size))), CliOptions(), _ => ())
        window.show()
        window.model.checkpoint.set(checkpoint.toString)
        if wildcards then
          window.plugins(Mode.Txt2Img).selectedScript.set("dynamic_prompts")
          window.plugins(Mode.Txt2Img).form("dynamic_prompts").get.props("wildcard_dirs").set(wildcardDir.toString)
      }
      val outputs = Seq(false, true).map { enabled =>
        fx {
          window.plugins(Mode.Txt2Img).form("pnginfo_injection").get.props("enabled").set(enabled.toString)
          window.statusBar.generateBtn.fire()
        }
        val result = finished.poll(120, TimeUnit.SECONDS)
        require(result != null, "native generation timed out")
        val image = result.fold(message => throw new IllegalStateException(message), identity)
        val fields = Infotext.parse(PngMetadata.readTexts(image.path)(PngMetadata.ParametersKey)).fields
        if enabled then require(ujson.read(fields("Hashes"))("model").str == loaded.hash.take(10), "saved model hash mismatch")
        if enabled && wildcards then
          for name <- Seq("positive", "negative") do
            val bytes = Files.readAllBytes(wildcardDir.resolve(s"$name.txt"))
            val expected = java.security.MessageDigest.getInstance("SHA-256").digest(bytes).map(b => f"${b & 0xff}%02x").mkString.take(10)
            require(ujson.read(fields("Hashes"))(s"wildcards_file:$name").str == expected, s"missing wildcard hash: $name")
        if !enabled then require(!fields.contains("Hashes"), "disabled injection added Hashes")
        image.path
      }
      val images = outputs.map(path => ImageIO.read(path.toFile))
      val arrays = images.map(image => image.getRGB(0, 0, image.getWidth, image.getHeight, null, 0, image.getWidth))
      require(java.util.Arrays.equals(arrays.head, arrays(1)), "metadata injection changed pixels")
      println(s"PASS native PNGINFO: Hashes saved, disabled omitted, exact RGB; ${outputs.mkString(", ")}")
    finally
      fx { if window != null then { window.dispose(); window.stage.close() } }
      services.shutdown()
      Platform.exit()
