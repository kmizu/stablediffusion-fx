package sdfx.ui

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path, Paths}
import java.security.MessageDigest
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import javafx.application.Platform
import javafx.beans.property.{SimpleDoubleProperty, SimpleStringProperty}
import javafx.embed.swing.SwingFXUtils
import javafx.scene.control.{Button, Label, TabPane}
import javafx.scene.image.ImageView
import javafx.stage.Stage
import javax.imageio.ImageIO
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.Await
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*
import scala.util.Using
import sdfx.core.config.{AppHome, Settings}
import sdfx.core.png.PngMetadata
import sdfx.engine.AppServices
import sdfx.plugins.StandardPlugins
import sdfx.ui.view.{PreviewImageLoader, TemporalLabTab}
import sdfx.ui.vm.{TemporalLabState, UiState}

/** Prior generated SD1.5/SDXL PNGs, real UI and temporal worker; no new inference.
  * Args: Python with OpenCV/diffusers, output directory, card preview evidence directory.
  */
object RealTemporalModelImageCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A] { def call(): A = body })
    Platform.runLater(task); task.get(30, TimeUnit.SECONDS)
  private def eventually(label: String, timeout: Long = 90)(condition: => Boolean): Unit =
    val deadline = System.nanoTime + TimeUnit.SECONDS.toNanos(timeout)
    var yes = false
    while !yes && System.nanoTime < deadline do
      yes = condition
      if !yes then Thread.sleep(20)
    require(yes, s"Timeout: $label")
  private def field[A](obj: AnyRef, suffix: String): A =
    val matches = obj.getClass.getDeclaredFields.filter(_.getName.endsWith(suffix))
    require(matches.length == 1, s"Missing unique field $suffix: ${matches.map(_.getName).mkString(",")}")
    matches.head.setAccessible(true); matches.head.get(obj).asInstanceOf[A]
  private def sha(path: Path): String =
    MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(path)).map(b => f"${b & 255}%02x").mkString
  private def darken(source: Path, target: Path): Unit =
    val image = ImageIO.read(source.toFile)
    require(image != null, s"Cannot decode $source")
    for y <- 0 until image.getHeight; x <- 0 until image.getWidth do
      val rgb = image.getRGB(x, y)
      val r = ((rgb >> 16) & 255) * 7 / 10
      val g = ((rgb >> 8) & 255) * 7 / 10
      val b = (rgb & 255) * 7 / 10
      image.setRGB(x, y, (rgb & 0xff000000) | (r << 16) | (g << 8) | b)
    require(ImageIO.write(image, "png", target.toFile))
  private def copyPython(source: Path, dest: Path): Unit =
    Using.resource(Files.walk(source)) { stream =>
      stream.iterator.asScala.filter(p => Files.isRegularFile(p) && p.getFileName.toString.endsWith(".py")).foreach { p =>
        val target = dest.resolve(source.relativize(p))
        Files.createDirectories(target.getParent); Files.copy(p, target)
      }
    }
  private def frames(directory: Path): Vector[Path] =
    if !Files.isDirectory(directory) then Vector.empty
    else Using.resource(Files.list(directory))(_.iterator.asScala.filter(p => p.getFileName.toString.matches("frame[0-9]{6}\\.png")).toVector.sortBy(_.toString))

  def main(args: Array[String]): Unit =
    require(args.length == 3, "python-executable output-directory card-preview-evidence-directory")
    val python = Paths.get(args(0)).toAbsolutePath.normalize()
    val destination = Paths.get(args(1)).toAbsolutePath.normalize()
    Files.createDirectories(destination)
    val run = Files.createTempDirectory(destination, "temporal-model-image-")
    val home = AppHome(Files.createDirectory(run.resolve("home")))
    copyPython(Paths.get("python/sdfx").toAbsolutePath, home.pythonDir.resolve("sdfx"))
    val out = Files.createDirectory(run.resolve("outputs"))
    val evidence = Paths.get(args(2)).toAbsolutePath.normalize()
    val inputs = Vector("sd15", "sdxl").map { mode =>
      val reportPath = evidence.resolve(s"$mode-core/card-preview-report.json")
      val report = ujson.read(Files.readString(reportPath))
      require(report("status").str == "passed" && report("mode").str == mode, s"invalid evidence $reportPath")
      val record = report("runs").arr.find(_("label").str == "model-control").get
      val source = Paths.get(record("image").str).toAbsolutePath.normalize()
      require(Files.isRegularFile(source) && sha(source) == record("imageSha256").str, s"source hash mismatch $source")
      val reference = Files.createDirectory(run.resolve(s"$mode-reference"))
      val generated = Files.createDirectory(run.resolve(s"$mode-generated"))
      for i <- 1 to 3 do
        Files.copy(source, reference.resolve(f"frame$i%04d.png"))
        if i == 2 then darken(source, generated.resolve(f"frame$i%04d.png"))
        else Files.copy(source, generated.resolve(f"frame$i%04d.png"))
      val image = ImageIO.read(source.toFile)
      val metadata = ujson.Obj.from(PngMetadata.readTexts(source).map((k, v) => k -> ujson.Str(v)))
      val info = ujson.Obj("mode" -> mode, "priorReport" -> reportPath.toString,
        "priorReportSha256" -> sha(reportPath), "model" -> report("model"), "source" -> source.toString,
        "sourceSha256" -> sha(source), "width" -> image.getWidth, "height" -> image.getHeight,
        "pngTexts" -> metadata, "sequence" -> "copies of same prior image; middle generated frame darkened to 70%")
      (mode, source, sha(source), reference, generated, info)
    }
    val checks = ArrayBuffer.empty[String]
    val screenshots = ArrayBuffer.empty[String]
    val cleanup = ArrayBuffer.empty[String]
    var failure: Throwable = null
    var services: AppServices = null
    var window: MainWindow = null
    var fxStarted = false
    def screenshot(name: String): Unit = fx {
      val scene = window.stage.getScene
      scene.getRoot.applyCss(); scene.getRoot.layout()
      val path = run.resolve(name + ".png")
      require(ImageIO.write(SwingFXUtils.fromFXImage(scene.snapshot(null), null), "png", path.toFile))
      screenshots += path.toString
    }
    try
      services = new AppServices(home, Settings(pythonExe = Some(python.toString)), dryRun = false,
        builtins = StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false)
      val ready = new CountDownLatch(1)
      Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
      require(ready.await(30, TimeUnit.SECONDS), "JavaFX startup timeout")
      fxStarted = true
      fx {
        window = new MainWindow(new Stage(), services, UiState(), CliOptions(), _ => ())
        window.show()
        val tabs = window.stage.getScene.lookup("#main-tabs").asInstanceOf[TabPane]
        tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getText == "Temporal Lab").get)
        window.startEngine()
      }
      eventually("real worker ready") { services.supervisor.state.isUsable }
      require(!services.supervisor.hello.exists(_.dryRun), "unexpected dry-run engine")
      val tab = fx { window.temporalLab }
      val state = fx { field[TemporalLabState](tab, "state") }
      val mode = fx { field[TabPane](tab, "workTabs") }
      val outputParent = fx { field[SimpleStringProperty](tab, "outputParent") }
      val outputName = fx { field[SimpleStringProperty](tab, "outputName") }
      val status = fx { field[Label](tab, "status") }
      def setString(name: String, value: String): Unit = fx { field[SimpleStringProperty](tab, name).set(value) }
      def choose(name: String): Unit = fx { mode.getSelectionModel.select(mode.getTabs.asScala.find(_.getText == name).get) }
      def press(id: String): Unit = fx {
        val b = window.stage.getScene.lookup("#" + id).asInstanceOf[Button]
        require(b != null && !b.isDisabled, s"Missing/disabled button $id")
        b.fire()
        require(state.busy, s"$id did not dispatch: ${status.getText}")
      }
      def done(label: String): Unit =
        eventually(label) { fx { !state.busy } }
        require(fx { state.result.nonEmpty }, s"$label failed: ${fx { status.getText }}")
      for (model, source, hash, reference, generated, info) <- inputs do
        fx { outputParent.set(out.toString); outputName.set(s"$model-dfi"); field[SimpleStringProperty](tab, "reference").set(reference.toString); field[SimpleStringProperty](tab, "generated").set(generated.toString) }
        choose("DFI"); press("temporal-preview"); done(s"$model preview")
        eventually(s"$model mask preview displayed") { fx { field[ImageView](tab, "mask").getImage != null } }
        screenshot(s"$model-preview")
        press("temporal-dfi"); done(s"$model dfi")
        val dfi = frames(out.resolve(s"$model-dfi"))
        require(dfi.size == 3, s"$model DFI frame count")
        val first = ImageIO.read(dfi.head.toFile)
        val original = ImageIO.read(source.toFile)
        require(first != null && first.getWidth == original.getWidth && first.getHeight == original.getHeight &&
          first.getRGB(0, 0) == original.getRGB(0, 0), s"$model DFI first frame")
        checks += s"$model actual worker DFI: three frames, dimensions and first pixel"
        fx { outputName.set(s"$model-deflicker"); field[SimpleStringProperty](tab, "frames").set(generated.toString); field[SimpleDoubleProperty](tab, "blend").set(.5) }
        choose("Deflicker"); press("temporal-deflicker"); done(s"$model deflicker")
        val deflicker = frames(out.resolve(s"$model-deflicker"))
        require(deflicker.size == 3, s"$model deflicker frame count")
        val result = ImageIO.read(deflicker(1).toFile)
        val dark = ImageIO.read(generated.resolve("frame0002.png").toFile)
        require(result != null && result.getWidth == original.getWidth && result.getHeight == original.getHeight, s"$model deflicker dimensions")
        require((0 until result.getHeight).exists(y => (0 until result.getWidth).exists(x => result.getRGB(x,y) != dark.getRGB(x,y))), s"$model deflicker did not change frame")
        eventually(s"$model result displayed") { fx { field[ImageView](tab, "image").getImage != null } }
        screenshot(s"$model-deflicker")
        info("dfiOutputs") = ujson.Arr.from(dfi.map(p => ujson.Obj("path" -> p.toString, "sha256" -> sha(p))))
        info("deflickerOutputs") = ujson.Arr.from(deflicker.map(p => ujson.Obj("path" -> p.toString, "sha256" -> sha(p))))
        require(sha(source) == hash, s"$model source modified")
        checks += s"$model actual worker deflicker changed darkened frame; source intact"
      fx { window.dispose(); window.stage.close() }
      checks += "native owner disposed"
    catch case e: Throwable => failure = e
    finally
      if fxStarted && window != null then try fx { window.dispose(); window.stage.close() } catch case e: Throwable => cleanup += e.toString
      if services != null then try services.shutdown() catch case e: Throwable => cleanup += e.toString
      try PreviewImageLoader.shutdownProcess() catch case e: Throwable => cleanup += e.toString
      if fxStarted then Platform.exit()
      val report = ujson.Obj("status" -> (if failure == null && cleanup.isEmpty then "passed" else "failed"),
        "checks" -> ujson.Arr.from(checks), "screenshots" -> ujson.Arr.from(screenshots),
        "inputs" -> ujson.Arr.from(inputs.map(_._6)),
        "limits" -> ujson.Arr("prior generated model outputs only; no new inference or GPU generation", "static source copies, artificially darkened middle frame; not natural video", "path selection injected; operations dispatched via native buttons"),
        "failure" -> Option(failure).map(_.toString).getOrElse(""), "cleanupErrors" -> ujson.Arr.from(cleanup))
      Files.writeString(run.resolve("temporal-model-image-report.json"), ujson.write(report, indent = 2))
    if failure != null || cleanup.nonEmpty then throw new IllegalStateException(s"Temporal model-image check failed: ${Option(failure).map(_.toString).getOrElse("")}", failure)
