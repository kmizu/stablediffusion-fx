package sdfx.ui

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path, Paths}
import java.security.MessageDigest
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import javafx.application.Platform
import javafx.beans.property.{SimpleDoubleProperty, SimpleStringProperty}
import javafx.embed.swing.SwingFXUtils
import javafx.scene.control.{Button, Label, TabPane, TextField}
import javafx.scene.image.ImageView
import javafx.stage.Stage
import javax.imageio.ImageIO
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.Await
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*
import scala.util.Using
import sdfx.core.config.{AppHome, Settings}
import sdfx.engine.AppServices
import sdfx.plugins.StandardPlugins
import sdfx.ui.view.{PreviewImageLoader, TemporalLabTab}
import sdfx.ui.vm.{TemporalLabState, UiState}

/** Native presentation interaction with real temporal worker, synthetic fixtures, no model/GPU inference. */
object RealTemporalPresentationCheck:
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
  private def writeImage(path: Path, rgb: Int, changed: Boolean = false): Path =
    val image = new BufferedImage(64, 64, BufferedImage.TYPE_INT_RGB)
    for y <- 0 until 64; x <- 0 until 64 do
      image.setRGB(x, y, if changed && x >= 24 && x < 40 && y >= 24 && y < 40 then 0xc0c0c0 else rgb)
    require(ImageIO.write(image, "png", path.toFile)); path
  private def copyPython(source: Path, dest: Path): Unit =
    Using.resource(Files.walk(source)) { stream =>
      stream.iterator.asScala.filter(p => Files.isRegularFile(p) && p.getFileName.toString.endsWith(".py")).foreach { p =>
        val target = dest.resolve(source.relativize(p))
        Files.createDirectories(target.getParent); Files.copy(p, target)
      }
    }
  private def makeVideo(python: Path, path: Path): Unit =
    val code = "import cv2,numpy as np,sys; w=cv2.VideoWriter(sys.argv[1],cv2.VideoWriter_fourcc(*'MJPG'),10,(64,64)); assert w.isOpened(); [w.write(np.full((64,64,3),v,dtype=np.uint8)) for v in (10,50,90,130,170,210)]; w.release()"
    val log = path.getParent.resolve("video-fixture.log")
    val process = new ProcessBuilder(python.toString, "-c", code, path.toString)
      .redirectErrorStream(true).redirectOutput(log.toFile).start()
    require(process.waitFor(60, TimeUnit.SECONDS), s"video fixture generator still running: ${process.pid()}")
    require(process.exitValue() == 0 && Files.isRegularFile(path), s"video fixture failed: $log")
  private def frames(directory: Path): Vector[Path] =
    if !Files.isDirectory(directory) then Vector.empty
    else Using.resource(Files.list(directory))(_.iterator.asScala.filter(p => p.getFileName.toString.matches("frame[0-9]{6}\\.png")).toVector.sortBy(_.toString))

  def main(args: Array[String]): Unit =
    require(args.length == 2, "python-executable output-directory")
    val python = Paths.get(args(0)).toAbsolutePath.normalize()
    val destination = Paths.get(args(1)).toAbsolutePath.normalize()
    Files.createDirectories(destination)
    val run = Files.createTempDirectory(destination, "temporal-presentation-")
    val home = AppHome(Files.createDirectory(run.resolve("home")))
    copyPython(Paths.get("python/sdfx").toAbsolutePath, home.pythonDir.resolve("sdfx"))
    val reference = Files.createDirectory(run.resolve("reference"))
    val generated = Files.createDirectory(run.resolve("generated"))
    val out = Files.createDirectory(run.resolve("outputs"))
    val sources = Vector(
      writeImage(reference.resolve("frame0001.png"), 0x404040),
      writeImage(reference.resolve("frame0002.png"), 0x404040, changed = true),
      writeImage(reference.resolve("frame0003.png"), 0x404040, changed = true),
      writeImage(generated.resolve("frame0001.png"), 0x202020),
      writeImage(generated.resolve("frame0002.png"), 0xb0b0b0),
      writeImage(generated.resolve("frame0003.png"), 0x707070))
    val alternate = Files.createDirectory(run.resolve("alternate"))
    val wide = new BufferedImage(80, 40, BufferedImage.TYPE_INT_RGB)
    require(ImageIO.write(wide, "png", alternate.resolve("frame0001.png").toFile))
    val inputHashes = sources.map(p => p -> sha(p)).toMap
    val video = run.resolve("fixture.avi")
    makeVideo(python, video)
    val videoHash = sha(video)
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
      val info = fx { window.stage.getScene.lookup("#temporal-input-info").asInstanceOf[Label] }
      require(info != null, "missing input info")
      def edit(id: String, value: String): Unit = fx {
        val control = window.stage.getScene.lookup("#" + id).asInstanceOf[TextField]
        require(control != null, s"missing input field $id")
        control.setText(value)
      }
      edit("temporal-reference-frames", reference.toString)
      edit("temporal-generated-frames", generated.toString)
      eventually("matching frame counts and sizes") { fx {
        val t = info.getText.toLowerCase
        t.contains("reference: 3") && t.contains("generated: 3") && t.contains("64 × 64")
      }}
      checks += "actual text fields asynchronously report matching counts and dimensions"
      // Switch twice in the same FX turn. The invalid intermediate scan must never overwrite the latest.
      fx {
        val control = window.stage.getScene.lookup("#temporal-reference-frames").asInstanceOf[TextField]
        control.setText(run.resolve("missing").toString)
        control.setText(alternate.toString)
      }
      eventually("latest changed input and crop notice") { fx {
        val t = info.getText.toLowerCase
        t.contains("reference: 1") && t.contains("80 × 40") && t.contains("generated: 3") &&
          t.contains("counts differ") && t.contains("center-cropped") && t.contains("resized")
      }}
      Thread.sleep(300)
      require(fx { info.getText.contains("80 × 40") }, "stale scan replaced latest input")
      checks += "rapid source change shows latest count, mismatch and crop/resize notice"
      def labels: String = fx { window.stage.getScene.getRoot.lookupAll(".label").asScala.collect { case l: Label => l.getText }.mkString("\n").toLowerCase }
      require(labels.contains("nearest integer") && labels.contains("next odd"), "DFI help missing from actual UI")
      def requireUnclipped(label: Label): Unit = fx {
        val root = window.stage.getScene.getRoot
        root.applyCss(); root.layout()
        val width = label.getWidth
        require(width > 0 && label.isWrapText && label.getHeight + 1 >= label.prefHeight(width),
          s"notice clipped: width=$width height=${label.getHeight} required=${label.prefHeight(width)} text=${label.getText}")
      }
      def dfiHelp: Label = fx { window.stage.getScene.getRoot.lookupAll(".label").asScala.collectFirst {
        case l: Label if l.getText.contains("nearest integer") => l
      }.get }
      fx { window.stage.setWidth(1500); window.stage.setHeight(1000) }
      Thread.sleep(600)
      requireUnclipped(info); requireUnclipped(dfiHelp)
      screenshot("presentation-normal")
      fx { window.stage.setWidth(window.stage.getMinWidth); window.stage.setHeight(window.stage.getMinHeight) }
      Thread.sleep(600)
      requireUnclipped(info); requireUnclipped(dfiHelp)
      screenshot("presentation-minimum")
      choose("Deflicker")
      eventually("deflicker help visible") { labels.contains("first pair") }
      val stages = labels
      require(stages.contains("blend") && stages.contains("overlay") && stages.contains("normalize") && stages.contains("twice"), "deflicker help missing from actual UI")
      screenshot("presentation-deflicker")
      checks += "actual UI shows DFI quantization and deflicker stage and duplicate explanations"
      fx { outputParent.set(out.toString); outputName.set("extract"); field[SimpleStringProperty](tab, "video").set(video.toString); field[SimpleDoubleProperty](tab, "fps").set(5.0) }
      choose("Extract"); press("temporal-extract"); done("extract")
      val text = fx { status.getText.toLowerCase }
      require(text.contains("source fps: 10") && text.contains("output fps: 5") && text.contains("duration: 0.6"), s"missing extraction metadata: $text")
      require(frames(out.resolve("extract")).size == 3, "unexpected extracted frames")
      checks += "actual extraction UI displays source/output FPS and duration"
      screenshot("presentation-extract")
      require(inputHashes.forall((path, hash) => sha(path) == hash) && sha(video) == videoHash, "input modified")
      checks += "synthetic source files unchanged"
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
        "limits" -> ujson.Arr("synthetic frames and video", "no SD model or GPU generation", "paths typed into real text fields", "source switch exercises live asynchronous header inspection"),
        "failure" -> Option(failure).map(_.toString).getOrElse(""), "cleanupErrors" -> ujson.Arr.from(cleanup))
      Files.writeString(run.resolve("temporal-presentation-report.json"), ujson.write(report, indent = 2))
    if failure != null || cleanup.nonEmpty then throw new IllegalStateException(s"Temporal presentation check failed: ${Option(failure).map(_.toString).getOrElse("")}", failure)
