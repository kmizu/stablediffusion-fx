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
import sdfx.engine.AppServices
import sdfx.plugins.StandardPlugins
import sdfx.ui.view.{PreviewImageLoader, TemporalLabTab}
import sdfx.ui.vm.{TemporalLabState, UiState}

/** Real native workspace and real Python temporal worker, synthetic inputs only; no model/GPU generation.
  * Args: Python with OpenCV/diffusers, output directory. Creates a small MJPEG fixture via that Python.
  */
object RealTemporalLabCheck:
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
    val run = Files.createTempDirectory(destination, "temporal-native-")
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
      fx { outputParent.set(out.toString); outputName.set("dfi"); field[SimpleStringProperty](tab, "reference").set(reference.toString); field[SimpleStringProperty](tab, "generated").set(generated.toString) }
      choose("DFI"); press("temporal-preview"); done("preview")
      eventually("mask preview displayed") { fx { field[ImageView](tab, "mask").getImage != null } }
      require(!Files.exists(out.resolve("dfi")), "preview wrote final output")
      checks += "actual worker DFI mask preview; no output"
      screenshot("temporal-preview")
      press("temporal-dfi"); done("dfi")
      val dfi = frames(out.resolve("dfi"))
      require(dfi.size == 3 && ImageIO.read(dfi.head.toFile).getRGB(0, 0) ==
        ImageIO.read(generated.resolve("frame0001.png").toFile).getRGB(0, 0), "DFI output count or first frame")
      checks += "actual worker DFI three frames and first pixels"
      fx { outputName.set("deflicker"); field[SimpleStringProperty](tab, "frames").set(generated.toString); field[SimpleDoubleProperty](tab, "blend").set(.5) }
      choose("Deflicker"); press("temporal-deflicker"); done("deflicker")
      val deflicker = frames(out.resolve("deflicker"))
      require(deflicker.size == 3 && ImageIO.read(deflicker(1).toFile).getRGB(0, 0) != ImageIO.read(generated.resolve("frame0002.png").toFile).getRGB(0, 0), "deflicker output")
      checks += "actual worker deflicker changed second frame"
      fx { outputName.set("fuse"); field[SimpleStringProperty](tab, "style").set(reference.toString); field[SimpleStringProperty](tab, "videoFrames").set(generated.toString) }
      choose("Fuse"); press("temporal-fuse"); done("fuse")
      require(frames(out.resolve("fuse")).size == 3, "fuse frame count")
      checks += "actual worker fuse three frames"
      fx { outputName.set("extract"); field[SimpleStringProperty](tab, "video").set(video.toString); field[SimpleDoubleProperty](tab, "fps").set(5.0) }
      choose("Extract"); press("temporal-extract"); done("extract")
      require(frames(out.resolve("extract")).size == 3, "extract 10→5 FPS count")
      checks += "actual MJPEG extraction at reduced FPS"
      eventually("playback frame displayed") { fx { field[ImageView](tab, "image").getImage != null } }
      screenshot("temporal-result")
      require(inputHashes.forall((path, hash) => sha(path) == hash) && sha(video) == videoHash, "input modified")
      checks += "all synthetic inputs unchanged"
      // Exercise owner cancellation with enough local frames to leave the worker active.
      val longFrames = Files.createDirectory(run.resolve("cancel-input"))
      for i <- 1 to 1200 do Files.copy(generated.resolve("frame0001.png"), longFrames.resolve(f"frame$i%06d.png"))
      fx { outputName.set("cancelled"); field[SimpleStringProperty](tab, "frames").set(longFrames.toString); field[SimpleDoubleProperty](tab, "blend").set(.5) }
      choose("Deflicker")
      val protection = Await.result(services.client.interruptionProtection(Some(true)), 10.seconds)
      require(protection.enabled, "failed to enable interruption protection")
      val blockedSeen = new java.util.concurrent.atomic.AtomicBoolean(false)
      fx { status.textProperty.addListener((_, _, text) => if text.toLowerCase.contains("blocked") then blockedSeen.set(true)) }
      press("temporal-deflicker")
      fx { window.stage.getScene.lookup("#temporal-cancel").asInstanceOf[Button].fire() }
      eventually("blocked cancellation") { blockedSeen.get() }
      checks += "UI reports blocked cancellation"
      require(Await.result(services.client.interruptionProtection(Some(false)), 10.seconds).enabled == false, "disable protection")
      fx { window.stage.getScene.lookup("#temporal-cancel").asInstanceOf[Button].fire() }
      eventually("cancelled operation finishes") { fx { !state.busy } }
      require(!Files.exists(out.resolve("cancelled")), "cancelled job published partial output")
      require(fx { state.result.isEmpty }, "cancelled job reported success")
      checks += "owned cancellation leaves no final output"
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
        "limits" -> ujson.Arr("synthetic frames and video", "no SD model or GPU generation"),
        "failure" -> Option(failure).map(_.toString).getOrElse(""), "cleanupErrors" -> ujson.Arr.from(cleanup))
      Files.writeString(run.resolve("temporal-native-report.json"), ujson.write(report, indent = 2))
    if failure != null || cleanup.nonEmpty then throw new IllegalStateException(s"Temporal native check failed: ${Option(failure).map(_.toString).getOrElse("")}", failure)
