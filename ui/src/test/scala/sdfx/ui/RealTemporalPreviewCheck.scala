package sdfx.ui

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path, Paths}
import java.util.Base64
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.scene.control.{Label, TabPane, TextField}
import javafx.scene.image.ImageView
import javafx.stage.Stage
import javax.imageio.ImageIO
import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters.*
import sdfx.core.config.{AppHome, Settings}
import sdfx.core.protocol.TemporalLabResult
import sdfx.engine.AppServices
import sdfx.plugins.StandardPlugins
import sdfx.ui.view.{PreviewImageLoader, TemporalLabTab}
import sdfx.ui.vm.UiState

/** JavaFX preview lifecycle regression; injects result data and does not launch an inference worker. */
object RealTemporalPreviewCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A] { def call(): A = body })
    Platform.runLater(task); task.get(30, TimeUnit.SECONDS)
  private def eventually(label: String)(condition: => Boolean): Unit =
    val deadline = System.nanoTime + TimeUnit.SECONDS.toNanos(30)
    var yes = false
    while !yes && System.nanoTime < deadline do
      yes = condition
      if !yes then Thread.sleep(25)
    require(yes, s"Timeout: $label")
  private def field[A](obj: AnyRef, suffix: String): A =
    val matches = obj.getClass.getDeclaredFields.filter(_.getName.endsWith(suffix))
    require(matches.length == 1, s"Missing unique field $suffix")
    matches.head.setAccessible(true); matches.head.get(obj).asInstanceOf[A]
  private def call(obj: AnyRef, suffix: String, arg: AnyRef): Unit =
    val matches = obj.getClass.getDeclaredMethods.filter(_.getName.endsWith(suffix))
    require(matches.length == 1, s"Missing unique method $suffix")
    matches.head.setAccessible(true); matches.head.invoke(obj, arg); ()
  private def image(path: Path, rgb: Int): Path =
    val img = new BufferedImage(40, 30, BufferedImage.TYPE_INT_RGB)
    for y <- 0 until 30; x <- 0 until 40 do img.setRGB(x, y, rgb)
    require(ImageIO.write(img, "png", path.toFile)); path
  def main(args: Array[String]): Unit =
    require(args.length == 1, "output-directory")
    val root = Paths.get(args(0)).toAbsolutePath.normalize()
    Files.createDirectories(root)
    val run = Files.createTempDirectory(root, "temporal-preview-")
    val home = AppHome(Files.createDirectory(run.resolve("home")))
    val a = image(run.resolve("previous.png"), 0x123456)
    val b = image(run.resolve("current.png"), 0x654321)
    val bad = Files.createDirectory(run.resolve("bad"))
    Files.writeString(bad.resolve("frame000001.png"), "not an image")
    val good = Files.createDirectory(run.resolve("good"))
    image(good.resolve("frame000001.png"), 0xaabbcc)
    val report = ArrayBuffer.empty[String]
    var services: AppServices = null
    var window: MainWindow = null
    var started = false
    var failure: Throwable = null
    try
      services = new AppServices(home, Settings(), true, StandardPlugins.withOwnAxes,
        launchOverride = Some(() => throw new IllegalStateException("unexpected engine launch")),
        persistSettings = false, installShutdownHook = false)
      val ready = new CountDownLatch(1)
      Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
      require(ready.await(30, TimeUnit.SECONDS), "JavaFX startup timeout")
      started = true
      fx {
        window = new MainWindow(new Stage(), services, UiState(), CliOptions(), _ => ())
        window.show()
        val pane = window.stage.getScene.getRoot.lookup("#main-tabs").asInstanceOf[TabPane]
        pane.getSelectionModel.select(pane.getTabs.asScala.find(_.getText == "Temporal Lab").get)
      }
      val tab = fx { window.temporalLab }
      val mask = fx { field[ImageView](tab, "mask") }
      val previous = fx { field[ImageView](tab, "previousImage") }
      val current = fx { field[ImageView](tab, "currentImage") }
      val frame = fx { field[ImageView](tab, "image") }
      val description = fx { field[Label](tab, "previewDescription") }
      val encoded = Base64.getEncoder.encodeToString(Files.readAllBytes(a))
      val result = TemporalLabResult("preview", maskPngBase64 = Some(encoded), frameIndex = Some(1),
        generatedPrevious = Some(a.toString), generatedCurrent = Some(b.toString))
      fx { call(tab, "showMask", result) }
      eventually("mask and two comparison images displayed") { fx {
        mask.getImage != null && previous.getImage != null && current.getImage != null &&
          previous.getImage.getProgress >= 1 && current.getImage.getProgress >= 1
      }}
      fx {
        val input = window.stage.getScene.lookup("#temporal-reference-frames").asInstanceOf[TextField]
        require(input != null, "reference field missing")
        input.setText("changed-input")
      }
      eventually("actual input edit clears old preview") { fx {
        mask.getImage == null && previous.getImage == null && current.getImage == null &&
          !description.getText.contains("Reference transition")
      }}
      report += "input edit clears all previous mask and pair previews"
      fx { call(tab, "loadFrames", bad.toString) }
      eventually("actual image decode error displayed") { fx { description.getText.toLowerCase.contains("failed") } }
      require(fx { frame.getImage == null }, "invalid frame retained image")
      report += "corrupt frame shows visible error"
      fx { call(tab, "loadFrames", good.toString) }
      eventually("image display recovers") { fx { frame.getImage != null && !frame.getImage.isError && frame.getImage.getProgress >= 1 } }
      report += "good frame recovers after an error"
      fx {
        val screenshot = run.resolve("temporal-preview.png")
        window.stage.getScene.getRoot.applyCss(); window.stage.getScene.getRoot.layout()
        require(ImageIO.write(SwingFXUtils.fromFXImage(window.stage.getScene.snapshot(null), null), "png", screenshot.toFile))
        tab.dispose()
        window.dispose(); window.stage.close()
      }
      report += "workspace disposal"
    catch case e: Throwable => failure = e
    finally
      if started && window != null then try fx { window.dispose(); window.stage.close() } catch case _: Throwable => ()
      if services != null then services.shutdown()
      PreviewImageLoader.shutdownProcess()
      if started then Platform.exit()
      Files.writeString(run.resolve("report.json"), ujson.write(ujson.Obj("status" -> (if failure == null then "passed" else "failed"),
        "checks" -> ujson.Arr.from(report), "failure" -> Option(failure).map(_.toString).getOrElse(""),
        "limits" -> ujson.Arr("injected preview results", "no worker or GPU")), indent = 2))
    if failure != null then throw new IllegalStateException("Temporal preview check failed", failure)
