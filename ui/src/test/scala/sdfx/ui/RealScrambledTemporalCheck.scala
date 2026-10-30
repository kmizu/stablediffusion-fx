package sdfx.ui

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import javafx.application.Platform
import javafx.beans.property.SimpleStringProperty
import javafx.scene.control.{Button, Label, TabPane}
import javafx.scene.image.ImageView
import javafx.stage.Stage
import sdfx.core.config.{AppHome, Settings}
import sdfx.core.png.{PngMetadata, ScrambledPng}
import sdfx.engine.AppServices
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{TemporalLabState, UiState}
import scala.jdk.CollectionConverters.*

/** Real native Temporal Lab and Python worker; synthetic frame fixtures, no diffusion model. */
object RealScrambledTemporalCheck:
  def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A] { def call(): A = body })
    Platform.runLater(task); task.get(30, TimeUnit.SECONDS)
  def eventually(label: String)(test: => Boolean): Unit =
    val end = System.nanoTime() + TimeUnit.SECONDS.toNanos(90)
    while !test && System.nanoTime() < end do Thread.sleep(50)
    require(test, s"Timeout: $label")
  def field[A](obj: AnyRef, suffix: String): A =
    val f = obj.getClass.getDeclaredFields.filter(_.getName.endsWith(suffix))
    require(f.length == 1, s"Missing field $suffix"); f.head.setAccessible(true); f.head.get(obj).asInstanceOf[A]
  def main(args: Array[String]): Unit =
    require(args.length == 2, "python output-dir")
    val root = Path.of(args(1)).toAbsolutePath; Files.createDirectories(root)
    val run = Files.createTempDirectory(root, "temporal-protected-")
    val home = AppHome(Files.createDirectories(run.resolve("home"))); Files.createDirectories(home.pythonDir)
    val input = Files.createDirectories(run.resolve("inputs")); val output = Files.createDirectories(run.resolve("outputs"))
    val originals = (1 to 3).map { index =>
      val image = new BufferedImage(64, 64, BufferedImage.TYPE_INT_RGB)
      for y <- 0 until 64; x <- 0 until 64 do image.setRGB(x, y, (x << 16) | (y << 8) | (index * 30))
      val path = input.resolve(f"frame$index%04d.png")
      ScrambledPng.write(image, path, Nil, Some("temporal-secret")); (path, Files.readAllBytes(path).toVector, image)
    }
    val services = new AppServices(home, Settings(pythonExe = Some(Path.of(args(0)).toAbsolutePath.toString)),
      dryRun = false, builtins = StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false)
    val ready = new CountDownLatch(1); Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    try
      services.start(); services.imageScrambleSession.unlock("temporal-secret")
      window = fx {
        val w = new MainWindow(new Stage(), services, UiState(), CliOptions(), _ => ())
        w.show(); val tabs = w.stage.getScene.lookup("#main-tabs").asInstanceOf[TabPane]
        tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getText == "Temporal Lab").get)
        w.ctx.imageAccessChanged(); w
      }
      val tab = fx(window.temporalLab)
      val state = fx(field[TemporalLabState](tab, "state"))
      val picture = fx(field[ImageView](tab, "image"))
      val status = fx(field[Label](tab, "status"))
      fx {
        field[SimpleStringProperty](tab, "frames").set(input.toString)
        field[SimpleStringProperty](tab, "outputParent").set(output.toString)
        field[SimpleStringProperty](tab, "outputName").set("protected")
        val tabs = field[TabPane](tab, "workTabs")
        tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getText == "Deflicker").get)
        val button = window.stage.getScene.lookup("#temporal-deflicker").asInstanceOf[Button]
        require(button != null && !button.isDisabled); button.fire()
      }
      eventually("processing") { fx(!state.busy) }
      require(fx(state.result.nonEmpty), s"Temporal failed: ${fx(status.getText)}")
      val first = output.resolve("protected/frame000001.png")
      require(PngMetadata.readTexts(first).get("Encrypt").contains("pixel_shuffle_3"))
      val decoded = ScrambledPng.read(first, Some("temporal-secret")).image
      require(decoded.getRGB(4, 5) == originals.head._3.getRGB(4, 5))
      eventually("decoded preview") { fx(picture.getImage != null && picture.getImage.getPixelReader.getArgb(4, 5) == decoded.getRGB(4, 5)) }
      fx { services.imageScrambleSession.lock(); window.ctx.imageAccessChanged() }
      eventually("locked preview cleared") { fx(picture.getImage == null) }
      fx { services.imageScrambleSession.unlock("temporal-secret"); window.ctx.imageAccessChanged() }
      eventually("unlocked preview restored") { fx(picture.getImage != null && picture.getImage.getPixelReader.getArgb(4, 5) == decoded.getRGB(4, 5)) }
      require(originals.forall((p, bytes, _) => Files.readAllBytes(p).toVector == bytes))
      println(s"PASS native Temporal Lab protected processing, pixel preview, lock and recovery: $run")
    finally
      if window != null then fx { window.dispose(); window.stage.close() }
      services.shutdown(); Platform.exit()
