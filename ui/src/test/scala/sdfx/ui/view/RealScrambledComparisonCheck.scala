package sdfx.ui.view

import javafx.application.Platform
import javafx.scene.Scene
import javafx.scene.control.Label
import javafx.stage.Stage
import sdfx.core.config.{AppHome, Settings}
import sdfx.core.png.{PngMetadata, ScrambledPng}
import sdfx.engine.AppServices
import sdfx.ui.vm.CatalogVM
import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import javax.imageio.ImageIO

object RealScrambledComparisonCheck:
  def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A] { def call(): A = body })
    Platform.runLater(task); task.get(30, TimeUnit.SECONDS)
  def main(args: Array[String]): Unit =
    val root = Path.of(args(0)).toAbsolutePath; Files.createDirectories(root)
    val image = new BufferedImage(32, 24, BufferedImage.TYPE_INT_RGB)
    for y <- 0 until 24; x <- 0 until 32 do image.setRGB(x, y, (x << 16) | (y << 8) | 7)
    val plain = root.resolve("plain.png"); val encoded = root.resolve("scrambled.png")
    PngMetadata.write(image, plain, Seq("parameters" -> "cat")); ScrambledPng.write(image, encoded, Seq("parameters" -> "cat"), Some("secret"))
    val service = new AppServices(AppHome(Files.createTempDirectory(root, "home-")), Settings(), dryRun = true,
      builtins = Nil, persistSettings = false, installShutdownHook = false)
    val ready = new CountDownLatch(1); Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var stage: Stage = null; var tab: ImageComparisonTab = null; var context: UiContext = null
    def awaitPair(a: Path, b: Path): Unit =
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while !fx(tab.loadedPaths.contains(a -> b)) && System.nanoTime() < deadline do Thread.sleep(50)
      require(fx(tab.loadedPaths.contains(a -> b)))
    try
      fx {
        stage = new Stage(); val ctx = new UiContext(service, new CatalogVM, stage); context = ctx
        tab = new ImageComparisonTab(ctx, _ => None, () => None)
        stage.setScene(new Scene(tab.node, 900, 800)); stage.show()
        service.imageScrambleSession.unlock("secret")
        tab.compare(plain, plain)
      }
      awaitPair(plain, plain)
      val expected = root.resolve("ordinary-export.png"); fx(tab.exportTo(expected))
      fx(tab.compare(plain, encoded)); awaitPair(plain, encoded)
      val actual = root.resolve("scrambled-export.png"); fx(tab.exportTo(actual))
      require(PngMetadata.readTexts(expected).get("Encrypt").contains("pixel_shuffle_3"))
      require(PngMetadata.readTexts(actual).get("Encrypt").contains("pixel_shuffle_3"))
      val a = ScrambledPng.read(expected, Some("secret")).image
      val b = ScrambledPng.read(actual, Some("secret")).image
      require(a.getWidth == b.getWidth && a.getHeight == b.getHeight)
      require(a.getRGB(0, 0, a.getWidth, a.getHeight, null, 0, a.getWidth).sameElements(b.getRGB(0, 0, b.getWidth, b.getHeight, null, 0, b.getWidth)))
      // Model a session change before the export destination is accepted. The OS chooser
      // itself is not automated; exportTo exercises its equivalent stale-session guard.
      val existing = root.resolve("stale-export.png")
      val sentinel = "existing target".getBytes(java.nio.charset.StandardCharsets.UTF_8)
      Files.write(existing, sentinel)
      fx(service.imageScrambleSession.unlock("wrong"))
      val rejected = scala.util.Try(fx(tab.exportTo(existing))).failed.toOption
      require(rejected.exists(e => Option(e.getCause).exists(_.isInstanceOf[IllegalArgumentException])),
        "stale comparison export was not rejected")
      require(Files.readAllBytes(existing).sameElements(sentinel), "stale export overwrote target")
      fx(context.imageAccessChanged())
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while !fx(Option(stage.getScene.lookup("#comparison-status")).exists(_.asInstanceOf[Label].getText.startsWith("Cannot compare"))) && System.nanoTime() < deadline do Thread.sleep(50)
      require(fx(tab.loadedPaths.isEmpty), "old decoded comparison remained loaded")
      fx(stage.close()); println("PASS native comparison: decoded parity and stale-session export rejection (OS chooser not automated)")
    finally
      if stage != null then fx(stage.close())
      service.shutdown(); Platform.exit()
