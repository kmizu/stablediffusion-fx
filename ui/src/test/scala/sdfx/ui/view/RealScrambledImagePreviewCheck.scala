package sdfx.ui.view

import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.scene.Scene
import javafx.scene.layout.VBox
import javafx.stage.Stage
import sdfx.core.config.{AppHome, Settings}
import sdfx.core.png.ScrambledPng
import sdfx.engine.AppServices
import sdfx.ui.vm.{CatalogVM, Mode}
import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}

object RealScrambledImagePreviewCheck:
  def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A] { def call(): A = body })
    Platform.runLater(task); task.get(30, TimeUnit.SECONDS)
  def main(args: Array[String]): Unit =
    val root = Path.of(args(0)).toAbsolutePath
    Files.createDirectories(root)
    val image = new BufferedImage(32, 24, BufferedImage.TYPE_INT_RGB)
    for y <- 0 until 24; x <- 0 until 32 do image.setRGB(x, y, (x << 16) | (y << 8) | 7)
    val protectedPath = root.resolve("scrambled.png")
    ScrambledPng.write(image, protectedPath, Nil, Some("secret"))
    val service = new AppServices(AppHome(Files.createTempDirectory(root, "home-")), Settings(), dryRun = true,
      builtins = Nil, persistSettings = false, installShutdownHook = false)
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var stage: Stage = null; var preview: PreviewPane = null; var context: UiContext = null
    try
      fx {
        stage = new Stage()
        val ctx = new UiContext(service, new CatalogVM, stage); context = ctx
        val actions = new ImageActions:
          def sendTo(mode: Mode, path: Path, info: Option[String]): Unit = ()
          def restore(text: String): Unit = ()
          def delete(path: Path): Unit = ()
        preview = new PreviewPane(ctx, actions)
        stage.setScene(new Scene(new VBox(preview.imageNode, preview.infoNode), 700, 600)); stage.show()
        service.imageScrambleSession.unlock("secret")
        preview.show(protectedPath)
      }
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while fx(preview.zoom.image == null) && System.nanoTime() < deadline do Thread.sleep(50)
      val decoded = fx(SwingFXUtils.fromFXImage(preview.zoom.image, null))
      require(image.getRGB(0, 0, 32, 24, null, 0, 32).sameElements(decoded.getRGB(0, 0, 32, 24, null, 0, 32)))
      fx { service.imageScrambleSession.unlock("wrong"); context.imageAccessChanged() }
      val failureDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while !fx(preview.imageStatus.startsWith("Cannot")) && System.nanoTime() < failureDeadline do Thread.sleep(50)
      require(fx(preview.zoom.image == null && preview.imageStatus.startsWith("Cannot")))
      fx(stage.close())
      println("PASS native preview: protected pixels decoded; wrong password shows no scrambled pixels")
    finally
      if stage != null then fx(stage.close())
      service.shutdown(); Platform.exit()
