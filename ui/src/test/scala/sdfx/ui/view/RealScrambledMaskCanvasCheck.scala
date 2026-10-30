package sdfx.ui.view

import javafx.application.Platform
import javafx.beans.property.{SimpleBooleanProperty, SimpleStringProperty}
import javafx.scene.{Node, Parent, Scene}
import javafx.scene.control.Label
import javafx.scene.image.ImageView
import javafx.stage.Stage
import sdfx.core.config.{AppHome, Settings}
import sdfx.core.png.ScrambledPng
import sdfx.engine.AppServices
import sdfx.ui.vm.CatalogVM
import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import scala.jdk.CollectionConverters.*

object RealScrambledMaskCanvasCheck:
  def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A] { def call(): A = body })
    Platform.runLater(task); task.get(30, TimeUnit.SECONDS)
  def main(args: Array[String]): Unit =
    val root = Path.of(args(0)).toAbsolutePath; Files.createDirectories(root)
    val img = new BufferedImage(64, 48, BufferedImage.TYPE_INT_RGB)
    for y <- 0 until 48; x <- 0 until 64 do img.setRGB(x, y, (x << 16) | (y << 8) | 7)
    val path = root.resolve("scrambled.png"); ScrambledPng.write(img, path, Nil, Some("secret"))
    val service = new AppServices(AppHome(Files.createTempDirectory(root, "home-")), Settings(), dryRun = true,
      builtins = Nil, persistSettings = false, installShutdownHook = false)
    val ready = new CountDownLatch(1); Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var stage: Stage = null; var canvas: MaskCanvas = null; var context: UiContext = null; val hasMask = new SimpleBooleanProperty(false)
    def nodes(n: Node): Seq[Node] = Seq(n) ++ (n match
      case p: Parent => p.getChildrenUnmodifiable.asScala.toSeq.flatMap(nodes)
      case _ => Nil)
    def failed = nodes(canvas.node).collect { case l: Label if l.isVisible => l.getText }.exists(_.contains("Cannot display"))
    try
      fx {
        stage = new Stage(); val ctx = new UiContext(service, new CatalogVM, stage); context = ctx
        service.imageScrambleSession.unlock("secret")
        canvas = new MaskCanvas(ctx, new SimpleStringProperty(path.toString), hasMask)
        stage.setScene(new Scene(canvas.node, 600, 520)); stage.show()
      }
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while !fx({ canvas.paintDemo(); hasMask.get }) && System.nanoTime() < deadline do Thread.sleep(50)
      require(fx(hasMask.get), "mask image did not load")
      val displayed = fx(javafx.embed.swing.SwingFXUtils.fromFXImage(nodes(canvas.node).collectFirst { case v: ImageView => v }.get.getImage, null))
      require(img.getRGB(0, 0, 64, 48, null, 0, 64).sameElements(displayed.getRGB(0, 0, 64, 48, null, 0, 64)), "input pixels differ")
      val beforePath = fx(canvas.exportMask().get); val before = Files.readAllBytes(beforePath)
      fx { service.imageScrambleSession.unlock("wrong"); context.imageAccessChanged() }
      val failDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while !fx(failed) && System.nanoTime() < failDeadline do Thread.sleep(50)
      require(fx(failed && hasMask.get), "mask was lost on failed unlock")
      require(fx(nodes(canvas.node).collectFirst { case v: ImageView => v }.exists(_.getImage == null)), "protected image remained visible")
      fx { service.imageScrambleSession.unlock("secret"); context.imageAccessChanged() }
      val recoverDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      def baseReady = nodes(canvas.node).collectFirst { case v: ImageView => v }.exists(v => v.isVisible && v.getImage != null)
      while !fx(baseReady) && System.nanoTime() < recoverDeadline do Thread.sleep(50)
      require(fx(baseReady), "image did not reload after unlocking")
      val afterPath = fx(canvas.exportMask().get)
      require(java.util.Arrays.equals(before, Files.readAllBytes(afterPath)), "mask pixels changed")
      fx(stage.close()); println("PASS native mask canvas: scramble access, failed unlock and retained mask")
    finally
      if stage != null then fx(stage.close())
      service.shutdown(); Platform.exit()
