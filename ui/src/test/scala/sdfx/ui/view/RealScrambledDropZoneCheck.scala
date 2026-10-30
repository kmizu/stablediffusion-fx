package sdfx.ui.view

import javafx.application.Platform
import javafx.beans.property.SimpleStringProperty
import javafx.embed.swing.SwingFXUtils
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

object RealScrambledDropZoneCheck:
  def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A] { def call(): A = body })
    Platform.runLater(task); task.get(30, TimeUnit.SECONDS)
  def main(args: Array[String]): Unit =
    val root = Path.of(args(0)).toAbsolutePath; Files.createDirectories(root)
    val image = new BufferedImage(32, 24, BufferedImage.TYPE_INT_RGB)
    for y <- 0 until 24; x <- 0 until 32 do image.setRGB(x, y, (x << 16) | (y << 8) | 7)
    val path = root.resolve("scrambled.png"); ScrambledPng.write(image, path, Nil, Some("secret"))
    val service = new AppServices(AppHome(Files.createTempDirectory(root, "home-")), Settings(), dryRun = true,
      builtins = Nil, persistSettings = false, installShutdownHook = false)
    val ready = new CountDownLatch(1); Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var stage: Stage = null; var zone: ImageDropZone = null; var context: UiContext = null
    def nodes(n: Node): Seq[Node] = Seq(n) ++ (n match
      case p: Parent => p.getChildrenUnmodifiable.asScala.toSeq.flatMap(nodes)
      case _ => Nil)
    def view: ImageView = nodes(zone.node).collectFirst { case v: ImageView => v }.get
    def failed = nodes(zone.node).collect { case l: Label if l.isVisible => l.getText }.exists(_.contains("Cannot display"))
    try
      fx {
        stage = new Stage(); val ctx = new UiContext(service, new CatalogVM, stage); context = ctx
        service.imageScrambleSession.unlock("secret")
        zone = new ImageDropZone(ctx, new SimpleStringProperty(path.toString), "Input")
        stage.setScene(new Scene(zone.node, 450, 300)); stage.show()
      }
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while !fx(view.getImage != null) && System.nanoTime() < deadline do Thread.sleep(50)
      val actual = fx(SwingFXUtils.fromFXImage(view.getImage, null))
      require(image.getRGB(0, 0, 32, 24, null, 0, 32).sameElements(actual.getRGB(0, 0, 32, 24, null, 0, 32)))
      fx { service.imageScrambleSession.unlock("wrong"); context.imageAccessChanged() }
      val badDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while !fx(failed) && System.nanoTime() < badDeadline do Thread.sleep(50)
      require(fx(view.getImage == null && failed))
      fx(stage.close()); println("PASS native image input: decoded pixels and wrong-password rejection")
    finally
      if stage != null then fx(stage.close())
      service.shutdown(); Platform.exit()
