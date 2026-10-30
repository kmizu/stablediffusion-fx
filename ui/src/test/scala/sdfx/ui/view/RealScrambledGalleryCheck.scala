package sdfx.ui.view

import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.scene.{Node, Parent, Scene}
import javafx.scene.image.ImageView
import javafx.stage.Stage
import sdfx.core.config.{AppHome, Settings}
import sdfx.core.gallery.GalleryItem
import sdfx.core.png.ScrambledPng
import sdfx.engine.AppServices
import sdfx.ui.vm.{CatalogVM, GalleryVM, Mode}
import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import scala.jdk.CollectionConverters.*

object RealScrambledGalleryCheck:
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
    var stage: Stage = null; var gallery: GalleryView = null; var context: UiContext = null
    def nodes(n: Node): Seq[Node] = Seq(n) ++ (n match
      case p: Parent => p.getChildrenUnmodifiable.asScala.toSeq.flatMap(nodes)
      case _ => Nil)
    def thumb: Option[ImageView] = nodes(gallery.node).collectFirst { case v: ImageView if v.getUserData == path => v }
    try
      fx {
        stage = new Stage(); val ctx = new UiContext(service, new CatalogVM, stage); context = ctx; val vm = new GalleryVM
        val actions = new ImageActions:
          def sendTo(mode: Mode, p: Path, info: Option[String]): Unit = ()
          def restore(text: String): Unit = ()
          def delete(p: Path): Unit = ()
        gallery = new GalleryView(ctx, vm, actions, _ => ())
        vm.setAll(Seq(GalleryItem(path, 0, Files.size(path), None)))
        service.imageScrambleSession.unlock("secret")
        stage.setScene(new Scene(gallery.node, 700, 260)); stage.show()
      }
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while !fx(thumb.exists(_.getImage != null)) && System.nanoTime() < deadline do Thread.sleep(50)
      val actual = fx(SwingFXUtils.fromFXImage(thumb.get.getImage, null))
      require(image.getRGB(0, 0, 32, 24, null, 0, 32).sameElements(actual.getRGB(0, 0, 32, 24, null, 0, 32)))
      fx { service.imageScrambleSession.unlock("wrong"); context.imageAccessChanged() }
      val failureDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while !fx(thumb.exists(v => Option(v.getAccessibleText).exists(_.startsWith("Cannot")))) && System.nanoTime() < failureDeadline do Thread.sleep(50)
      require(fx(thumb.exists(v => v.getImage == null && Option(v.getAccessibleText).exists(_.startsWith("Cannot")))))
      fx(stage.close()); println("PASS native gallery: decrypted thumbnail and wrong-password rejection")
    finally
      if stage != null then fx(stage.close())
      service.shutdown(); Platform.exit()
