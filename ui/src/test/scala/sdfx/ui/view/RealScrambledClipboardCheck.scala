package sdfx.ui.view

import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.scene.input.{Clipboard, ClipboardContent}
import javafx.stage.Stage
import sdfx.core.config.{AppHome, Settings}
import sdfx.core.png.ScrambledPng
import sdfx.engine.AppServices
import sdfx.ui.vm.CatalogVM
import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}

object RealScrambledClipboardCheck:
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
    var stage: Stage = null; var ctx: UiContext = null
    var original: Map[javafx.scene.input.DataFormat, Object] = Map.empty
    try
      fx {
        stage = new Stage(); ctx = new UiContext(service, new CatalogVM, stage)
        stage.setScene(new javafx.scene.Scene(new javafx.scene.layout.StackPane(ctx.toaster.layer), 300, 200)); stage.show()
        val clipboard = Clipboard.getSystemClipboard
        import scala.jdk.CollectionConverters.*
        original = clipboard.getContentTypes.asScala.flatMap(f => Option(clipboard.getContent(f)).map(v => f -> v.asInstanceOf[Object])).toMap
        val diagnostic = new ClipboardContent(); diagnostic.putImage(SwingFXUtils.toFXImage(image, null))
        println("direct clipboard write=" + clipboard.setContent(diagnostic) + " hasImage=" + clipboard.hasImage)
        service.imageScrambleSession.unlock("secret")
        val c = new ClipboardContent(); c.putString("before"); Clipboard.getSystemClipboard.setContent(c)
        ImageActions.copyImage(ctx, path)
      }
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while !fx(Clipboard.getSystemClipboard.hasImage) && System.nanoTime() < deadline do Thread.sleep(50)
      require(fx(Clipboard.getSystemClipboard.hasImage), "copy failed; toast=" + fx(ctx.toaster.layer.getChildren.toString))
      val actual = fx(SwingFXUtils.fromFXImage(Clipboard.getSystemClipboard.getImage, null))
      require(image.getRGB(0, 0, 32, 24, null, 0, 32).sameElements(actual.getRGB(0, 0, 32, 24, null, 0, 32)))
      require(!fx(Clipboard.getSystemClipboard.hasFiles), "scrambled source file was copied alongside decoded pixels")
      fx {
        val c = new ClipboardContent(); c.putString("sentinel"); Clipboard.getSystemClipboard.setContent(c)
        service.imageScrambleSession.unlock("wrong")
        ImageActions.copyImage(ctx, path)
      }
      def errorVisible: Boolean =
        import scala.jdk.CollectionConverters.*
        ctx.toaster.layer.getChildren.asScala.exists {
          case p: javafx.scene.Parent => p.getChildrenUnmodifiable.asScala.exists {
            case l: javafx.scene.control.Label => Option(l.getText).exists(_.startsWith("Cannot copy image"))
            case _ => false
          }
          case _ => false
        }
      val errorDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while !fx(errorVisible) && System.nanoTime() < errorDeadline do Thread.sleep(50)
      require(fx(errorVisible), "wrong password did not report an error")
      require(fx(Clipboard.getSystemClipboard.getString == "sentinel" && !Clipboard.getSystemClipboard.hasImage), "wrong password replaced clipboard")
      println("PASS native clipboard: decoded pixels, wrong-password preservation and no scrambled file")
    finally
      fx {
        val data = new ClipboardContent()
        original.foreach((format, value) => data.put(format, value))
        Clipboard.getSystemClipboard.setContent(data)
        if stage != null then stage.close()
      }
      service.shutdown(); Platform.exit()
