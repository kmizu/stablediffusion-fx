package sdfx.ui.view

import javafx.application.Platform
import javafx.scene.{Node, Parent, Scene}
import javafx.scene.control.Button
import javafx.stage.Stage
import sdfx.core.config.{AppHome, Settings}
import sdfx.core.png.{PngMetadata, ScrambledPng}
import sdfx.engine.AppServices
import sdfx.ui.vm.CatalogVM
import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import javax.imageio.ImageIO
import scala.jdk.CollectionConverters.*

object RealScrambledExtrasCheck:
  def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A] { def call(): A = body })
    Platform.runLater(task); task.get(30, TimeUnit.SECONDS)
  def main(args: Array[String]): Unit =
    require(args.length == 2)
    val root = Path.of(args(0)).toAbsolutePath; Files.createDirectories(root)
    val image = new BufferedImage(32, 24, BufferedImage.TYPE_INT_RGB)
    for y <- 0 until 24; x <- 0 until 32 do image.setRGB(x, y, (x << 16) | (y << 8) | 7)
    val plain = root.resolve("plain.png"); val encoded = root.resolve("scrambled.png")
    PngMetadata.write(image, plain, Nil); ScrambledPng.write(image, encoded, Nil, Some("secret"))
    val home = AppHome(Files.createTempDirectory(root, "home-")); Files.createDirectories(home.pythonDir)
    val service = new AppServices(home, Settings(pythonExe = Some(Path.of(args(1)).toAbsolutePath.toString)),
      dryRun = true, builtins = Nil, persistSettings = false, installShutdownHook = false)
    val ready = new CountDownLatch(1); Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS)); service.start()
    var stage: Stage = null; var tab: ExtrasTab = null
    def nodes(n: Node): Seq[Node] = Seq(n) ++ (n match
      case p: Parent => p.getChildrenUnmodifiable.asScala.toSeq.flatMap(nodes)
      case _ => Nil)
    def press(): Unit = nodes(tab.node).collectFirst { case b: Button if b.getText == "⤢ Upscale" => b }.get.fire()
    def outputFor(source: Path): Path =
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      var out: Option[Path] = None
      while out.isEmpty && System.nanoTime() < deadline do
        Thread.sleep(50); out = fx(tab.comparisonPair.collect { case (`source`, result) => result })
      out.getOrElse(throw new IllegalStateException("upscale did not produce result for " + source.getFileName))
    try
      fx {
        stage = new Stage(); val catalog = new CatalogVM; catalog.upscalers.add("Lanczos")
        val ctx = new UiContext(service, catalog, stage)
        tab = new ExtrasTab(ctx, "Lanczos", 2.0)
        stage.setScene(new Scene(tab.node, 800, 700)); stage.show()
        service.imageScrambleSession.unlock("secret")
        tab.image.set(plain.toString); press()
      }
      val ordinaryPath = outputFor(plain)
      require(PngMetadata.readTexts(ordinaryPath).get("Encrypt").contains("pixel_shuffle_3"), "upscale output was not scrambled")
      val ordinary = ScrambledPng.read(ordinaryPath, Some("secret")).image
      fx { tab.image.set(encoded.toString); press() }
      val protectedPath = outputFor(encoded)
      require(PngMetadata.readTexts(protectedPath).get("Encrypt").contains("pixel_shuffle_3"), "upscale output was not scrambled")
      val protectedResult = ScrambledPng.read(protectedPath, Some("secret")).image
      require(ordinary.getWidth == protectedResult.getWidth && ordinary.getHeight == protectedResult.getHeight)
      require(ordinary.getRGB(0, 0, ordinary.getWidth, ordinary.getHeight, null, 0, ordinary.getWidth).sameElements(
        protectedResult.getRGB(0, 0, protectedResult.getWidth, protectedResult.getHeight, null, 0, protectedResult.getWidth)))
      def correct(): Unit = nodes(tab.node).collectFirst { case b: Button if b.getText == "Apply color correction" => b }.get.fire()
      fx {
        service.imageScrambleSession.lock()
        tab.image.set(plain.toString); tab.correctionReference.set(plain.toString)
        nodes(tab.node).collectFirst { case c: javafx.scene.control.ComboBox[?] if c.getItems.contains("AdaIN") => c.asInstanceOf[javafx.scene.control.ComboBox[String]] }.get.setValue("AdaIN")
        correct()
      }
      val normalPath = outputFor(plain)
      val normal = PngMetadata.readImage(normalPath)
      fx {
        service.imageScrambleSession.unlock("secret")
        tab.image.set(encoded.toString); tab.correctionReference.set(encoded.toString); correct()
      }
      val protectedCorrection = outputFor(encoded)
      require(PngMetadata.readTexts(protectedCorrection).get("Encrypt").contains("pixel_shuffle_3"), "correction output was not scrambled")
      val corrected = ScrambledPng.read(protectedCorrection, Some("secret")).image
      require(normal.getRGB(0, 0, normal.getWidth, normal.getHeight, null, 0, normal.getWidth).sameElements(
        corrected.getRGB(0, 0, corrected.getWidth, corrected.getHeight, null, 0, corrected.getWidth)), "color correction mismatch")
      fx(stage.close()); println("PASS native Extras: protected upscale and color correction match ordinary inputs")
    finally
      if stage != null then fx(stage.close())
      service.shutdown(); Platform.exit()
