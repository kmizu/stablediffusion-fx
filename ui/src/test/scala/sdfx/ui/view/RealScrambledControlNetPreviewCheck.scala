package sdfx.ui.view

import javafx.application.Platform
import javafx.scene.{Node, Parent, Scene}
import javafx.scene.control.Button
import javafx.scene.image.ImageView
import javafx.stage.Stage
import sdfx.core.config.{AppHome, Settings}
import sdfx.core.plugins.api.ChoiceSource
import sdfx.core.png.{PngMetadata, ScrambledPng}
import sdfx.engine.AppServices
import sdfx.ui.vm.{CatalogVM, GenerationVM, Mode}
import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import scala.jdk.CollectionConverters.*

object RealScrambledControlNetPreviewCheck:
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
    val service = new AppServices(home, Settings(pythonExe = Some(Path.of(args(1)).toAbsolutePath.toString)), dryRun = true,
      builtins = Nil, persistSettings = false, installShutdownHook = false)
    val ready = new CountDownLatch(1); Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS)); service.start()
    var stage: Stage = null; var rootNode: Node = null; var vm: GenerationVM = null; var context: UiContext = null
    def nodes(n: Node): Seq[Node] = Seq(n) ++ (n match
      case p: Parent => p.getChildrenUnmodifiable.asScala.toSeq.flatMap(nodes)
      case _ => Nil)
    def button = nodes(rootNode).collectFirst { case b: Button if b.getText == "◉ Preview preprocessor" => b }.get
    def preview = nodes(rootNode).collectFirst { case v: ImageView if v.getId == "controlnet-preprocessor-image" => v }
    def awaitResult(): Unit =
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while !fx(preview.exists(v => v.getImage != null && v.getImage.getProgress >= 1)) && System.nanoTime() < deadline do Thread.sleep(50)
      require(fx(preview.exists(v => v.getImage != null && !v.getImage.isError && v.getImage.getProgress >= 1)),
        "preprocessor preview missing: " + fx(nodes(context.toaster.layer).collect { case l: javafx.scene.control.Label => l.getText }.mkString(" | ")))
    try
      fx {
        stage = new Stage(); val catalog = new CatalogVM; catalog.choices(ChoiceSource.Preprocessors).add("invert")
        val ctx = new UiContext(service, catalog, stage); context = ctx; vm = new GenerationVM(Mode.Img2Img)
        rootNode = ExtraSections.controlNet(ctx, vm, expanded = true)
        stage.setScene(new Scene(rootNode.asInstanceOf[javafx.scene.Parent], 700, 900)); stage.show()
        service.imageScrambleSession.unlock("secret")
        vm.controlNet.foreach { unit => unit.preprocessor.set("invert"); unit.image.set(encoded.toString) }
        button.fire()
      }
      awaitResult()
      val decoded = fx(javafx.embed.swing.SwingFXUtils.fromFXImage(preview.get.getImage, null))
      require((decoded.getRGB(9, 8) & 0xffffff) == ((~image.getRGB(9, 8)) & 0xffffff), "protected input not decoded before preprocessing")
      fx { button.fire(); vm.controlNet.foreach(_.image.set(plain.toString)) }
      val end = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while fx(button.isDisabled) && System.nanoTime() < end do Thread.sleep(50)
      require(fx(!button.isDisabled), "preview did not finish")
      require(fx(preview.forall(_.getImage == null)), "stale preview shown after input changed")
      fx(stage.close()); println("PASS native ControlNet: protected input, preview pixels and stale-result suppression")
    finally
      if stage != null then fx(stage.close())
      service.shutdown(); Platform.exit()
