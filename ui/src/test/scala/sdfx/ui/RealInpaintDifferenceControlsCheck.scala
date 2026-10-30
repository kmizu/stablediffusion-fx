package sdfx.ui.view

import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.scene.{Node, Parent, Scene}
import javafx.scene.control.{Button, CheckBox, Label}
import javafx.stage.Stage
import sdfx.core.config.{AppHome, Settings}
import sdfx.engine.AppServices
import sdfx.plugins.InpaintDifferencePlugin
import sdfx.ui.vm.{CatalogVM, GenerationVM, InpaintDifferenceInputs, Mode, PluginValuesVM}

import java.awt.image.BufferedImage
import java.nio.file.{Files, Paths}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import javax.imageio.ImageIO
import scala.jdk.CollectionConverters.*

/** Actual JavaFX controls and real dry-run JSONL preview worker; no diffusion model. */
object RealInpaintDifferenceControlsCheck:
  private def fx[A](f: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = f)
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)

  def main(args: Array[String]): Unit =
    require(args.length == 2, "output directory and Python executable")
    val output = Paths.get(args(0)).toAbsolutePath
    Files.createDirectories(output)
    def fixture(name: String, color: Int) =
      val image = new BufferedImage(96, 80, BufferedImage.TYPE_INT_RGB)
      val g = image.createGraphics()
      try
        g.setColor(new java.awt.Color(color)); g.fillRect(0, 0, 96, 80)
        if name == "altered" then
          g.setColor(java.awt.Color.WHITE); g.fillRect(20, 15, 30, 35)
      finally g.dispose()
      val path = output.resolve(name + ".png")
      ImageIO.write(image, "png", path.toFile)
      path
    val base = fixture("base", 0x203040)
    val altered = fixture("altered", 0x203040)
    val home = AppHome(Files.createTempDirectory(output, "home-"))
    Files.createDirectories(home.pythonDir)
    val services = new AppServices(home,
      Settings(pythonExe = Some(Paths.get(args(1)).toAbsolutePath.toString)), dryRun = true,
      builtins = Nil, persistSettings = false, installShutdownHook = false)
    services.start()
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var stage: Stage = null
    var controls: InpaintDifferenceControls = null
    var form: PluginValuesVM = null
    var inputs: InpaintDifferenceInputs = null
    var vm: GenerationVM = null
    var painted: MaskCanvas = null
    def nodes(n: Node): Seq[Node] = Seq(n) ++ (n match
      case p: Parent => p.getChildrenUnmodifiable.asScala.toSeq.flatMap(nodes)
      case _ => Nil)
    try
      fx {
        stage = new Stage()
        val ctx = new UiContext(services, new CatalogVM, stage)
        vm = new GenerationVM(Mode.Inpaint)
        vm.initImage.set(base.toString)
        form = new PluginValuesVM(new InpaintDifferencePlugin)
        inputs = new InpaintDifferenceInputs(vm.initImage, form.props("enabled"))
        painted = new MaskCanvas(ctx, inputs.painted, vm.hasMask)
        controls = new InpaintDifferenceControls(ctx, vm, form, inputs)
        stage.setScene(new Scene(new javafx.scene.control.ScrollPane(new javafx.scene.layout.VBox(controls.node, painted.node)), 850, 900))
        stage.show()
      }
      val maskDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while !fx(vm.hasMask.get) && System.nanoTime() < maskDeadline do
        fx(painted.loadMask(altered.toString)); Thread.sleep(100)
      require(fx(vm.hasMask.get), "painted mask did not load")
      fx {
        nodes(controls.node).collectFirst { case c: CheckBox if c.getId == "difference-enabled" => c }.get.fire()
        form.props("base_path").set(base.toString)
        inputs.altered.set(altered.toString)
      }
      def awaitPreview(): Unit =
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(90)
        var text = ""
        while !text.startsWith("Difference mask") && System.nanoTime() < deadline do
          Thread.sleep(100)
          text = fx(nodes(controls.node).find(_.getId == "difference-preview-status").get.asInstanceOf[Label].getText)
          require(!text.startsWith("Preview unavailable"), text)
        require(text.startsWith("Difference mask"), s"Preview did not finish: $text")
      awaitPreview()
      fx {
        require(vm.initImage.get == altered.toString)
        form.props("threshold").set("0.5")
        form.props("dilation").set("2")
        form.props("erosion").set("1")
      }
      awaitPreview()
      fx {
        nodes(controls.node).collectFirst { case b: Button if b.getId == "difference-swap" => b }.get.fire()
        require(form.props("base_path").get == altered.toString && vm.initImage.get == base.toString)
      }
      awaitPreview()
      fx {
        form.props("show_under").set("false")
        vm.maskBlur.set(0)
        vm.maskInvert.set(true)
      }
      awaitPreview()
      fx {
        val preview = nodes(controls.node).find(_.getId == "difference-preview").get.asInstanceOf[javafx.scene.image.ImageView].getImage
        val pixels = preview.getPixelReader
        require((pixels.getArgb(0, 0) & 0xffffff) == 0xffffff, "inverted exterior must be white")
        require((pixels.getArgb(30, 30) & 0xffffff) == 0x000000, "inverted changed area must be black")
        vm.maskInvert.set(false)
        form.props("show_under").set("true")
      }
      awaitPreview()
      fx {
        controls.node.asInstanceOf[Parent].applyCss()
        ImageIO.write(SwingFXUtils.fromFXImage(controls.node.snapshot(null, null), null), "png", output.resolve("controls.png").toFile)
        nodes(controls.node).collectFirst { case c: CheckBox if c.getId == "difference-enabled" => c }.get.fire()
        require(vm.initImage.get == base.toString && vm.hasMask.get, "painted image or mask was not preserved")
        controls.dispose()
      }
      println("PASS native difference controls, preview, changes, swap and paint input preservation")
    finally
      fx { if controls != null then controls.dispose(); if stage != null then stage.close() }
      services.shutdown()
      Platform.exit()
