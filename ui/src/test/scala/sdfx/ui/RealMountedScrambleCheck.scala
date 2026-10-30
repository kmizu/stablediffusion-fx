package sdfx.ui

import javafx.application.Platform
import javafx.stage.Stage
import javafx.scene.{Node, Parent}
import javafx.scene.control.{Button, PasswordField, SplitPane, TabPane, TextField, Label}
import scala.jdk.CollectionConverters.*
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.png.{PngMetadata, ScrambledPng}
import sdfx.core.plugins.api.OutputImage
import sdfx.engine.{AppServices, JobEvent}
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{Mode, UiState}
import java.nio.file.{Files, Path}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, LinkedBlockingQueue, TimeUnit}

/** Actual MainWindow Generate button and Python dry-run worker, not a diffusion-model test. */
object RealMountedScrambleCheck:
  def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A] { def call(): A = body })
    Platform.runLater(task); task.get(30, TimeUnit.SECONDS)
  def main(args: Array[String]): Unit =
    require(args.length == 2)
    val root = Path.of(args(0)).toAbsolutePath; Files.createDirectories(root)
    val home = AppHome(Files.createTempDirectory(root, "home-")); Files.createDirectories(home.pythonDir)
    val models = Files.createDirectories(home.root.resolve("test-models"))
    val checkpoint = models.resolve("fixture.safetensors")
    Files.write(checkpoint, Array[Byte](2, 0, 0, 0, 0, 0, 0, 0, 123, 125))
    val service = new AppServices(home, Settings(pythonExe = Some(Path.of(args(1)).toAbsolutePath.toString),
      modelDirs = ModelDirs(checkpoints = Seq(models.toString))), dryRun = true,
      builtins = StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false)
    val events = new LinkedBlockingQueue[Either[String, OutputImage]]()
    service.queue.addListener {
      case e: JobEvent.JobFinished => events.offer(Right(e.output.images.head)); ()
      case e: JobEvent.JobFailed => events.offer(Left(e.message)); ()
      case _ => ()
    }
    val ready = new CountDownLatch(1); Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    var serviceStopped = false
    def nodes(node: Node): Seq[Node] = Seq(node) ++ (node match
      case p: Parent => p.getChildrenUnmodifiable.asScala.toSeq.flatMap(nodes)
      case _ => Nil)
    def find(id: String): Node = nodes(window.root).find(_.getId == id).getOrElse(sys.error(s"missing $id"))
    def selectTab(pane: TabPane, id: String): Unit =
      pane.getSelectionModel.select(pane.getTabs.asScala.find(_.getId == id).getOrElse(sys.error(s"missing tab $id")))
    def next(): OutputImage =
      val result = events.poll(90, TimeUnit.SECONDS)
      require(result != null, "generation timed out")
      result.fold(e => throw new IllegalStateException(e), identity)
    def idle(): Unit =
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while (!service.queue.isIdle || fx(window.progress.get.running)) && System.nanoTime() < deadline do Thread.sleep(50)
      require(service.queue.isIdle && fx(!window.progress.get.running))
    def screenshot(page: Int, width: Double, divider: Double, narrow: Boolean, name: String): Unit =
      fx {
        selectTab(find("main-tabs").asInstanceOf[TabPane], "image-scrambling-tab")
        find("scramble-pages").asInstanceOf[TabPane].getSelectionModel.select(page)
        window.stage.setWidth(width)
        window.stage.setHeight(900)
      }
      Thread.sleep(250) // Allow the window manager and JavaFX layout to apply the resize.
      fx { window.root.lookup(".main-split").asInstanceOf[SplitPane].setDividerPositions(divider) }
      Thread.sleep(250)
      fx {
        window.root.applyCss(); window.root.layout()
        require(find("main-tabs").asInstanceOf[TabPane].getSelectionModel.getSelectedItem.getId == "image-scrambling-tab")
        require(find(if page == 0 then "scramble-session-status" else "scramble-status").isVisible)
        val panelWidth = find("main-tabs").getBoundsInParent.getWidth
        if narrow then require(panelWidth >= 419 && panelWidth <= 450, s"Expected about 420px, got $panelWidth")
        else require(panelWidth >= 700, s"Expected wide panel, got $panelWidth")
        val image = window.scene.snapshot(null)
        javax.imageio.ImageIO.write(javafx.embed.swing.SwingFXUtils.fromFXImage(image, null), "png", root.resolve(name).toFile)
        println(s"Screenshot $name: main panel width $panelWidth")
      }
    try
      service.start(); service.loadCheckpoint(checkpoint.toString, remember = false)
      fx {
        window = new MainWindow(new Stage(), service, UiState(), CliOptions(), _ => ())
        window.show(); window.select(Mode.Txt2Img); window.model.checkpoint.set(checkpoint.toString)
        val vm = window.vms(Mode.Txt2Img)
        vm.prompt.set("test portrait"); vm.seed.set(173); vm.steps.set(5); vm.width.set(64); vm.height.set(64)
        window.statusBar.generateBtn.fire()
      }
      val plain = next(); idle()
      fx {
        selectTab(find("main-tabs").asInstanceOf[TabPane], "image-scrambling-tab")
        find("scramble-session-password").asInstanceOf[PasswordField].setText("secret")
        find("scramble-session-enable").asInstanceOf[Button].fire()
        require(service.imageScrambleSession.snapshot.nonEmpty)
        require(find("scramble-session-password").asInstanceOf[PasswordField].getText.isEmpty)
        window.statusBar.generateBtn.fire()
        find("scramble-session-lock").asInstanceOf[Button].fire()
        require(service.imageScrambleSession.snapshot.isEmpty)
      }
      val protectedImage = next(); idle()
      require(PngMetadata.readTexts(protectedImage.path).get("Encrypt").contains("pixel_shuffle_3"), "queued secret not captured")
      val expected = PngMetadata.readImage(plain.path); val actual = ScrambledPng.read(protectedImage.path, Some("secret")).image
      require(expected.getRGB(0, 0, 64, 64, null, 0, 64).sameElements(actual.getRGB(0, 0, 64, 64, null, 0, 64)), "generated pixels differ")
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while !fx(window.preview.imageStatus.startsWith("Cannot")) && System.nanoTime() < deadline do Thread.sleep(50)
      require(fx(window.preview.zoom.image == null && window.preview.imageStatus.startsWith("Cannot")), "locked image remained visible")
      screenshot(0, 1500, 0.60, false, "scramble-session-wide.png")
      screenshot(0, 1280, 0.0, true, "scramble-session-narrow.png")
      screenshot(1, 1500, 0.60, false, "scramble-batch-wide.png")
      screenshot(1, 1280, 0.0, true, "scramble-batch-narrow.png")
      val batchInput = Files.createTempDirectory(root, "batch-input-")
      val batchOutput = Files.createTempDirectory(root, "batch-output-")
      Files.copy(plain.path, batchInput.resolve("plain.png"))
      fx {
        find("scramble-pages").asInstanceOf[TabPane].getSelectionModel.select(1)
        find("scramble-source").asInstanceOf[TextField].setText(batchInput.toString)
        find("scramble-destination").asInstanceOf[TextField].setText(batchOutput.toString)
        find("scramble-password").asInstanceOf[PasswordField].setText("batch-secret")
        find("scramble-start").asInstanceOf[Button].fire()
      }
      val batchDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while !fx(find("scramble-status").asInstanceOf[Label].getText.startsWith("Completed")) && System.nanoTime() < batchDeadline do Thread.sleep(50)
      require(fx(find("scramble-status").asInstanceOf[Label].getText.startsWith("Completed: 1 written")))
      require(ScrambledPng.read(batchOutput.resolve("plain.png"), Some("batch-secret")).scrambled)
      fx {
        val batchPassword = find("scramble-password").asInstanceOf[PasswordField]
        find("scramble-pages").asInstanceOf[TabPane].getSelectionModel.select(0)
        find("scramble-session-password").asInstanceOf[PasswordField].setText("dispose-secret")
        find("scramble-session-enable").asInstanceOf[Button].fire()
        window.dispose()
        require(service.imageScrambleSession.snapshot.nonEmpty, "window disposal must not lock a shared service session")
        require(batchPassword.getText.isEmpty)
      }
      service.shutdown()
      serviceStopped = true
      require(service.imageScrambleSession.snapshot.isEmpty, "service shutdown must lock the session")
      println("PASS mounted MainWindow controls: queued snapshot, preview refresh, batch conversion, disposal and service shutdown")
    finally
      if window != null then fx { window.stage.close() }
      if !serviceStopped then service.shutdown()
      Platform.exit()
