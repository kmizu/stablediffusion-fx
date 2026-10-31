package sdfx.ui

import java.awt.{Color, Graphics2D}
import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, LinkedBlockingQueue, TimeUnit}
import javafx.application.Platform
import javafx.scene.{Node, Parent}
import javafx.scene.control.{Button, CheckBox, ComboBox, PasswordField, TabPane, TitledPane}
import javafx.stage.Stage
import javax.imageio.ImageIO
import scala.concurrent.Await
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.infotext.Infotext
import sdfx.core.png.{PngMetadata, ScrambledPng}
import sdfx.core.plugins.api.OutputImage
import sdfx.engine.{AppServices, JobEvent}
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{Mode, UiState}

/** Real MainWindow img2img/inpaint, protected session and owned cancellation. */
object RealModelKeywordModesCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A] { def call(): A = body })
    Platform.runLater(task); task.get(30, TimeUnit.SECONDS)
  def main(args: Array[String]): Unit =
    require(args.length == 4, "empty-home python checkpoint size")
    val home = AppHome(Path.of(args(0)).toAbsolutePath)
    if Files.exists(home.root) then
      val entries = Files.list(home.root)
      try require(!entries.findAny().isPresent, "acceptance home must be empty") finally entries.close()
    Files.createDirectories(home.pythonDir)
    val model = Path.of(args(2)).toAbsolutePath; val size = args(3).toInt
    val services = new AppServices(home, Settings(pythonExe = Some(Path.of(args(1)).toAbsolutePath.toString),
      modelDirs = ModelDirs(checkpoints = Seq(model.getParent.toString))), dryRun = false,
      builtins = StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false)
    val finished = new LinkedBlockingQueue[Either[String, Seq[OutputImage]]]()
    val cancelled = new LinkedBlockingQueue[Seq[OutputImage]]()
    val progress = new java.util.concurrent.atomic.AtomicReference[CountDownLatch]()
    services.queue.addListener {
      case e: JobEvent.JobFinished => finished.offer(Right(e.output.images)); ()
      case e: JobEvent.JobFailed => finished.offer(Left(e.message)); ()
      case e: JobEvent.JobCancelled => cancelled.offer(e.partial.images); ()
      case e: JobEvent.Progress if e.progress.pass == "base" && e.progress.step >= 2 =>
        Option(progress.get()).foreach(_.countDown())
      case _ => ()
    }
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    def nodes(node: Node): Seq[Node] = Seq(node) ++ (node match
      case p: Parent => p.getChildrenUnmodifiable.asScala.toSeq.flatMap(nodes)
      case _ => Nil)
    def find(root: Node, id: String): Node = nodes(root).find(_.getId == id).getOrElse(sys.error(s"Missing $id"))
    def waitFor(label: String, seconds: Int = 45)(f: => Boolean): Unit =
      val end = System.nanoTime() + TimeUnit.SECONDS.toNanos(seconds)
      while !f && System.nanoTime() < end do Thread.sleep(50)
      require(f, s"Timed out: $label")
    def idle(): Unit = waitFor("queue idle")(services.queue.isIdle && fx(!window.progress.get.running))
    def generate(): Seq[OutputImage] =
      idle(); fx(window.statusBar.generateBtn.fire())
      val result = finished.poll(300, TimeUnit.SECONDS)
      require(result != null, "generation timed out")
      result.fold(e => throw new IllegalStateException(e), identity)
    def pixels(image: java.awt.image.BufferedImage): Array[Int] =
      image.getRGB(0, 0, image.getWidth, image.getHeight, null, 0, image.getWidth)
    def enable(mode: Mode): Unit = fx {
      window.select(mode)
      val panel = window.panels(mode).node
      find(panel, "plugin-model_keyword-section").asInstanceOf[TitledPane].setExpanded(true)
      find(panel, "plugin-model_keyword-enabled").asInstanceOf[CheckBox].setSelected(true)
      find(panel, "plugin-model_keyword-model_choice").asInstanceOf[ComboBox[String]].setValue("red")
      require(window.plugins(mode).form("model_keyword").get.isActive)
      val vm = window.vms(mode)
      vm.prompt.set("a ceramic teapot"); vm.negativePrompt.set("low quality")
      vm.seed.set(173); vm.steps.set(6); vm.width.set(size); vm.height.set(size)
      vm.batchSize.set(1); vm.batchCount.set(1)
    }
    def keyword(image: OutputImage): Unit =
      val info = Infotext.parse(PngMetadata.readParameters(image.path).get)
      require(info.prompt == "red a ceramic teapot", s"unexpected keyword prompt: ${info.prompt}")
    try
      services.start(); val loaded = services.loadCheckpoint(model.toString, remember = false)
      Await.result(services.client.raw("model_keyword", ujson.Obj("action" -> "save", "kind" -> "checkpoint",
        "model_path" -> model.toString, "mapping_dir" -> home.configDir.resolve("model-keyword").toString,
        "keywords" -> "red|blue")), 60.seconds)
      fx {
        window = new MainWindow(new Stage(), services, UiState(), CliOptions(), _ => ())
        window.show(); window.model.checkpoint.set(loaded.name)
      }
      enable(Mode.Txt2Img)
      val plain = generate().head; keyword(plain)
      fx {
        val tabs = find(window.root, "main-tabs").asInstanceOf[TabPane]
        tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getId == "image-scrambling-tab").get)
        find(window.root, "scramble-session-password").asInstanceOf[PasswordField].setText("acceptance-secret")
        find(window.root, "scramble-session-enable").asInstanceOf[Button].fire()
        require(services.imageScrambleSession.snapshot.nonEmpty)
        window.select(Mode.Txt2Img); window.statusBar.generateBtn.fire()
        tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getId == "image-scrambling-tab").get)
        find(window.root, "scramble-session-lock").asInstanceOf[Button].fire()
      }
      val securedResult = finished.poll(300, TimeUnit.SECONDS)
      require(securedResult != null, "protected generation timed out")
      val secured = securedResult.fold(e => throw new IllegalStateException(e), _.head)
      require(PngMetadata.readTexts(secured.path).get("Encrypt").contains("pixel_shuffle_3"))
      require(scala.util.Try(ScrambledPng.read(secured.path, Some("wrong-secret"))).isFailure,
        "incorrect password accepted")
      require(pixels(PngMetadata.readImage(plain.path)).sameElements(pixels(ScrambledPng.read(secured.path, Some("acceptance-secret")).image)),
        "protected keyword result differs")
      fx {
        val tabs = find(window.root, "main-tabs").asInstanceOf[TabPane]
        tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getId == "image-scrambling-tab").get)
        find(window.root, "scramble-session-password").asInstanceOf[PasswordField].setText("acceptance-secret")
        find(window.root, "scramble-session-enable").asInstanceOf[Button].fire()
        require(services.imageScrambleSession.snapshot.nonEmpty)
        find(window.root, "scramble-session-lock").asInstanceOf[Button].fire()
        require(services.imageScrambleSession.snapshot.isEmpty)
      }
      idle(); enable(Mode.Img2Img)
      fx { val vm = window.vms(Mode.Img2Img); vm.initImage.set(plain.path.toString); vm.denoise.set(.55) }
      val img = generate().head; keyword(img)
      enable(Mode.Inpaint)
      fx { val vm = window.vms(Mode.Inpaint); vm.initImage.set(plain.path.toString); vm.denoise.set(.55) }
      val mask = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB)
      val g = mask.createGraphics()
      try { g.setColor(Color.BLACK); g.fillRect(0, 0, size, size); g.setColor(Color.WHITE); g.fillRect(size/4, size/4, size/2, size/2) }
      finally g.dispose()
      val maskPath = home.root.resolve("test-mask.png"); ImageIO.write(mask, "png", maskPath.toFile)
      waitFor("mask loaded") {
        if !fx(window.vms(Mode.Inpaint).hasMask.get) then fx(window.panels(Mode.Inpaint).mask.get.loadMask(maskPath.toString))
        fx(window.vms(Mode.Inpaint).hasMask.get)
      }
      val painted = generate().head; keyword(painted)
      idle()
      enable(Mode.Txt2Img)
      val latch = new CountDownLatch(1); progress.set(latch)
      fx { window.vms(Mode.Txt2Img).steps.set(40); window.statusBar.generateBtn.fire() }
      require(latch.await(180, TimeUnit.SECONDS), "sampling did not begin")
      fx(window.interrupt())
      val partial = cancelled.poll(120, TimeUnit.SECONDS)
      require(partial != null && partial.isEmpty, "cancelled job should not have finished images")
      progress.set(null); idle()
      fx(window.vms(Mode.Txt2Img).steps.set(6))
      val recovery = generate().head; keyword(recovery)
      require(pixels(PngMetadata.readImage(plain.path)).sameElements(pixels(PngMetadata.readImage(recovery.path))), "recovery differs")
      Files.writeString(home.root.resolve("keyword-modes-report.json"), ujson.Obj(
        "plain" -> plain.path.toString, "protected" -> secured.path.toString, "img2img" -> img.path.toString,
        "inpaint" -> painted.path.toString, "recovery" -> recovery.path.toString,
        "protected_rgb_equal" -> true, "wrong_password_rejected" -> true,
        "session_reenabled_and_locked" -> true, "recovery_rgb_equal" -> true, "cancelled" -> true).render(indent = 2))
      println("PASS mounted keyword img2img, inpaint, protected session and cancellation recovery")
    finally
      if window != null then fx { window.dispose(); window.stage.close() }
      services.shutdown(); Platform.exit()
