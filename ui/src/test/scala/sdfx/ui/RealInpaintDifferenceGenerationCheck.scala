package sdfx.ui

import javafx.application.Platform
import javafx.stage.Stage
import javafx.embed.swing.SwingFXUtils
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.plugins.api.OutputImage
import sdfx.engine.{AppServices, JobEvent}
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{Mode, UiState}

import java.awt.{Color, Graphics2D}
import java.awt.image.BufferedImage
import java.nio.file.{Files, Path, Paths}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, LinkedBlockingQueue, TimeUnit}
import javax.imageio.ImageIO

/** Actual MainWindow Generate with difference mode compared to its explicit painted-mask equivalent. */
object RealInpaintDifferenceGenerationCheck:
  private def fx[A](f: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = f)
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)
  private def pixels(p: Path): Array[Int] =
    val image = ImageIO.read(p.toFile)
    image.getRGB(0, 0, image.getWidth, image.getHeight, null, 0, image.getWidth)
  def main(args: Array[String]): Unit =
    require(args.length == 3, "home python checkpoint")
    val home = AppHome(Paths.get(args(0)).toAbsolutePath)
    val model = Paths.get(args(2)).toAbsolutePath
    Files.createDirectories(home.pythonDir)
    def image(name: String, f: Graphics2D => Unit): Path =
      val data = new BufferedImage(512, 512, BufferedImage.TYPE_INT_RGB)
      val g = data.createGraphics()
      try f(g) finally g.dispose()
      val path = home.root.resolve(name + ".png")
      ImageIO.write(data, "png", path.toFile)
      path
    def scene(g: Graphics2D): Unit =
      g.setColor(new Color(0x96b6ce)); g.fillRect(0, 0, 512, 512)
      g.setColor(new Color(0x9a7958)); g.fillRect(0, 310, 512, 202)
      g.setColor(new Color(0xe1c19e)); g.fillOval(145, 90, 220, 260)
    val base = image("base", scene)
    val altered = image("altered", g => { scene(g); g.setColor(new Color(0xe65050)); g.fillRect(205, 170, 101, 121) })
    val mask = image("reference-mask", g => {
      g.setColor(Color.BLACK); g.fillRect(0, 0, 512, 512)
      g.setColor(Color.WHITE); g.fillRect(204, 169, 103, 123)
    })
    val services = new AppServices(home, Settings(pythonExe = Some(Paths.get(args(1)).toAbsolutePath.toString),
      modelDirs = ModelDirs(checkpoints = Seq(model.getParent.toString))), dryRun = false,
      builtins = StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false)
    val finished = new LinkedBlockingQueue[Either[String, Seq[OutputImage]]]()
    val cancelled = new LinkedBlockingQueue[Seq[OutputImage]]()
    val reached = new java.util.concurrent.atomic.AtomicReference[CountDownLatch]()
    services.queue.addListener {
      case e: JobEvent.JobFinished => finished.offer(Right(e.output.images)); ()
      case e: JobEvent.JobFailed => finished.offer(Left(e.message)); ()
      case e: JobEvent.JobCancelled => cancelled.offer(e.partial.images); ()
      case e: JobEvent.Progress if e.progress.pass == "base" && e.progress.step >= 2 =>
        Option(reached.get()).foreach(_.countDown())
      case _ => ()
    }
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    def generate(): OutputImage =
      val idleUntil = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while (!services.queue.isIdle || fx(window.progress.get.running)) && System.nanoTime() < idleUntil do Thread.sleep(50)
      require(services.queue.isIdle && fx(!window.progress.get.running), "queue did not become idle")
      fx(window.statusBar.generateBtn.fire())
      val result = finished.poll(240, TimeUnit.SECONDS)
      require(result != null, "generation timed out")
      result.fold(message => throw new IllegalStateException(message), _.head)
    try
      services.start(); services.loadCheckpoint(model.toString, remember = false)
      fx {
        window = new MainWindow(new Stage(), services, UiState(), CliOptions(), _ => ())
        window.show(); window.select(Mode.Inpaint); window.model.checkpoint.set(model.toString)
        val vm = window.vms(Mode.Inpaint)
        vm.prompt.set("a portrait of a woman, detailed painting"); vm.negativePrompt.set("low quality")
        vm.seed.set(173); vm.steps.set(12); vm.width.set(512); vm.height.set(512)
        vm.cfgScale.set(5); vm.denoise.set(.6); vm.maskBlur.set(4)
        val form = window.plugins(Mode.Inpaint).form("inpaint_difference").get
        form.props("enabled").set("true")
        form.props("base_path").set(base.toString)
        form.props("dilation").set("2"); form.props("erosion").set("1")
        vm.initImage.set(altered.toString)
      }
      val difference = generate()
      val previewDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while !fx(Option(window.preview.zoom.image).exists(i => !i.isError && i.getWidth > 0)) && System.nanoTime() < previewDeadline do
        Thread.sleep(100)
      require(fx(Option(window.preview.zoom.image).exists(i => !i.isError && i.getWidth > 0)), "generated preview did not load")
      fx {
        ImageIO.write(SwingFXUtils.fromFXImage(window.stage.getScene.snapshot(null), null), "png", home.root.resolve("difference-window.png").toFile)
        window.plugins(Mode.Inpaint).form("inpaint_difference").get.props("enabled").set("false")
        window.vms(Mode.Inpaint).initImage.set(altered.toString)
      }
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while !fx(window.vms(Mode.Inpaint).hasMask.get) && System.nanoTime() < deadline do
        fx(window.panels(Mode.Inpaint).mask.get.loadMask(mask.toString))
        Thread.sleep(100)
      require(fx(window.vms(Mode.Inpaint).hasMask.get), "reference mask did not load")
      val explicit = generate()
      require(java.util.Arrays.equals(pixels(difference.path), pixels(explicit.path)), "difference and explicit mask results differ")
      val latch = new CountDownLatch(1)
      reached.set(latch)
      fx {
        val form = window.plugins(Mode.Inpaint).form("inpaint_difference").get
        form.props("enabled").set("true")
        window.vms(Mode.Inpaint).steps.set(40)
        window.statusBar.generateBtn.fire()
      }
      require(latch.await(120, TimeUnit.SECONDS), "difference inpaint did not reach sampling")
      fx(window.statusBar.generateBtn.fire())
      val partial = cancelled.poll(90, TimeUnit.SECONDS)
      require(partial != null && partial.isEmpty, "incomplete difference image was not cancelled cleanly")
      reached.set(null)
      require(finished.isEmpty, "cancelled request also reported success")
      fx {
        window.plugins(Mode.Inpaint).form("inpaint_difference").get.props("enabled").set("false")
        window.vms(Mode.Inpaint).steps.set(12)
      }
      val recovery = generate()
      require(java.util.Arrays.equals(pixels(explicit.path), pixels(recovery.path)), "ordinary inpaint did not recover exactly")
      Files.writeString(home.root.resolve("report.json"), ujson.Obj("status" -> "passed", "difference" -> difference.path.toString,
        "explicit" -> explicit.path.toString, "recovery" -> recovery.path.toString, "rgb_exact" -> true,
        "cancelled_without_partial" -> true, "recovery_rgb_exact" -> true).render(indent = 2))
      println("PASS MainWindow difference, explicit mask, cancellation and exact recovery")
    finally
      if window != null then fx { window.dispose(); window.stage.close() }
      services.shutdown(); Platform.exit()
