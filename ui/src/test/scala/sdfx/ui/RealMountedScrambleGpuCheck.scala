package sdfx.ui

import java.nio.file.{Files, Path}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, LinkedBlockingQueue, TimeUnit}
import java.util.concurrent.atomic.AtomicReference
import javafx.application.Platform
import javafx.scene.{Node, Parent}
import javafx.scene.control.{Button, PasswordField, TabPane}
import javafx.stage.Stage
import scala.jdk.CollectionConverters.*
import scala.concurrent.duration.*
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.png.{PngMetadata, ScrambledPng}
import sdfx.core.plugins.api.OutputImage
import sdfx.engine.{AppServices, EngineTimeouts, JobEvent}
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{Mode, UiState}

/** Real GPU acceptance for the mounted controls. Args: output-dir python checkpoint-file.
  * Run separately with SD1.5 and SDXL; inspect the generated images as part of acceptance.
  */
object RealMountedScrambleGpuCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A] { def call(): A = body })
    Platform.runLater(task); task.get(45, TimeUnit.SECONDS)
  private def waitFor(label: String, seconds: Int = 60)(condition: => Boolean): Unit =
    val end = System.nanoTime() + TimeUnit.SECONDS.toNanos(seconds)
    while !condition && System.nanoTime() < end do Thread.sleep(100)
    require(condition, s"Timed out: $label")
  def main(args: Array[String]): Unit =
    require(args.length == 3, "output-dir python checkpoint-file")
    val root = Path.of(args(0)).toAbsolutePath.normalize()
    val python = Path.of(args(1)).toAbsolutePath.normalize()
    val checkpoint = Path.of(args(2)).toAbsolutePath.normalize()
    Seq(python, checkpoint).foreach(p => require(Files.isRegularFile(p), s"Missing $p"))
    Files.createDirectories(root)
    val run = Files.createTempDirectory(root, "mounted-scramble-gpu-")
    val home = AppHome(Files.createDirectories(run.resolve("home")))
    Files.createDirectories(home.pythonDir)
    val services = new AppServices(home, Settings(pythonExe = Some(python.toString), hfOffline = true,
      modelDirs = ModelDirs(checkpoints = Seq(checkpoint.getParent.toString))), dryRun = false,
      builtins = StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false,
      timeouts = EngineTimeouts(load = 10.minutes))
    val finished = new LinkedBlockingQueue[Either[String, Seq[OutputImage]]]()
    val cancelled = new LinkedBlockingQueue[Seq[OutputImage]]()
    val reached = new AtomicReference[CountDownLatch]()
    services.queue.addListener {
      case e: JobEvent.JobFinished => finished.offer(Right(e.output.images)); ()
      case e: JobEvent.JobFailed => finished.offer(Left(e.message)); ()
      case e: JobEvent.JobCancelled => cancelled.offer(e.partial.images); ()
      case e: JobEvent.Progress if e.progress.step >= 2 => Option(reached.get()).foreach(_.countDown())
      case _ => ()
    }
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS), "JavaFX startup timed out")
    var window: MainWindow = null
    def nodes(node: Node): Seq[Node] = Seq(node) ++ (node match
      case p: Parent => p.getChildrenUnmodifiable.asScala.toSeq.flatMap(nodes)
      case _ => Nil)
    def find(id: String): Node = nodes(window.root).find(_.getId == id).getOrElse(sys.error(s"Missing control $id"))
    def sessionTab(): Unit =
      val main = find("main-tabs").asInstanceOf[TabPane]
      main.getSelectionModel.select(main.getTabs.asScala.find(_.getId == "image-scrambling-tab").get)
      find("scramble-pages").asInstanceOf[TabPane].getSelectionModel.select(0)
    def unlock(password: String): Unit = fx {
      sessionTab()
      val field = find("scramble-session-password").asInstanceOf[PasswordField]
      field.setText(password); find("scramble-session-enable").asInstanceOf[Button].fire()
      require(field.getText.isEmpty && services.imageScrambleSession.snapshot.nonEmpty)
    }
    def lock(): Unit = fx {
      sessionTab(); find("scramble-session-lock").asInstanceOf[Button].fire()
      require(services.imageScrambleSession.snapshot.isEmpty)
    }
    def idle(): Unit = waitFor("queue idle")(services.queue.isIdle && fx(!window.progress.get.running))
    def result(): OutputImage =
      val value = finished.poll(300, TimeUnit.SECONDS)
      require(value != null, "Generation timed out")
      val images = value.fold(message => throw new IllegalStateException(message), identity)
      require(images.size == 1, s"Expected one image, got ${images.size}")
      images.head
    def generate(): OutputImage =
      idle(); fx(window.statusBar.generateBtn.fire()); val image = result(); idle(); image
    def pixels(path: Path, password: Option[String]): Array[Int] =
      val im = ScrambledPng.read(path, password).image
      im.getRGB(0, 0, im.getWidth, im.getHeight, null, 0, im.getWidth)
    try
      val hello = services.start(); require(!hello.dryRun, "Expected real worker")
      val loaded = services.loadCheckpoint(checkpoint.toString, remember = false)
      require(Set("sd15", "sdxl")(loaded.kind), s"Unexpected model family ${loaded.kind}")
      fx {
        window = new MainWindow(new Stage(), services, UiState(), CliOptions(), _ => ())
        window.show(); window.select(Mode.Txt2Img); window.model.checkpoint.set(loaded.name)
        val vm = window.vms(Mode.Txt2Img)
        vm.prompt.set("blue ceramic teapot on a wooden table, sunlight")
        vm.seed.set(173); vm.steps.set(5); vm.width.set(512); vm.height.set(512)
      }
      val ordinary = generate()
      require(!PngMetadata.readTexts(ordinary.path).contains("Encrypt"))
      unlock("native-secret")
      idle()
      fx { window.statusBar.generateBtn.fire(); sessionTab(); find("scramble-session-lock").asInstanceOf[Button].fire() }
      val protectedImage = result(); idle()
      require(PngMetadata.readTexts(protectedImage.path).get("Encrypt").contains("pixel_shuffle_3"))
      require(java.util.Arrays.equals(pixels(ordinary.path, None), pixels(protectedImage.path, Some("native-secret"))), "RGB differs")
      waitFor("locked preview rejected")(fx(window.preview.zoom.image == null && window.preview.imageStatus.startsWith("Cannot")))
      unlock("wrong")
      waitFor("wrong password rejected")(fx(window.preview.zoom.image == null && window.preview.imageStatus.contains("Incorrect image password")))
      unlock("native-secret")
      waitFor("correct password preview")(fx(window.preview.zoom.image != null))
      val latch = new CountDownLatch(1); reached.set(latch)
      fx { window.vms(Mode.Txt2Img).steps.set(50); window.statusBar.generateBtn.fire() }
      require(latch.await(180, TimeUnit.SECONDS), "Cancellation job never reached sampling")
      fx(window.statusBar.generateBtn.fire())
      require(cancelled.poll(120, TimeUnit.SECONDS) != null, "Native interruption did not cancel")
      reached.set(null); idle(); require(finished.isEmpty, "Cancelled job also finished")
      fx(window.vms(Mode.Txt2Img).steps.set(5))
      val recovered = generate()
      require(java.util.Arrays.equals(pixels(ordinary.path, None), pixels(recovered.path, Some("native-secret"))), "Recovery RGB differs")
      lock()
      waitFor("preview clears on lock")(fx(window.preview.zoom.image == null && window.preview.imageStatus.startsWith("Cannot")))
      Files.writeString(run.resolve("result.json"), ujson.Obj("family" -> loaded.kind,
        "checkpoint" -> checkpoint.toString, "ordinary" -> ordinary.path.toString,
        "protected" -> protectedImage.path.toString, "recovery" -> recovered.path.toString,
        "rgb_exact" -> true, "cancelled" -> true).render(2))
      println(s"PASS mounted session GPU ${loaded.kind}: parity, lock, wrong/correct password, cancellation and recovery; $run")
    finally
      if window != null then fx { window.dispose(); window.stage.close() }
      services.shutdown(); Platform.exit()
