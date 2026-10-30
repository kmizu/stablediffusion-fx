package sdfx.ui

import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.scene.{Node, Parent}
import javafx.scene.control.*
import javafx.stage.Stage
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.model.{GenerationParams, HiresParams, Img2ImgParams}
import sdfx.core.infotext.Infotext
import sdfx.core.png.{PngMetadata, ScrambledPng}
import sdfx.core.plugins.api.OutputImage
import sdfx.engine.{AppServices, EngineLaunch, EngineTimeouts, JobEvent}
import sdfx.plugins.{ConditioningHighresFixPlugin, StandardPlugins}
import sdfx.ui.vm.{Mode, UiState}
import java.nio.file.{Files, Paths}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, LinkedBlockingQueue, TimeUnit}
import java.util.concurrent.atomic.AtomicReference
import javax.imageio.ImageIO
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

/** Actual mounted controls, real image input, PNG replay and protected-session acceptance. */
object RealConditioningHighresCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body)
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)
  private def eventually(label: String)(test: => Boolean): Unit =
    val end = System.nanoTime() + 120.seconds.toNanos
    while !test && System.nanoTime() < end do Thread.sleep(40)
    require(test, s"Timed out: $label")
  private def nodes(node: Node): Seq[Node] = Seq(node) ++ (node match
    case p: TitledPane => Option(p.getContent).toSeq.flatMap(nodes) ++ Option(p.getGraphic).toSeq.flatMap(nodes)
    case p: ScrollPane => Option(p.getContent).toSeq.flatMap(nodes)
    case p: Parent => p.getChildrenUnmodifiable.asScala.toSeq.flatMap(nodes)
    case _ => Nil)

  def main(args: Array[String]): Unit =
    require(args.length == 4, "python checkpoint photo output")
    val python = Paths.get(args(0)).toAbsolutePath.normalize()
    val checkpoint = Paths.get(args(1)).toAbsolutePath.normalize()
    val photo = Paths.get(args(2)).toAbsolutePath.normalize()
    val output = Paths.get(args(3)).toAbsolutePath.normalize()
    Files.createDirectories(output)
    val engineDir = AppHome.detect().pythonDir
    val home = AppHome(Files.createTempDirectory(output, "conditioning-home-"))
    val services = new AppServices(home, Settings(pythonExe = Some(python.toString), vramMode = "model_offload",
      modelDirs = ModelDirs(checkpoints = Seq(checkpoint.getParent.toString)),
      outputDir = output.resolve("images").toString, mcpHttpEnabled = false), dryRun = false,
      builtins = StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false,
      launchOverride = Some(() => EngineLaunch(Seq(python.toString, "-m", "sdfx"), engineDir,
        Map("PYTHONPATH" -> engineDir.toString, "USE_TF" -> "0"))), timeouts = EngineTimeouts(load = 10.minutes))
    val finished = new LinkedBlockingQueue[Either[String, Seq[OutputImage]]]()
    val cancelled = new LinkedBlockingQueue[Seq[OutputImage]]()
    val reached = new AtomicReference[CountDownLatch]()
    services.queue.addListener {
      case e: JobEvent.JobFinished => finished.offer(Right(e.output.images)); ()
      case e: JobEvent.JobFailed => finished.offer(Left(e.message)); ()
      case e: JobEvent.JobCancelled => cancelled.offer(e.partial.images); ()
      case e: JobEvent.Progress if e.progress.step >= 1 => Option(reached.get()).foreach(_.countDown())
      case _ => ()
    }
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    var images = Vector.empty[String]
    def idle(): Unit = eventually("native queue idle")(services.queue.isIdle && fx(!window.progress.get.running))
    def generate(): OutputImage =
      idle(); fx(window.statusBar.generateBtn.fire())
      val result = finished.poll(300, TimeUnit.SECONDS)
      require(result != null, "Native conditioning generation timed out")
      val image = result.fold(message => throw new IllegalStateException(message), _.last)
      images :+= image.path.toString
      idle(); image
    def rgb(image: OutputImage, password: Option[String] = None): Array[Int] =
      val decoded = ScrambledPng.read(image.path, password).image
      decoded.getRGB(0, 0, decoded.getWidth, decoded.getHeight, null, 0, decoded.getWidth)
    def controls(mode: Mode): (CheckBox, ComboBox[String]) =
      window.select(mode); window.root.applyCss(); window.root.layout()
      val pane = nodes(window.panels(mode).node).collectFirst {
        case p: TitledPane if p.getText == "Conditioning Highres Fix" => p
      }.getOrElse(sys.error("Conditioning native section missing"))
      pane.setExpanded(true)
      val combo = nodes(pane.getContent).collectFirst { case c: ComboBox[?] => c.asInstanceOf[ComboBox[String]] }.get
      (pane.getGraphic.asInstanceOf[CheckBox], combo)
    def configure(mode: Mode, enabled: Boolean, strength: String = "high"): Unit = fx {
      val (check, combo) = controls(mode)
      check.setSelected(enabled); combo.setValue(strength)
      val form = window.plugins(mode).form("conditioning_highres_fix").get
      require(form.isActive == enabled && form.values.str("mode") == strength)
    }
    def session(password: Option[String]): Unit = fx {
      val tabs = window.scene.lookup("#main-tabs").asInstanceOf[TabPane]
      tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getId == "image-scrambling-tab").get)
      window.scene.lookup("#scramble-pages").asInstanceOf[TabPane].getSelectionModel.select(0)
      password match
        case Some(value) =>
          window.scene.lookup("#scramble-session-password").asInstanceOf[PasswordField].setText(value)
          window.scene.lookup("#scramble-session-enable").asInstanceOf[Button].fire()
          require(services.imageScrambleSession.snapshot.nonEmpty)
        case None =>
          window.scene.lookup("#scramble-session-lock").asInstanceOf[Button].fire()
          require(services.imageScrambleSession.snapshot.isEmpty)
    }
    try
      val hello = services.start()
      require(!hello.dryRun && hello.features.contains("conditioning_highres_fix"))
      val loaded = services.loadCheckpoint(checkpoint.toString, remember = false)
      require(Set("sd15-inpaint", "sdxl-inpaint")(loaded.kind))
      val size = if loaded.kind.startsWith("sdxl") then 768 else 512
      val base = GenerationParams(prompt = "portrait photograph of a woman astronaut in an orange flight suit, detailed, natural light",
        negativePrompt = "cartoon, illustration, blurry, distorted", seed = 173, steps = 24, width = size, height = size,
        cfgScale = 5, samplerName = "Euler", img2img = Some(Img2ImgParams(initImage = photo.toString, denoisingStrength = .5)))
      window = fx {
        val result = new MainWindow(new Stage(), services, UiState(), CliOptions())
        result.show(); result.model.checkpoint.set(loaded.name); result.select(Mode.Img2Img)
        result.vms(Mode.Img2Img).load(base); result
      }
      for mode <- Mode.values; strength <- ConditioningHighresFixPlugin.modes do
        configure(mode, true, strength)
        fx {
          val (_, combo) = controls(mode)
          require(combo.getItems.asScala.toSeq == ConditioningHighresFixPlugin.modes)
          window.plugins(mode).restoreInfotext(s"test\nSteps: 24, Conditioning Highres Fix: $strength")
          require(window.plugins(mode).form("conditioning_highres_fix").get.values.str("mode") == strength)
          window.plugins(mode).restoreInfotext("test\nSteps: 24, Conditioning Highres Fix: invalid")
          require(!window.plugins(mode).form("conditioning_highres_fix").get.isActive)
        }
      configure(Mode.Img2Img, false)
      val disabled = generate()
      require(!Infotext.parse(PngMetadata.readParameters(disabled.path).get).fields.contains("Conditioning Highres Fix"))
      configure(Mode.Img2Img, true)
      val active = generate()
      val info = PngMetadata.readParameters(active.path).get
      require(Infotext.parse(info).get("Conditioning Highres Fix").contains("high"))
      configure(Mode.Img2Img, false)
      fx(window.restore(info))
      require(fx(window.plugins(Mode.Img2Img).form("conditioning_highres_fix").get.isActive))
      val replay = generate()
      require(java.util.Arrays.equals(rgb(active), rgb(replay)), "PNG replay changed RGB")
      eventually("replayed preview")(fx(window.preview.currentPath.contains(replay.path) && window.preview.zoom.image != null))
      fx {
        val (_, combo) = controls(Mode.Img2Img)
        window.root.applyCss(); window.root.layout()
        val scroll = window.panels(Mode.Img2Img).node
        val content = scroll.getContent
        val y = content.sceneToLocal(combo.localToScene(combo.getBoundsInLocal)).getMinY - 80
        val available = content.getLayoutBounds.getHeight - scroll.getViewportBounds.getHeight
        if available > 0 then scroll.setVvalue(math.max(0, math.min(1, y / available)))
      }
      Thread.sleep(400)
      fx {
        window.root.applyCss(); window.root.layout()
        ImageIO.write(SwingFXUtils.fromFXImage(window.scene.snapshot(null), null), "png", output.resolve("mainwindow.png").toFile)
      }
      session(Some("conditioning-native-only"))
      fx(window.select(Mode.Img2Img))
      val protectedImage = generate()
      require(PngMetadata.readTexts(protectedImage.path).get("Encrypt").contains("pixel_shuffle_3"))
      require(java.util.Arrays.equals(rgb(active), rgb(protectedImage, Some("conditioning-native-only"))), "protected output changed RGB")
      session(None); fx(window.select(Mode.Img2Img))
      fx(window.vms(Mode.Img2Img).initImage.set(active.path.toString))
      val plainInput = generate()
      session(Some("conditioning-native-only"))
      fx { window.select(Mode.Img2Img); window.vms(Mode.Img2Img).initImage.set(protectedImage.path.toString) }
      val protectedInput = generate()
      require(java.util.Arrays.equals(rgb(plainInput), rgb(protectedInput, Some("conditioning-native-only"))), "protected source changed RGB")
      session(None)
      fx { window.select(Mode.Img2Img); window.vms(Mode.Img2Img).initImage.set(photo.toString) }
      val latch = new CountDownLatch(1); reached.set(latch)
      fx { window.vms(Mode.Img2Img).steps.set(60); window.statusBar.generateBtn.fire() }
      require(latch.await(180, TimeUnit.SECONDS)); fx(window.statusBar.generateBtn.fire())
      require(cancelled.poll(120, TimeUnit.SECONDS) != null, "Native interruption did not cancel")
      reached.set(null); idle()
      fx(window.restore(PngMetadata.readParameters(disabled.path).get))
      require(!fx(window.plugins(Mode.Img2Img).form("conditioning_highres_fix").get.isActive))
      require(java.util.Arrays.equals(rgb(disabled), rgb(generate())), "disabled recovery changed RGB")
      fx { window.select(Mode.Inpaint); window.vms(Mode.Inpaint).load(base) }
      eventually("inpaint mask") {
        fx { window.panels(Mode.Inpaint).mask.get.paintDemo(); window.vms(Mode.Inpaint).hasMask.get }
      }
      configure(Mode.Inpaint, true); generate()
      fx {
        window.select(Mode.Txt2Img)
        window.vms(Mode.Txt2Img).load(base.copy(img2img = None, width = 512, height = 512,
          hires = HiresParams(enabled = true, scale = 1.5, upscaler = "Lanczos", secondPassSteps = 16, denoisingStrength = .5)))
      }
      configure(Mode.Txt2Img, true); generate()
      Files.writeString(output.resolve("native-report.json"), ujson.Obj("status" -> "passed-pending-visual-inspection",
        "model" -> loaded.kind, "images" -> ujson.Arr.from(images), "png_replay_exact" -> true,
        "protected_output_exact" -> true, "protected_input_exact" -> true,
        "cancelled" -> true, "disabled_recovery_exact" -> true).render(2))
    catch case error: Throwable =>
      Files.writeString(output.resolve("native-report.json"), ujson.Obj("status" -> "failed", "failure" -> error.toString).render(2))
      throw error
    finally
      if window != null then fx { window.dispose(); window.stage.close() }
      services.shutdown(); Platform.exit()
