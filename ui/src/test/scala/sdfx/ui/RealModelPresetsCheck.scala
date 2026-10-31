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
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{Mode, UiState}
import java.nio.file.{Files, Paths}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, LinkedBlockingQueue, TimeUnit}
import java.util.concurrent.atomic.AtomicReference
import javax.imageio.ImageIO
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

/** Real model-bound preset controls and fixed-seed generation equivalence in every native mode. */
object RealModelPresetsCheck:
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
    val home = AppHome(Files.createTempDirectory(output, "model-presets-home-"))
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
    var lastBaseline: OutputImage = null
    def idle(): Unit = eventually("native queue idle")(services.queue.isIdle && fx(!window.progress.get.running))
    def generate(): OutputImage =
      idle(); fx(window.statusBar.generateBtn.fire())
      val result = finished.poll(300, TimeUnit.SECONDS)
      require(result != null, "Native model preset generation timed out")
      val image = result.fold(message => throw new IllegalStateException(message), _.last)
      images :+= image.path.toString
      idle(); image
    def rgb(image: OutputImage, password: Option[String] = None): Array[Int] =
      val decoded = ScrambledPng.read(image.path, password).image
      decoded.getRGB(0, 0, decoded.getWidth, decoded.getHeight, null, 0, decoded.getWidth)
    def button(id: String): Button = window.scene.lookup("#model-presets-" + id).asInstanceOf[Button]
    def field(id: String): TextInputControl = window.scene.lookup("#model-presets-" + id).asInstanceOf[TextInputControl]
    def library(): Unit = fx {
      val tabs = window.scene.lookup("#main-tabs").asInstanceOf[TabPane]
      tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getId == "model-presets-tab").get)
      window.root.applyCss(); window.root.layout()
    }
    def saved(): Unit = eventually("preset save")(fx(!button("save").isDisabled))
    try
      val hello = services.start()
      require(!hello.dryRun)
      val loaded = services.loadCheckpoint(checkpoint.toString, remember = false)
      require(Set("sd15", "sdxl")(loaded.kind), loaded.kind)
      val size = if loaded.kind == "sdxl" then 768 else 512
      val base = GenerationParams(prompt = "portrait photograph of a woman astronaut in an orange flight suit, detailed, natural light",
        negativePrompt = "cartoon, illustration, blurry, distorted", seed = 173, steps = 20,
        width = size, height = size, cfgScale = 5, samplerName = "Euler")
      window = fx {
        val result = new MainWindow(new Stage(), services, UiState(), CliOptions())
        result.show(); result.model.checkpoint.set(loaded.name); result
      }
      library(); fx(window.modelPresets.loadModel(checkpoint, autoFetch = false)); saved()
      for mode <- Mode.values do
        fx {
          window.select(mode)
          window.vms(mode).load(base.copy(img2img = if mode == Mode.Txt2Img then None
            else Some(Img2ImgParams(initImage = photo.toString, denoisingStrength = .5))))
        }
        if mode == Mode.Inpaint then eventually("inpaint mask")(fx {
          window.panels(mode).mask.get.paintDemo(); window.vms(mode).hasMask.get
        })
        val baseline = generate()
        lastBaseline = baseline
        val info = PngMetadata.readParameters(baseline.path).get
        eventually("baseline preview")(fx(window.preview.currentPath.contains(baseline.path)))
        library()
        fx {
          button("capture").fire()
          require(field("editor").getText == info, "Capture changed raw infotext")
          field("name").setText(mode.key)
          button("save").fire()
        }
        saved(); fx(button("default").fire()); saved()
        fx(window.modelPresets.loadModel(checkpoint, autoFetch = false)); saved()
        require(fx(field("name").getText) == mode.key)
        fx {
          button("share").fire()
          val doc = sdfx.core.modelpresets.ModelPresetDocument.parseShared(field("sharing").getText)
          require(doc.presets(mode.key) == info)
          window.vms(mode).prompt.set("deliberately different")
          window.vms(mode).seed.set(999)
          button("send-" + mode.key).fire()
        }
        val replay = generate()
        require(java.util.Arrays.equals(rgb(baseline), rgb(replay)), s"${mode.key}: preset restore changed RGB")
        fx(window.restore(PngMetadata.readParameters(replay.path).get))
        require(java.util.Arrays.equals(rgb(baseline), rgb(generate())), s"${mode.key}: PNG restore changed RGB")
        println(s"PASS ${loaded.kind} ${mode.key} capture/save/default/reload/share/send/PNG exact RGB")
      fx {
        val tabs = window.scene.lookup("#main-tabs").asInstanceOf[TabPane]
        tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getId == "image-scrambling-tab").get)
        window.scene.lookup("#scramble-pages").asInstanceOf[TabPane].getSelectionModel.select(0)
        window.scene.lookup("#scramble-session-password").asInstanceOf[PasswordField].setText("presets-acceptance-only")
        window.scene.lookup("#scramble-session-enable").asInstanceOf[Button].fire()
        require(services.imageScrambleSession.snapshot.nonEmpty)
        window.select(Mode.Inpaint)
      }
      val protectedImage = generate()
      require(PngMetadata.readTexts(protectedImage.path).get("Encrypt").contains("pixel_shuffle_3"))
      require(java.util.Arrays.equals(rgb(lastBaseline), rgb(protectedImage, Some("presets-acceptance-only"))))
      eventually("protected preview")(fx(window.preview.currentPath.contains(protectedImage.path)))
      library()
      fx {
        button("capture").fire()
        field("name").setText("protected")
        button("save").fire()
      }
      saved()
      fx {
        button("share").fire()
        require(!field("sharing").getText.contains("presets-acceptance-only"), "Image password leaked into preset")
        button("send-inpaint").fire()
      }
      val protectedReplay = generate()
      require(java.util.Arrays.equals(rgb(lastBaseline), rgb(protectedReplay, Some("presets-acceptance-only"))))
      println(s"PASS ${loaded.kind} protected output and protected-image preset capture/replay exact RGB; no password in share")
      library()
      Thread.sleep(400)
      fx {
        window.root.applyCss(); window.root.layout()
        ImageIO.write(SwingFXUtils.fromFXImage(window.scene.snapshot(null), null), "png", output.resolve("mainwindow.png").toFile)
        window.modelPresets.node.setVvalue(1.0)
      }
      Thread.sleep(400)
      fx(ImageIO.write(SwingFXUtils.fromFXImage(window.scene.snapshot(null), null), "png", output.resolve("mainwindow-actions.png").toFile))
      Files.writeString(output.resolve("native-report.json"), ujson.Obj("status" -> "passed-pending-visual-inspection",
        "model" -> loaded.kind, "images" -> ujson.Arr.from(images), "all_modes_preset_replay_exact" -> true,
        "protected_capture_replay_exact" -> true).render(2))
    catch case error: Throwable =>
      Files.writeString(output.resolve("native-report.json"), ujson.Obj("status" -> "failed", "failure" -> error.toString).render(2))
      throw error
    finally
      if window != null then fx { window.dispose(); window.stage.close() }
      services.shutdown(); Platform.exit()
