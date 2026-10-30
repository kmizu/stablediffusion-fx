package sdfx.ui

import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.scene.Node
import javafx.scene.control.*
import javafx.stage.Stage
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.engine.{AppServices, EngineLaunch, EngineTimeouts, JobEvent}
import sdfx.core.plugins.api.OutputImage
import sdfx.core.model.{GenerationParams, HiresParams}
import sdfx.core.infotext.Infotext
import sdfx.core.png.PngMetadata
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{UiState, Mode}
import java.nio.file.{Files, Paths}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit, LinkedBlockingQueue}
import javax.imageio.ImageIO
import scala.concurrent.duration.*

/** Manual real MainWindow acceptance. Args: python model-dir model-name output screenshot. */
object RealEmbeddingMergeCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body
    )
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)

  private def eventually(label: String)(test: => Boolean): Unit =
    val end = System.nanoTime() + 120.seconds.toNanos
    while !test && System.nanoTime() < end do Thread.sleep(50)
    require(test, s"Timed out: $label")

  def main(args: Array[String]): Unit =
    require(args.length == 5, "python model-dir model-name output screenshot")
    val python = Paths.get(args(0)).toAbsolutePath.normalize()
    val output = Paths.get(args(3)).toAbsolutePath.normalize()
    Files.createDirectories(output)
    val engineDir = AppHome.detect().pythonDir
    val home = AppHome(Files.createTempDirectory(output, "merge-app-home-"))
    val settings = Settings(pythonExe = Some(python.toString),
      modelDirs = ModelDirs(checkpoints = Seq(args(1)), embeddings = Seq(output.resolve("embeddings").toString)),
      outputDir = output.resolve("images").toString, mcpHttpEnabled = false)
    val services = new AppServices(home, settings, dryRun = false, StandardPlugins.withOwnAxes,
      launchOverride = Some(() => EngineLaunch(Seq(python.toString, "-m", "sdfx"), engineDir,
        Map("PYTHONPATH" -> engineDir.toString, "USE_TF" -> "0"))), persistSettings = false,
      timeouts = EngineTimeouts(load = 10.minutes), installShutdownHook = false)
    val finished = new LinkedBlockingQueue[Either[String, Seq[OutputImage]]]()
    services.queue.addListener {
      case event: JobEvent.JobFinished => finished.offer(Right(event.output.images)); ()
      case event: JobEvent.JobFailed => finished.offer(Left(event.message)); ()
      case _ => ()
    }
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    var failure: Throwable = null
    var checks = Vector.empty[String]
    var imagePaths = Vector.empty[String]
    def find[A <: Node](id: String): A =
      val node = window.scene.lookup(s"#$id")
      require(node != null, s"Missing control #$id")
      node.asInstanceOf[A]
    def run(action: String): Unit =
      fx { find[Button](s"embedding-merge-$action").fire() }
      eventually(action) { fx { !find[Button]("embedding-merge-inspect").isDisabled } }
      val status = fx { find[Label]("embedding-merge-status").getText }
      require(!status.toLowerCase.contains("error") && !status.toLowerCase.contains("failed"), status)
    try
      window = fx {
        val result = new MainWindow(new Stage(), services, UiState(selectedTab = "embedding_merge"), CliOptions())
        result.show()
        result
      }
      val hello = services.start()
      require(!hello.dryRun && hello.features.contains("embedding_merge"))
      val loaded = services.loadCheckpoint(args(2), remember = false)
      require(Set("sd15", "sdxl").contains(loaded.kind))
      for grouping <- Seq("none", "comma", "parts", "words", "tokens", "vectors") do
        fx {
          find[TextArea]("embedding-merge-input").setText("a cat, 日本語 café")
          find[ComboBox[String]]("embedding-merge-grouping").setValue(grouping)
        }
        run("inspect")
        checks :+= s"grouping-$grouping"
      val expression = "'red hair' * 0.7 + 'blue hair' * 0.3"
      fx {
        find[TextArea]("embedding-merge-input").setText(expression)
        find[TextField]("embedding-merge-name").setText("native47")
        find[TextField]("embedding-merge-output-dir").setText(output.resolve("embeddings").toString)
        find[CheckBox]("embedding-merge-safetensors").setSelected(true)
      }
      run("inspect")
      require(!Files.exists(output.resolve("embeddings/native47.safetensors")), "Preview wrote a file")
      run("save")
      require(Files.isRegularFile(output.resolve("embeddings/native47.safetensors")))
      require(fx { find[Label]("embedding-merge-status").getText.toLowerCase.contains("active") })
      checks :+= "save-live-active"
      fx {
        find[TextArea]("embedding-merge-input").setText("<'EM_1'>\nSteps: 20, Seed: 173, CFG scale: 7, EmbeddingMerge: <'EM_1'>=<'cat'>")
      }
      run("inspect")
      require(fx { find[TextArea]("embedding-merge-input").getText.startsWith("<'cat'>") })
      checks :+= "pasted-png-restore"
      fx { find[TextArea]("embedding-merge-input").setText(expression) }
      run("inspect")
      val streamNames = if loaded.kind == "sdxl" then Seq("clip_l", "clip_g") else Seq("clip_l")
      for (stream, index) <- streamNames.zipWithIndex do
        fx {
          val tabs = find[TabPane]("embedding-merge-results")
          require(tabs.getTabs.size == streamNames.size)
          require(tabs.getTabs.get(index).getText == stream)
          tabs.getSelectionModel.select(index)
          window.root.applyCss(); window.root.layout()
          val table = find[TableView[?]](s"embedding-merge-$stream-table")
          require(!table.getItems.isEmpty, s"Missing visible $stream vectors")
        }
      fx { find[TabPane]("embedding-merge-results").getSelectionModel.select(0) }
      // TabPane animates newly replaced headers; capture the settled UI rather
      // than an intermediate clipped header from its opening animation.
      Thread.sleep(400)
      checks :+= "visible-encoder-tabs"
      fx {
        window.root.applyCss(); window.root.layout()
        val screenshot = Paths.get(args(4)).toAbsolutePath.normalize()
        Option(screenshot.getParent).foreach(Files.createDirectories(_))
        require(ImageIO.write(SwingFXUtils.fromFXImage(window.scene.snapshot(null), null), "png", screenshot.toFile))
      }
      def generate(params: GenerationParams): OutputImage =
        eventually("native queue idle") { services.queue.isIdle && fx(!window.progress.get.running) }
        fx {
          window.model.checkpoint.set(loaded.path)
          window.vms(Mode.Txt2Img).load(params)
          window.statusBar.generateBtn.fire()
        }
        val result = finished.poll(300, TimeUnit.SECONDS)
        require(result != null, "Native expression generation timed out")
        val image = result.fold(error => throw new IllegalStateException(error), _.last)
        imagePaths :+= image.path.toString
        image
      val size = if loaded.kind == "sdxl" then 768 else 512
      val raw = GenerationParams(prompt = s"a woman with <$expression> in a garden, detailed portrait",
        negativePrompt = "blurry, low quality", seed = 173, steps = 8, width = size, height = size, cfgScale = 4,
        hires = HiresParams(enabled = true, scale = 1.25, secondPassSteps = 4, denoisingStrength = .6,
          prompt = s"a woman with {$expression} in a garden, detailed portrait"))
      val first = generate(raw)
      val metadata = PngMetadata.readParameters(first.path).getOrElse(throw new IllegalStateException("PNG parameters missing"))
      require(!metadata.contains("__nui_em_"), "Ephemeral token leaked into PNG")
      val restored = Infotext.toParams(Infotext.parse(metadata))
      require(restored.prompt == raw.prompt && restored.hires.prompt == raw.hires.prompt, metadata)
      val replay = generate(restored)
      val left = ImageIO.read(first.path.toFile)
      val right = ImageIO.read(replay.path.toFile)
      require(left.getWidth == right.getWidth && left.getHeight == right.getHeight)
      require(java.util.Arrays.equals(left.getRGB(0, 0, left.getWidth, left.getHeight, null, 0, left.getWidth),
        right.getRGB(0, 0, right.getWidth, right.getHeight, null, 0, right.getWidth)), "Native PNG replay RGB changed")
      checks :+= "actual-mainwindow-hires-raw-png-exact-replay"
      Files.writeString(output.resolve("native-report.json"), ujson.write(ujson.Obj(
        "status" -> "passed-pending-visual-inspection", "model" -> loaded.kind,
        "checks" -> ujson.Arr.from(checks), "images" -> ujson.Arr.from(imagePaths)), indent = 2))
    catch case error: Throwable =>
      failure = error
      Files.writeString(output.resolve("native-report.json"), ujson.write(ujson.Obj("status" -> "failed", "failure" -> error.toString), indent = 2))
      throw error
    finally
      NativeAcceptanceSafety.cleanup(failure)(
        () => { if window != null then fx { window.dispose() } },
        () => { if window != null then fx { window.stage.close() } },
        () => services.shutdown(),
        () => Platform.exit())
