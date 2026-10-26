package sdfx.ui

import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.scene.Node
import javafx.scene.chart.LineChart
import javafx.scene.control.*
import javafx.scene.layout.FlowPane
import javafx.stage.Stage
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.model.GenerationParams
import sdfx.core.png.PngMetadata
import sdfx.engine.{AppServices, EngineLaunch, EngineState, EngineTimeouts, Job, SystemInfoService}
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.UiState
import java.nio.file.{Files, Path, Paths}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import javax.imageio.ImageIO
import scala.concurrent.Await
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

/** MANUAL actual-model native acceptance. Args: python checkpoint-dir checkpoint-name output-dir screenshot.png.
  * Exercises a quick benchmark, cancellation, polling, local history and metadata. NEVER presses Submit.
  */
object RealSystemInfoCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body
    )
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)
  private def eventually(label: String, seconds: Int = 90)(condition: => Boolean): Unit =
    val until = System.nanoTime() + seconds.seconds.toNanos
    while !condition && System.nanoTime() < until do Thread.sleep(100)
    require(condition, s"Timed out: $label")
  private def sameRgb(a: Path, b: Path): Boolean =
    val left = ImageIO.read(a.toFile)
    val right = ImageIO.read(b.toFile)
    left.getWidth == right.getWidth && left.getHeight == right.getHeight &&
      (0 until left.getHeight).forall(y => (0 until left.getWidth).forall(x => left.getRGB(x, y) == right.getRGB(x, y)))
  def main(args: Array[String]): Unit =
    require(args.length == 5, "python checkpoint-dir checkpoint-name output-dir screenshot.png")
    val Array(python, checkpoints, checkpoint, out, screenshotName) = args
    val output = Paths.get(out).toAbsolutePath.normalize()
    Files.createDirectories(output)
    val screenshot = Paths.get(screenshotName).toAbsolutePath.normalize()
    Option(screenshot.getParent).foreach(Files.createDirectories(_))
    val home = AppHome(Files.createTempDirectory(output, "system-info-home-"))
    val engineDir = AppHome.detect().pythonDir
    val services = new AppServices(home, Settings(pythonExe = Some(python), outputDir = output.resolve("images").toString,
      modelDirs = ModelDirs(checkpoints = Seq(Paths.get(checkpoints).toAbsolutePath.toString))), false, StandardPlugins.withOwnAxes,
      launchOverride = Some(() => EngineLaunch(Seq(python, "-m", "sdfx"), engineDir, Map("PYTHONPATH" -> engineDir.toString, "USE_TF" -> "0"))),
      persistSettings = false, timeouts = EngineTimeouts(load = 10.minutes), installShutdownHook = false)
    val started = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); started.countDown() })
    require(started.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    var checks = Vector.empty[String]
    def find(root: Node, id: String): Option[Node] =
      if root == null then None
      else if root.getId == id then Some(root)
      else
        val children: Seq[Node] = root match
          case pane: TabPane => pane.getTabs.asScala.flatMap(t => Option(t.getContent)).toSeq
          case pane: TitledPane => Option(pane.getContent).toSeq
          case pane: ScrollPane => Option(pane.getContent).toSeq
          case parent: javafx.scene.Parent => parent.getChildrenUnmodifiable.asScala.toSeq
          case _ => Nil
        children.iterator.flatMap(child => find(child, id)).take(1).toSeq.headOption
    def node[A <: Node](id: String): A =
      find(window.root, id).getOrElse(throw new IllegalStateException(s"Missing native control $id")).asInstanceOf[A]
    def snapshot(path: Path): Unit = fx {
      window.root.applyCss()
      window.root.layout()
      require(ImageIO.write(SwingFXUtils.fromFXImage(window.scene.snapshot(null), null), "png", path.toFile))
    }
    try
      window = fx {
        val w = new MainWindow(new Stage(), services, UiState(selectedTab = "systeminfo"), CliOptions())
        w.show()
        w.startEngine()
        w
      }
      eventually("native engine startup and catalog") {
        services.supervisor.state.isInstanceOf[EngineState.Ready] &&
          fx { window.catalogVM.snapshot.get.checkpoints.exists(_.name == checkpoint) }
      }
      fx { window.loadCheckpoint(checkpoint) }
      eventually("native checkpoint load and Ready indicator", 600) {
        services.supervisor.loadedModel.nonEmpty && services.supervisor.state.isInstanceOf[EngineState.Ready] && fx {
          window.model.checkpoint.get == checkpoint &&
            window.topBar.node.lookupAll(".chip-ready").asScala.exists {
              case label: Label => label.getText.startsWith("● Ready")
              case _ => false
            }
        }
      }
      val loaded = services.supervisor.loadedModel.get
      // A full report can have completed before the model loaded. Use the real Refresh handler
      // and wait for its model-bearing result, rather than accepting that stale startup report.
      eventually("full refresh available") { fx { !node[Button]("system-info-refresh").isDisabled } }
      fx { node[Button]("system-info-refresh").fire() }
      eventually("full native loaded-model snapshot") { fx {
        val text = node[TextArea]("system-info-json").getText
        text.nonEmpty && scala.util.Try(ujson.read(text)).toOption.exists { data =>
          data.obj.contains("networks") && data.obj.get("model").exists(_.obj.get("name").contains(ujson.Str(loaded.name)))
        }
      } }
      val full = Await.result(services.client.systemInfo(), 30.seconds)
      require(full("model")("name").str == loaded.name)
      require(!full("runtime")("dry_run").bool)
      fx { require(window.state.selectedTab == "systeminfo") }
      checks :+= "native startup and checkpoint handlers, Ready indicator, loaded-model refresh and tab restoration"
      eventually("real GPU chart sample") { fx {
        val chart = node[LineChart[Number, Number]]("system-info-chart")
        chart.getData.get(1).getData.size > 0
      } }
      snapshot(screenshot)
      fx { node[TabPane]("main-tabs").getSelectionModel.select(0) }
      Thread.sleep(500) // let a request already in flight finish
      val hidden = fx { node[Label]("system-info-status").getText }
      Thread.sleep(6000)
      fx {
        require(node[Label]("system-info-status").getText == hidden, "Hidden tab continued polling")
        val tabs = node[TabPane]("main-tabs")
        tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getId == "system-info-tab").get)
        node[TabPane]("system-info-sections").getSelectionModel.select(2)
        node[ComboBox[String]]("system-benchmark-level").setValue("quick")
        node[ComboBox[String]]("system-benchmark-steps").setValue("turbo")
        node[Spinner[Integer]]("system-benchmark-width").getValueFactory.setValue(256)
        node[Spinner[Integer]]("system-benchmark-height").getValueFactory.setValue(256)
        node[Button]("system-benchmark-run").fire()
      }
      checks :+= "visible-only polling and real bounded GPU memory graph"
      val historyFile = home.configDir.resolve("system-benchmarks.json")
      eventually("quick benchmark and persisted local history", 600) { Files.isRegularFile(historyFile) && fx { !node[Button]("system-benchmark-run").isDisabled } }
      val history = SystemInfoService.load(historyFile)
      require(history.size == 1 && history.head("results").arr.head("status").str == "complete")
      require(history.head("results").arr.head("iterations_per_second").num > 0)
      fx {
        node[TextField]("system-benchmark-username").setText("validation-no-submit")
        node[Button]("system-benchmark-preview-button").fire()
      }
      eventually("submission preview, without sending") { fx { node[TextArea]("system-benchmark-preview").getText.contains("logs3.papertrailapp.com") } }
      val preview = fx { node[TextArea]("system-benchmark-preview").getText }
      require(preview.contains("anonymous") && !preview.contains("sent_unconfirmed"))
      snapshot(output.resolve("system-benchmark.png"))
      checks :+= "real quick benchmark, wall-time score, local history, submission preview only"
      fx {
        node[ComboBox[String]]("system-benchmark-level").setValue("normal")
        node[ComboBox[String]]("system-benchmark-steps").setValue("long")
        node[Button]("system-benchmark-run").fire()
      }
      eventually("benchmark running") {
        val data = Await.result(services.client.systemStatus(), 5.seconds)
        data("state").obj.get("request_type").contains(ujson.Str("system_benchmark"))
      }
      fx { node[Button]("system-info-interrupt").fire() }
      eventually("native benchmark interruption") { fx { !node[Button]("system-benchmark-run").isDisabled } }
      require(SystemInfoService.load(historyFile).head("timestamp") == history.head("timestamp"), "Interrupted run replaced completed history")
      checks :+= "native interrupt stopped benchmark and retained the completed history"
      val params = GenerationParams(prompt = "a cat", seed = 71, steps = 1, width = 256, height = 256, samplerName = "Euler")
      def generate() = Await.result(services.queue.submit(Job(params)), 10.minutes).images.head
      val plain = generate()
      require(!plain.infotext.contains("System Info:"), "Metadata was enabled without explicit selection")
      fx {
        val fields = node[FlowPane]("system-info-metadata-fields")
        fields.getChildren.asScala.collectFirst { case c: CheckBox if c.getText == "platform.python" => c }.get.fire()
      }
      val opted = generate()
      require(opted.infotext.contains("System Info:") && opted.infotext.contains("platform.python"))
      require(PngMetadata.readTexts(opted.path).getOrElse("parameters", "").contains("System Info:"))
      require(sameRgb(plain.path, opted.path), "Metadata selection changed generated RGB pixels")
      services.updateSettings(_.copy(systemInfoMetadataFields = Nil))
      val restored = generate()
      require(!restored.infotext.contains("System Info:") && sameRgb(plain.path, restored.path))
      checks :+= "explicit PNG metadata selection and RGB-invariant disable/restoration"
      Files.writeString(output.resolve("system-info-report.json"), ujson.Obj("success" -> true, "checks" -> ujson.Arr.from(checks),
        "snapshot" -> full, "benchmark" -> history.head, "metadata_png" -> opted.path.toString,
        "submission_sent" -> false, "home" -> home.root.toString).render(indent = 2))
      println(s"System Info native acceptance passed: ${loaded.kind}; no cloud transmission")
    finally
      try fx { if window != null then { window.dispose(); window.stage.close() } }
      finally
        try services.shutdown()
        finally Platform.exit()
