package sdfx.ui

import javafx.application.Platform
import javafx.scene.{Node, Parent}
import javafx.scene.control.{Button, Label, TabPane, TextField}
import javafx.stage.Stage
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.engine.AppServices
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.UiState
import java.nio.file.{Files, Path}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import scala.jdk.CollectionConverters.*

/** Native mounting and worker error path in dry-run; no model accuracy claim. */
object RealEmbeddingInspectorCheck:
  def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A] { def call(): A = body })
    Platform.runLater(task); task.get(30, TimeUnit.SECONDS)
  def main(args: Array[String]): Unit =
    require(args.length == 2)
    val root = Path.of(args(0)).toAbsolutePath; Files.createDirectories(root)
    val home = AppHome(Files.createTempDirectory(root, "home-")); Files.createDirectories(home.pythonDir)
    val source = Path.of("python", "sdfx").toAbsolutePath
    val target = home.pythonDir.resolve("sdfx")
    val sources = Files.walk(source)
    try sources.iterator().asScala.filterNot(_.toString.contains("__pycache__")).foreach { path =>
      val out = target.resolve(source.relativize(path).toString)
      if Files.isDirectory(path) then Files.createDirectories(out)
      else Files.copy(path, out)
    }
    finally sources.close()
    val models = Files.createDirectories(home.root.resolve("test-models"))
    val checkpoint = models.resolve("fixture.safetensors")
    Files.write(checkpoint, Array[Byte](2, 0, 0, 0, 0, 0, 0, 0, 123, 125))
    val service = new AppServices(home, Settings(pythonExe = Some(Path.of(args(1)).toAbsolutePath.toString),
      modelDirs = ModelDirs(checkpoints = Seq(models.toString))), dryRun = true,
      builtins = StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false)
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    def nodes(node: Node): Seq[Node] = Seq(node) ++ (node match
      case p: Parent => p.getChildrenUnmodifiable.asScala.toSeq.flatMap(nodes)
      case _ => Nil)
    def find(id: String): Node = nodes(window.root).find(_.getId == id).getOrElse(sys.error(s"missing $id"))
    try
      service.start(); service.loadCheckpoint(checkpoint.toString, remember = false)
      fx {
        window = new MainWindow(new Stage(), service, UiState(), CliOptions(), _ => ())
        window.show()
        val tabs = find("main-tabs").asInstanceOf[TabPane]
        tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getId == "embedding-inspector-tab").get)
        find("embedding-inspector-input").asInstanceOf[TextField].setText("cat")
        find("embedding-inspector-inspect").asInstanceOf[Button].fire()
      }
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while fx(find("embedding-inspector-status").asInstanceOf[Label].getText).startsWith("Queued") && System.nanoTime() < deadline do Thread.sleep(50)
      val status = fx(find("embedding-inspector-status").asInstanceOf[Label].getText)
      require(status.contains("real loaded model"), s"unexpected status: $status")
      fx {
        window.stage.setWidth(1500); window.stage.setHeight(900)
        window.root.applyCss(); window.root.layout()
        javax.imageio.ImageIO.write(javafx.embed.swing.SwingFXUtils.fromFXImage(window.scene.snapshot(null), null),
          "png", root.resolve("inspector-wide.png").toFile)
      }
      println("PASS mounted inspector: queued dry-run refusal shown")
    finally
      if window != null then fx { window.dispose(); window.stage.close() }
      service.shutdown(); Platform.exit()
