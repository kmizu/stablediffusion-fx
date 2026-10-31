package sdfx.ui

import java.nio.file.{Files, Path}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import javafx.application.Platform
import javafx.scene.{Node, Parent, Scene}
import javafx.scene.control.{Button, Label, TabPane, TextField}
import javafx.stage.Stage
import scala.jdk.CollectionConverters.*
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.engine.{AppServices, EngineLaunch}
import sdfx.plugins.StandardPlugins
import sdfx.ui.view.ModelKeywordMappingsPane
import sdfx.ui.vm.UiState

/** Manual native dry-run mapping action check. Args: output-dir python-executable. */
object RealModelKeywordMappingsCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A] { def call(): A = body })
    Platform.runLater(task); task.get(30, TimeUnit.SECONDS)
  private def waitFor(label: String)(f: => Boolean): Unit =
    val end = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
    while !f && System.nanoTime() < end do Thread.sleep(50)
    require(f, s"Timed out: $label")
  def main(args: Array[String]): Unit =
    require(args.length == 2, "output-dir python-executable")
    val root = Path.of(args(0)).toAbsolutePath; Files.createDirectories(root)
    val home = AppHome(Files.createTempDirectory(root, "keyword-ui-"))
    val models = Files.createDirectories(home.root.resolve("models"))
    val checkpoint = models.resolve("fixture.safetensors")
    Files.write(checkpoint, Array[Byte](2, 0, 0, 0, 0, 0, 0, 0, 123, 125))
    val python = Path.of(args(1)).toAbsolutePath
    val source = Path.of("python").toAbsolutePath
    val launch = () => EngineLaunch(Seq(python.toString, "-m", "sdfx", "--dry-run"), source,
      Map("PYTHONPATH" -> source.toString))
    val services = new AppServices(home, Settings(pythonExe = Some(python.toString),
      modelDirs = ModelDirs(checkpoints = Seq(models.toString))), dryRun = true,
      builtins = StandardPlugins.withOwnAxes, launchOverride = Some(launch), persistSettings = false,
      installShutdownHook = false)
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    def nodes(node: Node): Seq[Node] = Seq(node) ++ (node match
      case p: Parent => p.getChildrenUnmodifiable.asScala.toSeq.flatMap(nodes)
      case _ => Nil)
    def find(id: String): Node = nodes(window.stage.getScene.getRoot).find(_.getId == id).getOrElse(sys.error(s"Missing $id"))
    def status: String = fx(find("model-keyword-mapping-status").asInstanceOf[Label].getText)
    try
      val hello = services.start(); require(hello.dryRun)
      val model = services.loadCheckpoint(checkpoint.toString, remember = false)
      fx {
        window = new MainWindow(new Stage(), services, UiState(), CliOptions(), _ => ())
        window.show(); window.model.checkpoint.set(model.name)
        val tabs = find("main-tabs").asInstanceOf[TabPane]
        tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getId == "model-keyword-tab").get)
      }
      waitFor("mounted mapping editor")(fx(nodes(window.stage.getScene.getRoot).exists(_.getId == "model-keyword-mapping")))
      waitFor("initial mapping check")(!status.contains("Working"))
      fx {
        find("model-keyword-mapping").asInstanceOf[TextField].setText("blue|green")
        find("model-keyword-save").asInstanceOf[Button].fire()
      }
      waitFor("save mapping")(status.contains("custom") && status.contains("blue | green"))
      require(fx(window.ctx.modelKeywordChoices.asScala.toSeq) == Seq("keyword1, keyword2", "random", "iterate", "blue", "green"))
      fx(find("model-keyword-delete").asInstanceOf[Button].fire())
      waitFor("delete mapping")(status.contains("custom mapping deleted"))
      require(Files.isRegularFile(home.configDir.resolve("model-keyword/custom-mappings.txt")))
      println(s"PASS native model keyword mapping save/delete: ${home.root}")
    finally
      if window != null then fx { window.dispose(); window.stage.close() }
      services.shutdown(); Platform.exit()
