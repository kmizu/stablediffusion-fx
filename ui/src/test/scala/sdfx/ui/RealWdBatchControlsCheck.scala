package sdfx.ui

import java.nio.file.{Files, Paths}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.scene.control.{CheckBox, TextField}
import javafx.stage.Stage
import javax.imageio.ImageIO
import sdfx.core.config.{AppHome, Settings}
import sdfx.engine.{AppServices, WdBatchSpec}
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{JobBuilder, Mode, UiState}

/** Real JavaFX controls, no worker or model; chooser result is injected into the field. */
object RealWdBatchControlsCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A] { def call(): A = body })
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)

  def main(args: Array[String]): Unit =
    require(args.length == 1, "artifact output directory required")
    val out = Paths.get(args(0)).toAbsolutePath.normalize()
    Files.createDirectories(out)
    val run = Files.createTempDirectory(out, "wd-batch-native-")
    val home = AppHome(Files.createDirectories(run.resolve("home")))
    val images = Files.createDirectories(run.resolve("input images"))
    val services = new AppServices(home, Settings(), dryRun = true,
      builtins = StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false)
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS), "JavaFX startup timed out")
    var window: MainWindow = null
    try
      window = fx {
        val value = new MainWindow(new Stage(), services, UiState(), CliOptions(), _ => ())
        value.show(); value.select(Mode.Img2Img)
        value.stage.setWidth(800); value.stage.setHeight(650)
        value
      }
      fx {
        val panel = window.panels(Mode.Img2Img)
        val vm = window.vms(Mode.Img2Img)
        val root = panel.content
        val enabled = root.lookup("#wd-batch-enabled").asInstanceOf[CheckBox]
        val input = root.lookup("#wd-batch-input").asInstanceOf[TextField]
        val outputName = root.lookup("#wd-batch-output-name").asInstanceOf[TextField]
        require(enabled != null && input != null && outputName != null, "WD controls missing")
        require(!enabled.isSelected && input.isDisabled && outputName.isDisabled, "WD must default off")
        require(vm.validate(_ => false).exists(_.contains("init image")))
        vm.prompt.set("snapshot prompt")
        enabled.fire()
        require(!input.isDisabled && !outputName.isDisabled)
        input.setText(images.toString)
        outputName.setText("batch-one")
        require(vm.validate(_ => false, batchImages = true).isEmpty)
        val original = JobBuilder.build(vm, window.plugins(Mode.Img2Img))
          .copy(wdBatch = Some(WdBatchSpec(input.getText, home.root.toString, outputName.getText)))
        vm.prompt.set("later edit"); input.setText("later folder"); outputName.setText("later output")
        require(original.params.prompt == "snapshot prompt")
        require(original.wdBatch.exists(s => s.imagesRoot == images.toString && s.outputName == "batch-one"))
        input.setText(images.toString); outputName.setText("batch-one")
        window.stage.getScene.getRoot.applyCss(); window.stage.getScene.getRoot.layout()
        val targetY = panel.content.sceneToLocal(enabled.localToScene(enabled.getBoundsInLocal)).getMinY - 35
        val scrollable = panel.content.getLayoutBounds.getHeight - panel.node.getViewportBounds.getHeight
        panel.node.setVvalue(if scrollable > 0 then math.max(0, math.min(1, targetY / scrollable)) else 0)
        window.stage.getScene.getRoot.applyCss(); window.stage.getScene.getRoot.layout()
        require(ImageIO.write(SwingFXUtils.fromFXImage(window.stage.getScene.snapshot(null), null), "png",
          run.resolve("wd-batch-narrow.png").toFile))
        enabled.fire()
        require(input.isDisabled && outputName.isDisabled && !panel.wdBatchEnabled.get)
        require(vm.validate(_ => false).exists(_.contains("init image")))
        require(services.supervisor.pid.isEmpty, "unexpected worker launch")
      }
      println(s"WD_BATCH_UI_QA_OK artifacts=$run chooser=field-injected worker=not-started queue=not-submitted")
    finally
      if window != null then fx { window.dispose(); window.stage.close() }
      services.shutdown()
      Platform.exit()
