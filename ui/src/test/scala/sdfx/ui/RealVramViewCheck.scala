package sdfx.ui

import javafx.application.Platform
import javafx.stage.Stage
import javafx.scene.Scene
import javafx.scene.control.{Button, ComboBox, Spinner}
import javafx.scene.chart.LineChart
import javafx.scene.layout.VBox
import javafx.embed.swing.SwingFXUtils
import javax.imageio.ImageIO
import java.nio.file.{Files, Paths}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import sdfx.core.config.{AppHome, Settings, ModelDirs}
import sdfx.core.model.GenerationParams
import sdfx.engine.AppServices
import sdfx.ui.view.{UiContext, VramEstimatorTab, VramEstimatePane}
import sdfx.ui.vm.CatalogVM

object RealVramViewCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body)
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)
  private def until(body: => Boolean): Unit =
    val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(180)
    while !body && System.nanoTime() < deadline do Thread.sleep(100)
    require(body, "Native VRAM view condition timed out")
  def main(args: Array[String]): Unit =
    val home = AppHome(Paths.get(args(0)).toAbsolutePath)
    val model = Paths.get(args(2)).toAbsolutePath
    Files.createDirectories(home.pythonDir)
    val services = new AppServices(home, Settings(pythonExe = Some(Paths.get(args(1)).toAbsolutePath.toString),
      modelDirs = ModelDirs(checkpoints = Seq(model.getParent.toString))), dryRun = false,
      builtins = Nil, persistSettings = false, installShutdownHook = false)
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var stage: Stage = null
    var tab: VramEstimatorTab = null
    var estimate: VramEstimatePane = null
    var generation = GenerationParams(width = 288, height = 288)
    try
      services.start(); services.loadCheckpoint(model.toString, remember = false)
      fx {
        stage = new Stage()
        val ctx = new UiContext(services, new CatalogVM, stage)
        tab = new VramEstimatorTab(ctx)
        estimate = new VramEstimatePane(ctx, () => generation)
        stage.setScene(new Scene(new VBox(estimate.node, tab.node), 1150, 850))
        stage.show()
        val root = stage.getScene.getRoot
        root.applyCss(); root.layout()
        root.lookup("#vram-maximum-size").asInstanceOf[Spinner[Integer]].getValueFactory.setValue(320)
        root.lookup("#vram-maximum-batch").asInstanceOf[Spinner[Integer]].getValueFactory.setValue(1)
        root.lookup("#vram-calibrate").asInstanceOf[Button].fire()
      }
      until(fx {
        val combo = stage.getScene.lookup("#vram-profile-history").asInstanceOf[ComboBox[String]]
        !combo.getItems.isEmpty && !stage.getScene.lookup("#vram-calibrate").isDisabled
      })
      fx {
        Vector("txt2img", "txt2img_hr", "txt2img_hr_latent", "img2img").foreach { mode =>
          val chart = stage.getScene.lookup(s"#vram-chart-$mode").asInstanceOf[LineChart[Number, Number]]
          require(chart.getData.size == 1 && chart.getData.get(0).getData.size == 2, s"Missing $mode measurements")
        }
        estimate.setActive(true)
      }
      until(fx(estimate.node.getText.startsWith("Vanilla VRAM estimate:") && estimate.node.getText.contains("288×288")))
      fx { generation = generation.copy(width = 304, height = 304) }
      until(fx(estimate.node.getText.startsWith("Vanilla VRAM estimate:") && estimate.node.getText.contains("304×304")))
      fx {
        ImageIO.write(SwingFXUtils.fromFXImage(stage.getScene.snapshot(null), null), "png", home.root.resolve("native-vram-view.png").toFile)
        estimate.setActive(false)
      }
      println("PASS native VRAM view: calibration button, four measured charts, persisted history, live compatible estimate, disposal")
    finally
      fx { if tab != null then tab.dispose(); if estimate != null then estimate.dispose(); if stage != null then stage.close() }
      services.shutdown()
      Platform.exit()
