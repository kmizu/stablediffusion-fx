package sdfx.ui

import javafx.application.Platform
import javafx.scene.Scene
import javafx.scene.control.*
import javafx.stage.Stage
import java.nio.file.Files
import java.util.concurrent.{CountDownLatch, TimeUnit}
import java.util.concurrent.atomic.AtomicReference
import sdfx.core.config.{AppHome, Settings}
import sdfx.engine.AppServices
import sdfx.plugins.StandardPlugins
import sdfx.ui.view.{EmbeddingMergeTab, UiContext}
import sdfx.ui.vm.CatalogVM

/** CPU-only native controls smoke; actual model/actions use RealEmbeddingMergeCheck. */
object NativeEmbeddingMergeControlsCheck:
  def main(args: Array[String]): Unit =
    val done = new CountDownLatch(1)
    val failure = new AtomicReference[Throwable]()
    Platform.startup(() => {
      val home = AppHome(Files.createTempDirectory("embedding-merge-controls-"))
      val services = new AppServices(home, Settings(mcpHttpEnabled = false), dryRun = true,
        StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false)
      val stage = new Stage()
      var view: EmbeddingMergeTab = null
      try
        val context = new UiContext(services, new CatalogVM(), stage)
        view = new EmbeddingMergeTab(context)
        stage.setScene(new Scene(view.node, 1100, 760))
        stage.show()
        view.node.applyCss(); view.node.layout()
        val input = stage.getScene.lookup("#embedding-merge-input").asInstanceOf[TextArea]
        require(input != null)
        val grouping = stage.getScene.lookup("#embedding-merge-grouping").asInstanceOf[ComboBox[String]]
        require(grouping.getItems.size == 6)
        val save = stage.getScene.lookup("#embedding-merge-save").asInstanceOf[Button]
        require(save.isDisabled, "Empty input/name must not publish")
        input.setText("'cat' * 2")
        require(save.isDisabled, "Missing output name must not publish")
        val format = stage.getScene.lookup("#embedding-merge-safetensors").asInstanceOf[CheckBox]
        require(format.isSelected)
        println("Embedding Merge native controls passed")
      catch case error: Throwable => failure.set(error)
      finally
        if view != null then view.dispose()
        stage.close()
        services.shutdown()
        done.countDown()
    })
    require(done.await(60, TimeUnit.SECONDS), "Native controls timed out")
    Platform.exit()
    if failure.get() != null then throw failure.get()
