package sdfx.ui

import javafx.application.Platform
import javafx.scene.Scene
import javafx.stage.Stage
import javafx.scene.control.*
import java.nio.file.Files
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import sdfx.core.config.{AppHome, Settings}
import sdfx.engine.AppServices
import sdfx.ui.view.{UiContext, ModelPresetsTab}
import sdfx.ui.vm.{CatalogVM, Mode}

/** Real native controls without loading a model; later MainWindow GPU harness covers inference. */
object RealModelPresetsControlsCheck:
  def main(args: Array[String]): Unit =
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    def fx[A](f: => A): A =
      val task = new FutureTask[A](new Callable[A] { def call() = f })
      Platform.runLater(task); task.get(30, TimeUnit.SECONDS)
    def eventually(f: => Boolean): Unit =
      val end = System.nanoTime() + TimeUnit.SECONDS.toNanos(20)
      while !f && System.nanoTime() < end do Thread.sleep(20)
      require(f, "Native preset condition timed out")
    val root = Files.createTempDirectory("model-preset-controls-")
    val model = root.resolve("model.safetensors")
    Files.write(model, Array[Byte](1, 2, 3))
    val services = new AppServices(AppHome(root), Settings(), dryRun = true, builtins = Nil, persistSettings = false)
    var stage: Stage = null
    var tab: ModelPresetsTab = null
    var applied = ""
    try
      fx {
        stage = new Stage()
        val ctx = new UiContext(services, new CatalogVM, stage)
        tab = new ModelPresetsTab(ctx, () => Some(model), () => Some("portrait\nSteps: 20, Seed: 173"),
          (_, text, _) => applied = text)
        stage.setScene(new Scene(tab.node, 900, 780)); stage.show()
        tab.loadModel(model, autoFetch = false)
      }
      def button(id: String) = stage.getScene.lookup("#" + id).asInstanceOf[Button]
      eventually(fx(!button("model-presets-save").isDisabled))
      fx {
        stage.getScene.lookup("#model-presets-name").asInstanceOf[TextField].setText("native")
        stage.getScene.lookup("#model-presets-editor").asInstanceOf[TextArea].setText("猫\nSteps: 20, Seed: 173")
        button("model-presets-save").fire()
      }
      eventually(fx(!button("model-presets-save").isDisabled))
      fx(button("model-presets-default").fire())
      eventually(fx(!button("model-presets-save").isDisabled))
      fx(button("model-presets-send-txt2img").fire())
      require(applied == "猫\nSteps: 20, Seed: 173")
      fx(tab.loadModel(model, autoFetch = false))
      eventually(fx(!button("model-presets-save").isDisabled))
      require(fx(stage.getScene.lookup("#model-presets-name").asInstanceOf[TextField].getText) == "native")
      fx {
        stage.getScene.lookup("#model-presets-editor").asInstanceOf[TextArea].setText("unsaved draft")
        stage.getScene.lookup("#model-presets-url").asInstanceOf[TextField].setText("https://example.com/model")
        button("model-presets-save-info").fire()
      }
      eventually(fx(!button("model-presets-save").isDisabled))
      require(fx(stage.getScene.lookup("#model-presets-editor").asInstanceOf[TextArea].getText) == "unsaved draft", "Saving model info lost the unsaved prompt")
      fx {
        val choices = stage.getScene.lookup("#model-presets-list").asInstanceOf[ComboBox[String]]
        choices.setValue("default"); choices.setValue("native")
        require(stage.getScene.lookup("#model-presets-editor").asInstanceOf[TextArea].getText == "unsaved draft", "Switching presets lost the draft")
        button("model-presets-share").fire()
        val snapshot = sdfx.core.modelpresets.ModelPresetDocument.parseShared(stage.getScene.lookup("#model-presets-sharing").asInstanceOf[TextArea].getText)
        require(snapshot.presets("native") == "unsaved draft", "Sharing omitted the visible draft")
      }
      fx {
        stage.getScene.lookup("#model-presets-name").asInstanceOf[TextField].setText("renamed")
        button("model-presets-rename").fire()
      }
      eventually(fx(!button("model-presets-save").isDisabled))
      fx {
        button("model-presets-share").fire()
        val text = stage.getScene.lookup("#model-presets-sharing").asInstanceOf[TextArea].getText
        val doc = sdfx.core.modelpresets.ModelPresetDocument.parseShared(text)
        require(doc.defaultPreset == "renamed" && doc.presets("renamed") == "unsaved draft")
        require(!doc.presets.contains("native"))
        button("model-presets-delete").fire()
      }
      eventually(fx(!button("model-presets-save").isDisabled))
      val afterDelete = fx(stage.getScene.lookup("#model-presets-name").asInstanceOf[TextField].getText)
      require(afterDelete == "default", s"Delete fallback has unexpected draft name: $afterDelete")
      if args.length == 2 then
        val first = java.nio.file.Path.of(args(0))
        val second = java.nio.file.Path.of(args(1))
        require(Files.size(first) > 1024L * 1024 * 1024 && Files.size(second) > 1024L * 1024 * 1024)
        fx(tab.loadModel(first, autoFetch = false))
        Thread.sleep(150)
        require(fx(stage.getScene.lookup("#model-presets-editor").isDisabled), "Hash completed before cancellation test")
        fx(button("model-presets-cancel").fire())
        eventually(fx(!stage.getScene.lookup("#model-presets-editor").isDisabled))
        require(fx(stage.getScene.lookup("#model-presets-status").asInstanceOf[Label].getText).toLowerCase.contains("cancel"))
        require(fx(button("model-presets-save").isDisabled), "Cancelled load published a model identity")
        fx(tab.loadModel(first, autoFetch = false))
        Thread.sleep(150)
        fx(tab.loadModel(second, autoFetch = false))
        eventually(fx(!button("model-presets-save").isDisabled))
        require(fx(stage.getScene.lookup("#model-presets-identity").asInstanceOf[Label].getText).startsWith(second.getFileName.toString))
        fx {
          stage.getScene.lookup("#model-presets-editor").asInstanceOf[TextArea].setText("second model only")
          button("model-presets-save").fire()
        }
        eventually(fx(!button("model-presets-save").isDisabled))
        fx(tab.loadModel(first, autoFetch = false))
        eventually(fx(!button("model-presets-save").isDisabled))
        require(fx(stage.getScene.lookup("#model-presets-editor").asInstanceOf[TextArea].getText).isEmpty)
        fx {
          tab.dispose()
          tab = new ModelPresetsTab(new UiContext(services, new CatalogVM, stage), () => Some(second), () => None, (_, _, _) => ())
          stage.setScene(new Scene(tab.node, 900, 780))
          tab.loadModel(second, autoFetch = false)
        }
        eventually(fx(!button("model-presets-save").isDisabled))
        require(fx(stage.getScene.lookup("#model-presets-editor").asInstanceOf[TextArea].getText) == "second model only")
        println("PASS real checkpoint hash busy-cancel, superseded load, model isolation and recreated controls disk reload")
      println("PASS native Model Presets CRUD/default/reload/send/sharing and draft preservation")
    finally
      fx { if tab != null then tab.dispose(); if stage != null then stage.close() }
      services.shutdown(); Platform.exit()
