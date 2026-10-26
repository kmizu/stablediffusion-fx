package sdfx.ui

import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.scene.Node
import javafx.scene.control.*
import javafx.scene.layout.VBox
import javafx.stage.Stage
import sdfx.core.config.{AppHome, Settings}
import sdfx.core.payload.{PayloadKey, PayloadSnapshot, PayloadState}
import sdfx.engine.AppServices
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{Mode, UiState}
import sdfx.ui.view.{PayloadDocument, PayloadHistoryEntry}

import java.nio.file.{Files, Paths}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import javax.imageio.ImageIO

/** Root-only manual smoke for actual controls. No engine start, model load or system clipboard use.
  * Args: output-directory. The broader real dispatch/model acceptance is a separate helper.
  */
object RealRequestPayloadPaneCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body)
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)

  private def eventually(label: String)(condition: => Boolean): Unit =
    val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
    while !condition && System.nanoTime() < deadline do Thread.sleep(20)
    require(condition, s"Timed out: $label")

  def main(args: Array[String]): Unit =
    require(args.length == 1, "output-directory")
    val output = Paths.get(args(0)).toAbsolutePath.normalize()
    Files.createDirectories(output)
    val home = AppHome(Files.createTempDirectory(output, "payload-pane-home-"))
    val services = new AppServices(home, Settings(), dryRun = true,
      builtins = StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false)
    val started = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); started.countDown() })
    require(started.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    var copied = Vector.empty[String]
    var failure: Throwable = null
    def node[A <: Node](id: String): A =
      val stage = window.payloadInspectorStage.getOrElse(sys.error("Inspector stage unavailable"))
      val found = stage.getScene.lookup("#" + id)
      require(found != null, s"Missing native control #$id")
      found.asInstanceOf[A]
    try
      window = fx {
        val result = new MainWindow(new Stage(), services, UiState(), CliOptions(),
          text => copied :+= text)
        result.show()
        result
      }
      fx {
        Mode.values.foreach { mode =>
          window.select(mode)
          val entry = window.scene.lookup(s"#payload-entry-${mode.key}").asInstanceOf[TitledPane]
          require(entry != null && !entry.isExpanded, s"${mode.key} payload section must start collapsed")
        }
        window.select(Mode.Txt2Img)
        window.scene.lookup("#payload-entry-txt2img").asInstanceOf[TitledPane].setExpanded(true)
        window.scene.lookup("#payload-open-txt2img").asInstanceOf[Button].fire()
        require(node[Label]("payload-status").getText.toLowerCase.contains("no payload"))
        require(node[Button]("payload-copy").isDisabled)
        require(!node[CheckBox]("payload-include-images").isSelected)
      }
      require(services.supervisor.pid.isEmpty && services.supervisor.loadedModel.isEmpty,
        "Opening the inspector started an engine or loaded a model")

      val body = "{\"prompt\":\"<script>猫🦊</script>\",\"seed\":42}"
      val context = "{\"submitted\":{\"origin\":\"ui\",\"source_mode\":\"txt2img\",\"configuration\":{\"selected_script\":null}},\"reported_at_capture\":{\"model\":null}}"
      services.payloads.put(PayloadSnapshot(PayloadKey("native-1", 1), "txt2img", Some(body),
        System.currentTimeMillis(), Some(context), Some("{\"assets\":[]}"), PayloadState.Submitted,
        sourceMode = Some("txt2img")))
      eventually("history and raw preview") {
        fx {
          node[ComboBox[PayloadHistoryEntry]]("payload-history").getItems.size == 1 &&
          node[TextArea]("payload-raw-preview").getText == body
        }
      }
      fx {
        require(node[Label]("payload-status").getText.contains("native-1"))
        require(node[Label]("payload-context-facts").getText.contains("Origin: ui"))
        node[Button]("payload-copy").fire()
        require(copied == Vector(body))
        require(node[Label]("payload-copy-feedback").getText == "Copied")
        node[ComboBox[PayloadDocument]]("payload-document").setValue(PayloadDocument.Context)
      }
      eventually("context preview") { fx { node[TextArea]("payload-raw-preview").getText == context } }
      fx {
        node[Button]("payload-copy").fire()
        require(copied == Vector(body, context))
        node[CheckBox]("payload-include-images").fire()
        require(services.settings.apiPayloadIncludeImageContents)
        node[ComboBox[String]]("payload-scope").setValue("All captures")
        require(node[ComboBox[PayloadHistoryEntry]]("payload-history").getItems.size == 1)
        require(node[Label]("payload-rejected").getText.contains("0"))
      }
      val heavyWarnings = Vector.tabulate(32)(i => f"warning-$i%02d " + ("x" * 990))
      services.payloads.put(PayloadSnapshot(PayloadKey("warning-heavy", 1), "txt2img", Some(body),
        System.currentTimeMillis(), Some(context), None, PayloadState.Submitted,
        warnings = heavyWarnings, sourceMode = Some("txt2img")))
      eventually("bounded warning summary") {
        fx { node[Label]("payload-status").getText.contains("32 warnings") }
      }
      for (width, height) <- Seq(1280 -> 800, 1600 -> 1000) do
        fx {
          val inspector = window.payloadInspectorStage.get
          inspector.setWidth(width)
          inspector.setHeight(height)
        }
        eventually(s"${width}x${height} stage resize and client viewport") {
          fx {
            val inspector = window.payloadInspectorStage.get
            val scene = inspector.getScene
            math.abs(inspector.getWidth - width) < 1 && math.abs(inspector.getHeight - height) < 1 &&
              scene.getWidth >= width - 80 && scene.getHeight >= height - 80
          }
        }
        fx {
          val inspector = window.payloadInspectorStage.get
          val scene = inspector.getScene
          scene.getRoot.applyCss()
          scene.getRoot.layout()
          val warningView = node[TextArea]("payload-warnings")
          require(warningView.getText.contains("warning-00") && warningView.getText.contains("warning-31"),
            "Retained warning details are not fully inspectable")
          require(warningView.getHeight > 40 && warningView.getHeight < 150,
            "Warning details must have a bounded scroll viewport")
          val views = node[TabPane]("payload-views")
          views.getSelectionModel.select(1)
          scene.getRoot.layout()
          require(node[TextArea]("payload-raw-preview").getHeight > 160,
            "Raw preview was displaced by warning details")
          val help = node[Label]("payload-help")
          require(help.localToScene(help.getBoundsInLocal).getMaxY <= scene.getHeight,
            "Inspector help was pushed below the stage viewport")
          require(node[Label]("payload-status").getText.length < 1000,
            "Warning summary is too long for the header")
          require(node[ComboBox[String]]("payload-scope").getWidth >= 140,
            "All captures scope is too narrow to read")
          val bitmap = SwingFXUtils.fromFXImage(scene.snapshot(null), null)
          val clientWidth = math.ceil(scene.getWidth).toInt
          val clientHeight = math.ceil(scene.getHeight).toInt
          require(bitmap.getWidth == clientWidth && bitmap.getHeight == clientHeight,
            s"Screenshot ${bitmap.getWidth}x${bitmap.getHeight} differs from settled client ${scene.getWidth}x${scene.getHeight}")
          val screenshot = output.resolve(
            s"payload-warning-window-${width}x${height}-client-${clientWidth}x${clientHeight}.png")
          require(ImageIO.write(bitmap, "png", screenshot.toFile))
          val saved = ImageIO.read(screenshot.toFile)
          require(saved.getWidth == clientWidth && saved.getHeight == clientHeight,
            "Saved PNG dimensions differ from the actual client viewport")
          views.getSelectionModel.select(0)
          scene.getRoot.layout()
          require(node[VBox]("payload-tree-host").getHeight > 160,
            "Tree viewer was displaced by warning details")
        }
      require(services.supervisor.pid.isEmpty && services.supervisor.loadedModel.isEmpty,
        "Inspector interaction started an engine or loaded a model")
      println("Real request payload pane smoke passed: offline controls, history, raw, injected copy, setting")
    catch case error: Throwable => failure = error
    finally
      NativeAcceptanceSafety.cleanup(failure)(
        () => if window != null then fx { window.dispose(); window.stage.close() },
        () => services.shutdown(),
        () => Platform.exit())
    if failure != null then throw failure
