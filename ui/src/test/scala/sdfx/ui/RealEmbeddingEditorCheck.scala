package sdfx.ui

import javafx.application.Platform
import javafx.scene.{Node, Parent, Scene}
import javafx.scene.control.{Button, CheckBox, Label, Slider, TabPane, TextField}
import javafx.stage.Stage
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.engine.AppServices
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{UiState, Mode}
import java.nio.file.{Files, Path}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import scala.concurrent.Promise
import scala.jdk.CollectionConverters.*

/** Native mounting and worker error path in dry-run; no model accuracy claim. */
object RealEmbeddingEditorCheck:
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
        tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getId == "embedding-editor-tab").get)
        find("embedding-editor-name").asInstanceOf[TextField].setText("cat")
        find("embedding-editor-open").asInstanceOf[Button].fire()
        require(find("embedding-editor-name").isDisabled && find("embedding-editor-open").isDisabled,
          "editor controls must be disabled until the queued response completes")
      }
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while fx(find("embedding-editor-status").asInstanceOf[Label].getText).startsWith("Queued") && System.nanoTime() < deadline do Thread.sleep(50)
      val status = fx(find("embedding-editor-status").asInstanceOf[Label].getText)
      require(status.contains("real loaded model"), s"unexpected status: $status")
      require(fx(!find("embedding-editor-name").isDisabled && !find("embedding-editor-open").isDisabled),
        "editor controls must be restored after the failed request")
      val openReply = Promise[ujson.Value]()
      val guidanceReply = Promise[ujson.Value]()
      val saveReply = Promise[ujson.Value]()
      var testStage: Stage = null
      var testRoot: Parent = null
      def findTest(id: String): Node = nodes(testRoot).find(_.getId == id).getOrElse(sys.error(s"missing test $id"))
      try
        fx {
          val panel = new sdfx.ui.view.EmbeddingEditorTab(window.ctx, _ => (), Some(params =>
            params("operation").str match
              case "open" => openReply.future
              case "guidance" => guidanceReply.future
              case _ => saveReply.future))
          testRoot = panel.node
          testStage = new Stage(); testStage.setScene(new Scene(panel.node, 900, 700)); testStage.show()
          findTest("embedding-editor-name").asInstanceOf[TextField].setText("cat")
          findTest("embedding-editor-open").asInstanceOf[Button].fire()
          require(findTest("embedding-editor-name").isDisabled, "open must lock editor controls")
        }
        openReply.success(ujson.Obj("name" -> "cat", "model_token" -> "model", "path" -> "file",
          "file_identity" -> ujson.Arr("identity"), "vectors" -> ujson.Obj("clip_l" -> ujson.Arr(ujson.Arr(1.0))),
          "ranges" -> ujson.Obj("clip_l" -> ujson.Arr(ujson.Arr(0.0, 2.0))), "guidance" -> ujson.Arr()))
        val openDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
        while fx(findTest("embedding-editor-name").isDisabled) && System.nanoTime() < openDeadline do Thread.sleep(20)
        fx {
          findTest("embedding-editor-coordinate-0").asInstanceOf[Slider].setValue(1.25)
          findTest("embedding-editor-guidance").asInstanceOf[TextField].setText("apple:#ff0000")
          findTest("embedding-editor-update-guidance").asInstanceOf[Button].fire()
          require(findTest("embedding-editor-name").isDisabled, "guidance request must own the editor")
        }
        guidanceReply.success(ujson.Obj("ranges" -> ujson.Obj("clip_l" -> ujson.Arr(ujson.Arr(0.0, 2.0))),
          "guidance" -> ujson.Arr(ujson.Obj("text" -> "apple", "color" -> "#FF0000",
            "markers" -> ujson.Obj("clip_l" -> ujson.Arr(0.5))))))
        val guidanceDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
        while fx(findTest("embedding-editor-name").isDisabled) && System.nanoTime() < guidanceDeadline do Thread.sleep(20)
        fx {
          val slider = findTest("embedding-editor-coordinate-0").asInstanceOf[Slider]
          require(slider.getValue == 1.25, "guidance refresh discarded unsaved coordinate")
          val bar = findTest("embedding-editor-guidance-bar-0").asInstanceOf[javafx.scene.layout.Pane]
          require(bar.getChildren.size == 1, "guidance marker missing")
          val entry = findTest("embedding-editor-coordinate-value-0").asInstanceOf[TextField]
          entry.requestFocus(); entry.setText("1.5")
          findTest("embedding-editor-name").requestFocus()
          require(slider.getValue == 1.5, "focus loss discarded typed coordinate")
          entry.requestFocus(); entry.setText("NaN")
          findTest("embedding-editor-name").requestFocus()
          findTest("embedding-editor-overwrite").asInstanceOf[CheckBox].setSelected(true)
          findTest("embedding-editor-save").asInstanceOf[Button].fire()
          require(!slider.isDisabled && slider.getValue == 1.5, "invalid coordinate must not queue save")
          entry.setText("1.75"); entry.fireEvent(new javafx.event.ActionEvent())
          slider.setValue(1.75)
          findTest("embedding-editor-save").asInstanceOf[Button].fire()
          require(slider.isDisabled, "save must lock edited controls while queued")
        }
        saveReply.failure(new RuntimeException("injected save failure"))
        val saveDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
        while fx(findTest("embedding-editor-name").isDisabled) && System.nanoTime() < saveDeadline do Thread.sleep(20)
        fx {
          val slider = findTest("embedding-editor-coordinate-0").asInstanceOf[Slider]
          require(!slider.isDisabled && slider.getValue == 1.75, "failed save lost unsaved edits")
          testRoot.applyCss(); testRoot.layout()
          javax.imageio.ImageIO.write(javafx.embed.swing.SwingFXUtils.fromFXImage(testStage.getScene.snapshot(null), null),
            "png", root.resolve("embedding-editor-guidance-fixture.png").toFile)
        }
      finally if testStage != null then fx { testStage.close() }
      fx {
        window.stage.setWidth(1500); window.stage.setHeight(900)
        window.root.applyCss(); window.root.layout()
        javax.imageio.ImageIO.write(javafx.embed.swing.SwingFXUtils.fromFXImage(window.scene.snapshot(null), null),
          "png", root.resolve("embedding-editor-wide.png").toFile)
      }
      // Reject incompatible editor previews before exporting a private mask.
      val input = root.resolve("editor-admission-image.png")
      javax.imageio.ImageIO.write(new java.awt.image.BufferedImage(64, 64, java.awt.image.BufferedImage.TYPE_INT_RGB), "png", input.toFile)
      fx {
        window.select(Mode.Inpaint)
        window.vms(Mode.Inpaint).initImage.set(input.toString)
        window.plugins(Mode.Inpaint).selectedScript.set("prompt_matrix")
      }
      val maskDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(20)
      while !fx { window.panels(Mode.Inpaint).mask.get.paintDemo(); window.vms(Mode.Inpaint).hasMask.get } && System.nanoTime() < maskDeadline do Thread.sleep(20)
      require(fx(window.vms(Mode.Inpaint).hasMask.get), "inpaint fixture was not ready")
      def masks(): Set[Path] =
        val stream = Files.list(window.ctx.tmpDir)
        try stream.iterator().asScala.filter(_.getFileName.toString.startsWith("mask-")).toSet
        finally stream.close()
      val beforeMasks = masks()
      fx { window.generate(false, Some("{}")) }
      def refused: Boolean = fx(nodes(window.root).exists {
        case label: Label => label.getText.contains("Embedding editor preview cannot run with a script")
        case _ => false
      })
      val refusalDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(20)
      while !refused && System.nanoTime() < refusalDeadline do Thread.sleep(20)
      require(refused, "incompatible preview was not refused")
      require(masks() == beforeMasks, "rejected preview leaked an exported mask")
      require(fx(window.vms(Mode.Inpaint).maskImage.get.isEmpty), "rejected preview retained an exported mask path")
      require(service.queue.isIdle, "rejected editor preview entered the queue")
      val lateMask = new CountDownLatch(1)
      fx {
        window.vms(Mode.Inpaint).maskImage.addListener((_, _, path) => if path.nonEmpty then lateMask.countDown())
        window.plugins(Mode.Inpaint).selectedScript.set("")
        window.generate(false, Some("{}"))
        window.plugins(Mode.Inpaint).selectedScript.set("prompt_matrix")
      }
      require(lateMask.await(20, TimeUnit.SECONDS), "late-selection fixture did not export a mask")
      fx { require(window.vms(Mode.Inpaint).maskImage.get.isEmpty, "late refusal retained mask path") }
      require(masks() == beforeMasks, "script selection during export leaked an owned mask")
      require(service.queue.isIdle, "late script selection admitted an editor preview")
      println("PASS mounted editor: queued dry-run refusal and pre-export preview admission")
    finally
      if window != null then fx { window.dispose(); window.stage.close() }
      service.shutdown(); Platform.exit()
