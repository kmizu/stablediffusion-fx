package sdfx.ui

import java.nio.file.{Files, Path, Paths}
import java.security.MessageDigest
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import javafx.application.Platform
import javafx.collections.FXCollections
import javafx.embed.swing.SwingFXUtils
import javafx.event.Event
import javafx.scene.control.{Button, DialogPane, TabPane, TextArea}
import javafx.scene.input.{InputMethodEvent, InputMethodHighlight, InputMethodTextRun, KeyCode, KeyEvent}
import javafx.stage.{PopupWindow, Stage, Window}
import javax.imageio.ImageIO
import scala.jdk.CollectionConverters.*
import sdfx.core.config.{AppHome, Settings}
import sdfx.core.infotext.Infotext
import sdfx.core.png.PngMetadata
import sdfx.engine.AppServices
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{Mode, UiState}

/** Actual MainWindow/scene with local CSV and synthetic IME events. No model/worker needed.
  * OS chooser result is injected at the persisted settings boundary. Optional PNG is read only.
  */
object RealTagAutocompleteCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A] { def call(): A = body })
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)

  private def awaitFx(label: String)(condition: => Boolean): Unit =
    val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(15)
    while !fx(condition) && System.nanoTime() < deadline do Thread.sleep(40)
    require(fx(condition), s"Timed out: $label")

  def main(args: Array[String]): Unit =
    require(args.length == 1 || args.length == 2, "artifact directory [existing WD PNG] required")
    val out = Paths.get(args(0)).toAbsolutePath.normalize()
    Files.createDirectories(out)
    val run = Files.createTempDirectory(out, "tag-mainwindow-")
    val home = AppHome(Files.createDirectories(run.resolve("home")))
    val csv = run.resolve("local-tags.csv")
    Files.writeString(csv, "blue_hair,0,20,azure_hair\nblue_eyes,0,10,\n猫,0,1,\n")
    Settings.save(home.settingsFile, Settings(tagAutocompleteCsv = Some(csv.toString)))
    val services = new AppServices(home, Settings.load(home.settingsFile), dryRun = true,
      builtins = StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false)
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS), "JavaFX startup timed out")
    var window: MainWindow = null
    def shot(name: String): Unit =
      window.scene.getRoot.applyCss(); window.scene.getRoot.layout()
      require(ImageIO.write(SwingFXUtils.fromFXImage(window.scene.snapshot(null), null), "png", run.resolve(name + ".png").toFile))
    def key(area: TextArea, code: KeyCode, control: Boolean = false, released: Boolean = false): Unit =
      Event.fireEvent(area, new KeyEvent(if released then KeyEvent.KEY_RELEASED else KeyEvent.KEY_PRESSED,
        "", "", code, false, control, false, false))
    try
      window = fx {
        val value = new MainWindow(new Stage(), services, UiState(), CliOptions(), _ => ())
        value.show(); value.select(Mode.Txt2Img)
        value.stage.setWidth(1100); value.stage.setHeight(800)
        value.panels(Mode.Txt2Img).prompts.promptArea.requestFocus()
        value
      }
      awaitFx("CSV load") { window.ctx.tagDictionary.tags.size == 3 }
      val editor = fx { window.panels(Mode.Txt2Img).prompts }
      val area = editor.promptArea
      awaitFx("prompt focus") { area.isFocused }
      fx {
        require(services.settings.tagAutocompleteCsv.contains(csv.toString))
        area.setText("日本語, (blue:1.2), unchanged")
        area.positionCaret(area.getText.indexOf("blue") + 4)
        require(editor.completionVisible)
        shot("01-prompt-popup")
        Window.getWindows.asScala.collectFirst { case p: PopupWindow if p.isShowing => p }.foreach { popup =>
          require(ImageIO.write(SwingFXUtils.fromFXImage(popup.getScene.snapshot(null), null), "png", run.resolve("01-popup-detail.png").toFile))
        }
        key(area, KeyCode.DOWN); key(area, KeyCode.ENTER)
        require(area.getText == "日本語, (blue eyes:1.2), unchanged", area.getText)
        require(!editor.completionVisible && services.queue.isIdle)
        area.setText("blue, keep"); area.positionCaret(4)
        key(area, KeyCode.ESCAPE)
        require(!editor.completionVisible && area.getText == "blue, keep")
        area.setText("blue, IME"); area.positionCaret(4)
        Event.fireEvent(area, new InputMethodEvent(InputMethodEvent.INPUT_METHOD_TEXT_CHANGED,
          FXCollections.observableArrayList(new InputMethodTextRun("あ", InputMethodHighlight.UNSELECTED_RAW)), "", 1))
        require(editor.isComposingInput && !editor.completionVisible)
        val composedText = area.getText
        key(area, KeyCode.ENTER, control = true)
        require(services.queue.isIdle && area.getText == composedText, "Scene Ctrl+Enter escaped IME guard")
        Event.fireEvent(area, new InputMethodEvent(InputMethodEvent.INPUT_METHOD_TEXT_CHANGED,
          FXCollections.observableArrayList[InputMethodTextRun](), "あ", 0))
        require(editor.isComposingInput, "Commit-key guard missing")
        key(area, KeyCode.ENTER, control = true)
        require(services.queue.isIdle, "Scene Ctrl+Enter escaped commit guard")
        key(area, KeyCode.ENTER, released = true)
        require(!editor.isComposingInput)
        area.setText("blue, after IME"); area.positionCaret(4)
        key(area, KeyCode.TAB)
        require(area.getText == "blue hair, after IME", area.getText)
        editor.negativeArea.requestFocus()
      }
      awaitFx("negative prompt focus") { editor.negativeArea.isFocused }
      fx {
        val negative = editor.negativeArea
        negative.setText("bad, blue, 日本語"); negative.positionCaret(9)
        require(editor.completionVisible)
        key(negative, KeyCode.ENTER)
        require(negative.getText == "bad, blue hair, 日本語", negative.getText)
        shot("02-completed-prompts")
        require(services.queue.isIdle && services.supervisor.pid.isEmpty, "Unexpected generation")
      }
      // Real configuration dialog, OS chooser remains deliberately unautomated.
      Platform.runLater(() => editor.toolbar.getChildren.asScala.collectFirst {
        case b: Button if b.getText == "Tags…" => b
      }.get.fire())
      awaitFx("Tags dialog") { Window.getWindows.asScala.exists { case s: Stage => s.getTitle == "Tag autocomplete"; case _ => false } }
      fx {
        val dialog = Window.getWindows.asScala.collectFirst { case s: Stage if s.getTitle == "Tag autocomplete" => s }.get
        require(ImageIO.write(SwingFXUtils.fromFXImage(dialog.getScene.snapshot(null), null), "png", run.resolve("03-config-dialog.png").toFile))
        dialog.getScene.getRoot.asInstanceOf[DialogPane].lookupButton(javafx.scene.control.ButtonType.CANCEL).asInstanceOf[Button].fire()
      }
      args.lift(1).foreach { value =>
        val png = Paths.get(value).toAbsolutePath.normalize()
        def hash: Vector[Byte] = MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(png)).toVector
        val originalHash = hash
        val text = PngMetadata.readParameters(png).getOrElse(throw new IllegalArgumentException("PNG has no parameters"))
        val parsed = Infotext.parse(text)
        require(parsed.fields.contains("WD interrogators"), "Expected a WD output PNG")
        fx {
          window.select(Mode.Img2Img)
          window.vms(Mode.Img2Img).initImage.set("user-selected-init.png")
          val tabs = window.scene.lookup("#main-tabs").asInstanceOf[TabPane]
          tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getText == "PNG Info").get)
          window.pngInfo.image.set(png.toString)
        }
        awaitFx("PNG Info read") { window.pngInfo.node.lookupAll(".infotext").asScala.collect {
          case t: TextArea => t.getText
        }.exists(_ == text) }
        fx {
          shot("04-png-info-reopened")
          window.restore(text)
          val p = window.vms(Mode.Img2Img).toParams
          require(p.prompt == parsed.prompt && p.negativePrompt == parsed.negativePrompt)
          require(p.seed == parsed.fields("Seed").toLong)
          require(p.img2img.exists(_.initImage == "user-selected-init.png"))
          // Provenance stays in PNG Info; form Restore intentionally drops opaque extras.
          shot("05-png-restored")
        }
        require(hash == originalHash, "Read/Restore changed source PNG")
      }
      fx { require(services.queue.isIdle && services.supervisor.pid.isEmpty) }
      println(s"TAG_MAINWINDOW_QA_OK artifacts=$run chooser=settings-injected IME=synthetic optionalPNG=${args.length == 2}")
    finally
      if window != null then fx { window.dispose(); window.stage.close() }
      services.shutdown()
      Platform.exit()
