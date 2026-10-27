package sdfx.ui

import java.nio.file.{Files, Paths}
import java.util.concurrent.{CountDownLatch, TimeUnit}
import java.util.concurrent.atomic.AtomicReference
import javafx.application.Platform
import javafx.beans.property.SimpleStringProperty
import javafx.embed.swing.SwingFXUtils
import javafx.scene.Scene
import javafx.scene.control.{Button, DialogPane, Label}
import javafx.scene.layout.StackPane
import javafx.stage.{Stage, Window}
import javax.imageio.ImageIO
import scala.jdk.CollectionConverters.*
import sdfx.core.config.Settings
import sdfx.ui.view.SettingsDialog

/** Actual Settings dialog. Chooser results are injected at the property boundary;
  * the OS file chooser itself is not exercised. Settings live under a fresh
  * temporary home. No worker or model is loaded.
  */
object RealWdSettingsCheck:
  private def property(instance: AnyRef, suffix: String): SimpleStringProperty =
    val fields = instance.getClass.getDeclaredFields.filter(_.getName.endsWith(suffix))
    require(fields.length == 1, s"Expected one $suffix field")
    fields.head.setAccessible(true)
    fields.head.get(instance).asInstanceOf[SimpleStringProperty]

  def main(args: Array[String]): Unit =
    require(args.length == 1, "artifact output directory required")
    val output = Paths.get(args(0)).toAbsolutePath.normalize()
    Files.createDirectories(output)
    val run = Files.createTempDirectory(output, "wd-settings-native-")
    val home = Files.createDirectories(run.resolve("home"))
    val model = home.resolve("selected model.onnx")
    val tags = home.resolve("selected tags.csv")
    Files.writeString(model, "synthetic file, no model")
    Files.writeString(tags, "synthetic file, no tags")
    val done = new CountDownLatch(1)
    val failure = new AtomicReference[Throwable]()
    Platform.startup(() => {
      Platform.setImplicitExit(false)
      val owner = new Stage()
      owner.setScene(new Scene(new StackPane(), 640, 480))
      owner.show()
      val initial = Settings(lastModel = Some("preserved checkpoint"))
      val dialog = new SettingsDialog(owner, initial, 4)
      Platform.runLater(() => {
        try
          val stage = Window.getWindows.asScala.collectFirst {
            case s: Stage if s.isShowing && s.getTitle == "Settings" => s
          }.getOrElse(throw new IllegalStateException("Settings window missing"))
          val pane = stage.getScene.getRoot.asInstanceOf[DialogPane]
          def status: String = pane.lookup("#wd-v3-status").asInstanceOf[Label].getText
          def screenshot(name: String): Unit =
            pane.applyCss(); pane.layout()
            require(ImageIO.write(SwingFXUtils.fromFXImage(stage.getScene.snapshot(null), null), "png", run.resolve(name + ".png").toFile))
          require(status == "Not configured")
          screenshot("01-empty")
          property(dialog, "wdModel").set(model.toString)
          require(status.startsWith("Incomplete"))
          screenshot("02-incomplete")
          property(dialog, "wdTags").set(tags.toString)
          require(status == "Files selected")
          require(pane.lookup("#wd-v3-worker-status").asInstanceOf[Label].getText.contains("unknown"))
          stage.setWidth(520); stage.setHeight(440)
          screenshot("03-selected-narrow")
          pane.lookupButton(javafx.scene.control.ButtonType.OK).asInstanceOf[Button].fire()
        catch
          case e: Throwable =>
            failure.set(e)
            Window.getWindows.asScala.collect { case s: Stage if s.getTitle == "Settings" => s }.foreach(_.close())
      })
      try
        val result = dialog.showAndWait().getOrElse(throw new IllegalStateException("Settings save cancelled"))
        require(result.wdV3Model.contains(model.toString) && result.wdV3Tags.contains(tags.toString))
        require(result.lastModel == initial.lastModel)
        val path = home.resolve("settings.json")
        Settings.save(path, result)
        val reloaded = Settings.load(path)
        require(reloaded.wdV3Model == result.wdV3Model && reloaded.wdV3Tags == result.wdV3Tags)
        println(s"WD_SETTINGS_QA_OK artifacts=$run chooser=property-injected worker=not-started")
      catch case e: Throwable => failure.compareAndSet(null, e)
      finally
        owner.close()
        done.countDown()
    })
    require(done.await(60, TimeUnit.SECONDS), "Settings QA timed out")
    Platform.exit()
    if failure.get() != null then throw failure.get()
