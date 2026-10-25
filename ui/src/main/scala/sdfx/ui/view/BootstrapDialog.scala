package sdfx.ui.view

import javafx.geometry.{Insets, Pos}
import javafx.scene.Scene
import javafx.scene.control.*
import javafx.scene.layout.{HBox, Priority, VBox}
import javafx.stage.{FileChooser, Stage}
import sdfx.core.config.{AppHome, Settings}
import sdfx.engine.PythonEnv
import sdfx.ui.Fx

/** First-run setup of the Python environment. Calls `done` with the decision once the user picked a path. */
final class BootstrapDialog(home: AppHome, settings: Settings, stylesheets: Seq[String], themeClass: String):
  enum Result:
    case Installed
    case DryRun
    case UsePython(path: String)
    case Cancelled

  def show(done: Result => Unit): Unit =
    val stage = new Stage()
    stage.setTitle("SDFX — first start")
    var finished = false
    def finish(r: Result): Unit =
      if !finished then
        finished = true
        stage.close()
        done(r)

    val title = W.label("Set up the image generation engine", "dialog-title")
    val text = W.label(
      "SDFX runs Stable Diffusion in a local Python environment (python/.venv).\n" +
        "Install downloads PyTorch with CUDA and diffusers — several GB, this can take 5–20 minutes.\n\n" +
        "• Install — create the environment now (needs Python 3.10–3.12 on PATH)\n" +
        "• Use dry-run — try the UI with placeholder images, no GPU or downloads\n" +
        "• Choose Python… — use an existing interpreter that already has torch + diffusers",
      "dialog-text"
    )
    text.setWrapText(true)
    val log = new TextArea()
    log.setEditable(false)
    log.setWrapText(false)
    log.getStyleClass.add("log-area")
    log.setPrefRowCount(14)
    log.setVisible(false)
    log.setManaged(false)
    val progress = new ProgressBar()
    progress.setMaxWidth(Double.MaxValue)
    progress.setVisible(false)
    val status = W.label("", "hint")

    val install = W.button("⤓ Install", "Create python/.venv and install torch (CUDA) + diffusers", Seq("accent-button"))(())
    val dry = W.button("Use dry-run", "Start with the fake engine (UI only)")(finish(Result.DryRun))
    val choose = W.button("Choose Python…", "Pick python.exe of an environment that already has torch and diffusers")({
      val fc = new FileChooser()
      fc.setTitle("Choose Python executable")
      Option(fc.showOpenDialog(stage)).foreach(f => finish(Result.UsePython(f.getAbsolutePath)))
    })

    var task: Option[sdfx.engine.BootstrapTask] = None
    val cancelBtn = W.button("Cancel", "Stop the installation (kills pip)")(())
    cancelBtn.setVisible(false)
    cancelBtn.setManaged(false)

    install.setOnAction(_ =>
      install.setDisable(true)
      dry.setDisable(true)
      choose.setDisable(true)
      log.setVisible(true)
      log.setManaged(true)
      progress.setVisible(true)
      status.setText("Installing… you can keep this window open, it closes when done.")
      stage.sizeToScene()
      val buffer = new StringBuilder
      @volatile var scheduled = false
      def append(line: String): Unit =
        buffer.synchronized(buffer.append(line).append('\n'))
        if !scheduled then
          scheduled = true
          Fx.later {
            scheduled = false
            val chunk = buffer.synchronized { val s = buffer.toString; buffer.clear(); s }
            log.appendText(chunk)
            if log.getLength > 400000 then log.deleteText(0, log.getLength - 300000)
          }
      val t = PythonEnv.bootstrapAsync(home, append, settings)
      task = Some(t)
      cancelBtn.setVisible(true)
      cancelBtn.setManaged(true)
      Fx.onFx(t.result) { _ =>
        task = None
        progress.setProgress(1)
        finish(Result.Installed)
      } { e =>
        task = None
        cancelBtn.setVisible(false)
        cancelBtn.setManaged(false)
        progress.setVisible(false)
        status.setText(if t.isCancelled then "Installation cancelled. The environment is incomplete; Install again to resume." else s"Installation failed: ${Fx.message(e)}")
        if !t.isCancelled then status.getStyleClass.add("status-error")
        install.setDisable(false)
        install.setText("Retry")
        dry.setDisable(false)
        choose.setDisable(false)
      }
    )
    cancelBtn.setOnAction(_ =>
      task.foreach(_.cancel())
      status.setText("Cancelling…")
    )

    val buttons = new HBox(8, cancelBtn, install, dry, choose)
    buttons.setAlignment(Pos.CENTER_RIGHT)
    val root = new VBox(12, title, text, progress, status, log, buttons)
    VBox.setVgrow(log, Priority.ALWAYS)
    root.setPadding(new Insets(20))
    root.setPrefWidth(720)
    root.getStyleClass.addAll("bootstrap", themeClass)
    val scene = new Scene(root)
    scene.getStylesheets.setAll(stylesheets*)
    stage.setScene(scene)
    // closing the window cancels a running pip instead of orphaning it
    stage.setOnCloseRequest(_ =>
      task.foreach(_.cancel())
      finish(Result.Cancelled)
    )
    stage.show()
