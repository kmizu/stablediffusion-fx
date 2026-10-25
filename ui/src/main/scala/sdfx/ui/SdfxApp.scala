package sdfx.ui

import javafx.animation.PauseTransition
import javafx.application.{Application, Platform}
import javafx.embed.swing.SwingFXUtils
import javafx.stage.Stage
import javafx.util.Duration
import sdfx.core.config.{AppHome, Settings}
import sdfx.engine.{AppServices, PythonEnv}
import sdfx.plugins.StandardPlugins
import sdfx.ui.view.{BootstrapDialog, Theme}
import sdfx.ui.vm.{UiState, VramModes}

import java.nio.file.{Files, Paths}
import javax.imageio.ImageIO
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

/** JavaFX application: first-run setup, main window, engine start and orderly shutdown. */
final class SdfxApp extends Application:
  @volatile private var services: Option[AppServices] = None
  @volatile private var window: Option[MainWindow] = None
  @volatile private var closed = false

  override def start(stage: Stage): Unit =
    val opts = CliOptions.parse(getParameters.getRaw.asScala.toSeq)
    val home = opts.home.map(h => AppHome(Paths.get(h).toAbsolutePath.normalize())).getOrElse(AppHome.detect())
    // legacy VRAM modes (medvram / lowvram / cpu) are mapped to the engine's values
    val settings = VramModes.normalizeSettings(Settings.load(home.settingsFile))
    Platform.setImplicitExit(true)
    val needsSetup = !opts.dryRun && settings.pythonExe.forall(_.trim.isEmpty) &&
      (try PythonEnv.needsBootstrap(home) catch case NonFatal(_) => true)
    if needsSetup then
      val dlg = new BootstrapDialog(home, settings, Theme.stylesheets(settings.theme), Theme.themeClass(settings.theme))
      dlg.show {
        case dlg.Result.Installed    => launch(stage, home, Settings.load(home.settingsFile), dryRun = false, opts)
        case dlg.Result.DryRun       => launch(stage, home, settings, dryRun = true, opts)
        case dlg.Result.UsePython(p) =>
          val s = settings.copy(pythonExe = Some(p))
          try Settings.save(home.settingsFile, s)
          catch case NonFatal(e) => System.err.println(s"[sdfx] cannot save settings: ${e.getMessage}")
          launch(stage, home, s, dryRun = false, opts)
        case dlg.Result.Cancelled => Platform.exit()
      }
      scheduleScreenshot(opts, stage, None)
    else launch(stage, home, settings, opts.dryRun, opts)

  private def launch(stage: Stage, home: AppHome, settings: Settings, dryRun: Boolean, opts: CliOptions): Unit =
    var svc: AppServices = null
    // the X/Y/Z plot sees the axes of every enabled plugin (built-in and external)
    svc = new AppServices(home, settings, dryRun, StandardPlugins.all(() => if svc == null then Nil else svc.registry.xyzAxes))
    services = Some(svc)
    val w = new MainWindow(stage, svc, UiState.load(UiState.file(home)), opts)
    window = Some(w)
    stage.setOnCloseRequest(e =>
      val busy = w.progress.get.running || w.progress.get.pendingCount > 0
      val ok = !busy || {
        val a = new javafx.scene.control.Alert(javafx.scene.control.Alert.AlertType.CONFIRMATION,
          "A generation is still running. Quit anyway? Running and queued jobs are stopped.",
          javafx.scene.control.ButtonType.CANCEL, javafx.scene.control.ButtonType.OK)
        a.initOwner(stage)
        a.setHeaderText(null)
        a.getDialogPane.getStylesheets.setAll(w.scene.getStylesheets)
        val r = a.showAndWait()
        r.isPresent && r.get == javafx.scene.control.ButtonType.OK
      }
      if ok then shutdown() else e.consume()
    )
    w.show()
    w.startEngine { () =>
      // dev flags: paint a sample inpaint mask, then optionally press Generate once
      if opts.demoMask then w.panels(sdfx.ui.vm.Mode.Inpaint).mask.foreach(_.paintDemo())
      if opts.autogen then Fx.later(w.generate(false))
    }
    if opts.tab.exists(_.startsWith("settings")) then
      val open = new PauseTransition(Duration.seconds(1.5))
      open.setOnFinished(_ => Fx.later(w.openSettings(if opts.tab.contains("settings-mcp") then 3 else 0)))
      open.play()
    scheduleScreenshot(opts, stage, Some(w))
    if opts.screenshot.isEmpty then
      opts.exitAfter.foreach { secs =>
        val delay = new PauseTransition(Duration.seconds(secs))
        delay.setOnFinished(_ => shutdown())
        delay.play()
      }

  /** Dev flag `--screenshot <png> [--exit-after <s>]`: snapshot the topmost window after a delay, then exit.
    * With the main window on top, a full-height `<name>-panel.png` of the active generation form is written too.
    */
  private def scheduleScreenshot(opts: CliOptions, main: Stage, w: Option[MainWindow]): Unit =
    opts.screenshot.foreach { file =>
      val delay = new PauseTransition(Duration.seconds(opts.exitAfter.getOrElse(8.0)))
      delay.setOnFinished(_ =>
        try
          val showing = javafx.stage.Window.getWindows.asScala.filter(win => win.isShowing && win.isInstanceOf[Stage])
          val top = showing.filter(win => w.isEmpty || (win ne main)).lastOption
          val scene = top.map(_.getScene).orElse(w.map(_.scene))
          val out = Paths.get(file).toAbsolutePath
          Option(out.getParent).foreach(Files.createDirectories(_))
          scene.foreach(s => ImageIO.write(SwingFXUtils.fromFXImage(s.snapshot(null), null), "png", out.toFile))
          System.err.println(s"[sdfx] screenshot written to $out")
          for mw <- w if top.isEmpty do
            val form = mw.panels(mw.activeMode).content
            val full = out.resolveSibling(out.getFileName.toString.replaceFirst("\\.png$", "") + "-panel.png")
            ImageIO.write(SwingFXUtils.fromFXImage(form.snapshot(null, null), null), "png", full.toFile)
        catch case NonFatal(e) => System.err.println(s"[sdfx] screenshot failed: $e")
        if w.isDefined then shutdown() else { Platform.exit(); System.exit(0) }
      )
      delay.play()
    }

  private def shutdown(): Unit =
    if !closed then
      closed = true
      window.foreach { w =>
        w.persist()
        w.dispose()
      }
      val t = new Thread(() =>
        services.foreach(s =>
          try s.shutdown()
          catch case NonFatal(e) => System.err.println(s"[sdfx] shutdown: ${e.getMessage}")
        )
        Platform.exit()
        System.exit(0)
      , "sdfx-shutdown")
      t.setDaemon(false)
      t.start()

  override def stop(): Unit =
    if !closed then
      closed = true
      window.foreach(_.persist())
      services.foreach(s => try s.shutdown() catch case NonFatal(_) => ())
