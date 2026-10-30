package sdfx.ui.view

import javafx.collections.{FXCollections, ObservableList}
import javafx.stage.Stage
import sdfx.core.config.AtomicFiles
import sdfx.core.model.{ExtendedStyle, GenerationParams, PromptStyle, StyleLibrary, Styles}
import sdfx.engine.AppServices
import sdfx.ui.vm.{CatalogVM, Mode, PromptInsertionTarget}

import java.nio.file.{Files, Path}
import scala.util.control.NonFatal

/** Shared services and callbacks for the views. */
final class UiContext(val services: AppServices, val catalog: CatalogVM, val stage: Stage):
  val toaster = new Toaster
  val styles: ObservableList[PromptStyle] = FXCollections.observableArrayList()
  private var styleRefresh = 0L
  lazy val styleLibrary = new StyleLibrary(services.home.stylesFile)
  lazy val hardwareInfo = new sdfx.engine.HardwareInfoService(services)
  lazy val tagDictionary = new TagDictionary(this)
  var openStylesEditor: () => Unit = () => ()
  var openModelBrowser: (String, Option[PromptInsertionTarget]) => Unit = (_, _) => ()
  var editPositivePrompt: (String, Boolean) => Unit = (_, _) => ()
  var currentGeneration: () => GenerationParams = () => GenerationParams()
  /** Commits the named mode's existing numeric editor drafts before preset capture/apply. */
  var commitPresetEditors: Mode => Either[String, Unit] =
    _ => Left("Preset destination editors are not attached")
  var applyPromptStyle: PromptStyle => Unit = _ => ()
  var applyExtendedStyle: ExtendedStyle => Unit = _ => ()
  /** Held through the entire model/VAE load and form commit; direct callbacks must share the same guard. */
  val extendedStyleApplying = new javafx.beans.property.SimpleBooleanProperty(false)

  /** Wired by the main window: send an image (+ optional infotext) to a generation tab. */
  var sendToTab: (Mode, Path, Option[String]) => Unit = (_, _, _) => ()
  /** Wired by the main window: show an image in the big preview. */
  var showImage: Path => Unit = _ => ()
  /** Read-only comparison routes, wired by MainWindow. */
  var comparePair: (Path, Path) => Unit = (_, _) => ()
  /** Shared native cancellation route; MainWindow includes its job loop and truthful action feedback. */
  var interruptRuntime: () => Unit = () => services.client.interrupt()

  def info(msg: String): Unit = toaster.show(msg, Toaster.Kind.Info)
  def error(msg: String): Unit = toaster.show(msg, Toaster.Kind.Error)
  def success(msg: String): Unit = toaster.show(msg, Toaster.Kind.Success)

  def tmpDir: Path = services.home.tmpDir(services.settings)
  /** UI temp files (masks, pasted images): job-bound cleanup and sweeps. */
  val temps = new sdfx.ui.vm.TempTracker(() => tmpDir)
  def tmpFile(prefix: String, suffix: String): Path = Files.createTempFile(tmpDir, prefix, suffix)

  def loadStyles(): Unit =
    styleRefresh += 1
    val request = styleRefresh
    sdfx.ui.Fx.async(Styles.load(services.home.stylesFile))(xs => if request == styleRefresh then styles.setAll(xs*))(e => error(s"Cannot read styles.csv: ${e.getMessage}"))

  def saveStyle(s: PromptStyle, overwrite: Boolean = false): Unit =
    sdfx.ui.Fx.async {
      val document = styleLibrary.load()
      val matches = document.entries.filter(e => document.value(e, "name") == s.name)
      require(matches.size <= 1, s"Several existing rows use '${s.name}'. Give them distinct names in Styles before overwriting.")
      val existing = matches.headOption
      require(existing.isEmpty || overwrite, s"Style '${s.name}' already exists. Choose overwrite explicitly.")
      val fields = Map("name" -> s.name, "prompt" -> s.prompt, "negative_prompt" -> s.negativePrompt)
      val changed = existing.fold(document.append(fields))(e => document.edit(e.id, fields))
      styleLibrary.save(changed).promptStyles
    } { saved =>
      styles.setAll(saved*)
      loadStyles()
      success(s"Style '${s.name}' saved")
    }(e => error(s"Cannot save style: ${e.getMessage}"))
