package sdfx.ui.view

import javafx.collections.{FXCollections, ObservableList}
import javafx.stage.Stage
import sdfx.core.config.AtomicFiles
import sdfx.core.model.{PromptStyle, Styles}
import sdfx.engine.AppServices
import sdfx.ui.vm.{CatalogVM, Mode}

import java.nio.file.{Files, Path}
import scala.util.control.NonFatal

/** Shared services and callbacks for the views. */
final class UiContext(val services: AppServices, val catalog: CatalogVM, val stage: Stage):
  val toaster = new Toaster
  val styles: ObservableList[PromptStyle] = FXCollections.observableArrayList()

  /** Wired by the main window: send an image (+ optional infotext) to a generation tab. */
  var sendToTab: (Mode, Path, Option[String]) => Unit = (_, _, _) => ()
  /** Wired by the main window: show an image in the big preview. */
  var showImage: Path => Unit = _ => ()

  def info(msg: String): Unit = toaster.show(msg, Toaster.Kind.Info)
  def error(msg: String): Unit = toaster.show(msg, Toaster.Kind.Error)
  def success(msg: String): Unit = toaster.show(msg, Toaster.Kind.Success)

  def tmpDir: Path = services.home.tmpDir(services.settings)
  /** UI temp files (masks, pasted images): job-bound cleanup and sweeps. */
  val temps = new sdfx.ui.vm.TempTracker(() => tmpDir)
  def tmpFile(prefix: String, suffix: String): Path = Files.createTempFile(tmpDir, prefix, suffix)

  def loadStyles(): Unit =
    sdfx.ui.Fx.async(Styles.load(services.home.stylesFile))(xs => styles.setAll(xs*))(e => error(s"Cannot read styles.csv: ${e.getMessage}"))

  def saveStyle(s: PromptStyle): Unit =
    val idx = (0 until styles.size).find(i => styles.get(i).name == s.name)
    idx match
      case Some(i) => styles.set(i, s)
      case None    => styles.add(s)
    val csv = Styles.toCsv(styles.toArray(Array.empty[PromptStyle]).toSeq)
    sdfx.ui.Fx.async(AtomicFiles.writeString(services.home.stylesFile, csv))(_ => ())(e => error(s"Cannot write styles.csv: ${e.getMessage}"))
