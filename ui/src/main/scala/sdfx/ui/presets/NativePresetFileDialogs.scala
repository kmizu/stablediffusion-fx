package sdfx.ui.presets

import java.awt.Desktop
import java.nio.file.Path
import javafx.stage.{FileChooser, Window}

/** Only the platform picker and desktop Open actions cross this UI boundary. */
trait NativePresetFileDialogs:
  def chooseOpen(owner: Window): Option[Path]
  def chooseSave(owner: Window): Option[Path]
  def openExternal(path: Path): Unit

object NativePresetFileDialogs:
  val system: NativePresetFileDialogs = new NativePresetFileDialogs:
    def chooseOpen(owner: Window): Option[Path] =
      val chooser = new FileChooser()
      chooser.setTitle("Import preset data")
      Option(chooser.showOpenDialog(owner)).map(_.toPath)

    def chooseSave(owner: Window): Option[Path] =
      val chooser = new FileChooser()
      chooser.setTitle("Export preset data")
      Option(chooser.showSaveDialog(owner)).map(_.toPath)

    def openExternal(path: Path): Unit =
      Desktop.getDesktop.open(path.toFile)
