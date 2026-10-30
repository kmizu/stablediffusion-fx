package sdfx.ui.vm

import sdfx.core.config.{AppHome, AtomicFiles}
import sdfx.core.json.SnakePickle
import sdfx.core.json.SnakePickle.{ReadWriter => RW, *}
import sdfx.core.model.GenerationParams

import java.nio.file.{Files, Path}
import scala.util.control.NonFatal

case class WindowState(
    x: Option[Double] = None,
    y: Option[Double] = None,
    width: Double = 1500,
    height: Double = 920,
    maximized: Boolean = false
) derives RW

/** Everything the UI restores on the next start (`config/ui-state.json`). */
case class UiState(
    window: WindowState = WindowState(),
    dividers: Map[String, Seq[Double]] = Map.empty,
    selectedTab: String = "txt2img",
    params: Map[String, GenerationParams] = Map.empty,
    pluginValues: Map[String, Map[String, Map[String, String]]] = Map.empty,
    selectedScript: Map[String, String] = Map.empty,
    expanded: Map[String, Boolean] = Map.empty,
    variationEnabled: Map[String, Boolean] = Map.empty,
    logVisible: Boolean = false,
    galleryQuery: String = "",
    lastSeed: Long = -1L,
    extrasUpscaler: String = "",
    extrasScale: Double = 2.0,
    lastHiresSeed: Option[Long] = None,
    lastHiresSubseed: Option[Long] = None
) derives RW

object UiState:
  def file(home: AppHome): Path = home.configDir.resolve("ui-state.json")

  def toJson(s: UiState): String = SnakePickle.write(s, indent = 2)
  def fromJson(json: String): UiState = SnakePickle.read[UiState](json)

  /** Missing or unreadable files give the defaults. */
  def load(path: Path): UiState =
    if !Files.isRegularFile(path) then UiState()
    else
      try fromJson(Files.readString(path))
      catch
        case NonFatal(e) =>
          System.err.println(s"[sdfx] ui-state unreadable, using defaults: ${e.getMessage}")
          UiState()

  def save(path: Path, s: UiState): Unit = AtomicFiles.writeString(path, toJson(s))
