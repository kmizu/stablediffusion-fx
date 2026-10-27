package sdfx.ui.vm

import sdfx.core.config.Settings

import java.nio.file.{Files, Path, Paths}
import java.util.concurrent.ConcurrentHashMap
import scala.jdk.CollectionConverters.*
import scala.util.Using
import scala.util.control.NonFatal

/** Temp files the UI creates in `AppHome.tmpDir` (masks, pasted / dropped images).
  * Masks are tied to the job that uses them and deleted when it ends; pasted images are swept at start / exit
  * unless still referenced by the form state.
  */
final class TempTracker(dir: () => Path):
  private val jobFiles = new ConcurrentHashMap[String, Seq[Path]]()

  def isUiTemp(p: Path): Boolean =
    val name = Option(p.getFileName).map(_.toString).getOrElse("")
    TempTracker.Prefixes.exists(name.startsWith) &&
      (try p.toAbsolutePath.normalize().getParent == dir().toAbsolutePath.normalize()
      catch case NonFatal(_) => false)

  def registerJob(jobId: String, files: Seq[Path]): Unit = if files.nonEmpty then jobFiles.merge(jobId, files, _ ++ _)

  /** Deletes the files registered for a finished / failed / cancelled job. */
  def releaseJob(jobId: String): Int =
    Option(jobFiles.remove(jobId)).toSeq.flatten.count(delete)

  def pendingJobs: Set[String] = jobFiles.keySet.asScala.toSet

  /** Deletes UI temp files in the temp dir that are not in `keep` (absolute path strings) and not held by a job. */
  def sweep(keep: Set[String]): Int =
    val d = dir()
    if !Files.isDirectory(d) then 0
    else
      val held = jobFiles.values.asScala.flatten.map(norm).toSet
      val keepN = keep.filter(_.nonEmpty).map(s => norm(Paths.get(s)))
      Using.resource(Files.list(d))(_.iterator().asScala.toList)
        .filter(isUiTemp)
        .filterNot(p => keepN.contains(norm(p)) || held.contains(norm(p)))
        .count(delete)

  private def norm(p: Path): String = p.toAbsolutePath.normalize().toString.toLowerCase
  private def delete(p: Path): Boolean =
    try Files.deleteIfExists(p)
    catch case NonFatal(_) => false

object TempTracker:
  val Prefixes: Seq[String] = Seq("mask-", "paste-", "drop-")

/** Keeps the MCP HTTP server in line with the settings: (re)start on enable / port / token change, stop on disable. */
final class McpController(start: (Int, Option[String]) => AutoCloseable):
  private var running: Option[(Int, Option[String], AutoCloseable)] = None

  def isRunning: Boolean = running.isDefined
  def port: Option[Int] = running.map(_._1)
  def handle: Option[AutoCloseable] = running.map(_._3)

  /** Applies the settings; returns true when something was started or stopped. Throws when starting fails. */
  def apply(enabled: Boolean, port: Int, token: Option[String]): Boolean = synchronized {
    val tok = token.map(_.trim).filter(_.nonEmpty)
    running match
      case Some((p, t, _)) if enabled && p == port && t == tok => false
      case None if !enabled                                    => false
      case _ =>
        close()
        if enabled then running = Some((port, tok, start(port, tok)))
        true
  }

  def close(): Unit = synchronized {
    running.foreach((_, _, h) =>
      try h.close()
      catch case NonFatal(_) => ()
    )
    running = None
  }

object SettingsMerge:
  /** Applies only the fields the settings dialog edits, so values changed meanwhile (lastModel...) survive. */
  def apply(current: Settings, edited: Settings): Settings =
    current.copy(
      modelDirs = edited.modelDirs,
      outputDir = edited.outputDir,
      pythonExe = edited.pythonExe,
      dtype = edited.dtype,
      vramMode = edited.vramMode,
      vaeTiling = edited.vaeTiling,
      previewEvery = edited.previewEvery,
      previewMethod = edited.previewMethod,
      hfOffline = edited.hfOffline,
      mcpHttpEnabled = edited.mcpHttpEnabled,
      mcpHttpPort = edited.mcpHttpPort,
      mcpToken = edited.mcpToken,
      theme = edited.theme,
      nsfwCardBlurDefault = edited.nsfwCardBlurDefault,
      modelPreviewFolders = edited.modelPreviewFolders
    )

/** At most one asynchronous operation at a time (e.g. mask export before submit). The flag is always released:
  * by the `finish` callback, or when `start` throws synchronously (then `onError` gets the exception).
  */
final class SingleFlight:
  private var busy = false
  def isBusy: Boolean = busy

  /** Returns false (and does nothing) while another run is in flight. */
  def run(start: (() => Unit) => Unit, onError: Throwable => Unit): Boolean =
    if busy then false
    else
      busy = true
      var finished = false
      val finish = () => if !finished then { finished = true; busy = false }
      try start(finish)
      catch
        case NonFatal(e) =>
          finish()
          onError(e)
      true

/** VRAM modes the Python engine accepts, with UI labels; legacy A1111-style values are mapped. */
object VramModes:
  val Modes: Seq[(String, String)] = Seq(
    "gpu" -> "GPU (fastest)",
    "model_offload" -> "Model offload (~8 GB VRAM)",
    "sequential_offload" -> "Sequential offload (lowest VRAM)"
  )
  def normalize(v: String): String = Option(v).map(_.trim.toLowerCase).getOrElse("") match
    case "medvram"                    => "model_offload"
    case "lowvram" | "cpu"            => "sequential_offload"
    case x if Modes.exists(_._1 == x) => x
    case _                            => "gpu"
  def index(v: String): Int = Modes.indexWhere(_._1 == normalize(v))
  def normalizeSettings(s: Settings): Settings = s.copy(vramMode = normalize(s.vramMode))

/** Idempotent cleanup which attempts every owned resource even when one release fails. */
private[ui] final class UiCleanup:
  private var done = false
  def run(steps: Seq[(String, () => Unit)]): Unit = synchronized {
    if !done then
      done = true
      steps.foreach { (name, close) =>
        try close()
        catch case NonFatal(e) => System.err.println(s"[sdfx] $name cleanup failed: ${e.getMessage}")
      }
  }

/** Process-owned final cleanup. Scheduling is nonjoining; every finalizer step is attempted once. */
private[ui] final class FinalAppCleanup(loader: () => Unit, schedule: (() => Unit) => Unit,
    services: () => Unit, exit: () => Unit):
  private val start = new UiCleanup
  private val finish = new UiCleanup
  private def finalizeProcess(): Unit = finish.run(Seq(
    "application services" -> services,
    "application exit" -> exit
  ))
  def run(): Unit = start.run(Seq(
    "process preview loader" -> loader,
    "application finalizer scheduling" -> (() =>
      try schedule(() => finalizeProcess())
      catch
        case NonFatal(e) =>
          System.err.println(s"[sdfx] cannot schedule shutdown: ${e.getMessage}")
          val fallback = new Thread(() => finalizeProcess(), "sdfx-shutdown-fallback")
          fallback.setDaemon(false)
          fallback.start())
  ))
