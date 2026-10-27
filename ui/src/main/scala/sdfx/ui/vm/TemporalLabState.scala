package sdfx.ui.vm

import java.nio.file.Path
import scala.collection.mutable
import sdfx.core.protocol.*

/** Pure validation shared by the native workspace controls. Paths are only interpreted by the worker. */
object TemporalLabForms:
  private def paths(xs: String*): Boolean = xs.forall(s => s != null && s.trim.nonEmpty)
  private def validName(s: String): Boolean =
    s != null && s.nonEmpty && s != "." && s != ".." && !s.exists(c => "\\/:*?\"<>|".contains(c) || c < ' ')
  private def finiteRange(x: Double, lo: Double, hi: Double): Boolean = x.isFinite && x >= lo && x <= hi
  private def options(d: TemporalLabDfi): Boolean =
    d.sourceDenoise >= 0 && d.sourceDenoise <= 30 && finiteRange(d.tolerance, .5, 20) &&
      d.expand >= 0 && d.expand <= 50 && d.smooth >= 1 && d.smooth <= 99 && d.smooth % 2 == 1 &&
      d.interDenoise >= 1 && d.interDenoise <= 25 && d.interDenoiseSize >= 1 && d.interDenoiseSize <= 25 && d.interDenoiseSize % 2 == 1 &&
      d.interDenoiseFrequency >= 1 && d.interDenoiseFrequency <= 15 && finiteRange(d.interBlur, 1, 5) &&
      d.refreshFrequency >= 0 && d.refreshFrequency <= 30 && finiteRange(d.preserve, 0, 100) &&
      d.frameLimit >= 0 && d.frameLimit <= 10000
  def dfi(d: TemporalLabDfi): Either[String, TemporalLabDfi] =
    if !paths(d.referenceDir, d.generatedDir, d.outputParent) || !validName(d.outputName) then Left("Select both input folders, an output folder and a valid new output name")
    else if !options(d) then Left("Invalid DFI controls") else Right(d)
  def preview(d: TemporalLabDfi, frame: Int): Either[String, TemporalLabPreview] =
    if !paths(d.referenceDir, d.generatedDir) || frame < 1 || !options(d) then Left("Select both sequences and a valid preview frame and controls")
    else Right(TemporalLabPreview(d.referenceDir, d.generatedDir, frame, d.sourceDenoise, d.tolerance, d.expand, d.smooth))
  def deflicker(d: TemporalLabDeflicker): Either[String, TemporalLabDeflicker] =
    if !paths(d.framesDir, d.outputParent) || !validName(d.outputName) || !finiteRange(d.blend, 0, 1) || !finiteRange(d.overlay, 0, 1) then Left("Invalid deflicker settings") else Right(d)
  def fuse(d: TemporalLabFuse): Either[String, TemporalLabFuse] =
    if !paths(d.styleDir, d.videoDir, d.outputParent) || !validName(d.outputName) || !finiteRange(d.strength, .1, 1) then Left("Invalid fusion settings") else Right(d)
  def extract(d: TemporalLabExtract): Either[String, TemporalLabExtract] =
    if !paths(d.videoPath, d.outputParent) || !validName(d.outputName) || !finiteRange(d.fps, 0, 1000) then Left("Invalid video extraction settings") else Right(d)

/** Owned by the JavaFX thread; requests and revisions are immutable. */
final class TemporalLabState:
  final case class Ticket(id: Long, revision: Long, request: TemporalLabRequest)
  private var serial = 0L
  private var revision = 0L
  private var closed = false
  private var current: Option[Ticket] = None
  private var value: Option[TemporalLabResult] = None
  private var text = ""
  private var fraction = 0.0
  def busy: Boolean = current.nonEmpty
  def result: Option[TemporalLabResult] = value
  def status: String = text
  def progressFraction: Double = fraction
  def start(request: TemporalLabRequest): Either[String, Ticket] =
    if closed then Left("Temporal Lab is closed")
    else if busy then Left("Temporal Lab is already running")
    else
      serial += 1
      val t = Ticket(serial, revision, request)
      current = Some(t); fraction = 0; text = "Starting"; value = None
      Right(t)
  def progress(ticket: Ticket, done: Int, total: Int): Unit =
    if !closed && current.contains(ticket) && ticket.revision == revision && total > 0 then
      fraction = (done.toDouble / total).max(0).min(1)
      text = s"Processing $done / $total"
  def invalidate(): Unit = if !closed then { revision += 1; value = None }
  def complete(ticket: Ticket, result: TemporalLabResult): Unit =
    if !closed && current.contains(ticket) then
      current = None
      if ticket.revision == revision then { value = Some(result); fraction = 1; text = "Complete" }
  def fail(ticket: Ticket, message: String): Unit =
    if !closed && current.contains(ticket) then
      current = None
      if ticket.revision == revision then { value = None; text = message }
  def cancelDecision(ticket: Ticket, decision: InterruptionDecision): Unit =
    if !closed && current.contains(ticket) && ticket.revision == revision then
      text = if decision.blocked then "Cancellation blocked" else if !decision.affected then "No matching active request" else "Cancellation requested"
  def dispose(): Unit = { closed = true; current = None; value = None }

/** Lightweight byte-budgeted LRU for playback assets; no decoded images held in the cache. */
final class TemporalPreviewCache(val maxFrames: Int, val maxBytes: Long):
  require(maxFrames > 0 && maxBytes > 0)
  private var frames = Vector.empty[Path]
  private var position = 0
  private var running = false
  private var bytes = 0L
  private val cache = mutable.LinkedHashMap.empty[Path, Array[Byte]]
  def playing: Boolean = running
  def current: Option[Path] = frames.lift(position)
  def index: Int = position
  def cachedCount: Int = cache.size
  def cachedBytes: Long = bytes
  def setFrames(value: Vector[Path]): Unit = { frames = value; position = 0; running = false; cache.clear(); bytes = 0 }
  def play(): Unit = if frames.nonEmpty then running = true
  def pause(): Unit = running = false
  def stop(): Unit = { running = false; position = 0 }
  def seek(index: Int): Unit = if frames.nonEmpty then position = index.max(0).min(frames.size - 1)
  def advance(): Unit = if running && frames.nonEmpty then position = (position + 1) % frames.size
  def remember(path: Path, data: Array[Byte]): Unit =
    cache.remove(path).foreach(v => bytes -= v.length)
    if frames.contains(path) && data.length <= maxBytes then
      while cache.nonEmpty && (cache.size >= maxFrames || bytes + data.length > maxBytes) do
        val (key, old) = cache.head
        cache.remove(key); bytes -= old.length
      cache(path) = data.clone()
      bytes += data.length
  def get(path: Path): Option[Array[Byte]] = cache.remove(path).map { data => cache(path) = data; data.clone() }
  def dispose(): Unit = { stop(); frames = Vector.empty; cache.clear(); bytes = 0 }

/** Coalesces to one in-flight and one latest pending load per fixed display slot. FX-thread owned. */
final class TemporalPreviewSlots[A](load: (String, Path, Either[String, A] => Unit) => (() => Unit)):
  private final case class Request(path: Path, revision: Long, ok: A => Unit, error: String => Unit)
  private final class Active(val request: Request):
    var cancel: () => Unit = () => ()
  private final class Slot:
    var running: Option[Active] = None
    var pending: Option[Request] = None
  private val slots = Map("previous" -> new Slot, "current" -> new Slot, "frame" -> new Slot)
  private var revision = 0L
  private var closed = false
  def show(name: String, path: Path, ok: A => Unit, error: String => Unit): Unit =
    require(slots.contains(name), "Unknown preview slot")
    if !closed then
      val slot = slots(name)
      val request = Request(path, revision, ok, error)
      if slot.running.nonEmpty then slot.pending = Some(request) else start(name, slot, request)
  private def start(name: String, slot: Slot, request: Request): Unit =
    val active = new Active(request)
    slot.running = Some(active)
    def done(result: Either[String, A]): Unit =
      if slot.running.contains(active) then
        slot.running = None
        // Playback may tick faster than a decode: still show completed frames
        // while retaining just the latest pending request. Static pair slots
        // remain latest-only so an older comparison never replaces a newer one.
        if !closed && request.revision == revision && (name == "frame" || slot.pending.isEmpty) then
          result match
            case Right(value) => request.ok(value)
            case Left(message) => request.error(message)
        val next = slot.pending
        slot.pending = None
        if !closed then next.foreach(r => start(name, slot, r))
    try
      val cancel = load(name, request.path, done)
      if slot.running.contains(active) then active.cancel = cancel
    catch case scala.util.control.NonFatal(e) => done(Left(Option(e.getMessage).getOrElse(e.getClass.getSimpleName)))
  def clear(): Unit =
    if !closed then
      revision += 1
      slots.values.foreach(_.pending = None)
  def dispose(): Unit =
    if !closed then
      closed = true; revision += 1
      slots.values.foreach { slot =>
        slot.pending = None
        slot.running.foreach { active => try active.cancel() catch case scala.util.control.NonFatal(_) => () }
        slot.running = None
      }
