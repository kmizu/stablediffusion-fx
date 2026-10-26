package sdfx.ui.view

import java.lang.ref.WeakReference
import scala.util.control.NonFatal
import sdfx.core.payload.{PayloadKey, PayloadSnapshot}
import sdfx.engine.RequestPayloadStore

/** Independent documents: the image diagnostic and context are never part of the native wire body. */
enum PayloadDocument(val label: String):
  case Body extends PayloadDocument("Request body")
  case Context extends PayloadDocument("Job and model context")
  case ImageContents extends PayloadDocument("Image contents")

final case class PayloadRawPreview(text: String, totalBytes: Long, clipped: Boolean):
  def notice: String =
    if clipped then s"Preview: first 64 KiB at most; Copy includes complete retained JSON ($totalBytes UTF-8 bytes)"
    else s"Complete retained JSON ($totalBytes UTF-8 bytes)"

/** A selection label never owns a retained document or parsed JSON tree. */
final case class PayloadHistoryEntry(key: PayloadKey, label: String, sourceMode: Option[String])

/** Pure state for the shared read-only inspector. Mutations belong to its UI thread; the store itself is
  * thread-safe. No control, engine, file, clipboard or background-executor dependency is introduced here.
  *
  * Only selection identity is retained. History and documents are read from the current store, so this
  * model cannot keep a second body/history cache after eviction. The weak stamp detects immutable snapshot
  * replacements without comparing multi-megabyte JSON strings or keeping the previous body alive.
  */
final class RequestPayloadPresentation(
    store: RequestPayloadStore,
    writeClipboard: String => Unit,
    clockMillis: () => Long
):
  private var currentStore: RequestPayloadStore = store
  private var currentWriter: String => Unit = writeClipboard
  private var disposed = false
  private var allCaptures = false
  private var sourceMode: Option[String] = None
  private var selectedKey: Option[PayloadKey] = None
  private var document = PayloadDocument.Body
  private var followLatest = true
  private var revision = 0L
  private var stampedMode: Option[String] = None
  private var stampedKey: Option[PayloadKey] = None
  private var stampedDocument = PayloadDocument.Body
  private var stampedSnapshot = new WeakReference[PayloadSnapshot](null)
  private var stampedAvailable = false
  private var feedback: Option[(String, Long)] = None

  /** Oldest to newest, using first-class submitted source mode rather than inferring Inpaint from type. */
  def snapshots: Vector[PayloadSnapshot] =
    if disposed then Vector.empty
    else if allCaptures then currentStore.snapshots
    else sourceMode.fold(Vector.empty[PayloadSnapshot])(mode => currentStore.snapshots.filter(_.sourceMode.contains(mode)))

  def historyEntries: Vector[PayloadHistoryEntry] = snapshots.map { snapshot =>
    val mode = snapshot.sourceMode.getOrElse("source mode unavailable")
    val key = snapshot.key
    val shortJob = RequestPayloadPresentation.shortLabel(
      if key.jobId.length <= 60 then key.jobId else key.jobId.take(59) + "…")
    val label = s"$mode · $shortJob · #${key.sequence} · ${snapshot.state.wireName}"
    PayloadHistoryEntry(key, label.take(160), snapshot.sourceMode)
  }

  def rejectedCaptureText: String =
    val count = if disposed then 0L else currentStore.rejectedCaptures
    s"Unavailable captures rejected by memory limits: $count"

  def selectedSnapshot: Option[PayloadSnapshot] =
    if disposed then None
    else selectedKey.flatMap(key => snapshots.find(_.key == key))

  def selectedDocument: PayloadDocument = document
  def isAllCaptures: Boolean = allCaptures
  def currentMode: Option[String] = sourceMode
  def warningDetails: Vector[String] = selectedSnapshot.map(_.warnings).getOrElse(Vector.empty)

  def selectedJson: Option[String] = selectedSnapshot.flatMap(documentJson)
  def copyEnabled: Boolean = selectedJson.nonEmpty

  /** A native mode entry starts at that mode's latest capture and at the exact request body. */
  def openMode(mode: String): Unit = if !disposed then
    allCaptures = false
    sourceMode = Some(mode)
    document = PayloadDocument.Body
    followLatest = true
    selectedKey = snapshots.lastOption.map(_.key)
    feedback = None
    updateRevision()

  def openAllCaptures(): Unit = if !disposed then
    allCaptures = true
    sourceMode = None
    document = PayloadDocument.Body
    followLatest = true
    selectedKey = snapshots.lastOption.map(_.key)
    feedback = None
    updateRevision()

  /** Call after marshaling a store notification to the UI thread. Do not pass the event's potentially
    * stale vector: this deliberately reads the store again at the point the view is refreshed.
    */
  def refresh(): Unit = if !disposed then
    if followLatest then selectedKey = snapshots.lastOption.map(_.key)
    updateRevision()

  def select(key: PayloadKey): Unit = if !disposed then
    val current = snapshots
    if current.exists(_.key == key) then
      selectedKey = Some(key)
      followLatest = current.lastOption.exists(_.key == key)
      feedback = None
      updateRevision()

  def chooseDocument(next: PayloadDocument): Unit = if !disposed then
    document = next
    feedback = None
    updateRevision()

  /** Snapshot reference identity changes on lifecycle/context updates. The check also catches eviction
    * before the queued UI refresh, and never hashes/compares the full document for stale-work admission.
    */
  def selectionRevision: Long =
    if !disposed then updateRevision()
    revision

  def acceptsRevision(candidate: Long): Boolean = !disposed && selectionRevision == candidate

  def statusText: String =
    if disposed then "Payload inspector closed"
    else selectedSnapshot match
      case Some(snapshot) =>
        val mode = snapshot.sourceMode.getOrElse("source mode unavailable")
        val header = s"sdfx protocol v1 · ${RequestPayloadPresentation.shortLabel(snapshot.requestType)} · job ${RequestPayloadPresentation.shortLabel(snapshot.key.jobId)} · sub-generation ${snapshot.key.sequence} · $mode · ${snapshot.state.wireName}"
        val unavailable = Option.when(documentJson(snapshot).isEmpty)(document match
          case PayloadDocument.Body => "Request body unavailable: no body was dispatched or retained"
          case PayloadDocument.Context => "Job and model context unavailable for this capture"
          case PayloadDocument.ImageContents => "Image contents diagnostic unavailable for this capture")
        val warningSummary = snapshot.warnings.headOption.map(first =>
          s"${snapshot.warnings.size} warnings retained; inspect warning details below\n${RequestPayloadPresentation.shortLabel(first)}")
        (Vector(header) ++ unavailable ++ warningSummary).mkString("\n")
      case None if selectedKey.nonEmpty => "Selected payload unavailable: it was evicted from memory-only history"
      case None => s"No payload captured for ${if allCaptures then "All captures" else sourceMode.getOrElse("this mode")}"

  /** Only an explicit Copy action calls the injected writer. Recheck the selected identity/document in
    * the current store without first following a newer entry: a delayed refresh must not copy another job.
    */
  def copySelected(): Unit = if !disposed then
    selectedJson match
      case Some(exact) =>
        try
          currentWriter(exact)
          feedback = Some("Copied" -> clockMillis())
        catch case NonFatal(_) => feedback = Some("Copy failed" -> clockMillis())
      case None => feedback = Some("Copy failed: document unavailable" -> clockMillis())

  def copyFeedback: Option[String] =
    if disposed then None
    else
      feedback match
        case Some((message, started)) =>
          val elapsed = clockMillis() - started
          if elapsed < 0L || elapsed >= 3000L then
            feedback = None
            None
          else Some(message)
        case None => None

  /** The shared store belongs to AppServices; closing a viewer never clears another viewer's history. */
  def dispose(): Unit = if !disposed then
    disposed = true
    sourceMode = None
    selectedKey = None
    stampedMode = None
    stampedKey = None
    stampedSnapshot.clear()
    stampedAvailable = false
    feedback = None
    currentStore = null
    currentWriter = null
    revision += 1

  private def documentJson(snapshot: PayloadSnapshot): Option[String] = document match
    case PayloadDocument.Body => snapshot.bodyJson
    case PayloadDocument.Context => snapshot.contextJson
    case PayloadDocument.ImageContents => snapshot.assetsJson

  private def updateRevision(): Unit =
    val current = selectedSnapshot.orNull
    if stampedMode != sourceMode || stampedKey != selectedKey || stampedDocument != document ||
        stampedAvailable != (current != null) || (stampedSnapshot.get() ne current) then
      revision += 1
      stampedMode = sourceMode
      stampedKey = selectedKey
      stampedDocument = document
      stampedSnapshot = new WeakReference[PayloadSnapshot](current)
      stampedAvailable = current != null

object RequestPayloadPresentation:
  private val MaxPreviewBytes = 64 * 1024

  private def shortLabel(text: String): String =
    val prefix = if text.length <= 160 then text else text.take(159) + "…"
    prefix.map(ch => if ch < ' ' then ' ' else ch)

  /** Run on the presentation worker alongside parsing, never on FX. Counts final UTF-8 bytes without
    * allocating a full encoded copy. Valid surrogate pairs stay whole; isolated surrogates use the JDK
    * UTF-8 encoder's one-byte replacement accounting, matching retained-store byte accounting.
    */
  def rawPreview(json: String): PayloadRawPreview =
    var index = 0
    var end = 0
    var total = 0L
    var prefixBytes = 0
    var clipped = false
    while index < json.length do
      val char = json.charAt(index)
      var chars = 1
      val width =
        if char <= 0x7f then 1
        else if char <= 0x7ff then 2
        else if Character.isHighSurrogate(char) then
          if index + 1 < json.length && Character.isLowSurrogate(json.charAt(index + 1)) then
            chars = 2
            4
          else 1
        else if Character.isLowSurrogate(char) then 1
        else 3
      total += width
      if !clipped then
        if prefixBytes + width <= MaxPreviewBytes then
          prefixBytes += width
          end = index + chars
        else clipped = true
      index += chars
    PayloadRawPreview(if clipped then json.substring(0, end) else json, total, clipped)
