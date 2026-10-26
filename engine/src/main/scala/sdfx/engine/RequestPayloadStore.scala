package sdfx.engine

import sdfx.core.payload.*
import scala.collection.mutable.ArrayBuffer
import scala.util.control.NonFatal

/** Frozen per submission. Only this preference persists; captured documents never do. */
final case class CaptureOptions(includeImageContents: Boolean = false)

/** Bounded memory-only history. No observer runs while the history lock is held.
  * Each replacement is normalized and budgeted atomically, including lifecycle diagnostics.
  */
final class RequestPayloadStore(val limits: RequestPayloadStore.Limits = RequestPayloadStore.Limits()):
  import RequestPayloadStore.*
  private var entries = Vector.empty[PayloadSnapshot]
  private var bytes = 0L
  private var rejectedCount = 0L
  private var closed = false
  private var version = 0L
  private final class Subscriber(var callback: Option[Vector[PayloadSnapshot] => Unit]):
    var busy = false
    var pending: Option[(Long, Vector[PayloadSnapshot])] = None
    var lastDeliveredVersion = -1L
  private val observers = ArrayBuffer.empty[Subscriber]

  def snapshots: Vector[PayloadSnapshot] = synchronized(entries)
  def retainedBytes: Long = synchronized(bytes)
  def rejectedCaptures: Long = synchronized(rejectedCount)
  def latestForMode(mode: String): Option[PayloadSnapshot] = synchronized {
    entries.reverseIterator.find(_.sourceMode.contains(mode))
  }

  def subscribe(observer: Vector[PayloadSnapshot] => Unit): () => Unit =
    val registered = synchronized {
      if closed then None
      else
        val subscriber = new Subscriber(Some(observer))
        observers += subscriber
        subscriber.pending = Some((version, entries))
        subscriber.busy = true
        Some(subscriber)
    }
    registered.foreach(drain)
    () => synchronized {
      registered.foreach { subscriber =>
        subscriber.callback = None
        subscriber.pending = None
        observers -= subscriber
      }
      ()
    }

  def put(snapshot: PayloadSnapshot): Unit =
    val ready = synchronized {
      val previous = entries.find(_.key == snapshot.key)
      // A delayed prepared event can never rewrite an already terminal subrequest.
      if !closed && !previous.exists(_.state.isTerminal) then retain(snapshot)
      queuePublication()
    }
    ready.foreach(drain)

  def transition(key: PayloadKey, state: PayloadState, completionJson: Option[String] = None): Unit =
    val ready = synchronized {
      entries.find(_.key == key).filterNot(s => closed || s.state.isTerminal).foreach { previous =>
        if state != PayloadState.Prepared || previous.state == PayloadState.Prepared then
          val next = previous.transition(state)
          val withCompletion = completionJson.fold(next) { completion =>
            val primaryBytes = size(next.copy(contextJson = None, assetsJson = None), limits.maxEntryBytes)
            val available = math.max(0L, limits.maxEntryBytes - primaryBytes)
            val contextBytes = next.contextJson.map(utf8Size(_, available)).getOrElse(2L)
            val completionBytes = utf8Size(completion, available)
            val merged =
              // Reserve the new key/braces before either document is parsed into an object graph.
              if contextBytes + completionBytes + 32 > available then Left("bytes_limit")
              else
                val budget = jsonLimits(available)
                val context = next.contextJson match
                  case None => Right(ujson.Obj())
                  case Some(json) => PayloadJsonBudget.admit(json, budget) {
                    ujson.read(json) match
                      case obj: ujson.Obj => obj
                      case _ => throw new IllegalArgumentException("context must be an object")
                  }
                context.flatMap { obj =>
                  PayloadJsonBudget.admit(completion, budget)(ujson.read(completion)).flatMap { facts =>
                    obj("reported_at_completion") = facts
                    PayloadJsonBudget.write(obj, budget)
                  }
                }
            merged match
              case Right(json) => next.copy(contextJson = Some(json))
              case Left(reason) => next.copy(contextJson = None, warnings = next.warnings :+ s"Completion context unavailable: $reason")
          }
          retain(withCompletion)
      }
      queuePublication()
    }
    ready.foreach(drain)

  /** The enclosing job only settles subrequests that have not already settled themselves. */
  def finishJob(jobId: String, state: PayloadState): Unit =
    val ready = synchronized {
      val pending = entries.filter(s => !closed && s.key.jobId == jobId && !s.state.isTerminal)
      pending.foreach(s => retain(s.transition(state)))
      queuePublication()
    }
    ready.foreach(drain)

  def clear(): Unit =
    val ready = synchronized {
      entries = Vector.empty
      bytes = 0L
      queuePublication()
    }
    ready.foreach(drain)

  def close(): Unit =
    synchronized {
      closed = true
      entries = Vector.empty
      bytes = 0L
      observers.foreach { subscriber =>
        subscriber.callback = None
        subscriber.pending = None
      }
      observers.clear()
    }

  private[engine] def warn(key: PayloadKey, text: String): Unit =
    val ready = synchronized {
      if !closed then entries.find(_.key == key).foreach(s => retain(s.copy(warnings = s.warnings :+ text)))
      queuePublication()
    }
    ready.foreach(drain)

  private def retain(raw: PayloadSnapshot): Unit =
    // If identity metadata alone cannot fit, no normalized version can keep its identity.
    val identity = raw.copy(bodyJson = None, contextJson = None, assetsJson = None,
      sourceMode = None, warnings = Vector.empty)
    if size(identity, limits.maxEntryBytes) > limits.maxEntryBytes then
      rejectCapture()
      return
    val normalized = normalize(raw)
    if size(normalized, limits.maxEntryBytes) > limits.maxEntryBytes then
      rejectCapture()
      return
    val index = entries.indexWhere(_.key == raw.key)
    if index >= 0 then entries = entries.updated(index, normalized)
    else entries = entries :+ normalized
    bytes = entries.iterator.map(s => size(s, limits.maxEntryBytes)).sum
    while entries.nonEmpty && (entries.size > limits.maxCaptures || bytes > limits.maxStoreBytes) do
      bytes -= size(entries.head, limits.maxEntryBytes)
      entries = entries.tail

  private def rejectCapture(): Unit =
    if rejectedCount < Long.MaxValue then rejectedCount += 1

  private def normalize(raw: PayloadSnapshot): PayloadSnapshot =
    var result = raw.copy(sourceMode = raw.sourceMode.filter(KnownModes), warnings = boundedWarnings(raw.warnings))
    if result.bodyJson.exists(s => utf8Size(s, limits.maxBodyBytes) > limits.maxBodyBytes) then
      result = result.copy(bodyJson = None, warnings = boundedWarnings(result.warnings :+ "Request body unavailable: exceeds exact-body byte limit"))
    result.contextJson.foreach { context =>
      PayloadJsonBudget.guard(context, jsonLimits(limits.maxEntryBytes)).left.foreach { reason =>
        result = result.copy(contextJson = None, warnings = boundedWarnings(result.warnings :+ s"Submitted context unavailable: $reason"))
      }
    }
    // Optional diagnostics cannot erase a valid exact primary body. Drop complete asset/context
    // documents in that order, never return a prefix while calling it the retained JSON document.
    if size(result, limits.maxEntryBytes) > limits.maxEntryBytes && result.assetsJson.nonEmpty then
      result = result.copy(assetsJson = None,
        warnings = boundedWarnings(result.warnings :+ "Image diagnostic unavailable: total retained entry byte limit"))
    if size(result, limits.maxEntryBytes) > limits.maxEntryBytes && result.contextJson.nonEmpty then
      result = result.copy(contextJson = None,
        warnings = boundedWarnings(result.warnings :+ "Submitted context unavailable: total retained entry byte limit"))
    if size(result, limits.maxEntryBytes) > limits.maxEntryBytes then
      result = result.copy(warnings = Vector("Payload unavailable: retained diagnostics exceed entry byte limit"))
    if size(result, limits.maxEntryBytes) > limits.maxEntryBytes then
      result = result.copy(bodyJson = None, warnings = Vector("Request body unavailable: exact body exceeds retained entry byte limit"))
    result

  private def jsonLimits(maxBytes: Long): PayloadJsonBudget.Limits =
    PayloadJsonBudget.Limits(maxBytes = math.min(maxBytes, Int.MaxValue.toLong - 1).toInt)

  private def boundedWarnings(warnings: Vector[String]): Vector[String] =
    val overflow = "([0-9]+) warnings omitted by diagnostic limit".r
    var priorOmitted = 0L
    val messages = warnings.filter {
      case overflow(count) => priorOmitted += count.toLongOption.getOrElse(0L); false
      case _ => true
    }
    val kept = messages.take(limits.maxWarnings).map { warning =>
      if utf8Size(warning, limits.maxWarningBytes) <= limits.maxWarningBytes then warning
      else clipUtf8(warning, math.max(0, limits.maxWarningBytes - 12)) + " [truncated]"
    }
    val omitted = priorOmitted + messages.size - kept.size
    if omitted > 0 then kept :+ s"$omitted warnings omitted by diagnostic limit" else kept

  /** Called with the store monitor held. A busy subscriber keeps only the latest publication. */
  private def queuePublication(): Vector[Subscriber] =
    if closed then Vector.empty
    else
      version += 1
      observers.iterator.flatMap { subscriber =>
        if subscriber.callback.isEmpty || version <= subscriber.lastDeliveredVersion then None
        else
          subscriber.pending = Some((version, entries))
          if subscriber.busy then None
          else
            subscriber.busy = true
            Some(subscriber)
      }.toVector

  /** One caller owns each subscriber's drain. User callbacks always run outside store locks. */
  private def drain(subscriber: Subscriber): Unit =
    var next = synchronized(takeNext(subscriber))
    while next.nonEmpty do
      val (callback, snapshot) = next.get
      safely(callback(snapshot))
      next = synchronized(takeNext(subscriber))

  private def takeNext(subscriber: Subscriber): Option[(Vector[PayloadSnapshot] => Unit, Vector[PayloadSnapshot])] =
    (subscriber.callback, subscriber.pending) match
      case (Some(callback), Some((newVersion, snapshot))) if !closed && newVersion > subscriber.lastDeliveredVersion =>
        subscriber.pending = None
        subscriber.lastDeliveredVersion = newVersion
        Some((callback, snapshot))
      case _ =>
        subscriber.pending = None
        subscriber.busy = false
        None

  private def safely(action: => Unit): Unit =
    try action
    catch
      case error: InterruptedException => System.err.println(s"[sdfx] Payload subscriber failed: ${error.getClass.getSimpleName}")
      case NonFatal(error) => System.err.println(s"[sdfx] Payload subscriber failed: ${error.getClass.getSimpleName}")

object RequestPayloadStore:
  final case class Limits(
      maxCaptures: Int = 50,
      maxBodyBytes: Long = 8L * 1024 * 1024,
      maxEntryBytes: Long = 24L * 1024 * 1024,
      maxStoreBytes: Long = 32L * 1024 * 1024,
      maxWarnings: Int = 32,
      maxWarningBytes: Int = 1024
  ):
    require(maxCaptures > 0 && maxBodyBytes > 0 && maxEntryBytes > 0 && maxStoreBytes > 0)
    require(maxWarnings > 0 && maxWarningBytes >= 64)

  private[engine] val KnownModes = Set("txt2img", "img2img", "inpaint")
  /** Exact for retained input; oversized input returns limit + 1 without materializing encoded bytes.
    * Isolated UTF-16 surrogates use the JDK UTF-8 encoder's one-byte replacement semantics.
    */
  private[engine] def utf8Size(text: CharSequence, maxBytes: Long): Long =
    require(maxBytes >= 0)
    val length = text.length()
    var index = 0
    var total = 0L
    while index < length do
      val char = text.charAt(index)
      val width =
        if char <= 0x7f then 1
        else if char <= 0x7ff then 2
        else if Character.isHighSurrogate(char) then
          if index + 1 < length && Character.isLowSurrogate(text.charAt(index + 1)) then
            index += 1
            4
          else 1
        else if Character.isLowSurrogate(char) then 1
        else 3
      total += width
      if total > maxBytes then return maxBytes + 1
      index += 1
    total

  private def size(snapshot: PayloadSnapshot, maxBytes: Long): Long =
    val strings = (Vector(snapshot.key.jobId, snapshot.requestType, snapshot.state.wireName) ++ snapshot.bodyJson ++
      snapshot.contextJson ++ snapshot.assetsJson ++ snapshot.sourceMode ++ snapshot.warnings).iterator
    var total = 0L
    while strings.hasNext do
      val count = utf8Size(strings.next(), maxBytes - total)
      total += count
      if total > maxBytes then return maxBytes + 1
    total

  /** Keeps valid code points and a strict final UTF-8 byte ceiling. */
  private def clipUtf8(text: String, limit: Int): String =
    val result = new java.lang.StringBuilder
    val points = text.codePoints().iterator()
    var bytes = 0
    var full = false
    while points.hasNext && !full do
      val cp = points.nextInt()
      val width = if cp <= 0x7f then 1 else if cp <= 0x7ff then 2 else if cp <= 0xffff then 3 else 4
      if bytes + width > limit then full = true
      else
        result.appendCodePoint(cp)
        bytes += width
    result.toString
