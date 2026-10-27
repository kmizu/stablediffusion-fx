package sdfx.engine

import sdfx.core.config.{AppHome, Settings}
import sdfx.core.modelcards.*
import java.util.concurrent.{Executor, Executors, ThreadFactory}
import scala.collection.mutable
import scala.concurrent.{Future, Promise}
import scala.util.control.NonFatal

/** Borrowed only inside withView; product consumers must not retain records after their lease closes. */
final class CardScanView private[engine] (private val result: CardScanResult):
  def revision: Long = result.revision
  def records: Vector[ModelCardRecord] = result.records
  def status: CardScanStatus = result.status
  def problems: Vector[String] = result.problems
  def problemCount: Long = result.problemCount
  def visited: Long = result.visited

final case class CardScanInfo(revision: Long, status: CardScanStatus, recordCount: Int,
    problemCount: Long, visited: Long, deliveryRefusals: Long = 0L)

private[engine] final class CardScanGraph(initial: CardScanResult, ledger: CardScanReservations, ticket: Long):
  private var result: CardScanResult = initial
  private var owners = 1 // catalog ownership
  def acquire(): Boolean = synchronized {
    if result == null then false else { owners += 1; true }
  }
  def borrow: Option[CardScanResult] = synchronized { Option(result) }
  def release(): Unit = synchronized {
    require(owners > 0, "duplicate graph release")
    owners -= 1
    if owners == 0 then
      val old = result
      result = null
      ledger.releaseResult(old)
      ledger.release(ticket)
  }

/** One bounded pool counts returned, listener, retained and derived graph owners. */
private[engine] final class CardScanLeasePool(ledger: CardScanReservations):
  private var count = 0
  def lease(graph: CardScanGraph): Either[String, CardScanLease] = synchronized {
    if count >= 128 then Left("Model card view capacity reached")
    else ledger.tryReserve("card-lease", 0L, 256L) match
      case None => Left("Model card metadata capacity reached")
      case Some(ticket) =>
        if !graph.acquire() then { ledger.release(ticket); Left("Model card view is closed") }
        else
          count += 1
          ledger.observer.materialize(ticket)
          Right(new CardScanLease(graph, this, ledger, ticket))
  }
  def borrow(graph: CardScanGraph): Either[String, Long] = synchronized {
    if count >= 128 then Left("Model card view capacity reached")
    else ledger.tryReserve("card-borrow", 0L, 256L) match
      case None => Left("Model card metadata capacity reached")
      case Some(ticket) =>
        if !graph.acquire() then { ledger.release(ticket); Left("Model card view is closed") }
        else
          count += 1
          ledger.observer.materialize(ticket)
          Right(ticket)
  }
  def derived(graph: CardScanGraph, chars: Long, bytes: Long): Either[String, CardDerivedLease] = synchronized {
    if chars < 0 || bytes < 0 || bytes > Long.MaxValue - 256L then Left("Invalid derived view size")
    else if count >= 128 then Left("Model card view capacity reached")
    else ledger.tryReserve("card-derived-view", chars, bytes + 256L) match
      case None => Left("Model card metadata capacity reached")
      case Some(ticket) =>
        if !graph.acquire() then { ledger.release(ticket); Left("Model card view is closed") }
        else
          count += 1
          ledger.observer.materialize(ticket)
          Right(new CardDerivedLease(graph, this, ledger, ticket))
  }
  def release(graph: CardScanGraph, ticket: Long): Unit = synchronized {
    count -= 1
    ledger.release(ticket)
    graph.release()
  }

final class CardScanLease private[engine] (initial: CardScanGraph, pool: CardScanLeasePool,
    ledger: CardScanReservations, ticket: Long):
  private var graph: CardScanGraph = initial
  val revision: Long = initial.borrow.map(_.revision).getOrElse(0L)
  private[engine] def hasGraphForTest: Boolean = synchronized { graph != null }
  def retain(): Either[String, CardScanLease] = synchronized {
    if graph == null then Left("Model card view is closed") else pool.lease(graph)
  }
  def reserveDerived(chars: Long, bytes: Long): Either[String, CardDerivedLease] = synchronized {
    if graph == null then Left("Model card view is closed") else pool.derived(graph, chars, bytes)
  }
  def withView[A](f: CardScanView => A): Either[String, A] =
    val admitted = synchronized {
      if graph == null then Left("Model card view is closed")
      else pool.borrow(graph).map(ticket => (graph, ticket))
    }
    admitted.flatMap { (owner, borrowTicket) =>
      try owner.borrow match
        case None => Left("Model card view is closed")
        case Some(value) => Right(f(new CardScanView(value)))
      finally pool.release(owner, borrowTicket)
    }

  def close(): Unit =
    val old = synchronized { val value = graph; graph = null; value }
    if old != null then pool.release(old, ticket)

final class CardDerivedLease private[engine] (initial: CardScanGraph, pool: CardScanLeasePool,
    ledger: CardScanReservations, ticket: Long):
  private var graph: CardScanGraph = initial
  def close(): Unit =
    val old = synchronized { val value = graph; graph = null; value }
    if old != null then pool.release(old, ticket)

/** Offline presentation owner; cached engine metadata is read on the scanner worker only. */
final class ModelCardCatalog(home: AppHome, settings: () => Settings, engineSnapshot: () => CatalogSnapshot,
    scan: (CardScanRequest, () => Boolean) => CardScanResult = null,
    worker: Executor = null, hooks: CardScanHooks = CardScanHooks()):
  private val ownedWorker = if worker == null then Some(Executors.newSingleThreadExecutor(new ThreadFactory:
    def newThread(r: Runnable): Thread =
      val thread = new Thread(r, "sdfx-model-card-scan")
      thread.setDaemon(true)
      thread
  )) else None
  private val executor: Executor = ownedWorker.getOrElse(worker)
  private val ledger = hooks.reservations.getOrElse(new CardScanReservations(
    hooks.limits.metadataChars, hooks.limits.metadataBytes, hooks.observer))
  private val pool = new CardScanLeasePool(ledger)
  private final class Registration(val id: Long, val listener: CardScanLease => Unit, val ticket: Long):
    var active = true
  private final case class Delivery(registration: Registration, lease: CardScanLease, ticket: Long)
  private var nextListenerId = 0L
  private val listeners = mutable.ArrayBuffer.empty[Registration]
  private final case class Work(request: CardScanRequest, promise: Promise[CardScanLease])
  private var nextRevision = 0L
  private var pending: Option[Work] = None
  private var active = false
  private var closed = false
  private var current: Option[CardScanGraph] = None
  private var info: Option[CardScanInfo] = None

  def snapshotInfo: Option[CardScanInfo] = synchronized { info }
  def acquireSnapshot(): Either[String, CardScanLease] = synchronized {
    if closed then Left("Model card catalog is closed")
    else current.toRight("No model card snapshot").flatMap(pool.lease)
  }
  def addListener(listener: CardScanLease => Unit): Either[String, () => Unit] = synchronized {
    if closed then Left("Model card catalog is closed")
    else if listeners.size >= 64 then Left("Model card listener capacity reached")
    else ledger.tryReserve("card-listener", 0L, 512L) match
      case None => Left("Model card metadata capacity reached")
      case Some(ticket) =>
        ledger.observer.materialize(ticket)
        nextListenerId += 1
        val registration = new Registration(nextListenerId, listener, ticket)
        listeners += registration
        Right(() => synchronized {
          if registration.active then
            registration.active = false
            listeners -= registration
            ledger.release(registration.ticket)
        })
  }

  def refresh(): Future[CardScanLease] =
    val prepared = synchronized {
      if closed then None
      else
        nextRevision += 1
        val promise = Promise[CardScanLease]()
        val work = Work(CardScanRequest(nextRevision, home.root, settings()), promise)
        val replaced = pending
        pending = Some(work)
        val start = !active
        if start then active = true
        Some((work, replaced, start))
    }
    prepared match
      case None => Future.failed(new IllegalStateException("Model card catalog is closed"))
      case Some((work, replaced, start)) =>
        replaced.foreach(_.promise.tryFailure(new java.util.concurrent.CancellationException("Superseded scan")))
        if start then
          try executor.execute(() => runLoop())
          catch
            case error: AssertionError => throw error
            case NonFatal(e) =>
              val stranded = synchronized {
                active = false
                val next = pending
                pending = None
                next
              }
              stranded.foreach(_.promise.tryFailure(e))
        work.promise.future

  private def cachedEntries(snapshot: CatalogSnapshot): Iterator[CardScanCachedEntry] =
    val checkpoint = snapshot.checkpoints.iterator.map(m =>
      CardScanCachedEntry("checkpoint", m.path, Option(m.kind).filter(k => k.nonEmpty && k != "unknown"), false,
        displayName = Option(m.name)))
    val lora = snapshot.loras.iterator.map(f => CardScanCachedEntry("lora", f.path, None, false, displayName = Option(f.name)))
    val embeddings = snapshot.embeddings.iterator.map(f => CardScanCachedEntry("embedding", f.path, None, false, displayName = Option(f.name)))
    val builtin = Set("face_restore", "svd", "zero123", "photomaker")
    val extras = snapshot.extraFiles.iterator.flatMap { (kind, files) =>
      files.iterator.map(f => CardScanCachedEntry(kind, f.path, None, !builtin(kind)))
    }
    checkpoint ++ lora ++ embeddings ++ extras

  private def runLoop(): Unit =
    var running = true
    while running do
      val next = synchronized {
        if closed then { active = false; None }
        else pending match
          case Some(work) => pending = None; Some(work)
          case None => active = false; None
      }
      next match
        case None => running = false
        case Some(work) =>
          val cancelled = () => synchronized { closed || work.request.revision != nextRevision }
          var result = try
            if scan != null then scan(work.request, cancelled)
            else
              val cached = engineSnapshot()
              ModelCardScanner.scan(work.request, cancelled, hooks.copy(reservations = Some(ledger), retainResult = true,
                cachedEntries = () => cachedEntries(cached)))
          catch
            case error: AssertionError => throw error
            case NonFatal(e) =>
              CardScanResult(work.request.revision, Vector.empty, CardScanStatus.Failed,
                Vector(s"Scan failed: ${Option(e.getMessage).getOrElse(e.toString)}".take(512)), 1L, 0L)
          val (published, completion): (Option[(CardScanGraph, Long)], Either[Throwable, CardScanLease]) = synchronized {
            if closed || work.request.revision != nextRevision then
              ledger.releaseResult(result)
              (None, Left(new java.util.concurrent.CancellationException("Superseded scan")))
            else ledger.tryReserve("catalog-graph", 0L, 512L) match
              case None =>
                ledger.releaseResult(result)
                (None, Left(new IllegalStateException("Model card metadata capacity reached")))
              case Some(ticket) =>
                ledger.observer.materialize(ticket)
                val graph = new CardScanGraph(result, ledger, ticket)
                val previous = current
                current = Some(graph)
                info = Some(CardScanInfo(result.revision, result.status, result.records.size, result.problemCount, result.visited))
                previous.foreach(_.release())
                val outcome = pool.lease(graph).left.map(reason => new IllegalStateException(reason))
                (Some((graph, nextListenerId)), outcome)
          }
          // Completion can run arbitrary parasitic callbacks: never invoke one under the catalog monitor.
          result = null
          completion match
            case Right(lease) => if !work.promise.trySuccess(lease) then lease.close()
            case Left(error) => work.promise.tryFailure(error)
          published.foreach { (graph, highWater) =>
            var previousId = 0L
            var delivering = true
            while delivering do
              val delivery = synchronized {
                if closed || nextRevision != work.request.revision then None
                else listeners.find(r => r.active && r.id > previousId && r.id <= highWater).flatMap { registration =>
                  previousId = registration.id
                  ledger.tryReserve("card-listener-delivery", 0L, 512L) match
                    case None =>
                      info = info.map(i => i.copy(deliveryRefusals = i.deliveryRefusals + 1L))
                      None
                    case Some(ticket) =>
                      ledger.observer.materialize(ticket)
                      pool.lease(graph) match
                        case Right(lease) => Some(Delivery(registration, lease, ticket))
                        case Left(_) =>
                          ledger.release(ticket)
                          info = info.map(i => i.copy(deliveryRefusals = i.deliveryRefusals + 1L))
                          None
                }
              }
              delivery match
                case None => delivering = false
                case Some(item) =>
                  try item.registration.listener(item.lease)
                  catch
                    case error: AssertionError => item.lease.close(); throw error
                    case NonFatal(_) => item.lease.close()
                  finally ledger.release(item.ticket)
          }

  /** Nonblocking with respect to scan workers; consumers retain their own graph leases. */
  def close(): Unit =
    val cancelled = synchronized {
      if closed then None
      else
        closed = true
        nextRevision += 1
        val oldPending = pending
        pending = None
        val old = current
        current = None
        info = None
        old.foreach(_.release())
        listeners.foreach { r => if r.active then { r.active = false; ledger.release(r.ticket) } }
        listeners.clear()
        ownedWorker.foreach(_.shutdown())
        oldPending
    }
    cancelled.foreach(_.promise.tryFailure(new java.util.concurrent.CancellationException("Catalog closed")))
