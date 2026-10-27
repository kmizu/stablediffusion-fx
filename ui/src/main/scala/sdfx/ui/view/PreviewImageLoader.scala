package sdfx.ui.view

import java.awt.image.BufferedImage
import java.util.concurrent.{ExecutorService, Executors}
import javafx.application.Platform
import javafx.scene.image.{Image, PixelFormat, WritableImage}
import javax.imageio.ImageIO
import javax.imageio.stream.ImageInputStreamImpl
import scala.collection.mutable
import scala.util.control.NonFatal
import sdfx.core.modelcards.*

trait PreviewTicket:
  /** Releases this consumer's delivered pixels as well as cancelling queued delivery. */
  def cancel(): Unit

sealed trait PreviewImageResult
object PreviewImageResult:
  final case class Ready(image: Image) extends PreviewImageResult
  final case class Unavailable(reason: String) extends PreviewImageResult

private[ui] final case class PreviewLoaderMetrics(activeWorkers: Int, admittedBuffers: Int,
    sourceBytes: Long, queued: Int, ownedImages: Int, ownedImageBytes: Long, cachedItems: Int,
    mailboxItems: Int, mailboxBytes: Long, visibleConsumers: Int, pendingFxDrains: Int,
    abandonedGenerations: Int, rejected: Long, completed: Long)

final class PreviewImageLoader private[view] (owner: PreviewDecodeOwner, generation: Long):
  def request(ref: PreviewReference, edge: Int, complete: PreviewImageResult => Unit): PreviewTicket =
    owner.request(generation, ref, edge, complete)
  def close(): Unit = owner.abandon(generation)
  private[ui] def metrics: PreviewLoaderMetrics = owner.metrics
  private[ui] def addCapacityListener(listener: () => Unit): () => Unit = owner.addCapacityListener(generation, listener)

final class PreviewLoaderLease private[view] (val loader: PreviewImageLoader, release: () => Unit):
  private var closed = false
  def close(): Unit = synchronized {
    if !closed then
      closed = true
      release()
  }

object PreviewImageLoader:
  private[ui] val visibleCapacityReason = "Preview visible capacity reached; waiting for another preview to close"
  // Never replace this owner on window close, generation close or shutdown.
  private lazy val owner = new PreviewDecodeOwner(PreviewDecodeDependencies.system)
  def acquire(): PreviewLoaderLease = owner.acquire()
  def shutdownProcess(): Unit = owner.shutdownProcess()
  private[ui] def metrics: PreviewLoaderMetrics = owner.metrics

private[ui] final case class PreviewAllocationEvent(id: Long, kind: String, bytes: Long, acquired: Boolean)
private[ui] final case class PreviewDecodeDependencies(
    readSnapshot: (PreviewReference, Array[Byte], () => Boolean) => Either[String, PreviewSourceSnapshot],
    openReader: PreviewSourceSnapshot => Either[String, PreviewImageReader],
    scheduleFx: (() => Unit) => Unit,
    executorFactory: () => ExecutorService,
    allocationEvent: PreviewAllocationEvent => Unit)

private[ui] object PreviewDecodeDependencies:
  def system: PreviewDecodeDependencies = PreviewDecodeDependencies(
    (ref, buffer, cancelled) => LocalPreviewFiles.scoped().readSnapshot(ref, buffer, cancelled),
    PreviewImageCodec.open,
    task => Platform.runLater(() => task()),
    () => Executors.newFixedThreadPool(2, (task: Runnable) =>
      val thread = new Thread(task, "sdfx-preview-decoder")
      thread.setDaemon(true)
      thread),
    _ => ())

private[ui] trait PreviewImageReader:
  def width: Int
  def height: Int
  def format: String
  /** Conservative JDK PNG/JPEG/GIF/BMP output reservation, including 16-bit RGBA. */
  def pixelBytes: Int = 8
  def read(frameIndex: Int, subsampling: Int): BufferedImage
  def close(): Unit

private[ui] object PreviewImageCodec:
  /** Seek directly in the admitted immutable buffer; no second encoded-image cache or path open. */
  private final class SnapshotStream(snapshot: PreviewSourceSnapshot) extends ImageInputStreamImpl:
    override def length(): Long = snapshot.length.toLong
    def read(): Int =
      checkClosed()
      bitOffset = 0
      if streamPos >= snapshot.length then -1
      else
        val value = snapshot.bytes(streamPos.toInt) & 0xff
        streamPos += 1
        value
    def read(target: Array[Byte], offset: Int, length: Int): Int =
      checkClosed()
      java.util.Objects.checkFromIndexSize(offset, length, target.length)
      bitOffset = 0
      if length == 0 then 0
      else if streamPos >= snapshot.length then -1
      else
        val count = math.min(length.toLong, snapshot.length - streamPos).toInt
        System.arraycopy(snapshot.bytes, streamPos.toInt, target, offset, count)
        streamPos += count
        count

  def open(snapshot: PreviewSourceSnapshot): Either[String, PreviewImageReader] =
    val stream = new SnapshotStream(snapshot)
    var reader: javax.imageio.ImageReader = null
    try
      val readers = ImageIO.getImageReaders(stream)
      if !readers.hasNext then
        stream.close()
        Left("Unsupported or corrupt preview image")
      else
        reader = readers.next()
        reader.setInput(stream, true, true)
        val selected = reader
        val w = selected.getWidth(0)
        val h = selected.getHeight(0)
        val codec = selected.getFormatName.toLowerCase(java.util.Locale.ROOT)
        if !Set("png", "jpeg", "jpg", "gif", "bmp").contains(codec) then
          selected.dispose()
          stream.close()
          Left("Unsupported preview image format")
        else Right(new PreviewImageReader:
          def width: Int = w
          def height: Int = h
          def format: String = codec
          def read(frameIndex: Int, subsampling: Int): BufferedImage =
            val params = selected.getDefaultReadParam
            params.setSourceSubsampling(subsampling, subsampling, 0, 0)
            selected.read(frameIndex, params)
          def close(): Unit =
            try selected.dispose()
            finally stream.close()
        )
    catch case NonFatal(error) =>
      try if reader != null then reader.dispose()
      finally stream.close()
      Left(s"Preview image cannot be read: ${Option(error.getMessage).getOrElse(error.getClass.getSimpleName).take(400)}")

/** One permanent admission domain. A fixture may inject an isolated owner; windows may only acquire(). */
private[ui] final class PreviewDecodeOwner(deps: PreviewDecodeDependencies):
  private val MiB = 1024L * 1024L
  private val sourceLimit = 32 * 1024 * 1024
  private val imageLimit = 32L * MiB
  private val executor = deps.executorFactory()
  private var terminating = false
  private var sequence = 0L
  private var current: Option[Generation] = None
  private var active = 0
  private var abandoned = 0
  private var rejectedCount = 0L
  private var completedCount = 0L
  private var drainScheduled = false
  private val deliveryBusy = "Preview delivery busy; refresh to retry"
  private val workersBusy = "Preview workers busy"
  private val imageBusy = "Preview image capacity reached; refresh to retry"
  private val allocations = mutable.LinkedHashMap.empty[Long, Allocation]
  private val queue = mutable.ArrayBuffer.empty[Work]
  private val running = mutable.Set.empty[Work]
  private val consumers = mutable.LinkedHashSet.empty[Ticket]
  private val mailbox = mutable.ArrayBuffer.empty[Delivery]
  private val cache = mutable.LinkedHashMap.empty[CacheKey, Pixels]
  private val errors = mutable.LinkedHashMap.empty[CacheKey, String]
  private val capacityListeners = mutable.LinkedHashMap.empty[Long, CapacityListener]

  private final class Generation(val id: Long):
    var leases = 0
    var live = true
    lazy val loader = new PreviewImageLoader(PreviewDecodeOwner.this, id)
  private final case class RequestKey(source: String, identity: LocalPreviewIdentity, edge: Int)
  private final case class CacheKey(fileKey: String, digest: String, edge: Int)
  private final class Work(val generation: Long, val ref: PreviewReference, val edge: Int):
    val key = RequestKey(ref.source, ref.identity, edge)
    val tickets = mutable.LinkedHashSet.empty[Ticket]
    var source: Option[Allocation] = None
    var aborted = false
    var accepting = true
  private final class Allocation(val id: Long, val kind: String, val bytes: Long):
    var live = true
  private final class Pixels(val image: Image, val allocation: Allocation):
    var references = 1
  private final case class Delivery(ticket: Ticket, result: Either[String, Pixels])
  private enum ConsumerStatus:
    case None, WorkersBusy, DeliveryBusy
  private final class CapacityListener(val id: Long, val generation: Long, val callback: () => Unit):
    var pending = false
  private final class Ticket(val generation: Long, val callback: PreviewImageResult => Unit) extends PreviewTicket:
    var live = true
    var work: Option[Work] = None
    var displayed: Option[Pixels] = None
    var status = ConsumerStatus.None
    def cancel(): Unit = PreviewDecodeOwner.this.synchronized { cancelTicket(this); pump() }

  private def nextId(): Long =
    sequence += 1
    sequence
  private def generationLive(id: Long): Boolean = current.exists(g => g.id == id && g.live) && !terminating
  private def wanted(work: Work): Boolean = !work.aborted && generationLive(work.generation) && work.tickets.exists(_.live)
  private def isWanted(work: Work): Boolean = synchronized { wanted(work) }

  def acquire(): PreviewLoaderLease = synchronized {
    val generation = current.filter(_.live).getOrElse {
      val next = new Generation(nextId())
      if terminating then next.live = false else current = Some(next)
      next
    }
    generation.leases += 1
    new PreviewLoaderLease(generation.loader, () => synchronized {
      generation.leases -= 1
      if generation.leases == 0 then abandon(generation.id)
    })
  }

  private[view] def addCapacityListener(generation: Long, listener: () => Unit): () => Unit = synchronized {
    require(generationLive(generation), "Preview loader closed")
    if capacityListeners.size >= 32 then throw new IllegalStateException("Preview capacity listener limit reached")
    val entry = new CapacityListener(nextId(), generation, listener)
    capacityListeners(entry.id) = entry
    () => synchronized { capacityListeners.remove(entry.id); () }
  }

  private def notifyCapacity(): Unit =
    capacityListeners.valuesIterator.filter(l => generationLive(l.generation)).foreach(_.pending = true)
    if capacityListeners.valuesIterator.exists(_.pending) then scheduleDrain()

  def request(generation: Long, ref: PreviewReference, edge: Int,
      complete: PreviewImageResult => Unit): PreviewTicket = synchronized {
    val ticket = new Ticket(generation, complete)
    val reason =
      if !generationLive(generation) then Some("Preview loader closed")
      else if edge <= 0 then Some("Invalid preview size")
      else if consumers.size >= 32 then Some(PreviewImageLoader.visibleCapacityReason)
      else None
    if reason.nonEmpty then
      ticket.live = false
      rejectedCount += 1
      safely(complete, PreviewImageResult.Unavailable(reason.get))
    else
      val boundedEdge = math.min(edge, 512)
      val key = RequestKey(ref.source, ref.identity, boundedEdge)
      val existing = (queue.iterator ++ running.iterator).find(w =>
        w.accepting && !w.aborted && w.generation == generation && w.key == key)
      if existing.isEmpty && queue.size >= 32 then
        ticket.live = false
        rejectedCount += 1
        safely(complete, PreviewImageResult.Unavailable("Preview request queue full; refresh to retry"))
      else
        consumers += ticket
        val work = existing.getOrElse {
          val created = new Work(generation, ref, boundedEdge)
          queue += created
          created
        }
        ticket.work = Some(work)
        work.tickets += ticket
        if active == 2 && running.forall(w => !wanted(w)) then
          ticket.status = ConsumerStatus.WorkersBusy
          scheduleDrain()
        pump()
    ticket
  }

  private def allocate(kind: String, bytes: Long): Option[Allocation] =
    if kind != "source" then
      def fits: Boolean =
        val pixels = allocations.valuesIterator.filter(_.kind != "source").toVector
        pixels.size < 128 && pixels.map(_.bytes).sum + bytes <= imageLimit
      while !fits && cache.values.exists(_.references == 1) do
        val key = cache.iterator.find(_._2.references == 1).get._1
        releasePixels(cache.remove(key).get)
      if !fits then return None
    val allocation = new Allocation(nextId(), kind, bytes)
    allocations(allocation.id) = allocation
    deps.allocationEvent(PreviewAllocationEvent(allocation.id, kind, bytes, true))
    Some(allocation)

  private def releaseAllocation(allocation: Allocation): Unit =
    if allocation.live then
      allocation.live = false
      allocations.remove(allocation.id)
      deps.allocationEvent(PreviewAllocationEvent(allocation.id, allocation.kind, allocation.bytes, false))
  private def retain(pixels: Pixels): Pixels =
    pixels.references += 1
    pixels
  private def releasePixels(pixels: Pixels): Unit =
    pixels.references -= 1
    require(pixels.references >= 0, "Preview ownership underflow")
    if pixels.references == 0 then releaseAllocation(pixels.allocation)

  private def pump(): Unit =
    queue.filterInPlace(wanted)
    while !terminating && active < 2 && queue.nonEmpty do
      val work = queue.remove(0)
      work.source = allocate("source", sourceLimit.toLong)
      active += 1
      running += work
      try executor.execute(() => run(work))
      catch case NonFatal(_) =>
        work.source.foreach(releaseAllocation)
        work.source = None
        running -= work
        active -= 1
        deliver(work, Left("Preview workers unavailable"))

  private def run(work: Work): Unit =
    try
      if isWanted(work) then
        val buffer = new Array[Byte](sourceLimit)
        val outcome = deps.readSnapshot(work.ref, buffer, () => !isWanted(work)).flatMap { snapshot =>
          if !(snapshot.bytes eq buffer) || snapshot.length < 0 || snapshot.length > sourceLimit ||
              snapshot.identity != work.ref.identity then Left("Preview snapshot identity changed")
          else decode(work, snapshot)
        }
        synchronized { deliver(work, outcome) }
    catch case NonFatal(error) =>
      synchronized { deliver(work, Left(s"Preview failed: ${Option(error.getMessage).getOrElse(error.getClass.getSimpleName).take(400)}")) }
    finally synchronized {
      work.source.foreach(releaseAllocation)
      work.source = None
      work.tickets.foreach(_.work = None)
      work.tickets.clear()
      running -= work
      val wasFull = active == 2
      active -= 1
      completedCount += 1
      pump()
      if wasFull && active < 2 then notifyCapacity()
    }

  private def decode(work: Work, snapshot: PreviewSourceSnapshot): Either[String, Pixels] =
    val key = CacheKey(snapshot.identity.fileKey, snapshot.sha256, work.edge)
    val previous: Option[Either[String, Pixels]] = synchronized {
      if !wanted(work) then Some(Left("Preview cancelled"))
      else cache.remove(key) match
        case Some(pixels) =>
          cache(key) = pixels
          Some(Right(retain(pixels)))
        case None => errors.get(key).map(Left(_))
    }
    previous.getOrElse {
      val result = deps.openReader(snapshot).flatMap { reader =>
        var produced: Option[Pixels] = None
        try
          val width = reader.width
          val height = reader.height
          if width <= 0 || height <= 0 || width > 16384 || height > 16384 || width.toLong * height > 32000000L then
            Left("Preview exceeds 32MP or 16384px dimension limit")
          else if !isWanted(work) then Left("Preview cancelled")
          else
            val sample = math.max(1, (math.max(width, height) + work.edge - 1) / work.edge)
            val w = (width + sample - 1) / sample
            val h = (height + sample - 1) / sample
            val result = raster(work, reader, sample, w, h)
            produced = result.toOption
            result
        finally
          try reader.close()
          catch case NonFatal(error) =>
            synchronized { produced.foreach(releasePixels) }
            throw error
      }
      synchronized {
        if wanted(work) then result match
          case Right(pixels) =>
            cache.remove(key).foreach(releasePixels)
            cache(key) = retain(pixels)
          case Left(reason) =>
            if reason != imageBusy && reason != "Preview cancelled" then
              errors.remove(key)
              errors(key) = reason.take(512)
              while errors.size > 128 do errors.remove(errors.head._1)
      }
      result
    }

  private def raster(work: Work, reader: PreviewImageReader, sample: Int,
      width: Int, height: Int): Either[String, Pixels] =
    val decodedReservation = synchronized { allocate("decoded", width.toLong * height * math.max(8, reader.pixelBytes)) }
    if decodedReservation.isEmpty then Left(imageBusy)
    else
      var decoded: BufferedImage = null
      var fxReservation: Option[Allocation] = None
      var rowReservation: Option[Allocation] = None
      var transferred = false
      try
        decoded = reader.read(0, sample)
        if decoded == null || decoded.getWidth != width || decoded.getHeight != height then
          Left("Preview decoder returned unexpected dimensions")
        else if !isWanted(work) then Left("Preview cancelled")
        else
          synchronized {
            fxReservation = allocate("fx", width.toLong * height * 4L)
            if fxReservation.nonEmpty then rowReservation = allocate("conversion", width.toLong * 4L)
          }
          if fxReservation.isEmpty || rowReservation.isEmpty then Left(imageBusy)
          else
            val image = new WritableImage(width, height)
            val row = new Array[Int](width)
            var y = 0
            while y < height do
              decoded.getRGB(0, y, width, 1, row, 0, width)
              image.getPixelWriter.setPixels(0, y, width, 1, PixelFormat.getIntArgbInstance(), row, 0, width)
              y += 1
            transferred = true
            Right(new Pixels(image, fxReservation.get))
      finally
        if decoded != null then decoded.flush()
        synchronized {
          rowReservation.foreach(releaseAllocation)
          if !transferred then fxReservation.foreach(releaseAllocation)
          releaseAllocation(decodedReservation.get)
        }

  private def deliver(work: Work, result: Either[String, Pixels]): Unit =
    work.accepting = false
    var admitted = false
    try
      if wanted(work) then work.tickets.toVector.filter(_.live).foreach { ticket =>
        ticket.status = ConsumerStatus.None
        val bytes = result.toOption.map(_.allocation.bytes).getOrElse(0L)
        if mailbox.size < 8 && mailboxBytes + bytes <= 8L * MiB then
          mailbox += Delivery(ticket, result.map(retain))
          admitted = true
        else
          rejectedCount += 1
          ticket.status = ConsumerStatus.DeliveryBusy
      }
      if !admitted then result.foreach { pixels =>
        cache.iterator.filter(_._2 eq pixels).map(_._1).toVector.foreach(key =>
          cache.remove(key).foreach(releasePixels))
      }
      if mailbox.nonEmpty || consumers.exists(_.status != ConsumerStatus.None) then scheduleDrain()
    finally result.foreach(releasePixels)

  private def mailboxBytes: Long = mailbox.iterator.flatMap(_.result.toOption).map(_.allocation.bytes).sum
  private def scheduleDrain(): Unit =
    if !drainScheduled then
      drainScheduled = true
      try deps.scheduleFx(() => drain())
      catch case NonFatal(_) =>
        drainScheduled = false
        mailbox.foreach(_.result.foreach(releasePixels))
        mailbox.clear()
        consumers.foreach(_.status = ConsumerStatus.None)
        capacityListeners.valuesIterator.foreach(_.pending = false)

  private def drain(): Unit =
    // One callback owns at most eight events, including overflow notifications and stale drops.
    var remaining = 8
    while remaining > 0 do
      remaining -= 1
      var capacityEvent: Option[CapacityListener] = None
      val next: Option[(Ticket, PreviewImageResult)] = synchronized {
        if mailbox.nonEmpty then
          val item = mailbox.remove(0)
          if item.ticket.live && generationLive(item.ticket.generation) then
            item.result match
              case Right(pixels) =>
                item.ticket.displayed.foreach(releasePixels)
                item.ticket.displayed = Some(pixels) // transfer mailbox reference to the view
                Some((item.ticket, PreviewImageResult.Ready(pixels.image)))
              case Left(reason) => Some((item.ticket, PreviewImageResult.Unavailable(reason)))
          else
            item.result.foreach(releasePixels)
            None
        else consumers.find(_.status != ConsumerStatus.None) match
          case Some(ticket) =>
            val reason = if ticket.status == ConsumerStatus.WorkersBusy then workersBusy else deliveryBusy
            ticket.status = ConsumerStatus.None
            Some((ticket, PreviewImageResult.Unavailable(reason)))
          case None =>
            capacityListeners.valuesIterator.find(_.pending) match
              case Some(listener) =>
                listener.pending = false
                capacityEvent = Some(listener)
              case None => remaining = 0
            None
      }
      next.foreach { case (ticket, result) => synchronized {
        if ticket.live && generationLive(ticket.generation) then safely(ticket.callback, result)
      }}
      capacityEvent.foreach(listener => synchronized {
        if generationLive(listener.generation) && capacityListeners.contains(listener.id) then
          try listener.callback()
          catch case NonFatal(_) => ()
      })
    synchronized {
      drainScheduled = false
      if mailbox.nonEmpty || consumers.exists(_.status != ConsumerStatus.None) ||
          capacityListeners.valuesIterator.exists(_.pending) then scheduleDrain()
    }

  private def safely(callback: PreviewImageResult => Unit, result: PreviewImageResult): Unit =
    try callback(result)
    catch case NonFatal(_) => ()

  private def cancelTicket(ticket: Ticket): Unit =
    if ticket.live then
      val consumersFull = consumers.size == 32
      ticket.live = false
      ticket.status = ConsumerStatus.None
      consumers -= ticket
      ticket.displayed.foreach(releasePixels)
      ticket.displayed = None
      mailbox.filter(_.ticket eq ticket).foreach(_.result.foreach(releasePixels))
      mailbox.filterInPlace(item => !(item.ticket eq ticket))
      ticket.work.foreach { work =>
        work.tickets -= ticket
        if work.tickets.isEmpty then work.aborted = true
      }
      ticket.work = None
      queue.filterInPlace(wanted)
      if consumersFull then notifyCapacity()

  def abandon(generation: Long): Unit = synchronized {
    current.filter(g => g.id == generation && g.live).foreach { g =>
      g.live = false
      abandoned += 1
      capacityListeners.filterInPlace((_, listener) => listener.generation != generation)
      consumers.toVector.filter(_.generation == generation).foreach(cancelTicket)
      queue.filterInPlace(_.generation != generation)
      cache.values.foreach(releasePixels)
      cache.clear()
      errors.clear()
      current = None
    }
  }

  def shutdownProcess(): Unit = synchronized {
    if !terminating then
      current.foreach(g => abandon(g.id))
      terminating = true
      // shutdown(), not an interrupt/termination claim. Active buffers remain owned until run exits.
      executor.shutdown()
  }

  def metrics: PreviewLoaderMetrics = synchronized {
    val sources = allocations.valuesIterator.filter(_.kind == "source").toVector
    val pixels = allocations.valuesIterator.filter(_.kind != "source").toVector
    PreviewLoaderMetrics(active, sources.size, sources.map(_.bytes).sum, queue.size,
      pixels.size, pixels.map(_.bytes).sum, cache.size, mailbox.size, mailboxBytes,
      consumers.size, if drainScheduled then 1 else 0, abandoned, rejectedCount, completedCount)
  }
