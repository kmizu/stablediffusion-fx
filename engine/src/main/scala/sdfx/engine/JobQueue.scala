package sdfx.engine

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.core.protocol.*

import java.nio.file.{Files, Path}
import java.util.UUID
import java.util.concurrent.atomic.{AtomicBoolean, AtomicLong}
import java.util.concurrent.{CopyOnWriteArrayList, LinkedBlockingQueue, TimeUnit}
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.{Future, Promise}
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

/** One queued unit of work. `script` takes over the run; `alwaysOn` plugins process every generation. */
final case class Job(
    id: String,
    params: GenerationParams,
    script: Option[(Plugin, Values)] = None,
    alwaysOn: Seq[(Plugin, Values)] = Nil,
    origin: String = "ui",
    limits: JobLimits = JobLimits()
)

object Job:
  private val counter = new AtomicLong(0)
  def newId(): String = s"job-${counter.incrementAndGet()}-${UUID.randomUUID().toString.take(8)}"
  def apply(params: GenerationParams): Job = Job(newId(), params)

/** Thrown into the job future when a job was cancelled or interrupted; `partial` holds images saved so far. */
final case class JobCancelledException(jobId: String, partial: JobOutput) extends Exception(s"job $jobId cancelled")

sealed trait JobEvent:
  def jobId: String

object JobEvent:
  final case class JobQueued(job: Job, position: Int) extends JobEvent { def jobId: String = job.id }
  final case class JobStarted(job: Job) extends JobEvent { def jobId: String = job.id }
  /** Engine progress plus the script status (`text`, `done`/`total` sub-generations). */
  final case class Progress(jobId: String, progress: EngineEvent.Progress, text: String, done: Int, total: Int) extends JobEvent
  final case class Status(jobId: String, text: String, done: Int, total: Int) extends JobEvent
  final case class Preview(jobId: String, jpegB64: String, step: Int) extends JobEvent
  final case class ImageSaved(jobId: String, image: OutputImage, grid: Boolean) extends JobEvent
  final case class Log(jobId: String, message: String) extends JobEvent
  final case class JobFinished(jobId: String, output: JobOutput) extends JobEvent
  final case class JobFailed(jobId: String, error: Throwable) extends JobEvent:
    def message: String = Option(error.getMessage).getOrElse(error.toString)
  final case class JobCancelled(jobId: String, partial: JobOutput) extends JobEvent

/** Runs a single job against a [[Generator]]. Stateless apart from the per-run listener. */
object JobRunner:
  /** Non-script jobs: one generation (AlwaysOn applied once inside the job context).
    * Script jobs: the script receives the raw params and every `ctx.generate` it makes gets AlwaysOn processing.
    */
  def run(job: Job, generator: Generator, listener: GenerationListener): JobOutput =
    val ctx = generator.forJob(listener, job.alwaysOn, job.limits)
    job.script match
      case Some((plugin, values)) => plugin.run(job.params, values, ctx)
      case None                   => JobOutput(ctx.generate(job.params))

  def isInterruption(e: Throwable): Boolean = e match
    case ee: EngineError => ee.isInterrupted
    case _: InterruptedException => true
    case _ =>
      val c = e.getCause
      c != null && (c ne e) && isInterruption(c)

/** FIFO job queue with a single worker thread. */
final class JobQueue(generator: Generator, interruptEngine: () => Unit):
  private final class Entry(val job: Job, val promise: Promise[JobOutput])

  private val queue = new LinkedBlockingQueue[Entry]()
  private val listeners = new CopyOnWriteArrayList[JobEvent => Unit]()
  @volatile private var running: Option[(Entry, AtomicBoolean)] = None
  @volatile private var stopped = false

  def addListener(l: JobEvent => Unit): () => Unit =
    listeners.add(l)
    () => { listeners.remove(l); () }

  private def emit(e: JobEvent): Unit =
    listeners.forEach { l =>
      try l(e)
      catch case ex: Throwable => System.err.println(s"[sdfx] job listener failed: $ex")
    }

  def submit(job: Job): Future[JobOutput] =
    val e = new Entry(job, Promise[JobOutput]())
    if stopped then e.promise.failure(new IllegalStateException("job queue is shut down"))
    else
      queue.put(e)
      emit(JobEvent.JobQueued(job, queue.size() + (if running.isDefined then 1 else 0)))
    e.promise.future

  /** Jobs waiting to run (excluding the current one). */
  def pending: Seq[Job] = queue.asScala.toSeq.map(_.job)
  def current: Option[Job] = running.map(_._1.job)
  def isIdle: Boolean = running.isEmpty && queue.isEmpty

  /** Stops the running job (its script loop and the engine generation). */
  def interruptCurrent(): Boolean =
    running match
      case Some((_, flag)) =>
        flag.set(true)
        try interruptEngine()
        catch case NonFatal(_) => ()
        true
      case None => false

  /** Cancels a pending job or interrupts it if it is running. */
  def cancel(jobId: String): Boolean =
    if running.exists(_._1.job.id == jobId) then interruptCurrent()
    else
      queue.asScala.find(_.job.id == jobId) match
        case Some(e) if queue.remove(e) =>
          e.promise.tryFailure(JobCancelledException(jobId, JobOutput(Nil)))
          emit(JobEvent.JobCancelled(jobId, JobOutput(Nil)))
          true
        case _ => false

  /** Drops all pending jobs; returns how many were removed. The running job continues. */
  def clear(): Int =
    val drained = new java.util.ArrayList[Entry]()
    queue.drainTo(drained)
    drained.asScala.foreach { e =>
      e.promise.tryFailure(JobCancelledException(e.job.id, JobOutput(Nil)))
      emit(JobEvent.JobCancelled(e.job.id, JobOutput(Nil)))
    }
    drained.size()

  def shutdown(): Unit =
    stopped = true
    clear()
    interruptCurrent()
    worker.interrupt()

  private val worker: Thread =
    val t = new Thread(() => loop(), "sdfx-job-worker")
    t.setDaemon(true)
    t.start()
    t

  private def loop(): Unit =
    while !stopped do
      val e =
        try queue.poll(500, TimeUnit.MILLISECONDS)
        catch case _: InterruptedException => null
      if e != null then
        // The worker must survive anything a job throws (OOM in a grid, LinkageError from a plugin jar...).
        try runEntry(e)
        catch
          case t: Throwable =>
            System.err.println(s"[sdfx] job worker error: $t")
            e.promise.tryFailure(t)

  private def runEntry(e: Entry): Unit =
    val flag = new AtomicBoolean(false)
    running = Some((e, flag))
    val id = e.job.id
    val images = ArrayBuffer.empty[OutputImage]
    val grids = ArrayBuffer.empty[OutputImage]
    val temps = java.util.concurrent.ConcurrentHashMap.newKeySet[Path]()
    @volatile var statusText = ""
    @volatile var done = 0
    @volatile var total = 0
    val listener = new GenerationListener:
      override def onProgress(p: EngineEvent.Progress): Unit = emit(JobEvent.Progress(id, p, statusText, done, total))
      override def onPreview(p: EngineEvent.Preview): Unit = emit(JobEvent.Preview(id, p.jpegB64, p.step))
      override def onLog(message: String): Unit = emit(JobEvent.Log(id, message))
      override def onStatus(text: String, d: Int, t: Int): Unit =
        statusText = text
        done = d
        total = t
        emit(JobEvent.Status(id, text, d, t))
      override def onImageSaved(image: OutputImage, grid: Boolean): Unit =
        images.synchronized { if grid then grids += image else images += image }
        emit(JobEvent.ImageSaved(id, image, grid))
      override def onTempFile(path: Path): Unit = temps.add(path.toAbsolutePath.normalize())
      override def isInterrupted: Boolean = flag.get()
    def partial = images.synchronized(JobOutput(images.toSeq, grids.toSeq))
    // Deletes every temp file registered during the job except the ones in its final output. Runs once,
    // before the job's terminal event / promise completion.
    val cleaned = new AtomicBoolean(false)
    def cleanup(keep: JobOutput): Unit =
      if cleaned.compareAndSet(false, true) then
        val kept = keep.all.map(_.path.toAbsolutePath.normalize()).toSet
        temps.asScala.filterNot(kept.contains).foreach { p =>
          try Files.deleteIfExists(p)
          catch case NonFatal(_) => ()
        }
    try
      emit(JobEvent.JobStarted(e.job))
      val out = JobRunner.run(e.job, generator, listener)
      if flag.get() then
        val o = if out.all.nonEmpty then out else partial
        cleanup(o)
        emit(JobEvent.JobCancelled(id, o))
        e.promise.tryFailure(JobCancelledException(id, o))
      else
        cleanup(out)
        emit(JobEvent.JobFinished(id, out))
        e.promise.trySuccess(out)
    catch
      case ex: Throwable if ex.isInstanceOf[InterruptedException] || (NonFatal(ex) && (flag.get() || JobRunner.isInterruption(ex))) =>
        val o = partial
        cleanup(o)
        emit(JobEvent.JobCancelled(id, o))
        e.promise.tryFailure(JobCancelledException(id, o))
      case ex: Throwable =>
        // Includes fatal errors (OutOfMemoryError, LinkageError): fail the job, keep the queue alive.
        try
          cleanup(partial)
          emit(JobEvent.JobFailed(id, ex))
        finally e.promise.tryFailure(ex)
    finally
      try cleanup(partial)
      finally running = None
