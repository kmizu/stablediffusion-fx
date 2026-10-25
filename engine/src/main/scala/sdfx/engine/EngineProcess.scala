package sdfx.engine

import sdfx.core.protocol.*

import java.io.{BufferedReader, BufferedWriter, InputStreamReader, OutputStreamWriter}
import java.nio.charset.StandardCharsets
import java.nio.file.Path
import java.util.concurrent.atomic.{AtomicBoolean, AtomicLong}
import java.util.concurrent.{ConcurrentHashMap, TimeUnit}
import scala.concurrent.duration.*
import scala.concurrent.{Await, Future, Promise}
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

/** How to launch the engine child process. */
final case class EngineLaunch(
    command: Seq[String],
    workDir: Path,
    env: Map[String, String] = Map.empty,
    helloTimeout: FiniteDuration = 180.seconds
)

object EngineProcess:
  val EngineCrashed = "engine_crashed"
  val EngineTimeout = "engine_timeout"
  val EngineStartFailed = "engine_start_failed"
  val EngineNotRunning = "engine_not_running"

  /** Starts the process and blocks until the `hello` line arrives (or throws [[EngineError]]).
    * `onExit(code, expected)` is invoked once when the process ends after a successful handshake;
    * `expected` is true when the exit was caused by [[EngineProcess.close]] or [[EngineProcess.kill]].
    */
  def start(launch: EngineLaunch, log: String => Unit = _ => (), onExit: (Int, Boolean) => Unit = (_, _) => ()): EngineProcess =
    val pb = new ProcessBuilder(launch.command.asJava)
    pb.directory(launch.workDir.toFile)
    val env = pb.environment()
    env.put("PYTHONUNBUFFERED", "1")
    env.put("PYTHONIOENCODING", "utf-8")
    env.put("PYTHONUTF8", "1")
    launch.env.foreach((k, v) => env.put(k, v))
    val proc =
      try pb.start()
      catch
        case NonFatal(e) =>
          throw EngineError(EngineStartFailed, s"cannot start engine: ${e.getMessage}", launch.command.mkString(" "))
    val ep = new EngineProcess(proc, log, onExit)
    ep.awaitHello(launch.helloTimeout)
    ep

/** One running engine child. Thread-safe. Use [[EngineProcess.start]]. */
final class EngineProcess private (proc: Process, log: String => Unit, onExit: (Int, Boolean) => Unit):
  import EngineProcess.*

  private case class Pending(promise: Promise[ujson.Value], onEvent: EngineEvent => Unit, tpe: String)

  private val pending = new ConcurrentHashMap[String, Pending]()
  private val ids = new AtomicLong(0)
  private val helloP = Promise[EngineHello]()
  private val closing = new AtomicBoolean(false)
  private val exited = new AtomicBoolean(false)
  private val exitP = Promise[Int]()
  @volatile private var forcedError: Option[EngineError] = None
  private val writer = new BufferedWriter(new OutputStreamWriter(proc.getOutputStream, StandardCharsets.UTF_8))
  private val stderrTail = new java.util.ArrayDeque[String]()

  // Listener failures must never kill the reader threads (that would look like an engine crash), hence Throwable.
  private def safeLog(s: String): Unit =
    try log(s)
    catch case _: Throwable => ()

  private def daemon(name: String)(body: => Unit): Thread =
    val t = new Thread(() => body, name)
    t.setDaemon(true)
    t.start()
    t

  private val stderrThread = daemon("sdfx-engine-stderr") {
    val r = new BufferedReader(new InputStreamReader(proc.getErrorStream, StandardCharsets.UTF_8))
    try
      var line = r.readLine()
      while line != null do
        stderrTail.synchronized {
          stderrTail.addLast(line)
          if stderrTail.size > 60 then stderrTail.removeFirst()
        }
        safeLog(line)
        line = r.readLine()
    catch case NonFatal(_) => ()
  }

  // BufferedReader.readLine grows its buffer as needed, so multi-megabyte lines (base64 previews) are fine.
  private val stdoutThread = daemon("sdfx-engine-stdout") {
    val r = new BufferedReader(new InputStreamReader(proc.getInputStream, StandardCharsets.UTF_8), 1 << 16)
    try
      var line = r.readLine()
      while line != null do
        try handleLine(line)
        catch case t: Throwable => safeLog(s"[sdfx] failed to handle engine line: $t")
        line = r.readLine()
    catch case NonFatal(_) => ()
    finally handleExit()
  }

  def pid: Long = proc.pid()
  def isAlive: Boolean = !exited.get() && proc.isAlive
  def hello: EngineHello =
    helloP.future.value.flatMap(_.toOption).getOrElse(throw EngineError(EngineNotRunning, "no handshake"))
  def pendingCount: Int = pending.size()
  def pendingTypes: Seq[String] = pending.values().asScala.map(_.tpe).toSeq
  /** Pending requests as (id, type), oldest first. */
  def pendingIds: Seq[(String, String)] =
    pending.entrySet().asScala.toSeq.map(e => (e.getKey, e.getValue.tpe)).sortBy((id, _) => id.drop(1).toLongOption.getOrElse(Long.MaxValue))
  def isPending(id: String): Boolean = pending.containsKey(id)
  /** Completion future of a pending request, if still pending. */
  def futureOf(id: String): Option[Future[ujson.Value]] = Option(pending.get(id)).map(_.promise.future)
  def exitFuture: Future[Int] = exitP.future
  def stderrLines: Seq[String] = stderrTail.synchronized(stderrTail.asScala.toList)

  private def awaitHello(timeout: FiniteDuration): Unit =
    try Await.result(helloP.future, timeout)
    catch
      case _: java.util.concurrent.TimeoutException =>
        kill()
        throw EngineError(EngineTimeout, s"engine did not say hello within ${timeout.toMillis} ms", stderrLines.mkString("\n"))
      case e: EngineError =>
        kill()
        throw e
      case NonFatal(e) =>
        kill()
        throw EngineError(EngineStartFailed, s"engine handshake failed: ${e.getMessage}", stderrLines.mkString("\n"))

  private def handleLine(line: String): Unit =
    val trimmed = line.trim
    if trimmed.isEmpty then ()
    else if !trimmed.startsWith("{") then safeLog(s"[stdout] $trimmed")
    else
      val js =
        try Some(ujson.read(trimmed))
        catch case NonFatal(_) => None
      js match
        case Some(o: ujson.Obj) => dispatch(o)
        case _                  => safeLog(s"[stdout] $trimmed")

  private def dispatch(o: ujson.Obj): Unit =
    val tpe = o.value.get("type").collect { case ujson.Str(s) => s }.getOrElse("")
    val id = o.value.get("id").collect { case ujson.Str(s) => s; case ujson.Num(n) => n.toLong.toString }
    tpe match
      case "hello" =>
        try
          val h = Json.fromJs[EngineHello](o)
          if !helloP.trySuccess(h) then safeLog("[engine] duplicate hello ignored")
        catch case NonFatal(e) => helloP.tryFailure(EngineError(EngineStartFailed, s"bad hello: ${e.getMessage}", o.toString))
      case "result" =>
        id.flatMap(i => Option(pending.remove(i))) match
          case Some(p) => p.promise.trySuccess(o.value.getOrElse("result", ujson.Obj()))
          case None    => safeLog(s"[engine] result for unknown id ${id.getOrElse("null")}")
      case "error" =>
        def s(k: String) = o.value.get(k).collect { case ujson.Str(v) => v }.getOrElse("")
        val code = if s("code").nonEmpty then s("code") else Protocol.ErrorCodes.Internal
        val err = EngineError(code, s("message"), s("detail"))
        id.flatMap(i => Option(pending.remove(i))) match
          case Some(p) => p.promise.tryFailure(err)
          case None    => safeLog(s"[engine] error: ${err.getMessage}")
      case "event" =>
        val ev =
          try EngineEvent.parse(o)
          catch case NonFatal(_) => None
        ev match
          case Some(EngineEvent.Log(level, msg)) => safeLog(s"[$level] $msg")
          case _                                 => ()
        for
          e <- ev
          i <- id
          p <- Option(pending.get(i))
        do
          try p.onEvent(e)
          catch case ex: Throwable => safeLog(s"[sdfx] event listener failed: $ex")
      case other => safeLog(s"[engine] unknown message type '$other'")

  private def handleExit(): Unit =
    if exited.compareAndSet(false, true) then
      val code =
        try
          if proc.waitFor(5, TimeUnit.SECONDS) then proc.exitValue()
          else
            proc.destroyForcibly()
            -1
        catch case NonFatal(_) => -1
      stderrThread.join(1000)
      val err = forcedError.getOrElse(EngineError(EngineCrashed, s"engine exited (code $code)", stderrLines.takeRight(20).mkString("\n")))
      helloP.tryFailure(EngineError(EngineStartFailed, s"engine exited before handshake (code $code)", stderrLines.mkString("\n")))
      failAll(err)
      exitP.trySuccess(code)
      if helloP.future.value.exists(_.isSuccess) then
        try onExit(code, closing.get())
        catch case e: Throwable => safeLog(s"[sdfx] exit listener failed: $e")

  private def failAll(err: EngineError): Unit =
    pending.keySet().asScala.toList.foreach { k =>
      Option(pending.remove(k)).foreach(_.promise.tryFailure(err))
    }

  /** Sends a request. Events for this request are delivered (on the reader thread) before the future completes. */
  def request(tpe: String, params: ujson.Value = ujson.Obj(), onEvent: EngineEvent => Unit = _ => ()): Future[ujson.Value] =
    requestWithId(tpe, params, onEvent)._2

  /** Like [[request]] but also returns the request id. */
  def requestWithId(tpe: String, params: ujson.Value = ujson.Obj(), onEvent: EngineEvent => Unit = _ => ()): (String, Future[ujson.Value]) =
    if !isAlive then ("", Future.failed(forcedError.getOrElse(EngineError(EngineNotRunning, "engine is not running"))))
    else
      val id = s"r${ids.incrementAndGet()}"
      val p = Pending(Promise[ujson.Value](), onEvent, tpe)
      pending.put(id, p)
      val msg = ujson.write(ujson.Obj("id" -> id, "type" -> tpe, "params" -> params))
      try
        writer.synchronized {
          writer.write(msg)
          writer.write('\n')
          writer.flush()
        }
      catch
        case NonFatal(e) =>
          pending.remove(id)
          p.promise.tryFailure(forcedError.getOrElse(EngineError(EngineCrashed, s"cannot write to engine: ${e.getMessage}")))
      if exited.get() then failAll(forcedError.getOrElse(EngineError(EngineCrashed, "engine exited")))
      (id, p.promise.future)

  /** Graceful stop: sends `shutdown`, waits up to `grace`, then kills. */
  def close(grace: FiniteDuration = 5.seconds): Unit =
    closing.set(true)
    if isAlive then
      try
        val f = request(Protocol.Types.Shutdown)
        try Await.ready(f, grace)
        catch case NonFatal(_) => ()
        try writer.close()
        catch case NonFatal(_) => ()
        if !proc.waitFor(grace.toMillis, TimeUnit.MILLISECONDS) then kill()
      catch case NonFatal(_) => kill()
    stdoutThread.join(3000)

  /** Kills the process now. Pending requests fail with `failWith` (default: engine_crashed). */
  def kill(failWith: Option[EngineError] = None): Unit =
    closing.set(true)
    failWith.foreach { e =>
      forcedError = Some(e)
      failAll(e)
    }
    try
      proc.descendants().forEach(h => { h.destroyForcibly(); () })
      proc.destroyForcibly()
    catch case NonFatal(_) => ()
