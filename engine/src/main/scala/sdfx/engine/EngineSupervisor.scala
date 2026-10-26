package sdfx.engine

import sdfx.core.protocol.*

import java.util.concurrent.{CopyOnWriteArrayList, Executors, ScheduledExecutorService, ThreadFactory, TimeUnit}
import scala.concurrent.duration.*
import scala.concurrent.{Await, ExecutionContext, Future, Promise}
import scala.util.control.NonFatal
import scala.util.{Failure, Success, Try}

enum EngineState:
  case Stopped
  case Starting
  case Ready(hello: EngineHello)
  case Busy
  case Crashed(message: String)

  def isUsable: Boolean = this match
    case Ready(_) | Busy => true
    case _               => false

/** Owns the engine process lifecycle: start/stop, crash detection with bounded auto-restart,
  * remembering the loaded model (re-issued after a restart) and interrupt with forced restart.
  *
  * State listeners are always invoked without internal locks held.
  *
  * @param launch evaluated on every (re)start so settings changes apply
  * @param interruptGrace how long a generation may keep running after `interrupt()` before the engine is restarted
  * @param loadInterruptGrace the same for a `load_model` / `set_vae` that is pending when `interrupt()` is called
  * @param startTimeout upper bound for [[start]] (handshake + model reload)
  * @param reloadTimeout upper bound for re-issuing the last model after a restart
  */
final class EngineSupervisor(
    launch: () => EngineLaunch,
    log: String => Unit = _ => (),
    maxRestarts: Int = 3,
    restartWindow: FiniteDuration = 5.minutes,
    interruptGrace: FiniteDuration = 10.seconds,
    shutdownGrace: FiniteDuration = 5.seconds,
    loadInterruptGrace: FiniteDuration = 60.seconds,
    startTimeout: FiniteDuration = 45.minutes,
    reloadTimeout: FiniteDuration = 30.minutes
):
  import EngineProcess.*

  /** Requests that make the engine "busy" (everything that may take long). */
  val heavyTypes: Set[String] = Set(Protocol.Types.Txt2Img, Protocol.Types.Img2Img, Protocol.Types.Upscale,
    Protocol.Types.Preprocess, Protocol.Types.LoadModel, Protocol.Types.SetVae, Protocol.Types.SystemBenchmark)
  private val generationTypes: Set[String] = Set(Protocol.Types.Txt2Img, Protocol.Types.Img2Img, Protocol.Types.Upscale, Protocol.Types.Preprocess,
    Protocol.Types.SystemBenchmark)
  private val loadTypes: Set[String] = Set(Protocol.Types.LoadModel, Protocol.Types.SetVae)

  private val lock = new Object
  @volatile private var proc: Option[EngineProcess] = None
  @volatile private var _state: EngineState = EngineState.Stopped
  @volatile private var _hello: Option[EngineHello] = None
  @volatile private var startFuture: Option[Future[EngineProcess]] = None
  @volatile private var stopped = true
  @volatile private var gaveUp = false
  @volatile private var restartRequested = false
  /** Incremented by stop(): a start attempt from an older epoch never installs its process or clears a newer start. */
  @volatile private var epoch = 0L
  @volatile private var lastLoad: Option[ujson.Value] = None
  @volatile private var lastVae: Option[ujson.Value] = None
  @volatile private var _loaded: Option[LoadedModel] = None
  private var crashTimes = List.empty[Long]
  private var inflight = 0
  private val listeners = new CopyOnWriteArrayList[EngineState => Unit]()

  private val scheduler: ScheduledExecutorService = Executors.newSingleThreadScheduledExecutor(daemonFactory("sdfx-supervisor"))
  private given ec: ExecutionContext = ExecutionContext.fromExecutor(Executors.newCachedThreadPool(daemonFactory("sdfx-supervisor-worker")))

  private def daemonFactory(name: String): ThreadFactory = r =>
    val t = new Thread(r, name)
    t.setDaemon(true)
    t

  def state: EngineState = _state
  def hello: Option[EngineHello] = _hello
  /** Model reported by the last successful load_model / set_vae. */
  def loadedModel: Option[LoadedModel] = _loaded
  def isRunning: Boolean = proc.exists(_.isAlive)
  def pid: Option[Long] = proc.filter(_.isAlive).map(_.pid)

  def addStateListener(l: EngineState => Unit): () => Unit =
    listeners.add(l)
    () => { listeners.remove(l); () }

  /** Updates the state and notifies listeners. Must not be called while holding `lock`. */
  private def setState(s: EngineState): Unit =
    val changed = lock.synchronized {
      val c = _state != s
      _state = s
      c
    }
    if changed then notifyState(s)

  private def notifyState(s: EngineState): Unit =
    listeners.forEach { l =>
      try l(s)
      catch case t: Throwable => log(s"[sdfx] state listener failed: $t")
    }

  /** Starts the engine (blocking until handshake and model reload). No-op when already running. */
  def start(): EngineHello =
    awaitStarted(beginStart())

  /** Like [[start]] without blocking the caller; a later [[stop]] wins over this start. */
  def startAsync(): Future[EngineHello] =
    val f = beginStart()
    Future(awaitStarted(f))

  private def beginStart(): Future[EngineProcess] =
    lock.synchronized {
      stopped = false
      gaveUp = false
      crashTimes = Nil
    }
    ensureStarted()

  private def awaitStarted(f: Future[EngineProcess]): EngineHello =
    try Await.result(f, startTimeout).hello
    catch
      case _: java.util.concurrent.TimeoutException =>
        throw EngineError(EngineTimeout, s"engine did not become ready within $startTimeout")

  private def ensureStarted(): Future[EngineProcess] =
    var launchP: Option[Promise[EngineProcess]] = None
    val f = lock.synchronized {
      proc.filter(_.isAlive) match
        case Some(p) if startFuture.isEmpty => Future.successful(p)
        case _ =>
          startFuture match
            case Some(f) => f
            case None =>
              if stopped then Future.failed(EngineError(EngineNotRunning, "engine is stopped"))
              else if gaveUp then Future.failed(EngineError(EngineCrashed, "engine crashed repeatedly; restart it"))
              else
                val p = Promise[EngineProcess]()
                startFuture = Some(p.future)
                _state = EngineState.Starting
                launchP = Some(p)
                p.future
    }
    launchP.foreach { p =>
      val myEpoch = epoch
      notifyState(EngineState.Starting)
      p.future.onComplete { r =>
        val (current, again) = lock.synchronized {
          val cur = epoch == myEpoch && startFuture.contains(p.future)
          if cur then startFuture = None
          val a = cur && restartRequested && !stopped && !gaveUp
          if cur then restartRequested = false
          (cur, a)
        }
        r match
          case Success(_) => ()
          case Failure(e) if current => setState(if stopped then EngineState.Stopped else EngineState.Crashed(e.getMessage))
          case Failure(_) => ()
        if again then
          log("[sdfx] engine died while starting; trying again")
          ensureStarted().failed.foreach(e => log(s"[sdfx] restart failed: ${e.getMessage}"))
      }
      Future {
        // Throwable, not Try: a fatal Error (e.g. NoClassDefFoundError) must still complete the start future.
        try p.trySuccess(launchNow(myEpoch))
        catch case t: Throwable => p.tryFailure(t)
      }
    }
    f

  private def launchNow(myEpoch: Long): EngineProcess =
    def live = !stopped && epoch == myEpoch
    val l = launch()
    log(s"[sdfx] starting engine: ${l.command.mkString(" ")} (cwd ${l.workDir})")
    lazy val ep: EngineProcess = EngineProcess.start(l, log, (code, expected) => onProcessExit(ep, code, expected))
    val p = ep
    val keep = lock.synchronized {
      if !live then false
      else
        proc = Some(p)
        _hello = Some(p.hello)
        true
    }
    if !keep then
      p.close(shutdownGrace)
      throw EngineError(EngineNotRunning, "engine was stopped while starting")
    log(s"[sdfx] engine ready: ${p.hello.engineVersion} device=${p.hello.device} dryRun=${p.hello.dryRun}")
    // Re-issue the last model (and VAE) so a restart is transparent to the user.
    // Policy: a reload that *times out* means the engine is wedged -> kill it and fail the start (state Crashed).
    // A reload that *fails* (e.g. checkpoint deleted) -> engine is usable: Ready, with loadedModel / remembered model cleared.
    def reload(tpe: String, params: ujson.Value, onOk: ujson.Value => Unit): Unit =
      try onOk(Await.result(p.request(tpe, params), reloadTimeout))
      catch
        case _: java.util.concurrent.TimeoutException =>
          lock.synchronized { if proc.contains(p) then proc = None }
          p.kill()
          throw EngineError(EngineTimeout, s"engine did not finish reloading ($tpe) within $reloadTimeout; restart it")
        case NonFatal(e) if !p.isAlive =>
          // Process died during the reload: keep the remembered model so the next restart retries it.
          throw EngineError(EngineCrashed, s"engine died while reloading the model ($tpe)", p.stderrLines.takeRight(20).mkString("\n"))
        case NonFatal(e) =>
          log(s"[sdfx] $tpe after restart failed, continuing without a model: ${e.getMessage}")
          forgetModel()
    lastLoad.foreach { params =>
      log("[sdfx] reloading last model after restart")
      reload(Protocol.Types.LoadModel, params, r => _loaded = Try(Json.fromJs[LoadedModel](r)).toOption)
    }
    lastVae.foreach { params =>
      reload(Protocol.Types.SetVae, params, r => Try(Json.fromJs[LoadedModel](r)).toOption.foreach(m => _loaded = Some(m)))
    }
    if !p.isAlive then throw EngineError(EngineCrashed, "engine died while reloading the model", p.stderrLines.takeRight(20).mkString("\n"))
    val ready = lock.synchronized(live && proc.contains(p))
    if !ready then
      p.close(shutdownGrace)
      throw EngineError(EngineNotRunning, "engine was stopped while starting")
    setState(EngineState.Ready(p.hello))
    p

  private def onProcessExit(p: EngineProcess, code: Int, expected: Boolean): Unit =
    val current = lock.synchronized(proc.contains(p))
    if current && !expected && !stopped then
      val msg = s"engine exited unexpectedly (code $code)"
      log(s"[sdfx] $msg")
      val now = System.currentTimeMillis()
      val (allowed, n, starting) = lock.synchronized {
        crashTimes = (now :: crashTimes).filter(t => now - t <= restartWindow.toMillis)
        inflight = 0
        proc = None
        val ok = crashTimes.size <= maxRestarts
        if !ok then gaveUp = true
        // A start in progress (e.g. died during the model reload) restarts once that attempt has failed.
        if ok && startFuture.isDefined then restartRequested = true
        (ok, crashTimes.size, startFuture.isDefined)
      }
      if !starting then setState(EngineState.Crashed(msg))
      if allowed then
        log(s"[sdfx] restarting engine ($n/$maxRestarts)")
        if !starting then ensureStarted().failed.foreach(e => log(s"[sdfx] restart failed: ${e.getMessage}"))
      else log(s"[sdfx] engine crashed more than $maxRestarts times within $restartWindow; not restarting")

  /** Sends a request, waiting for a pending (re)start first. Model bookkeeping happens before the returned
    * future completes, so `loadedModel` is current when a caller observes the result.
    */
  def request(tpe: String, params: ujson.Value = ujson.Obj(), onEvent: EngineEvent => Unit = _ => ()): Future[ujson.Value] =
    ensureStarted().flatMap { p =>
      val heavy = heavyTypes.contains(tpe)
      if heavy then markBusy(+1, p)
      p.request(tpe, params, onEvent).transform { r =>
        r match
          case Success(v) if tpe == Protocol.Types.LoadModel =>
            lastLoad = Some(params)
            lastVae = None
            _loaded = Try(Json.fromJs[LoadedModel](v)).toOption
          case Success(v) if tpe == Protocol.Types.SetVae =>
            lastVae = Some(params)
            Try(Json.fromJs[LoadedModel](v)).toOption.foreach(m => _loaded = Some(m))
          case Success(_) if tpe == Protocol.Types.Unload =>
            lastLoad = None
            lastVae = None
            _loaded = None
          case _ => ()
        if heavy then markBusy(-1, p)
        r
      }(using ExecutionContext.parasitic)
    }

  private def markBusy(delta: Int, p: EngineProcess): Unit =
    val next = lock.synchronized {
      // Completions of a process that is no longer current must not disturb the new process' counter.
      if proc.contains(p) then inflight = math.max(0, inflight + delta)
      (inflight, proc.filter(_.isAlive).map(_.hello))
    }
    next match
      case (n, Some(_)) if n > 0 && _state != EngineState.Busy => setState(EngineState.Busy)
      case (0, Some(h)) if _state == EngineState.Busy          => setState(EngineState.Ready(h))
      case _                                                    => ()

  /** Asks the engine to stop the current generation. If the request that was running at interrupt time is
    * still pending after the grace period the engine is force-restarted and that request fails with an
    * `interrupted` error. Later requests are never affected by an old interrupt.
    */
  def interrupt(): Unit =
    proc.filter(_.isAlive).foreach { p =>
      val pend = p.pendingIds
      val target = pend.find((_, t) => generationTypes.contains(t)).orElse(pend.find((_, t) => loadTypes.contains(t)))
      target match
        case Some((id, tpe)) =>
          p.request(Protocol.Types.Interrupt, ujson.Obj("target_id" -> id)).failed.foreach(e => log(s"[sdfx] interrupt failed: ${e.getMessage}"))
          val grace = if loadTypes.contains(tpe) then loadInterruptGrace else interruptGrace
          val task = scheduler.schedule(
            (() => {
              if proc.contains(p) && p.isAlive && p.isPending(id) then
                log(s"[sdfx] engine did not stop $tpe within $grace; restarting it")
                forceRestart(EngineError(Protocol.ErrorCodes.Interrupted, "generation interrupted (engine restarted)"))
            }): Runnable,
            grace.toMillis,
            TimeUnit.MILLISECONDS
          )
          p.futureOf(id) match
            case Some(f) => f.onComplete(_ => task.cancel(false))(using ExecutionContext.parasitic)
            case None    => task.cancel(false)
        case None => p.request(Protocol.Types.Interrupt)
    }

  /** Kills the running engine (pending requests fail with `reason`), waits briefly for it to exit (VRAM release)
    * and starts a new one. No-op (failed future) when the supervisor is stopped.
    */
  def forceRestart(reason: EngineError = EngineError(EngineCrashed, "engine restarted")): Future[EngineHello] =
    if stopped then Future.failed(EngineError(EngineNotRunning, "engine is stopped"))
    else
      val old = lock.synchronized {
        val o = proc
        proc = None
        inflight = 0
        o
      }
      old.foreach(_.kill(Some(reason)))
      gaveUp = false
      Future {
        old.foreach(o =>
          try Await.ready(o.exitFuture, 5.seconds)
          catch case NonFatal(_) => ()
        )
      }.flatMap(_ => ensureStarted()).map(_.hello)

  /** Restarts the engine gracefully (e.g. after settings change). */
  def restart(): EngineHello =
    val old = lock.synchronized {
      val o = proc
      proc = None
      o
    }
    old.foreach(_.close(shutdownGrace))
    start()

  def stop(): Unit =
    val (old, starting) = lock.synchronized {
      stopped = true
      epoch += 1
      val o = proc
      val sf = startFuture
      proc = None
      inflight = 0
      // A later start() must never be handed this attempt's future; it closes its own process.
      startFuture = None
      restartRequested = false
      (o, sf)
    }
    old.foreach(_.close(shutdownGrace))
    // A start in flight closes its own process when it notices `stopped`; wait for it (bounded).
    starting.foreach(f =>
      try Await.ready(f, shutdownGrace * 3)
      catch case NonFatal(_) => ()
    )
    setState(EngineState.Stopped)

  /** Forget the remembered model (e.g. after the user unloads it). */
  def forgetModel(): Unit =
    lastLoad = None
    lastVae = None
    _loaded = None

  def close(): Unit =
    stop()
    scheduler.shutdownNow()
