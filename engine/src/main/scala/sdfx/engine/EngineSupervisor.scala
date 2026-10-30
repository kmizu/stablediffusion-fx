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
    Protocol.Types.Preprocess, Protocol.Types.LoadModel, Protocol.Types.SetVae, Protocol.Types.SystemBenchmark, Protocol.Types.VramEstimator, Protocol.Types.VaeTools,
    Protocol.Types.TemporalLab)
  private val readerTypes: Set[String] = Set(Protocol.Types.Ping, Protocol.Types.Interrupt,
    Protocol.Types.InterruptionProtection, Protocol.Types.RestartAuthorization, Protocol.Types.SystemStatus, Protocol.Types.Shutdown)
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

  private def ensureStarted(expectedEpoch: Option[Long] = None): Future[EngineProcess] =
    var launchP: Option[(Promise[EngineProcess], Long)] = None
    val f = lock.synchronized {
      if expectedEpoch.exists(_ != epoch) then Future.failed(EngineError(EngineNotRunning, "start was superseded by stop"))
      else proc.filter(_.isAlive) match
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
                launchP = Some((p, epoch))
                p.future
    }
    launchP.foreach { (p, myEpoch) =>
      notifyState(EngineState.Starting)
      def complete(r: Try[EngineProcess]): Unit =
        val message = r match
          case Failure(e) =>
            try Option(e.getMessage).getOrElse(e.getClass.getName)
            catch case _: Throwable => e.getClass.getName
          case Success(_) => ""
        // Await callers can resume as soon as the promise fails. Publish the terminal
        // state and finish this attempt's bookkeeping before exposing that completion.
        val (changedState, again) = lock.synchronized {
          val cur = epoch == myEpoch && startFuture.contains(p.future)
          val next: Option[EngineState] = r match
            case Failure(_) if cur => Some(if stopped then EngineState.Stopped else EngineState.Crashed(message))
            case _ => None
          val changed = next.filter(_ != _state)
          next.foreach(s => _state = s)
          if cur then startFuture = None
          val a = cur && restartRequested && !stopped && !gaveUp
          if cur then restartRequested = false
          (changed, a)
        }
        p.tryComplete(r)
        // Listeners may call stop/restart: never invoke them under the lock or while
        // this attempt's promise is unfinished, or they could wait on themselves.
        changedState.foreach(notifyState)
        if again then
          log("[sdfx] engine died while starting; trying again")
          ensureStarted().failed.foreach(e => log(s"[sdfx] restart failed: ${e.getMessage}"))
      Future {
        // Throwable, not Try: a fatal Error (e.g. NoClassDefFoundError) must still complete the start future.
        val result = try Success(launchNow(myEpoch)) catch case t: Throwable => Failure(t)
        complete(result)
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
      try
        val value = Await.result(p.request(tpe, params), reloadTimeout)
        lock.synchronized {
          if live && proc.contains(p) then onOk(value)
        }
      catch
        case _: java.util.concurrent.TimeoutException =>
          lock.synchronized { if proc.contains(p) then proc = None }
          p.kill()
          throw EngineError(EngineTimeout, s"engine did not finish reloading ($tpe) within $reloadTimeout; restart it")
        case NonFatal(e) if !p.isAlive =>
          // Process died during the reload: keep the remembered model so the next restart retries it.
          throw EngineError(EngineCrashed, s"engine died while reloading the model ($tpe)", p.stderrLines.takeRight(20).mkString("\n"))
        case NonFatal(e) =>
          val forgotten = lock.synchronized {
            if live && proc.contains(p) then
              forgetModel()
              true
            else false
          }
          if forgotten then log(s"[sdfx] $tpe after restart failed, continuing without a model: ${e.getMessage}")
    lastLoad.foreach { params =>
      log("[sdfx] reloading last model after restart")
      reload(Protocol.Types.LoadModel, params, r => _loaded = Try(Json.fromJs[LoadedModel](r)).toOption)
    }
    lastVae.foreach { params =>
      reload(Protocol.Types.SetVae, params, r => Try(Json.fromJs[LoadedModel](r)).toOption.foreach(m => _loaded = Some(m)))
    }
    if !p.isAlive then throw EngineError(EngineCrashed, "engine died while reloading the model", p.stderrLines.takeRight(20).mkString("\n"))
    val ready = EngineState.Ready(p.hello)
    val published = lock.synchronized {
      if live && proc.contains(p) then
        val changed = _state != ready
        _state = ready
        Some(changed)
      else None
    }
    if published.isEmpty then
      p.close(shutdownGrace)
      throw EngineError(EngineNotRunning, "engine was stopped while starting")
    if published.contains(true) then notifyState(ready)
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
    requestOwned(tpe, params, onEvent, None)

  private[engine] def requestOwned(tpe: String, params: ujson.Value, onEvent: EngineEvent => Unit,
                                   owner: Option[String]): Future[ujson.Value] =
    ensureStarted().flatMap { p =>
      val heavy = heavyTypes.contains(tpe)
      if heavy then markBusy(+1, p)
      p.requestOwned(tpe, params, onEvent, owner).transform { r =>
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

  /** Inform the cancellation gate only after the owned envelope has been written. */
  private[engine] def requestTemporal(params: ujson.Value, onEvent: EngineEvent => Unit,
                                      owner: String, beforeDispatch: () => Future[Boolean],
                                      onDispatched: () => Unit): Future[ujson.Value] =
    ensureStarted().flatMap { p =>
      beforeDispatch().flatMap { admitted =>
        if !admitted then Future.failed(EngineError(Protocol.ErrorCodes.Interrupted, "interrupted before start"))
        else
          markBusy(+1, p)
          val result = p.requestOwned(Protocol.Types.TemporalLab, params, onEvent, Some(owner))
          onDispatched()
          result.transform { r =>
            markBusy(-1, p)
            r
          }(using ExecutionContext.parasitic)
      }
    }

  private[engine] def requestExtras(kind: String, params: ujson.Value, onEvent: EngineEvent => Unit,
      owner: String, beforeDispatch: () => Future[Boolean], onDispatched: () => Unit,
      onProcess: EngineProcess => Unit = _ => (), beforeWrite: () => Boolean = () => true): Future[ujson.Value] =
    ensureStarted().flatMap { p =>
      beforeDispatch().flatMap { admitted =>
        if !admitted then Future.failed(EngineError(Protocol.ErrorCodes.Interrupted, "Extras cancelled before start").markWorkerResponse())
        else
          onProcess(p)
          if !beforeWrite() then Future.failed(EngineError(Protocol.ErrorCodes.Interrupted, "Extras cancelled before write").markWorkerResponse())
          else
            markBusy(+1, p)
            val result = p.requestOwned(kind, params, onEvent, Some(owner))
            onDispatched()
            result.transform { r => markBusy(-1, p); r }(using ExecutionContext.parasitic)
      }
    }

  /** Reader controls bypass the worker/reload future and never block the calling (possibly FX) thread. */
  private[engine] def readerControl(tpe: String, params: ujson.Value): Future[ujson.Value] =
    require(tpe == Protocol.Types.InterruptionProtection)
    val result = Promise[ujson.Value]()
    try
      val timeout = scheduler.schedule(
        (() => { result.tryFailure(EngineError(EngineTimeout, "reader control timed out")); () }): Runnable,
        5, TimeUnit.SECONDS)
      result.future.onComplete(_ => timeout.cancel(false))(using ExecutionContext.parasitic)
      // A live reader is usable before the model reload completes. Snapshot it inside
      // the worker so calling this method never performs a pipe write on the FX thread.
      Future(proc.filter(_.isAlive)).flatMap {
        case Some(p) => Future.successful(p)
        case None => ensureStarted()
      }.flatMap { p =>
        if result.isCompleted then Future.failed(EngineError(EngineTimeout, "reader control timed out"))
        else p.request(tpe, params)
      }.onComplete(result.tryComplete)
    catch case NonFatal(e) => result.tryFailure(e)
    result.future

  /** Passive status lookup for one already-running process. Never starts/restarts an engine,
    * never migrates to another process, and never writes to stdio on the caller's FX thread.
    */
  def systemStatusIfRunning(expectedPid: Long, timeout: FiniteDuration = 5.seconds): Future[ujson.Value] =
    val target = lock.synchronized {
      proc.filter(p => !stopped && p.isAlive && p.pid == expectedPid)
    }
    target match
      case None => Future.failed(EngineError(EngineNotRunning, "Hardware status engine is no longer running"))
      case Some(p) =>
        val result = Promise[ujson.Value]()
        try
          val deadline = scheduler.schedule(
            (() => { result.tryFailure(EngineError(EngineTimeout, "Hardware status lookup timed out")); () }): Runnable,
            timeout.toMillis, TimeUnit.MILLISECONDS)
          Future {
            if result.isCompleted then throw EngineError(EngineTimeout, "Hardware status lookup timed out")
            val current = lock.synchronized(!stopped && proc.contains(p) && p.isAlive)
            if !current then throw EngineError(EngineNotRunning, "Hardware status engine changed before dispatch")
            p.request(Protocol.Types.SystemStatus)
          }.flatten.onComplete { response =>
            val current = lock.synchronized(!stopped && proc.contains(p) && p.isAlive)
            if current then result.tryComplete(response)
            else result.tryFailure(EngineError(EngineNotRunning, "Hardware status engine changed during lookup"))
            deadline.cancel(false)
          }(using ExecutionContext.parasitic)
        catch case NonFatal(error) => result.tryFailure(error)
        result.future

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
  /** Admission and backend cancellation are one reader operation. No timer exists until permission arrives. */
  def requestInterrupt(owner: Option[String] = None): Future[InterruptionDecision] =
    captureInterrupt(owner)()

  private[engine] def interruptAfterStartupOwned(owner: String, restartOnTimeout: Boolean = true): Future[InterruptionDecision] =
    ensureStarted().flatMap(_ => captureInterrupt(Some(owner), restartOnTimeout)())

  /** Capture before asynchronous dispatch: neither a later request nor a replacement
    * process can become the target of this already initiated user action.
    */
  private[engine] def captureInterrupt(owner: Option[String] = None, restartOnTimeout: Boolean = true): () => Future[InterruptionDecision] =
    val captured = proc.filter(_.isAlive).map { p =>
      val target = p.pendingIdsFor(owner).find((_, t) => !readerTypes.contains(t))
      (p, target)
    }
    () => captured match
      case None => Future.successful(InterruptionDecision(true, false, InterruptionProtection()))
      case Some((p, _)) if !proc.contains(p) || !p.isAlive =>
        Future.successful(InterruptionDecision(true, false, InterruptionProtection()))
      case Some((p, target)) =>
        // An empty unowned snapshot is deliberately not an unscoped interrupt.
        // Native request IDs are r<integer>, so this sentinel can never match.
        val params = target.fold(if owner.isDefined then ujson.Obj() else ujson.Obj("target_id" -> "no-captured-request"))(
          (id, _) => ujson.Obj("target_id" -> id))
        owner.foreach(value => params("owner_id") = value)
        p.request(Protocol.Types.Interrupt, params).map { raw =>
          val decision = Json.fromJs[InterruptionDecision](raw)
          require(decision.validAdmission, "Invalid interruption admission response")
          if decision.blocked then log(decision.message.getOrElse("Interruptions are blocked"))
          else if restartOnTimeout && decision.accepted && decision.affected then target.filter { (id, _) =>
            // A queued request is cancelled immediately in reader state but may
            // await unrelated active work before its error envelope is emitted.
            // Only actual active admission may arm a forced-restart grace timer.
            raw.obj.get("active_id").contains(ujson.Str(id))
          }.foreach { (id, tpe) =>
            val grace = if loadTypes.contains(tpe) then loadInterruptGrace else interruptGrace
            val task = scheduler.schedule(
              (() => {
                if proc.contains(p) && p.isAlive && p.isPending(id) then
                  log(s"[sdfx] engine did not stop $tpe within $grace; restarting it")
                  forceRestartIfCurrent(p, EngineError(Protocol.ErrorCodes.Interrupted, "generation interrupted (engine restarted)"))
              }): Runnable,
              grace.toMillis, TimeUnit.MILLISECONDS)
            p.futureOf(id) match
              case Some(f) => f.onComplete(_ => task.cancel(false))(using ExecutionContext.parasitic)
              case None => task.cancel(false)
          }
          decision
        }

  /** Compatibility callback: callers needing the outcome use requestInterrupt rather than assume cancellation. */
  def interrupt(): Unit = interruptOwned(None)

  private[engine] def interruptOwned(owner: Option[String]): Unit =
    requestInterrupt(owner).failed.foreach(e => log(s"[sdfx] interrupt failed: ${e.getMessage}"))

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

  /** Stop only the process captured for an owned request; stale timeouts cannot kill replacements. */
  private[engine] def processSnapshot: Option[EngineProcess] = lock.synchronized(proc.filter(_.isAlive))

  private[engine] def forceRestartIfCurrent(expected: EngineProcess, reason: EngineError): Future[Boolean] =
    val removed = lock.synchronized {
      if stopped || !proc.contains(expected) then false
      else
        proc = None
        inflight = 0
        true
    }
    if !removed then Future.successful(false)
    else
      expected.kill(Some(reason))
      gaveUp = false
      Future {
        try Await.ready(expected.exitFuture, 5.seconds)
        catch case NonFatal(_) => ()
      }.flatMap(_ => ensureStarted()).map(_ => true)

  /** Restarts the engine gracefully (e.g. after settings change). */
  def restart(): EngineHello =
    val (requested, requestedEpoch) = lock.synchronized((proc, epoch))
    requested.filter(_.isAlive).foreach { p =>
      val decision = Json.fromJs[InterruptionDecision](Await.result(p.request(Protocol.Types.RestartAuthorization), 5.seconds))
      require(decision.accepted != decision.blocked, "Invalid restart admission response")
      if !decision.accepted then throw EngineError(Protocol.ErrorCodes.InterruptionBlocked,
        decision.message.getOrElse("Interruptions are blocked"))
    }
    val (old, restartEpoch) = lock.synchronized {
      if epoch != requestedEpoch then throw EngineError(EngineNotRunning, "restart was superseded by stop")
      if proc != requested then throw EngineError(EngineCrashed, "Engine changed during restart authorization; retry")
      // Establish this explicit start intent before waiting for shutdown. A later
      // stop increments the epoch and cannot be undone by the continuation.
      stopped = false
      gaveUp = false
      crashTimes = Nil
      val o = proc
      proc = None
      inflight = 0
      // An installed live reader may still have a remembered-model initializer
      // in flight. That attempt belongs to the process being detached, not its
      // replacement: invalidate it before closing so late completion cannot
      // clear the replacement future or publish its terminal state.
      epoch += 1
      startFuture = None
      restartRequested = false
      (o, epoch)
    }
    old.foreach(_.close(shutdownGrace))
    awaitStarted(ensureStarted(Some(restartEpoch)))

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
