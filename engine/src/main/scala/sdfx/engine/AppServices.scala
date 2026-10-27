package sdfx.engine

import sdfx.core.config.{AppHome, Settings, SettingsRecovery, SettingsSaveIntent}
import sdfx.core.gallery.GalleryIndex
import sdfx.core.plugins.api.Plugin
import sdfx.core.protocol.*

import java.nio.file.{Files, Path}
import java.util.concurrent.CopyOnWriteArrayList
import scala.concurrent.duration.*
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.util.control.NonFatal

/** Wires the engine side of the app together. Shared by the JavaFX UI and the MCP server.
  *
  * @param launchOverride replaces the Python discovery (tests, custom engines)
  */
final class AppServices(
    val home: AppHome,
    initialSettings: Settings,
    val dryRun: Boolean,
    builtins: Seq[Plugin],
    launchOverride: Option[() => EngineLaunch] = None,
    persistSettings: Boolean = true,
    timeouts: EngineTimeouts = EngineTimeouts(),
    installShutdownHook: Boolean = true,
    initialRecovery: Option[SettingsRecovery] = None,
    settingsWriter: (Path, Settings) => Unit = Settings.save
):
  @volatile private var _settings = initialSettings
  @volatile private var _settingsRecovery = initialRecovery
  private var holdLogged = false
  private val logListeners = new CopyOnWriteArrayList[String => Unit]()
  private given ExecutionContext = ExecutionContext.global

  def settings: Settings = _settings
  def settingsRecovery: Option[SettingsRecovery] = _settingsRecovery
  val cardPreviews: CardPreviewSession = new CardPreviewSession(initialSettings.nsfwCardBlurDefault)

  /** Updates (and saves) the settings. */
  def updateSettings(f: Settings => Settings): Settings = synchronized {
    _settings = f(_settings)
    if persistSettings && _settingsRecovery.nonEmpty then
      if !holdLogged then
        holdLogged = true
        log(s"[sdfx] settings persistence held: ${_settingsRecovery.get.reason}")
    else if persistSettings then
      try
        Settings.validatePreviewConfiguration(_settings).fold(reason => throw new IllegalArgumentException(reason), _ => ())
        settingsWriter(home.settingsFile, _settings)
      catch case NonFatal(e) => log(s"[sdfx] cannot save settings: ${e.getMessage}")
    _settings
  }

  /** Checked changes are persisted before publication; recovery requires a real successful write. */
  def updateSettingsChecked(f: Settings => Settings, intent: SettingsSaveIntent = SettingsSaveIntent.Explicit): Either[String, Settings] = synchronized {
    if _settingsRecovery.nonEmpty && intent == SettingsSaveIntent.Explicit then
      Left(s"settings recovery required: ${_settingsRecovery.get.reason}")
    else if intent == SettingsSaveIntent.Recover && !persistSettings then
      Left("cannot recover settings without persistence")
    else
      try
        val candidate = f(_settings)
        Settings.validatePreviewConfiguration(candidate) match
          case Left(reason) => Left(reason)
          case Right(_) =>
            if persistSettings then settingsWriter(home.settingsFile, candidate)
            _settings = candidate
            if intent == SettingsSaveIntent.Recover then
              _settingsRecovery = None
              holdLogged = false
            Right(candidate)
      catch case NonFatal(e) => Left(s"cannot save settings: ${Option(e.getMessage).getOrElse(e.toString)}")
  }

  def addLogListener(l: String => Unit): () => Unit =
    logListeners.add(l)
    () => { logListeners.remove(l); () }

  def log(line: String): Unit =
    if logListeners.isEmpty then System.err.println(line)
    else logListeners.forEach(l =>
      try l(line)
      catch case NonFatal(_) => ()
    )

  val supervisor: EngineSupervisor = new EngineSupervisor(
    launchOverride.getOrElse(() => PythonEnv.engineLaunch(home, _settings, dryRun)),
    log
  )
  val client: EngineClient = new EngineClient(supervisor)
  val catalog: ModelCatalog = new ModelCatalog(client, home, () => _settings)
  val modelCards: ModelCardCatalog = new ModelCardCatalog(home, () => _settings, () => catalog.snapshot)
  val payloads: RequestPayloadStore = new RequestPayloadStore()
  val generator: Generator = new Generator(client, catalog, home, () => _settings, timeouts = timeouts, payloads = payloads)
  val queue: JobQueue = new JobQueue(generator, owner => supervisor.requestInterrupt(owner))
  /** Global user interrupt: capture owned queue work or the actual nonqueue request
    * now, then perform admission I/O off the caller/FX thread. Queue-specific APIs
    * continue to affect only the native entry they captured.
    */
  def interruptRuntime(): Future[QueueInterruption] =
    val backend = supervisor.captureInterrupt()
    val dispatch = queue.captureRuntimeInterrupt(backend)
    Future(dispatch()).flatten

  val registry: PluginRegistry = new PluginRegistry(builtins, home, () => _settings)
  def gallery: GalleryIndex = galleryIndex
  @volatile private var galleryIndex = new GalleryIndex(home.outputs(_settings))

  // Ensures the Python child does not outlive the JVM when shutdown() is never reached.
  private val shutdownHook: Option[Thread] =
    if !installShutdownHook then None
    else
      val t = new Thread(() =>
        try supervisor.close()
        catch case _: Throwable => ()
      , "sdfx-engine-shutdown")
      try
        Runtime.getRuntime.addShutdownHook(t)
        Some(t)
      catch case NonFatal(_) => None

  /** Starts the engine, refreshes the catalog and loads `settings.lastModel` / `lastVae`. Blocking.
    * Stale engine temp files in outputs/.tmp are removed first.
    */
  def start(): EngineHello =
    val cleaned = generator.cleanTempDir()
    if cleaned > 0 then log(s"[sdfx] removed $cleaned stale temp files")
    val hello = supervisor.start()
    try catalog.refresh()
    catch case NonFatal(e) => log(s"[sdfx] model list refresh failed: ${e.getMessage}")
    _settings.lastModel.filter(_.nonEmpty).foreach { m =>
      try loadCheckpoint(m, remember = false)
      catch case NonFatal(e) => log(s"[sdfx] cannot load last model '$m': ${e.getMessage}")
    }
    _settings.lastVae.filter(v => v.nonEmpty && v != "Automatic" && v != "None").foreach { v =>
      try setVae(Some(v), remember = false)
      catch case NonFatal(e) => log(s"[sdfx] cannot set last VAE '$v': ${e.getMessage}")
    }
    hello

  def startAsync(): Future[EngineHello] = Future(start())

  /** Loads a checkpoint by catalog name/title/path and remembers it as `lastModel`. */
  def loadCheckpoint(name: String, remember: Boolean = true): LoadedModel =
    val path = catalog.findCheckpoint(name).map(_.path)
      .orElse(Some(home.resolve(name)).filter(f => Files.isRegularFile(f) && generator.withinDirs(f, _settings.modelDirs.checkpoints)).map(_.toString))
      .getOrElse(throw EngineError(Protocol.ErrorCodes.InvalidParams, s"checkpoint not found: $name"))
    val s = _settings
    val m = awaitOrRestart(client.loadModel(LoadModelRequest(path, s.dtype, s.vramMode, None, s.vaeTiling)), timeouts.load, "load_model")
    catalog.rememberHash(m)
    if remember then updateSettings(_.copy(lastModel = Some(catalog.findCheckpoint(name).map(_.name).getOrElse(name))))
    m

  /** Sets the VAE (None / "Automatic" = checkpoint default) and remembers it as `lastVae`. */
  def setVae(name: Option[String], remember: Boolean = true): LoadedModel =
    val path = name.filter(n => n.nonEmpty && n != "Automatic" && n != "None").map { n =>
      catalog.findVae(n).map(_.path).getOrElse(throw EngineError(Protocol.ErrorCodes.InvalidParams, s"VAE not found: $n"))
    }
    val m = awaitOrRestart(client.setVae(path), timeouts.load, "set_vae")
    if remember then updateSettings(_.copy(lastVae = name))
    m

  private def awaitOrRestart[T](f: Future[T], timeout: FiniteDuration, what: String): T =
    try Await.result(f, timeout)
    catch
      case _: java.util.concurrent.TimeoutException =>
        val err = EngineError(EngineProcess.EngineTimeout, s"$what did not finish within $timeout")
        supervisor.forceRestart(err)
        throw err

  /** Re-creates the gallery index after the output directory changed. */
  def resetGallery(): GalleryIndex =
    galleryIndex = new GalleryIndex(home.outputs(_settings))
    galleryIndex

  def shutdown(): Unit =
    shutdownHook.foreach(t =>
      try Runtime.getRuntime.removeShutdownHook(t)
      catch case _: IllegalStateException => ()
    )
    try modelCards.close()
    catch case NonFatal(_) => ()
    try queue.shutdown()
    catch case NonFatal(_) => ()
    try supervisor.close()
    catch case NonFatal(_) => ()
    registry.close()
    payloads.close()
