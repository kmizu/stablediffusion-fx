package sdfx.engine

import sdfx.core.config.{AppHome, Settings}
import sdfx.core.gallery.GalleryIndex
import sdfx.core.plugins.api.Plugin
import sdfx.core.protocol.*

import java.nio.file.Files
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
    installShutdownHook: Boolean = true
):
  @volatile private var _settings = initialSettings
  private val logListeners = new CopyOnWriteArrayList[String => Unit]()
  private given ExecutionContext = ExecutionContext.global

  def settings: Settings = _settings

  /** Updates (and saves) the settings. */
  def updateSettings(f: Settings => Settings): Settings = synchronized {
    _settings = f(_settings)
    if persistSettings then
      try Settings.save(home.settingsFile, _settings)
      catch case NonFatal(e) => log(s"[sdfx] cannot save settings: ${e.getMessage}")
    _settings
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
  val generator: Generator = new Generator(client, catalog, home, () => _settings, timeouts = timeouts)
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
    try queue.shutdown()
    catch case NonFatal(_) => ()
    try supervisor.close()
    catch case NonFatal(_) => ()
    registry.close()
