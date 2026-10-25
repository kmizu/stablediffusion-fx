package sdfx.engine

import sdfx.core.config.{AppHome, Settings}

import java.io.{InputStream, InputStreamReader}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.{ConcurrentHashMap, CopyOnWriteArrayList, TimeUnit}
import scala.concurrent.duration.*
import scala.concurrent.{ExecutionContext, Future}
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

/** A running [[PythonEnv.bootstrapAsync]]; `cancel()` kills the current pip/venv child process. */
final class BootstrapTask private[engine] (val result: Future[Unit], cancelFlag: AtomicBoolean, procs: CopyOnWriteArrayList[Process]):
  def cancel(): Unit =
    cancelFlag.set(true)
    procs.forEach(p => PythonEnv.destroyTree(p))
  def isCancelled: Boolean = cancelFlag.get()

/** Locates the Python interpreter for the engine and bootstraps `python/.venv`. */
object PythonEnv:
  val isWindows: Boolean = System.getProperty("os.name").toLowerCase.contains("win")
  val TorchIndexUrl = "https://download.pytorch.org/whl/cu128"
  val TorchRequirements = "requirements-torch.txt"
  val Requirements = "requirements.txt"
  val StampFile = ".sdfx-stamp"
  val ProbeTimeout: FiniteDuration = 20.seconds
  /** Per-command limit during bootstrap (torch wheels are large). */
  val BootstrapCommandTimeout: FiniteDuration = 2.hours
  private val PipFlags = Seq("--progress-bar", "off", "--disable-pip-version-check")

  def venvDir(home: AppHome): Path = home.pythonDir.resolve(".venv")

  def venvPython(home: AppHome): Path =
    if isWindows then venvDir(home).resolve("Scripts").resolve("python.exe")
    else venvDir(home).resolve("bin").resolve("python")

  // Only successes are cached: a failed probe (still installing, AV scan, timeout under load) is retried next time.
  private val probeCache = ConcurrentHashMap.newKeySet[Seq[String]]()

  /** True when `cmd --version` runs successfully within [[ProbeTimeout]] (successes cached). */
  def works(cmd: Seq[String]): Boolean =
    if probeCache.contains(cmd) then true
    else
      val ok = probe(cmd)
      if ok then probeCache.add(cmd)
      ok

  private def probe(cmd: Seq[String]): Boolean =
    try
      val p = new ProcessBuilder((cmd :+ "--version").asJava).redirectErrorStream(true).start()
      drain(p.getInputStream, _ => ())
      if p.waitFor(ProbeTimeout.toMillis, TimeUnit.MILLISECONDS) then p.exitValue() == 0
      else
        destroyTree(p)
        false
    catch case NonFatal(_) => false

  /** Reads a child's output on a daemon thread, splitting on \n and \r (pip progress), dropping blank lines. */
  private def drain(in: InputStream, line: String => Unit): Thread =
    val t = new Thread(() =>
      try
        val r = new InputStreamReader(in, StandardCharsets.UTF_8)
        val sb = new StringBuilder
        val buf = new Array[Char](8192)
        var n = r.read(buf)
        while n >= 0 do
          var i = 0
          while i < n do
            val c = buf(i)
            if c == '\n' || c == '\r' then
              if sb.nonEmpty then line(sb.toString)
              sb.clear()
            else sb += c
            i += 1
          n = r.read(buf)
        if sb.nonEmpty then line(sb.toString)
      catch case NonFatal(_) => ()
    , "sdfx-python-output")
    t.setDaemon(true)
    t.start()
    t

  private[engine] def destroyTree(p: Process): Unit =
    try
      p.descendants().forEach(h => { h.destroyForcibly(); () })
      p.destroyForcibly()
    catch case NonFatal(_) => ()

  /** System interpreters in preference order (no venv). */
  def systemCandidates: Seq[Seq[String]] =
    (if isWindows then Seq(Seq("py", "-3.12"), Seq("py", "-3.10")) else Nil) ++
      Seq(Seq("python3"), Seq("python"))

  def findSystemPython(): Option[Seq[String]] = systemCandidates.find(works)

  private def configured(settings: Settings): Option[Seq[String]] = settings.pythonExe.map(_.trim).filter(_.nonEmpty).map(Seq(_))

  /** A message when `settings.pythonExe` is set but cannot be run. */
  def configuredProblem(settings: Settings): Option[String] =
    configured(settings).filterNot(works).map(c => s"configured python_exe '${c.mkString(" ")}' does not run")

  /** settings.pythonExe (exclusively, when set), else the app venv, else system interpreters.
    * An invalid configured interpreter yields None instead of silently using another Python.
    */
  def locate(home: AppHome, settings: Settings): Option[Seq[String]] =
    configured(settings) match
      case Some(c) => Some(c).filter(works)
      case None =>
        Some(venvPython(home)).filter(Files.isRegularFile(_)).map(p => Seq(p.toString)).filter(works).orElse(findSystemPython())

  def engineCommand(home: AppHome, settings: Settings, dryRun: Boolean): Seq[String] =
    val py = locate(home, settings).getOrElse(
      throw new IllegalStateException(configuredProblem(settings).getOrElse(
        "no Python interpreter found (set python_exe in settings or install Python 3.10+)"))
    )
    py ++ Seq("-m", "sdfx") ++ (if dryRun then Seq("--dry-run") else Nil)

  def engineEnv(settings: Settings): Map[String, String] =
    Map("PYTHONUNBUFFERED" -> "1", "PYTHONIOENCODING" -> "utf-8") ++
      (if settings.hfOffline then Map("HF_HUB_OFFLINE" -> "1", "TRANSFORMERS_OFFLINE" -> "1") else Map.empty) ++
      // custom_code (A1111 --allow-code): only when the user opted in; the engine refuses run_code otherwise
      (if settings.allowCustomCode then Map("SDFX_ALLOW_CODE" -> "1") else Map.empty)

  def engineLaunch(home: AppHome, settings: Settings, dryRun: Boolean): EngineLaunch =
    EngineLaunch(engineCommand(home, settings, dryRun), home.pythonDir, engineEnv(settings))

  /** Python version recorded in the venv's pyvenv.cfg ("" when there is no venv). */
  def venvPythonVersion(home: AppHome): String =
    val cfg = venvDir(home).resolve("pyvenv.cfg")
    try
      if !Files.isRegularFile(cfg) then ""
      else
        Files.readAllLines(cfg).asScala.map(_.trim).collectFirst {
          case l if l.startsWith("version") && l.contains("=") => l.substring(l.indexOf('=') + 1).trim.split('.').take(2).mkString(".")
        }.getOrElse("")
    catch case NonFatal(_) => ""

  /** sha256 over both requirement files, the torch index URL and the venv Python major.minor. */
  def requirementsStamp(home: AppHome): String =
    val md = MessageDigest.getInstance("SHA-256")
    Seq(TorchRequirements, Requirements).foreach { n =>
      val f = home.pythonDir.resolve(n)
      md.update(n.getBytes(StandardCharsets.UTF_8))
      if Files.isRegularFile(f) then md.update(Files.readAllBytes(f))
    }
    md.update(TorchIndexUrl.getBytes(StandardCharsets.UTF_8))
    md.update(venvPythonVersion(home).getBytes(StandardCharsets.UTF_8))
    md.digest().map(b => f"$b%02x").mkString

  def stampPath(home: AppHome): Path = venvDir(home).resolve(StampFile)

  /** True when the venv is missing or its requirements stamp is outdated. */
  def needsBootstrap(home: AppHome): Boolean =
    !Files.isRegularFile(venvPython(home)) ||
      !Files.isRegularFile(stampPath(home)) ||
      Files.readString(stampPath(home)).trim != requirementsStamp(home)

  /** The commands [[bootstrap]] runs after the venv exists, in order (torch first, from the CUDA index). */
  def bootstrapCommands(home: AppHome): Seq[Seq[String]] =
    val py = venvPython(home).toString
    Seq(Seq(py, "-m", "pip", "install", "--upgrade", "pip") ++ PipFlags) ++
      Option.when(Files.isRegularFile(home.pythonDir.resolve(TorchRequirements)))(
        Seq(py, "-m", "pip", "install", "-r", TorchRequirements, "--index-url", TorchIndexUrl) ++ PipFlags) ++
      Option.when(Files.isRegularFile(home.pythonDir.resolve(Requirements)))(
        Seq(py, "-m", "pip", "install", "-r", Requirements) ++ PipFlags)

  /** Creates `python/.venv` and installs requirements. Blocking; streams output to `log`. Throws on failure. */
  def bootstrap(home: AppHome, log: String => Unit, settings: Settings = Settings()): Unit =
    bootstrapImpl(home, log, settings, new AtomicBoolean(false), new CopyOnWriteArrayList[Process]())

  /** Cancellable [[bootstrap]] on a background thread. */
  def bootstrapAsync(home: AppHome, log: String => Unit, settings: Settings = Settings()): BootstrapTask =
    val flag = new AtomicBoolean(false)
    val procs = new CopyOnWriteArrayList[Process]()
    val f = Future(bootstrapImpl(home, log, settings, flag, procs))(using ExecutionContext.global)
    new BootstrapTask(f, flag, procs)

  private def bootstrapImpl(home: AppHome, log: String => Unit, settings: Settings, cancel: AtomicBoolean, procs: CopyOnWriteArrayList[Process]): Unit =
    val vpy = venvPython(home)
    if !Files.isRegularFile(vpy) then
      val base = configured(settings).filter(works).orElse(findSystemPython()).getOrElse(
        throw new IllegalStateException("no system Python found to create the virtual environment")
      )
      log(s"creating virtual environment with ${base.mkString(" ")}")
      runImpl(base ++ Seq("-m", "venv", venvDir(home).toString), home.pythonDir, log, BootstrapCommandTimeout, cancel, procs)
    bootstrapCommands(home).foreach(c => runImpl(c, home.pythonDir, log, BootstrapCommandTimeout, cancel, procs))
    Files.writeString(stampPath(home), requirementsStamp(home))
    log("python environment ready")

  def run(cmd: Seq[String], cwd: Path, log: String => Unit): Unit =
    runImpl(cmd, cwd, log, BootstrapCommandTimeout, new AtomicBoolean(false), new CopyOnWriteArrayList[Process]())

  private def runImpl(cmd: Seq[String], cwd: Path, log: String => Unit, timeout: FiniteDuration, cancel: AtomicBoolean,
      procs: CopyOnWriteArrayList[Process]): Unit =
    if cancel.get() then throw new InterruptedException("bootstrap cancelled")
    log(s"$$ ${cmd.mkString(" ")}")
    val pb = new ProcessBuilder(cmd.asJava).directory(cwd.toFile).redirectErrorStream(true)
    pb.environment().put("PYTHONIOENCODING", "utf-8")
    pb.environment().put("PYTHONUNBUFFERED", "1")
    val p = pb.start()
    procs.add(p)
    try
      val reader = drain(p.getInputStream, log)
      val finished = p.waitFor(timeout.toMillis, TimeUnit.MILLISECONDS)
      if !finished then
        destroyTree(p)
        throw new IllegalStateException(s"command timed out after $timeout: ${cmd.mkString(" ")}")
      reader.join(2000)
      if cancel.get() then throw new InterruptedException("bootstrap cancelled")
      val code = p.exitValue()
      if code != 0 then throw new IllegalStateException(s"command failed (exit $code): ${cmd.mkString(" ")}")
    finally procs.remove(p)

  /** Absolute path helper for user supplied strings. */
  def absolute(home: AppHome, p: String): Path = home.resolve(p).toAbsolutePath.normalize()
